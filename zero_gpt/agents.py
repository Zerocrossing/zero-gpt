from typing import List, Type, TypeVar, overload

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import BaseModel
from instructor import OpenAISchema

from zero_gpt.storage import load_history, save_messages

from .models import ChatHistory, ChatMessage, ChatRole, Voices
from .settings import settings
from .tools import OpenAIMessageTool


def get_client():
    """Get the OpenAI client.

    Uses the system OPENAI_API_KEY environment variable to authenticate.
    """
    return OpenAI()


StructuredResponseType = TypeVar("StructuredResponseType", bound=BaseModel)


class OpenAIChatAgent:
    """OpenAI Chat Agent

    A simple chat agent that uses the OpenAI API to generate responses.
    This base class should be subclassed to add tools and other functionality.
    """

    def __init__(self, user_id: str | None = None):
        self.user_id = user_id
        self.client = get_client()
        self.model = self._get_model()
        self.prompt = self._get_prompt()
        self.history: ChatHistory = self._get_history()
        self.tools: List[OpenAIMessageTool] = self._get_tools()
        self._outgoing_messages: List[ChatMessage] = []
        self._response_model: Type[BaseModel] | None = None
        self._voice: Voices | None = None

    # region internal methods

    def _get_history(self):
        if self.user_id is None:
            return ChatHistory()
        return load_history(self.user_id)

    def _save_messages(self, additional_messages: List[ChatMessage] | None = None):
        """Saves outgoing messages that should be added to the history as well as any passed messages"""
        if self.user_id is None:
            return

        to_save = [msg for msg in self._outgoing_messages if msg.include_in_history]
        if additional_messages:
            to_save.extend(additional_messages)
        save_messages(self.user_id, to_save)

    def _get_model(self):
        return settings.default_gpt_model

    def _get_prompt(self):
        return settings.default_prompt

    def _make_prompt_message(self):
        prompt = self.prompt
        return ChatMessage(role=ChatRole.system, content=prompt)

    def _get_tools(self):
        return []

    def _format_tools_for_openai(self):
        tools = [tool.openai_schema() for tool in self.tools]
        if not tools:
            return None
        return tools

    def _get_message_from_tool_call(self, tool_call):
        function_name = tool_call.function.name
        # find the tool with the matching name
        chosen_tool = None
        for tool in self.tools:
            tool_name = tool.tool_name()
            if tool_name == function_name:
                chosen_tool = tool
                break
        if not chosen_tool:
            raise ValueError(f"Tool {function_name} not found")
        arguments = tool_call.function.arguments
        function_inputs = chosen_tool.FunctionInputs.model_validate_json(arguments)
        tool_message = chosen_tool.run(function_inputs)
        # if tool message is a string, convert it to a ChatMessage
        if isinstance(tool_message, str):
            tool_message = ChatMessage(role=ChatRole.tool, content=tool_message)
        return tool_message

    def _openai_chat_completion(self, messages):
        args = {"model": self.model, "messages": messages}
        if self.tools:
            args["tools"] = self._format_tools_for_openai()

        if self._response_model:
            # the new parse feature is sketchy, but this works at the moment
            if args.get("tools"):
                for tool in args.get("tools"):
                    tool["function"]["strict"] = True
                    tool["function"]["parameters"]["additionalProperties"] = False

            args["response_format"] = self._response_model
            completion = self.client.beta.chat.completions.parse(**args)
            return completion

        elif self._voice is not None:
            args = {
                "model": "gpt-4o-audio-preview",
                "tools": self._format_tools_for_openai(),
                "modalities": ["text", "audio"],
                "audio": {"voice": self._voice.value, "format": "mp3"},
                "messages": messages,
            }
            return self.client.chat.completions.create(**args)

        else:
            args = {
                "messages": messages,
                "model": self.model,
                "tools": self._format_tools_for_openai(),
            }
            return self.client.chat.completions.create(**args)

    def _construct_messages(self):
        """Creates the full message list to send to openAI"""
        messages = []
        messages.append(self._make_prompt_message().as_openai)
        messages.extend(self.history.as_openai)
        messages.extend([message.as_openai for message in self._outgoing_messages])
        return messages

    def _handle_tool_calls(self, messages, completion):
        while completion.choices[0].finish_reason == "tool_calls":
            tool_calls = completion.choices[0].message.tool_calls
            if not tool_calls:
                break
            # add the openai tool calling message to the history (temporarily)
            messages.append(completion.choices[0].message)
            for tool_call in tool_calls:
                tool_message = self._get_message_from_tool_call(tool_call)
                tool_response_message = tool_message.as_openai
                # append the tool_call_id
                tool_response_message["tool_call_id"] = tool_call.id
                messages.append(tool_response_message)
            completion = self._openai_chat_completion(messages)
        return completion

    # endregion internal methods

    def add_message(self, message: ChatMessage):
        """Add Message

        Adds a message to be sent to the agent.
        Messages added in this way won't be sent until `send_messages` is called.

        This is useful for adding multiple messages, in particular context messages
        or messages from multiple 'users' (using OpenAI's name parameter)
        some of which you may not want included in the message history
        """
        self._outgoing_messages.append(message)

    @overload
    def send_messages(self) -> str | None: ...

    @overload
    def send_messages(
        self, response_model: Type[StructuredResponseType]
    ) -> StructuredResponseType | None: ...

    def send_messages(
        self,
        response_model: Type[StructuredResponseType] | None = None,
        messages: List[ChatMessage] | None = None,
    ) -> str | StructuredResponseType | None:
        """Send Messages

        Sends all messages in the outgoing queue to the agent.
        The user messages and agent response will be added to the history
        unless the messages have `include_in_history` set to false.

        Optinal messages passed to this function will be added to the queue.
        """
        # send messages
        if messages:
            self._outgoing_messages.extend(messages)
        if response_model:
            self._response_model = response_model
        messages = self._construct_messages()
        completion: ChatCompletion = self._openai_chat_completion(messages)
        completion = self._handle_tool_calls(messages, completion)

        # handle history
        for message in self._outgoing_messages:
            if not message.include_in_history:
                continue
            self.history.add_message(message)
        agent_message = ChatMessage.from_openai(completion.choices[0].message)
        self.history.add_message(agent_message)

        # save history, clear queue, reset response model, return
        self._save_messages([agent_message])
        self._outgoing_messages.clear()
        self._response_model = None

        # parse output if required
        response_content = completion.choices[0].message.content
        if response_model and isinstance(response_content, str):
            try:
                return response_model.model_validate_json(response_content)
            except Exception as e:
                raise ValueError(
                    "Response could not be parsed into the structured model.", e
                )
        return response_content

    @overload
    def send_message(self, user_message: ChatMessage | str) -> str | None: ...

    @overload
    def send_message(
        self,
        user_message: ChatMessage | str,
        response_model: Type[StructuredResponseType],
    ) -> StructuredResponseType | None: ...

    def send_message(
        self,
        user_message: ChatMessage | str,
        response_model: Type[StructuredResponseType] | None = None,
    ) -> str | StructuredResponseType | None:
        """Send a single message to the chat agent and get the text of the response"""

        if isinstance(user_message, str):
            user_message = ChatMessage(role=ChatRole.user, content=user_message)

        self.add_message(user_message)
        if response_model:
            return self.send_messages(response_model=response_model)
        return self.send_messages()

    def send_messages_audio_response(
        self, messages: List[ChatMessage] | None = None, voice: Voices = Voices.ASH
    ) -> ChatCompletionMessage:
        """Send all messages to the chat agent and get the audio response

        The response object includes the audio as well as the text.
        """
        if messages:
            self._outgoing_messages.extend(messages)
        messages = self._construct_messages()
        completion: ChatCompletion = self._openai_chat_completion(messages)
        completion: ChatCompletion = self._handle_tool_calls(messages, completion)
        if not isinstance(completion, ChatCompletion):
            raise ValueError("Expected a ChatCompletion from OpenAI")

        # handle history
        for message in self._outgoing_messages:
            if not message.include_in_history:
                continue
            self.history.add_message(message)
        completion_message = completion.choices[0].message
        if not hasattr(completion_message, "audio") or not completion_message.audio:
            raise ValueError("Expected an audio response from OpenAI")
        agent_message = ChatMessage(
            role=ChatRole.assistant, 
            content=completion_message.audio.transcript
        )
        self.history.add_message(agent_message)

        # save history, clear queue, reset voice model, return
        self._save_messages([agent_message])
        self._outgoing_messages.clear()
        self._voice = None

        # parse output if required
        return completion_message

    def send_message_audio_response(
        self, user_message: ChatMessage | str, voice: Voices | str = Voices.ASH
    ) -> ChatCompletionMessage:
        """Send a single message to the chat agent and get the audio response"""
        if isinstance(user_message, str):
            user_message = ChatMessage(role=ChatRole.user, content=user_message)

        if isinstance(voice, str):
            voice = Voices(voice)
        self._voice = voice

        self.add_message(user_message)
        return self.send_messages_audio_response(voice=voice)
