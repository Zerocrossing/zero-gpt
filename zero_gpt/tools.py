from abc import ABC, abstractmethod

from instructor import OpenAISchema
from pydantic import BaseModel

from zero_gpt.models import ChatMessage


class OpenAIMessageTool(BaseModel, ABC):
    """OpenAI Message Tool

    This class defines a tool that may be called by the openai agent.
    It has a run method that invokes some logic and returns a Message to the agent.

    The inputs to the function are defined in the FunctionInputs class.
    This derives from Instructors OpenAISchema class which is a Pydantic model.

    For best results, give a descriptive docstring to the FunctionInputs or the outer class.
    Additionally, you should use Pydantic's Field() to give descriptions to all fields on the FunctionInputs

    If you are looking for your agent to return structured output instead of text,
    you should use a different class.
    """

    class FunctionInputs(OpenAISchema):
        pass

    @abstractmethod
    def run(self, inputs: OpenAISchema) -> ChatMessage | str:
        pass

    @classmethod
    def openai_schema(cls):
        schema = cls.FunctionInputs.openai_schema
        # rename the 'name' field to the child class name
        schema["name"] = cls.__name__
        # check if there is no docstring on inputmodel, if so set the schema 'description' to the outer class description
        if not cls.FunctionInputs.__doc__:
            schema["description"] = cls.__doc__
        return {"type": "function", "function": schema}

    @classmethod
    def tool_name(cls):
        return cls.__name__
