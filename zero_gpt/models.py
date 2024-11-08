from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRole(str, Enum):
    """Chat Role

    The different types of "Roles" an openAI message can have.
    """

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ChatMessage(BaseModel):
    """Chat Message

    Intended to work with OpenAI messages, this class
    Is a simple wrapper for passing message data around.
    """

    model_config = ConfigDict(from_attributes=True)

    role: ChatRole = Field(
        ...,
        description="The role of the message in the conversation. Can be 'system', 'user', 'assistant', or 'tool'.",
    )
    content: str = Field(
        ...,
        description="The actual content of the message. This is the text that will be processed.",
    )
    name: Optional[str] = Field(
        default=None,
        description="An optional name identifier for the message. May not contain spaces.",
    )
    include_in_history: bool = Field(
        default=True,
        description="Determines whether this message should be included in the conversation history.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="The timestamp when the message was created. Defaults to the current time.",
    )

    @field_validator("role", mode="before")
    def infer_role(cls, v) -> ChatRole:
        if isinstance(v, str):
            return ChatRole(v)
        return v

    @field_validator("name", mode="before")
    def no_spaces_in_names(cls, v: str | None):
        if v is None:
            return v
        if " " in v:
            raise ValueError("Name cannot contain spaces")
        return v

    @property
    def as_openai(self):
        """Return the message formatted for openai's API"""
        msg = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            msg["name"] = self.name
        return msg

    @classmethod
    def from_openai(cls, msg):
        return cls(
            role=msg.role,
            content=msg.content,
        )


class ChatHistory(BaseModel):
    """Chat History

    A simple wrapper for passing around a list of ChatMessages.
    Has a method to return the entire history as a list of openai messages.
    """

    model_config = ConfigDict(from_attributes=True)

    messages: List[ChatMessage] = []

    def add_message(self, message: ChatMessage):
        self.messages.append(message)

    @property
    def as_openai(self):
        return [msg.as_openai for msg in self.messages]
