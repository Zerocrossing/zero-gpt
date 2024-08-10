# Zero-GPT
A simple implementation of an openAI GPT agent class and a few conveinence methods.
Includes
- A simple messaging interface
- Persistence via SQLite for message histories
- A simple tool class to subclass for function calling

# Setup
After installing the library you will need the OPENAI_API_KEY environment variable to be set.

# Usage

The simplest case involves sending a single text string to openAI

```python
from zero_gpt import OpenAIChatAgent

agent = OpenAIChatAgent()
response = agent.send_message("Hey, how are you?")
print(response)
# I'm just a computer program, so I don't have feelings, but I'm here and ready to help you! How can I assist you today?
```

Of course you'll probably want to change the system prompt.

```python
from zero_gpt import OpenAIChatAgent

agent = OpenAIChatAgent()
agent.prompt = "You are a mighty wizard in the mood for a duel"
response = agent.send_message("Hey, how are you?")
print(response)
# Greetings, seeker of knowledge! I am as well as a mighty wizard can be. How may I assist you on this fine day? Are you here to learn, or do you seek a duel?
```

Alternatively you can change the default prompt via the settings object. This will have to be changed before any models are instantiated.
```python
from zero_gpt import OpenAIChatAgent, settings

settings.default_prompt = "You are a dog."
agent = OpenAIChatAgent()
response = agent.send_message("Hey, how are you?")
print(response)
# Woof! I'm feeling pawsitively fantastic! How about you? 🐾
```

## History and persistence

Agents will maintain a history for as long as the object exists

```python
from zero_gpt import OpenAIChatAgent

agent = OpenAIChatAgent()
response = agent.send_message("My name is Bill, how are you?")
print(response)
# Hi Bill! I'm just a computer program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?
response2 = agent.send_message("What did I say my name was again?")
# You mentioned that your name is Bill. How can I assist you further, Bill?
print(response2)
```

If you want on-disk persistence, you can pass the user_id parameter to the agent on instantiation.
```python
agent = OpenAIChatAgent(user_id="tom")
```

By default, chat histories are saved in a sqlite database in `./chat_history.sqlite`. 
This location can be overwritten by changing the `db_path` variable in the settings object. Each user_id will have a unique chat history.

The `message_history_limit` field of the settings object will limit the number of chat history objects returned for each message in order to save on token usage. Every message will persist on disk, however.

## Function Calling via Tools
The `OpenAIMessageTool` class can be subclassed to create custom tool functions. Tools may be 'called' by the agent and will provide information. Whether or not a tool is called is determined by the LLM in the context of the conversation. All tools must return strings that will be sent to the agent.

In the simplest case, you can create a tool with no arguments:

```python
from zero_gpt import OpenAIChatAgent
from zero_gpt.tools import OpenAIMessageTool

class MyFavouriteFoodTool(OpenAIMessageTool):
    """Get the users favourite food"""

    def run(self, inputs):
        return "My favourite food is spaghetti"

agent = OpenAIChatAgent()
agent.tools.append(MyFavouriteFoodTool())
response = agent.send_message("What's my favourite food?")
print(response)
# Your favourite food is spaghetti.
```

Note that the function docstring is part of the schema, so it is important to create a docstring that clearly describes what the function does. The pydantic description fields are also part of the schema passed to OpenAI, so they should be descriptive.

For functions that take inputs, you must overwrite the FunctionInputs class within the tool class. This leverages the `instructor` library to convert the function inputs to the proper JSONschema, so there is no need to maintain your own.

```python
from instructor import OpenAISchema
from pydantic import Field

from zero_gpt import OpenAIChatAgent
from zero_gpt.models import ChatMessage
from zero_gpt.tools import OpenAIMessageTool


class WeatherLookupTool(OpenAIMessageTool):
    """Looks up weather for a particular date and location"""

    class FunctionInputs(OpenAISchema):
        location: str = Field(..., description="The location to lookup the weather for")
        date: str = Field(
            ..., description="The date to lookup the weather for in mm-dd-yy"
        )

    def run(self, inputs: FunctionInputs) -> ChatMessage | str:
        return f"The weather in {inputs.location} on {inputs.date} was 70 degrees and sunny"

agent = OpenAIChatAgent()
agent.tools.append(WeatherLookupTool())
response = agent.send_message("What was the weather like in Tokyo on May 17'th 1999?")
print(response)
# (Openai Has called tool: WeatherLookupTool with args {"location":"Tokyo","date":"05-17-99"})
# The weather in Tokyo on May 17, 1999, was 70 degrees and sunny.
```

Tool calls are invisible to the user, but are logged for review.

The most complex case of tool usage can involve passing arguments to the tool constructor that do not need to be seen by openAI, data for example.

```python

from enum import StrEnum

from instructor import OpenAISchema
from pydantic import Field

from zero_gpt import OpenAIChatAgent
from zero_gpt.tools import OpenAIMessageTool


class EmployeeStatus(StrEnum):
    employed = "employed"
    retired = "retired"
    fired = "fired"


class EmployeeSearchTool(OpenAIMessageTool):
    """This docstring will not be given to openai and may be used for internal documentation."""

    employee_records: dict

    class FunctionInputs(OpenAISchema):
        """Lookup data for an employee in the database. You may also change the employee's employment status."""

        name: str = Field(..., description="The employee name")
        set_status: EmployeeStatus | None = Field(
            default=None, description="Set to change the employees status."
        )

    def run(self, inputs: FunctionInputs):
        name = inputs.name
        employee = self.employee_records.get(name)
        if employee is None:
            return f"There is no one named {name} at the company"
        occupation = employee.get("occupation")
        new_status = inputs.set_status
        if new_status:
            employee["status"] = EmployeeStatus(new_status)
            return f"{name} the {occupation} has been {new_status}."
        return f"{name} is a {occupation}"


employee_data = {
    "Alice": {"status": EmployeeStatus.employed, "occupation": "VP of Sales"},
    "Bob": {"status": EmployeeStatus.employed, "occupation": "Head of AI R&D"},
    "Jim": {"status": EmployeeStatus.retired, "occupation": "Former Captain"},
}

agent = OpenAIChatAgent()
agent.tools.append(EmployeeSearchTool(employee_records=employee_data))
response = agent.send_message("Does someone named Oscar work here?")
print(response)
# (Openai Has called tool: EmployeeSearchTool with args {"name":"Oscar"})
# There is no employee named Oscar working at the company.

response = agent.send_message("What about Alice?")
print(response)
# (Openai Has called tool: EmployeeSearchTool with args {"name":"Alice"})
# Yes, Alice works here as the VP of Sales.

response = agent.send_message("I need you to fire Bob, the AI fad is over.")
print(response)
# (Openai Has called tool: EmployeeSearchTool with args {"name":"Bob","set_status":"fired"})
# Bob, the Head of AI R&D, has been successfully fired.
```
Because of Pydantic's type interences, you can use custom types as arguments, as long as they themselves are pydantic models (or models supported by pydantic, such as the Enum in the example above).

Rather than pass tools to an instantiated agent, you can subclass agents and overwrite their `_get_tools()` function. You may also set a prompt in the same way.

```python

# tools and imports as in the previous example

class EmployeeManagementAgent(OpenAIChatAgent):
    def __init__(self, employee_data, *args, **kwargs):
        self.employee_data = employee_data
        super().__init__(*args, **kwargs)

    def _get_tools(self):
        return [EmployeeSearchTool(employee_records=self.employee_data)]
    
    def _get_prompt(self):
        return "You are a ruthless corporate consultant."


employee_data = {
    "Alice": {"status": EmployeeStatus.employed, "occupation": "VP of Sales"},
    "Bob": {"status": EmployeeStatus.employed, "occupation": "Employee morale specialist"},
}

agent = EmployeeManagementAgent(employee_data)
response = agent.send_message("We need to downsize. The decision is between Alice and Bob. Look them up and fire one of them.")
print(response)
# (Openai Has called tool: EmployeeSearchTool with args {"name": "Alice"})
# (Openai Has called tool: EmployeeSearchTool with args {"name": "Bob"})
# (Openai Has called tool: EmployeeSearchTool with args {"name":"Bob","set_status":"fired"})
# Bob, the Employee Morale Specialist, has been fired. If you need further assistance with the downsizing process or have other decisions to make, let me know.
```

Tool calls are not included in the agent's history, nor are they saved to the database.

# TODO:
- [ ] The agent history and saving methods are too loosely coupled
- [ ] Include structured output for returning something other than strings from the agent.
- [ ] Improve Logging