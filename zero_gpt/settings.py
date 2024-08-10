from dataclasses import dataclass


@dataclass
class ZeroGPTSettings:
    message_history_limit: int = 10
    default_gpt_model: str = "gpt-4o-mini"
    default_prompt: str = "You are a helpful assistant."
    db_path: str = "./chat_history.sqlite"


settings = ZeroGPTSettings()
