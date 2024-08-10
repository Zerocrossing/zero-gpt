"""Storage

Basic storage functionality for default models.
Saves chat messages to a table in a local sqlite
"""

import os
import sqlite3
from datetime import datetime
from typing import List

from zero_gpt.models import ChatHistory, ChatMessage, ChatRole
from zero_gpt.settings import settings


def save_messages(user_id: str, messages: List[ChatMessage]):
    """Save the chat history to the SQLite database."""
    # Ensure the directory exists
    if not os.path.exists(settings.db_path):
        conn = sqlite3.connect(settings.db_path)
        cursor = conn.cursor()

        # Create the tables if they don't exist
        cursor.execute("""CREATE TABLE IF NOT EXISTS chat_history (
                            user_id TEXT,
                            role TEXT,
                            content TEXT,
                            name TEXT,
                            include_in_history INTEGER,
                            created_at TEXT
                        )""")
        conn.commit()
    else:
        conn = sqlite3.connect(settings.db_path)
        cursor = conn.cursor()

    # Insert each message in the history into the database
    for message in messages:
        cursor.execute(
            """INSERT INTO chat_history (
                            user_id, role, content, name, include_in_history, created_at
                          ) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                message.role.value,
                message.content,
                message.name,
                int(message.include_in_history),
                message.created_at.isoformat(),
            ),
        )

    conn.commit()
    conn.close()


def load_history(user_id: str) -> ChatHistory:
    """Load the chat history from the SQLite database."""
    if not os.path.exists(settings.db_path):
        return ChatHistory()

    conn = sqlite3.connect(settings.db_path)
    cursor = conn.cursor()

    message_limit = settings.message_history_limit

    # Fetch the most recent messages for the given user_id, ordered by created_at
    cursor.execute(
        """SELECT role, content, name, include_in_history, created_at 
           FROM chat_history 
           WHERE user_id = ?
           ORDER BY datetime(created_at) DESC
           LIMIT ?""",
        (user_id, message_limit),
    )

    rows = cursor.fetchall()
    conn.close()

    # Reverse the list to maintain the original chronological order (oldest first)
    messages = [
        ChatMessage(
            role=ChatRole(role),
            content=content,
            name=name,
            include_in_history=bool(include_in_history),
            created_at=datetime.fromisoformat(created_at),
        )
        for role, content, name, include_in_history, created_at in reversed(rows)
    ]
    return ChatHistory(messages=messages)
