import sqlite3
from datetime import datetime

from harc_rag.memory.models import Conversation, Message
from harc_rag.memory.interfaces import MemoryStore


class SQLiteMemoryStore(MemoryStore):

    def __init__(self, database: str = "memory.db"):

        self.connection = sqlite3.connect(
            database,
            check_same_thread=False,
        )

        self.create_tables()

    def create_tables(self):

        cursor = self.connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations(
                conversation_id TEXT PRIMARY KEY
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT
            )
        """)

        self.connection.commit()

    # -------------------------
    # ADD THIS METHOD HERE
    # -------------------------
    def save(self, conversation: Conversation):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT OR IGNORE INTO conversations(conversation_id)
            VALUES(?)
            """,
            (conversation.conversation_id,),
        )

        cursor.execute(
            "DELETE FROM messages WHERE conversation_id=?",
            (conversation.conversation_id,),
        )

        for message in conversation.messages:

            cursor.execute(
                """
                INSERT INTO messages(
                    conversation_id,
                    role,
                    content,
                    timestamp
                )
                VALUES(?,?,?,?)
                """,
                (
                    conversation.conversation_id,
                    message.role,
                    message.content,
                    message.timestamp.isoformat(),
                ),
            )

        self.connection.commit()

    # -------------------------
    # ADD THIS METHOD HERE
    # -------------------------
    def load(self, conversation_id: str) -> Conversation:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT role, content, timestamp
            FROM messages
            WHERE conversation_id=?
            ORDER BY id
            """,
            (conversation_id,),
        )

        rows = cursor.fetchall()

        conversation = Conversation(
            conversation_id=conversation_id
        )

        for role, content, timestamp in rows:

            conversation.messages.append(
                Message(
                    role=role,
                    content=content,
                    timestamp=datetime.fromisoformat(timestamp),
                )
            )

        return conversation