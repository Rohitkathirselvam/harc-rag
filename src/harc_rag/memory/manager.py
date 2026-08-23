from harc_rag.memory.models import Conversation, Message
from harc_rag.memory.sqlite_store import SQLiteMemoryStore


class MemoryManager:

    def __init__(self, database: str = "memory.db"):

        self.store = SQLiteMemoryStore(database)

        self.conversations = {}

    def create(self, conversation_id: str) -> Conversation:

        conversation = Conversation(
            conversation_id=conversation_id
        )

        self.conversations[conversation_id] = conversation

        self.store.save(conversation)

        return conversation

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ):

        conversation = self.get(conversation_id)

        conversation.messages.append(
            Message(
                role=role,
                content=content,
            )
        )

        self.store.save(conversation)

    def get(
        self,
        conversation_id: str,
    ) -> Conversation:

        if conversation_id in self.conversations:

            return self.conversations[conversation_id]

        conversation = self.store.load(
            conversation_id
        )

        self.conversations[conversation_id] = conversation

        return conversation