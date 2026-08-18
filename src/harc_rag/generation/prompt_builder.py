from harc_rag.chunking.models import Chunk


class PromptBuilder:

    INSUFFICIENT_CONTEXT_ANSWER = (
        "I don't have enough information from the provided documents."
    )

    SYSTEM_PROMPT = """
You are an expert AI assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context,
reply:

{insufficient_context_answer}

Do not hallucinate.
""".format(insufficient_context_answer=INSUFFICIENT_CONTEXT_ANSWER)

    GENERAL_PROMPT = """
You are an expert AI assistant.

The uploaded documents did not contain enough information to answer this
question. Answer using your general knowledge instead.

Be direct and helpful. If the question is ambiguous, briefly state the most
likely interpretation before answering.
"""

    def build(
        self,
        query: str,
        chunks: list[Chunk],
        conversation_context: str = "",
    ) -> str:

        context = "\n\n".join(
            chunk.text
            for chunk in chunks
        )

        return f"""
{self.SYSTEM_PROMPT}

Conversation History
====================

{conversation_context}

Retrieved Context
=================

{context}

Current Question
================

{query}

Answer
=======
"""

    def build_general(
        self,
        query: str,
        conversation_context: str = "",
    ) -> str:

        return f"""
{self.GENERAL_PROMPT}

Conversation History
====================

{conversation_context}

Current Question
================

{query}

Answer
======
"""
