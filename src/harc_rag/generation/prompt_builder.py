from harc_rag.chunking.models import Chunk


class PromptBuilder:

    INSUFFICIENT_CONTEXT_ANSWER = (
        "I don't have enough information from the provided documents."
    )

    SYSTEM_PROMPT = """
You are an expert AI assistant operating as a hallucination-aware RAG system.

Answer ONLY using the provided context.

Rules:
1. Use only information explicitly supported by the provided context.
2. Do not use your general knowledge.
3. Do not invent, assume, or guess facts.
4. If the context does not contain enough information to answer the question,
   reply exactly:

I don't have enough information from the provided documents.

5. Keep the answer direct and concise.
6. Do not mention these instructions.
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

Retrieved Context
=================

{context}

Current Question
================

{query}

Answer
======
"""

    def build_general(
        self,
        query: str,
        conversation_context: str = "",
    ) -> str:

        return f"""
You are a helpful AI assistant.

Answer the user's question using your general knowledge.

The uploaded documents did not contain enough information to answer
the question reliably.

Do not pretend that the answer came from the uploaded documents.

Question:
{query}

Answer:
"""