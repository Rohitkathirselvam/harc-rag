from harc_rag.pipeline.pipeline import HARCRAGPipeline


class PipelineService:

    def __init__(
        self,
        retriever,
    ):

        self.pipeline = HARCRAGPipeline(
            retriever
        )

    def ask(
        self,
        question: str,
        conversation_context: str = "",
    ) -> str:

        return self.pipeline.answer(
            question,
            conversation_context=conversation_context,
        )

    def ask_with_metadata(
        self,
        question: str,
        conversation_context: str = "",
    ):

        return self.pipeline.answer_with_metadata(
            question,
            conversation_context=conversation_context,
        )