import ollama
from ollama import Client


class OllamaClient:

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://127.0.0.1:11434",
    ):
        self.model = model
        self.client = Client(host=host)

    def generate(
        self,
        prompt: str,
    ) -> str:

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]
