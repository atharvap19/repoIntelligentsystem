
from ollama import Client

from app.prompts.system_prompt import SYSTEM_PROMPT

class LLMService:
    """
    Handles communication with the local Qwen model.
    """

    def __init__(self):

        self.client = Client(
            host="http://localhost:11434"
        )

        self.model = "qwen3:4b"

    def build_context(self, retrieved_chunks: list) -> str:
        """
        Convert retrieved chunks into a prompt context.
        """

        context = ""

        for chunk in retrieved_chunks:
            context += (
                f"\nFile: {chunk['relative_path']}\n"
                f"{chunk['content']}\n"
            )

        return context

    def generate_response(
        self,
        question: str,
        retrieved_chunks: list,
    ) -> str:

        # Instead of rebuilding the context here...
        context = self.build_context(retrieved_chunks)

        prompt = f"""
Repository Context:

{context}

User Question:

{question}
"""

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response["message"]["content"]