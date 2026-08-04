from app.services.llm_service import LLMService

service = LLMService()

chunks = [
    {
        "relative_path": "routing.py",
        "content": """
class APIRouter:

    def include_router(self):
        pass
"""
    }
]

response = service.generate_response(
    question="What does APIRouter do?",
    retrieved_chunks=chunks,
)

print(response)