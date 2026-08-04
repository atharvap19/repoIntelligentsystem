from fastapi import APIRouter
from pydantic import BaseModel

from app.services.github_service import GitHubService

router = APIRouter(prefix="/github", tags=["GitHub"])

class GitHubRepoRequest(BaseModel):
    url: str

@router.post("/import")
def import_repository(request: GitHubRepoRequest):
    """
    Clone a GitHub repository and return its local path.
    """
    service = GitHubService()

    result = service.clone_repository(request.url)

    return result