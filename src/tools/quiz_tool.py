from smolagents import Tool
import requests
import json




class FetchCompletedQuizzesTool(Tool):
    name = "Fetch_Completed_Quizzes_Tool"
    description = (
        "Fetch recently completed quizzes for a user to personalize future explanations and quizzes."
    )
    inputs = {
        "email": {"type": "string", "description": "User email"},
        "limit": {"type": "integer", "description": "Max quizzes to fetch", "nullable": True},
    }
    output_type = "string"

    def __init__(self, api_base="http://localhost:5000"):
        super().__init__()
        self.api_base = api_base

    def forward(self, email: str, limit: int = 5) -> str:
        try:
            resp = requests.post(
                f"{self.api_base}/quizzes/history",
                data={
                    "email": email,
                    "limit": str(limit),
                },
                timeout=10,
            )
        except Exception as e:
            return f"Quiz history fetch failed: {e}"

        if resp.status_code != 200:
            return f"Quiz history error {resp.status_code}: {resp.text}"

        return resp.text   # JSON string: {"items":[...]}
