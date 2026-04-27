# tools/history_tools.py
from smolagents import Tool
import requests

class FetchUserHistoryTool(Tool):
    name = "Fetch_User_History_Tool"
    description = (
        "Fetch recent interactions for a user to provide context. "
        "Inputs: email (str), limit (int, optional; default 10). Returns recent queries with topics."
    )
    inputs = {
        "email": {"type": "string", "description": "User email"},
        "limit": {"type": "integer", "description": "How many recent items to fetch", "nullable": True},
    }
    output_type = "string"

    def __init__(self, api_base: str = "http://localhost:5000"):
        super().__init__()
        self.api_base = api_base

    def forward(self, email: str, limit: int = 10) -> str:
        r = requests.get(
            f"{self.api_base}/interactions/history",
                data={              # <-- form-encoded! same as an HTML form
                    "email": email,
                    "limit": str(limit),
                },
                timeout=10,
        )
        if r.status_code != 200:
            return f"History fetch failed: {r.status_code} {r.text}"
        return r.text  # server returns JSON string (list of {id,email,topic,query})
