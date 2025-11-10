from smolagents import Tool, DuckDuckGoSearchTool
from bs4 import BeautifulSoup
import csv
import requests


# will take in three inputs, needed to add the nullable because it was messing with smolagents
# the init will fill the rows to be of the csv file
# then the forward will use the input from the AI to pick out what from the rows
# then return them
# class EventsCSVTool(Tool):
#     name = "events_csv"
#     description = ("Query Utah Tech events from events.csv. "
#                    "Inputs: date(optional, YYYY-MM-DD), keywords(optional, comma-separated), "
#                    "and limit(optional, default 5). Returns a concise list.")
#     inputs = {
#         "date": {"type": "string", "description": "YYYY-MM-DD or empty","nullable": True},
#         "keywords": {"type": "string", "description": "e.g., 'football,science' or empty","nullable": True},
#         "limit": {"type": "integer", "description": "max results, default 5","nullable": True}
#     }
#     output_type = "string"

#     def __init__(self, csv_path="events.csv"):
#         super().__init__()
#         self.rows = []
#         with open(csv_path, newline='', encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             for r in reader:
#                 self.rows.append(r)
#     def forward(self, date: str = "", keywords: str = "", limit: int = 5) -> str:
#         items = self.rows
#         if date:
#             items = [r for r in items if r.get("date","").startswith(date)]
#         if keywords:
#             ks = [k.strip().lower() for k in keywords.split(",") if k.strip()]
#             def hit(r):
#                 blob = " ".join([r.get("tittle",""),r.get("location",""), r.get("category","")]).lower()
#                 return any(k in blob for k in ks)
#             items = [r for r in items if hit(r)]
#         items = items[: (limit or 5)]
#         if not items:
#             return "No matching events found"
#         lines = []
#         for r in items:
#             lines.append(f"- {r.get('date','')} | {r.get('title','')} @ {r.get('location','')} [{r.get('category','')}]")
#         return "\n".join(lines)

# # to make a more defined search for the duck duck go search engine dont need this in reality
class MachineLearningSearchTool(Tool):
    name = "Machine_Learning_Search_Tool"
    description = (
        "Search for machine learning and image processing papers across research sites."
    )
    inputs = {
        "question": {"type": "string", "description": "Natural-language research question"}
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self._ddg = DuckDuckGoSearchTool()

    def forward(self, question: str) -> str:
        # Base filters for ML paper sources
        research_sites = [
            "site:arxiv.org",
            "site:ieeexplore.ieee.org",
            "site:springer.com",
            "site:sciencedirect.com",
            "site:nature.com",
            "site:scholar.google.com",
        ]

        # Construct the combined site filter
        site_filter = " OR ".join(research_sites)

        # Add your topic focus
        topics = (
            '"machine learning" "image processing" "deep learning" '
            '"computer vision" "image classification"'
        )

        # Combine it all
        query = f"({site_filter}) {topics} {question}"

        # Run the search
        return self._ddg(query)


class ScrapePageTool(Tool):
    # will read the description to decide if it could use it.
    name = "scrape_page"
    description = "Fetch a web page and return a cleaned text summary (title + first ~500 characters)."
    inputs = {"url": {"type":"string","description":"HTTP/HTTPS URL to fetch"}}
    output_type = "string"

    # forward is what gets called by AI 
    def forward(self, url: str) -> str:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            return f"Request failed: {e}"
        soup = BeautifulSoup(resp.text, "html.parser")
        title = (soup.title.string.strip() if soup.title and soup.title.string else url)
        # crude extract of visible text
        for tag in soup(["script","style","noscript"]):
            tag.decompose()
        text = " ".join(t.get_text(" ", strip=True) for t in soup.find_all(["p","li","h1","h2","h3"]))
        text = " ".join(text.split())
        snippet = text[:500] + ("…" if len(text) > 500 else "")
        return f"{title}\n{snippet}"

