from smolagents import Tool, DuckDuckGoSearchTool
from bs4 import BeautifulSoup
import requests


# computer science tool
class ComputerScienceSearchTool(Tool):
    name = "Computer_Science_Search_Tool"
    description = (
        "Search for Computer Science and Machine Learning papers across technical research sites."
    )
    inputs = {
        "question": {"type": "string", "description": "Computer Science question or research query."}
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self._ddg = DuckDuckGoSearchTool()

    def forward(self, question: str) -> str:
        # Research sources for CS / AI / ML
        research_sites = [
            "site:arxiv.org",
            "site:ieeexplore.ieee.org",
            "site:acm.org",
            "site:springer.com",
            # "site:sciencedirect.com",  returns paywall
            "site:scholar.google.com",
        ]

        site_filter = " OR ".join(research_sites)

        # Focus terms for CS and AI
        topics = (
            '"computer science" "algorithms" "machine learning" '
            '"neural networks" "image classification" "data structures" '
            '"AI research" "computer vision" "software engineering"'
        )

        query = f"({site_filter}) {topics} {question}"
        return self._ddg(query)
    
    # med tool
class MedicalSearchTool(Tool):
    name = "Medical_Search_Tool"
    description = (
        "Search for Medical and Biomedical research papers across academic and healthcare databases."
    )
    inputs = {
        "question": {"type": "string", "description": "Medical or healthcare-related research question."}
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self._ddg = DuckDuckGoSearchTool()

    def forward(self, question: str) -> str:
        # Medical and healthcare research sources
        research_sites = [
            "site:pubmed.ncbi.nlm.nih.gov",
            "site:nature.com",
            # "site:sciencedirect.com", returns paywall
            "site:springer.com",
            "site:nejm.org",
            "site:who.int",
            "site:cdc.gov",
            "site:scholar.google.com",
        ]

        site_filter = " OR ".join(research_sites)

        # Focus terms for medicine and health
        topics = (
            '"medicine" "healthcare" "clinical study" "biomedical" '
            '"disease" "treatment" "diagnosis" "X-ray" "CT scan" '
            '"ultrasound" "pneumonia" "medical imaging"'
        )

        query = f"({site_filter})  {topics} {question}"
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

