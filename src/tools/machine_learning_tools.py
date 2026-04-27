from smolagents import Tool, DuckDuckGoSearchTool
from bs4 import BeautifulSoup
import requests


# computer science tool
class ComputerScienceSearchTool(Tool):
    name = "Computer_Science_Search_Tool"
    description = (
        "Broad web search for Computer Science topics. Handles general CS questions, systems, networking, "
        "programming, algorithms, and academic ML/AI when needed."
    )
    inputs = {
        "question": {"type": "string", "description": "Computer Science question or topic."}
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self._ddg = DuckDuckGoSearchTool()

    def forward(self, question: str) -> str:
        # Tier 1: General CS sites (best for most questions)
        general_sources = [
            "site:wikipedia.org",
            "site:geeksforgeeks.org",
            "site:stackexchange.com",
            "site:stackoverflow.com",
            "site:mdn.dev",
            "site:cloudflare.com/learning",
            "site:freecodecamp.org",
            "site:tutorialspoint.com",
        ]

        # Tier 2: Networking + systems (e.g., port forwarding)
        systems_sources = [
            "site:cisco.com",
            "site:redhat.com",
            "site:linuxize.com",
            "site:ubuntu.com",
        ]

        # Tier 3: Academic (use only for ML/AI/theory)
        academic_sources = [
            "site:arxiv.org",
            "site:ieeexplore.ieee.org",
            "site:acm.org",
            "site:nature.com",
        ]

        # Merge all
        site_filter = " OR ".join(general_sources + systems_sources + academic_sources)

        # No over-restrictive topic bias (kept minimal!)
        query = f"({site_filter}) {question}"

        return self._ddg(query)

    # med tool
class MedicalSearchTool(Tool):
    name = "Medical_Search_Tool"
    description = "Broad medical search across trusted sources, clinical sites, and academic research."
    inputs = {
        "question": {"type": "string", "description": "Medical or healthcare-related question."}
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self._ddg = DuckDuckGoSearchTool()

    def forward(self, question: str) -> str:
        consumer_health = [
            "site:mayoclinic.org",
            "site:clevelandclinic.org",
            "site:medlineplus.gov",
            "site:healthline.com",
        ]

        clinical_sites = [
            "site:cdc.gov",
            "site:who.int",
            "site:nih.gov",
        ]

        academic_sources = [
            "site:pubmed.ncbi.nlm.nih.gov",
            "site:nature.com",
            "site:springer.com"
        ]

        site_filter = " OR ".join(consumer_health + clinical_sites + academic_sources)

        query = f"({site_filter}) {question}"
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
            resp = requests.get(url, timeout=10,headers= { "User-Agent": "Mozilla/5.0 (compatible; AdaptiveTutorBot/1.0; +https://example.com/bot)"
})
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

