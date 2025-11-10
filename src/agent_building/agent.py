from smolagents import ToolCallingAgent
from agent_building import model_utils
from tools.vector_store import ChromaRetriever
from tools.retrieval_tool import RetrieveDocumentsTool
from tools.machine_learning_tools import MachineLearningSearchTool
from tools.machine_learning_tools import ScrapePageTool

def build_agent(verbose: int = 1) -> ToolCallingAgent:
    model = model_utils.google_build_reasoning_model()
    retriever =  ChromaRetriever(  persist_directory="vector_store",collection_name="ml_image_papers")
    tools = [
        RetrieveDocumentsTool(retriever=retriever),
        ScrapePageTool(),
        MachineLearningSearchTool()
    ]

    agent = ToolCallingAgent(
        tools=tools,
        model=model,
        verbosity_level=verbose,
        stream_outputs=False,
        instructions="""You are an agent to help users with questions about Machine Learning (like a tutor).
        When the user asks how does this (topic) work in machine learning, search the chroma database and evaluate the result. DO NOT GUESS.
        When searching the Web for resources.
        Always include a brief “Sources:” section with one URL if you used the web. If data is from the database, say “Source: Chroma.db”.
        """
    )
    return agent


