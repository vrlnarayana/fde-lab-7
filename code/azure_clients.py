"""
azure_clients.py
----------------
One place that builds all the Azure objects, so the other scripts stay short.
Reads settings from the .env file.

Provides:
  get_embeddings()      -> AzureOpenAIEmbeddings
  get_chat()            -> AzureChatOpenAI
  get_vector_store(search_type) -> LangChain AzureSearch vector store
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current folder

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch

SEMANTIC_CONFIG = "carepathways-semantic"


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val or val.startswith("YOUR-") or "YOUR-" in val:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Copy .env.example to .env and fill in your Azure values."
        )
    return val


def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
        api_key=_require("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_deployment=_require("AZURE_OPENAI_EMBED_DEPLOYMENT"),
    )


def get_chat(temperature: float = 0.0) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
        api_key=_require("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_deployment=_require("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        temperature=temperature,
    )


def get_vector_store(search_type: str = "hybrid") -> AzureSearch:
    """
    search_type is one of:
      "similarity"      -> pure vector search (our 'naive RAG' retriever)
      "hybrid"          -> BM25 keyword + vector, fused with RRF
      "semantic_hybrid" -> hybrid, then re-ranked by the Azure semantic ranker
    """
    return AzureSearch(
        azure_search_endpoint=_require("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=_require("AZURE_SEARCH_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX", "carepathways-idx"),
        embedding_function=get_embeddings().embed_query,
        search_type=search_type,
        semantic_configuration_name=SEMANTIC_CONFIG,
    )
