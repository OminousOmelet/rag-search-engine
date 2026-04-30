import json, os

from dotenv import load_dotenv
from typing import Any

GENAI_MODEL = "gemma-3-27b-it"

DEFAULT_SEARCH_LIMIT = 5
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3
DEFAULT_QUERY_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEF_SEARCH_CHUNK_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
MOVIES_PATH = os.path.join(DATA_PATH, "movies.json")
STOPWORDS_PATH = os.path.join(DATA_PATH, "stopwords.txt")
GOLDEN_DATA_PATH = os.path.join(DATA_PATH, "golden_dataset.json")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

load_dotenv()
DEBUG = os.environ.get('DEBUG', 'false').lower()
DEBUG_JSON_PATH = os.path.join(CACHE_DIR, "debug_log.json")
debug_data: list[dict] = [] # icky no consty

BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_ALPHA = 0.5
RRF_K = 60
RERANK_FACTOR = 5

def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()

def log_to_debug_file(obj):
    json_string = json.dumps(obj)
    with open(DEBUG_JSON_PATH, 'w', encoding="utf-8") as file:
        file.write(json_string)


# Used for some results, instructor's code (long, annoying story)
def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }

def print_doc_list(documents):
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc['title']}")
        
        if 'rerank' in doc:
            print(f"  Re-rank {doc['rerank']}")
        if 'cross_enc' in doc:
            print(f"  Cross Encoder Score: {doc['cross_enc']:.3f}")
        if 'rrf_score' in doc:
            print(f"  RRF Score: {doc['rrf_score']:.3f}")

        bm25_rank, sem_rank = None, None
        if 'bm25_rank' in doc:
            bm25_rank = doc['bm25_rank']
        if 'sem_rank' in doc:
            sem_rank = doc['sem_rank']

        print(f"  BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
        print(f"  {doc['document'][:100]}...\n")

def print_docs_with_llm_response(documents: list[dict], response: str, resp_title):
    print("Search Results:")
    for doc in documents:
        print(f"- {doc['title']}")
    print(f"\n{resp_title}:")
    print(response)