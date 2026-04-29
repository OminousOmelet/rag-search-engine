import argparse

from lib.utils import RRF_K, DEFAULT_SEARCH_LIMIT, print_docs_with_llm_response
from hybrid_search_cli import rrf_search_command
from lib.augmented_generation import rag_command, summarize_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    sum_parser = subparsers.add_parser("summarize", help="Summarize RAG responses")
    sum_parser.add_argument("query", type=str, help="Search query for RAG to summarize")
    sum_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="limit number of search results")
    args = parser.parse_args()

    query = args.query
    match args.command:
        case "rag":
            documents = rrf_search_command(query, RRF_K, DEFAULT_SEARCH_LIMIT)
            rag_response = rag_command(query, documents)
            print_docs_with_llm_response(documents, rag_response)
        case "summarize":
            documents = rrf_search_command(query, RRF_K, args.limit)
            summary = summarize_command(query, documents)
            print_docs_with_llm_response(documents, summary)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()