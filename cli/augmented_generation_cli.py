import argparse

from lib.utils import RRF_K, DEFAULT_SEARCH_LIMIT, print_docs_with_llm_response
from hybrid_search_cli import rrf_search_command
from lib.augmented_generation import rag_command, summarize_command, citations_command, question_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    sum_parser = subparsers.add_parser("summarize", help="Summarize RAG responses")
    sum_parser.add_argument("query", type=str, help="Search query for RAG to summarize")
    sum_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="limit number of search results")
    citation_parser = subparsers.add_parser("citations", help="Search with citations for results")
    citation_parser.add_argument("query", type=str, help="Query to be searched")
    citation_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="limit number of search results"
    )
    question_parser = subparsers.add_parser("question", help="Ask a movie-related question")
    question_parser.add_argument("question", type=str, help="Question to be asked")
    question_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="limit number of search results to base answer on"
    )

    args = parser.parse_args()

    if hasattr(args, 'question'):
        query = args.question
    else:
        query = args.query
    
    # can use ternary op here since DEFAULT_SEARCH_LIMIT always exists (and I like them, dammit!)
    limit = DEFAULT_SEARCH_LIMIT if not hasattr(args, 'limit') else args.limit
    
    documents = rrf_search_command(query, RRF_K, limit)
    match args.command:
        case "rag":
            rag_response = rag_command(query, documents)
            print_docs_with_llm_response(documents, rag_response, "RAG Response")
        case "summarize":
            summary = summarize_command(query, documents)
            print_docs_with_llm_response(documents, summary, "LLM Summary")
        case "citations":
            response = citations_command(query, documents)
            print_docs_with_llm_response(documents, response, "LLM Answer")
        case "question":
            answer = question_command(query, documents)
            print_docs_with_llm_response(documents, answer, "Answer")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()