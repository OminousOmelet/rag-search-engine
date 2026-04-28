import argparse, os

from lib.hybrid_search import *
from lib.utils import DEBUG, debug_data, DEFAULT_ALPHA, DEFAULT_QUERY_LIMIT, RRF_K, print_doc_list, log_to_debug_file

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparser.add_parser("normalize", help="Normalize scores for combining")
    normalize_parser.add_argument("list", nargs="+", type=float)
    weighted_parser = subparser.add_parser(
        "weighted-search", help="weighted hybrid keyword/semantic search"
    )
    weighted_parser.add_argument("query", type=str, help="Query for weighted search")
    weighted_parser.add_argument(
        "--DEFAULT_ALPHA", default=DEFAULT_ALPHA, type=float, 
        help="Weight value for priority keyword or semantic"
    )
    weighted_parser.add_argument(
        "--limit", default=DEFAULT_QUERY_LIMIT, type=int, help="Max number of results to return"
    )
    rrf_parser = subparser.add_parser("rrf-search", help="Run rrf search on query")
    rrf_parser.add_argument("query", type=str, help="Query to be searched")
    rrf_parser.add_argument(
        "-k", default=RRF_K, type=int, help="Weight value for the rrf search"
    )
    rrf_parser.add_argument(
        "--limit", default=DEFAULT_QUERY_LIMIT, type=int, help="Max number of results"
    )
    rrf_parser.add_argument(
        "--enhance", type=str, choices=["spell", "rewrite", "expand"], 
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method", type=str, choices=["individual", "batch", "cross_encoder", ""],
        help="method of reranking for fine-tuning search results"    
    )
    rrf_parser.add_argument("--evaluate", action="store_true", help="Use LLM to evaluate relevance of results")
    args = parser.parse_args()

    debug_type = None
    match args.command:
        case "normalize":
            norm_list = normalize_list(args.list)
            for num in norm_list:
                print(f"* {num:.4f}")
        case "weighted-search":
            if DEBUG:
                debug_type = "weighted_search"
                debug_data.append({'original_query': args.query})
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            if DEBUG:
                debug_type = "rrf_search"
                debug_data.append({'original_query': args.query})
            results = rrf_search_command(
                args.query, args.k, args.limit, args.enhance, args.rerank_method, args.evaluate
            )
            if not args.evaluate:
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={RRF_K}):\n")
                print_doc_list(results)
        case _:
            parser.print_help()

    if DEBUG:
        obj = {debug_type: debug_data}
        log_to_debug_file(obj)


if __name__ == "__main__":
    main()