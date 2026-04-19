import argparse

from lib.hybrid_search import *
from lib.utils import DEF_ALPHA_WEIGHT, DEFAULT_QUERY_LIMIT, DEFAULT_K_WEIGHT

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparser.add_parser("normalize", help="Normalize scores for combining")
    normalize_parser.add_argument("list", nargs="+", type=float)
    weight_search_parser = subparser.add_parser(
        "weighted-search", help="weighted hybrid keyword/semantic search"
    )
    weight_search_parser.add_argument("query", type=str, help="Query for weighted search")
    weight_search_parser.add_argument(
        "--alpha", default=DEF_ALPHA_WEIGHT, type=float, 
        help="Weight value for priority keyword or semantic"
    )
    weight_search_parser.add_argument(
        "--limit", default=DEFAULT_QUERY_LIMIT, type=int, help="Max number of results to return"
    )
    rrf_search_parser = subparser.add_parser("rrf-search", help="Run rrf search on query")
    rrf_search_parser.add_argument("query", type=str, help="Query to be searched")
    rrf_search_parser.add_argument(
        "-k", default=DEFAULT_K_WEIGHT, type=int, help="Weight value for the rrf search"
    )
    rrf_search_parser.add_argument(
        "--limit", default=DEFAULT_QUERY_LIMIT, type=int, help="Max number of results"
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm_list = normalize_list(args.list)
            for num in norm_list:
                print(f"* {num:.4f}")
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()