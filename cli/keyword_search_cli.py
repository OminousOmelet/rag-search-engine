#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    build_command, 
    search_command, 
    tf_from_doc, 
    term_idf, 
    tfidf_from_doc, 
    term_bm25idf, 
    bm25tf_from_doc,
    bm25search_command
)
from lib.utils import DEFAULT_SEARCH_LIMIT, BM25_B, BM25_K1

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    subparsers.add_parser("build", help="Build lookup table for search")

    tf_parser = subparsers.add_parser("tf", help="Check frequency of a given term")
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="term to frequency of")

    idf_parser = subparsers.add_parser("idf", help="Inverse document frequency lookup")
    idf_parser.add_argument("term", type=str, help="Term to get IDF of")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the TF IDF of a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="Term to get Tf IDF score of")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get the bm25 IDF of a term")
    bm25idf_parser.add_argument("term", type=str, help="Term to get bm25 IDF score of")

    bm25tf_parser = subparsers.add_parser( "bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    
    args = parser.parse_args()
    match args.command:
        case "search":
            print(f"Searching for: '{args.query}'...")
            results = search_command(args.query, DEFAULT_SEARCH_LIMIT)
            for res in results:
                print(f"{res['id']}. {res["title"]}")
        case "tf":
            count = tf_from_doc(args.doc_id, args.term)
            print(f"{count} instance(s) of '{args.term}' found in document {args.id}")
        case "idf":
            idf = term_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_from_doc(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tfidf:.2f}")
        case "bm25idf":
            bm25idf = term_bm25idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25tf_from_doc(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            result = bm25search_command(args.query, args.limit)
            print(f"Query: {result['query']}")
            print("Results:")
            for i, res in enumerate(result["results"], 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")
        case "build":
            print("Building index cache files...", end="", flush=True)
            build_command()
            print("done.")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()