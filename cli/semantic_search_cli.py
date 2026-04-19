#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model, verify_embeddings, embed_text, embed_query_text, search_command, chunk_command,
    semantic_chunk_text, embed_chunks_command, search_chunked_command
)
from lib.utils import DEFAULT_QUERY_LIMIT, DEFAULT_CHUNK_SIZE, DEF_SEARCH_CHUNK_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the sentence transformer model")
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedded text from input")
    embed_parser.add_argument("text", type=str, help="text to be embedded")
    subparsers.add_parser("verify_embeddings", help="Verify the cache of embeddings")
    embed_query_parser = subparsers.add_parser("embedquery", help="User query embedding")
    embed_query_parser.add_argument("query", type=str, help="query to embed")
    search_parser = subparsers.add_parser("search", help="semantic search")
    search_parser.add_argument("query", type=str, help="query to be searched")
    search_parser.add_argument(
        "--limit", default=DEFAULT_QUERY_LIMIT, type=int, help="limit length of results"
    )
    chunk_parser = subparsers.add_parser("chunk", help="break input text into chunks")
    chunk_parser.add_argument("text", type=str, help="text to break into chunks")
    chunk_parser.add_argument(
        "--chunk-size", default=DEFAULT_CHUNK_SIZE, type=int, help="amount of chars per chunk"
    )
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text on sentence boundaries to preserve meaning"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum size of each chunk in sentences",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences to overlap between chunks",
    )
    subparsers.add_parser("embed_chunks", help="Generate embeddings for chunked documents")
    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search using chunked embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=DEF_SEARCH_CHUNK_LIMIT, help="Number of results to return"
    )
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            sem_scores = search_command(args.query, args.limit)
            for i, item in enumerate(sem_scores):
                text = f"{i+1}. {item[0]['title']} (score: {item[1]})\n  {item[0]['description']}"
                print(text[:150] + "...")
        case "chunk":
            chunks = chunk_command(args.text, args.chunk_size)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embeddings = embed_chunks_command()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            result = search_chunked_command(args.query, args.limit)
            print(f"Query: {result['query']}")
            print("Results:")
            for i, res in enumerate(result["results"], 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()