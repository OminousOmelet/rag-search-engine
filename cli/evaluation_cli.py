import argparse, json

from lib.utils import GOLDEN_DATA_PATH, DEFAULT_QUERY_LIMIT, RRF_K
from lib.hybrid_search import rrf_search_command

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_QUERY_LIMIT,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    dataset = {}
    with open(GOLDEN_DATA_PATH, 'r') as file:
        dataset = json.load(file)

    results: list[dict] = []
    for i, set in enumerate(dataset.get('test_cases')):
        print(
            f"Comparing relevance test cases: {i+1}/{len(dataset.get('test_cases'))}\r", 
            end="", flush=True
        )

        results.append({
            'query': set.get('query'),
            'precision': 0.0,
            'retrieved': [],
            'relevant': []
        })
        retrieved: list = results[i].get('retrieved')
        relevant: list = results[i].get('relevant')
        documents = rrf_search_command(set.get('query'), RRF_K, limit)
        for doc in documents:
            retrieved.append(doc.get('title'))
            if doc['title'] in set.get('relevant_docs'):
                relevant.append(doc.get('title'))

        results[i]['precision'] = len(relevant) / len(retrieved)

    sorted_results = sorted(results, key=lambda x: x['precision'], reverse=True)
    print(f"k={limit}\n\n")
    for res in sorted_results:
        print(f"- Query: {res.get('query')}")
        print(f"  - Precision@{limit}: {res.get('precision'):.4f}")
        retrieved_str = ", ".join(res.get('retrieved'))
        print("  - Retrieved: " + retrieved_str)
        relevant_str = ", ".join(res.get('relevant'))
        print("  - Relevant: " + relevant_str)

if __name__ == "__main__":
    main()