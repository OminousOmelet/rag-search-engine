import os, time, json

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"

def enhance_query(query, enhancement):
    match enhancement:
        case "spell":
            response = client.models.generate_content(
                model=model, 
                contents=f"""
                    Fix any spelling errors in the user-provided movie search query below.
                    Correct only clear, high-confidence typos. 
                    Do not rewrite, add, remove, or reorder words.
                    Preserve punctuation and capitalization unless a change is required for a typo fix.
                    If there are no spelling errors, 
                    or if you're unsure, output the original query unchanged.
                    Output only the final query text, nothing else.
                    User query: "{query}"
                    """
            )
            enhanced_query = response.text.strip().strip('"')
        case "rewrite":
            response = client.models.generate_content(
                model=model, 
                contents=f"""
                Rewrite the user-provided movie search query below to be more specific and searchable.
                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> 
                "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}"
                """
            )
            enhanced_query = response.text.strip().strip('"')
        case "expand":
            response = client.models.generate_content(
                model=model, 
                contents=f"""
                Expand the user-provided movie search query below with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.

                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                User query: "{query}"
                """
            )
            expanded_terms = response.text.strip().strip('"')
            enhanced_query = f"{query} {expanded_terms}".strip()

    return enhanced_query



def rerank_func(query, documents: list[dict], limit, rerank_method):
    print(f"Re-ranking top {limit} results using {rerank_method} method...")
    match rerank_method:
        case "individual":
            return individual_rerank(query, documents, limit)
        case "batch":
            return batch_rerank(query, documents, limit)


def individual_rerank(query, documents: list[dict], limit):
    for i, doc in enumerate(documents, 1):
        print(f"Re-ranking initial results: {i}/{len(documents)}\r", end="", flush=True)
        response = client.models.generate_content(
            model=model, 
            contents=f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Output ONLY the number in your response, no other text or explanation.

            Score:
            """
        )
        doc['rerank_value'] = float(response.text)
        time.sleep(3)
    
    final_list = sorted(documents, key=lambda x: x['rerank_value'], reverse=True)[:limit]
    for doc in final_list:
        doc['rerank'] = f"Score: {doc['rerank_value']:.3f}/10"

    return final_list

def batch_rerank(query, documents: list[dict], limit):
    doc_list_str = json.dumps(documents)

    response = client.models.generate_content(
        model=model,
        contents=f"""
        Rank the movies listed below by relevance to the following search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

        For example:
        [75, 12, 34, 2, 1]

        Ranking:
        """
    )
    rank_list = json.loads(response.text)
    id_to_doc = {doc['id']: doc for doc in documents} # for mapping ranks sequentially to docs by ID
    doc_list = []
    for id in rank_list[:limit]:
        doc_list.append(id_to_doc.get(id))
    
    for i, doc in enumerate(doc_list, 1):
        doc['rerank'] = f" Rank: {i}"
    
    return doc_list
    