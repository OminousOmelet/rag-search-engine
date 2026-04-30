import os, json

from dotenv import load_dotenv
from google import genai
from .utils import GENAI_MODEL

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)
model = GENAI_MODEL

def rag_command(query, documents: list[dict]):
    docs_str_list = []
    for doc in documents:
        docs_str_list.append(json.dumps(doc))
    response = client.models.generate_content(
        model=GENAI_MODEL,
        contents=f"""
        You are a RAG agent for Hoopla, a movie streaming service.
        Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
        Provide a comprehensive answer that addresses the user's query.

        Query: {query}

        Documents:
        {chr(10).join(docs_str_list)}

        Answer:"""
    )
    return response.text

def summarize_command(query, documents: list[dict]):
    docs_str_list = []
    for doc in documents:
        docs_str_list.append(json.dumps(doc))
    response = client.models.generate_content(
        model=GENAI_MODEL,
        contents=f"""
        Provide information useful to the query below by synthesizing data from multiple search results in detail.

        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Search results:
        {chr(10).join(docs_str_list)}

        Provide a comprehensive 3 to 4 sentence answer that combines information from multiple sources:"""
    )
    return response.text

def citations_command(query, documents: list[dict]):
    docs_str_list = []
    for doc in documents:
        docs_str_list.append(json.dumps(doc))
    response = client.models.generate_content(
        model=GENAI_MODEL,
        contents=f"""
        Answer the query below and give information based on the provided documents.

        The answer should be tailored to users of Hoopla, a movie streaming service.
        If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

        Query: {query}

        Documents:
        {chr(10).join(docs_str_list)}

        Instructions:
        - Provide a comprehensive answer that addresses the query
        - Cite sources in the format [1], [2], etc. when referencing information
        - If sources disagree, mention the different viewpoints
        - If the answer isn't in the provided documents, say "I don't have enough information"
        - Be direct and informative

        Answer:"""
    )
    return response.text

def question_command(question, documents: list[dict]):
    docs_str_list = []
    for doc in documents:
        docs_str_list.append(json.dumps(doc))
    response = client.models.generate_content(
        model=GENAI_MODEL,
        contents=f"""
        Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

        Question: {question}

        Documents:
        {chr(10).join(docs_str_list)}

        Instructions:
        - Answer questions directly and concisely
        - Be casual and conversational
        - Don't use 'hype-y' or overly enthusiastic language
        - Talk like a person typically would in a chat conversation

        Answer:"""
    )
    return response.text