import os, mimetypes

from dotenv import load_dotenv
from google import genai
from .utils import GENAI_MODEL, PROJECT_ROOT

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)

def rewrite_query_for_img(image_path, query):
    full_path = os.path.join(PROJECT_ROOT, image_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image file not found: {full_path}")
    
    mime, _ = mimetypes.guess_type(full_path)
    mime = mime or "image/jpeg"
    with open(image_path, 'rb') as f:
        img = f.read()

    system_prompt = """
    Given the included image and text query, rewrite the text query to improve search results from a movie database. 
    Make sure to:
    
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """
    
    parts = [
        system_prompt,
        genai.types.Part.from_bytes(data=img, mime_type=mime),
        query.strip(),
    ]

    response = client.models.generate_content(
        model=GENAI_MODEL,
        contents=parts
    )

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
