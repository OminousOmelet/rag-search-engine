import os

from PIL import Image
from sentence_transformers import SentenceTransformer
from .utils import PROJECT_ROOT, DEFAULT_SEARCH_LIMIT, load_movies, cosine_similarity

class MultimodalSearch:
    def __init__(self, documents: list[dict] = [], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts: list[str] = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def search_with_image(self, image_path, limit):
        img_embedding = self.embed_image(image_path)
        for i, embed in enumerate(self.text_embeddings):
            self.documents[i]['cosine_sim'] = cosine_similarity(img_embedding, embed)
        return sorted(self.documents, key=lambda x: x['cosine_sim'], reverse=True)[:limit]
        
    def embed_image(self, image_path):
        full_path = os.path.join(PROJECT_ROOT, image_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")
        
        img = Image.open(full_path)
        return self.model.encode([img], show_progress_bar=True)[0]
    

def verify_image_embedding(image_path):
    mms = MultimodalSearch()
    embedding = mms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path, limit=DEFAULT_SEARCH_LIMIT):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    documents = load_movies()
    mms = MultimodalSearch(documents)
    return mms.search_with_image(image_path, limit)