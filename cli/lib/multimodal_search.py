import os

from PIL import Image
from sentence_transformers import SentenceTransformer
from .utils import PROJECT_ROOT

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path):
        full_path = os.path.join(PROJECT_ROOT, image_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")
        
        img = Image.open(full_path)
        return self.model.encode([img])[0]
    

def verify_image_embedding(image_path):
    mms = MultimodalSearch()
    embedding = mms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")