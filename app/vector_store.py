import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.config import settings


class VectorStore:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.document_metadata = []
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        
    def create_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        if not self.index:
            self.create_index()
            
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        if metadata:
            self.document_metadata.extend(metadata)
        else:
            self.document_metadata.extend([{"source": f"doc_{i}"} for i in range(len(documents))])
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.index or self.index.ntotal == 0:
            return []
            
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.document_metadata[idx],
                    "score": float(score),
                    "rank": i + 1
                })
        
        return results
    
    def save(self, path: str):
        # Ensure the directory exists
        dir_path = os.path.dirname(path)
        if dir_path:  # Only create directory if path has directory components
            os.makedirs(dir_path, exist_ok=True)
        
        # Create the main directory for the vector store
        os.makedirs(path, exist_ok=True)
        
        faiss.write_index(self.index, f"{path}/index.faiss")
        
        with open(f"{path}/documents.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.document_metadata
            }, f)
    
    def load(self, path: str):
        if os.path.exists(f"{path}/index.faiss"):
            self.index = faiss.read_index(f"{path}/index.faiss")
            
            with open(f"{path}/documents.pkl", "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.document_metadata = data["metadata"]
                
            return True
        return False


def load_product_documents(data_dir: str = "./data") -> List[Dict[str, Any]]:
    documents = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                documents.append({
                    "content": content,
                    "metadata": {
                        "source": filename,
                        "filepath": filepath
                    }
                })
    
    return documents


def initialize_vector_store() -> VectorStore:
    vector_store = VectorStore()
    
    # Check if we have a saved index
    if vector_store.load(settings.vector_store_path):
        print(f"Loaded existing vector store from {settings.vector_store_path}")
        return vector_store
    
    # Load and index documents
    product_docs = load_product_documents()
    if product_docs:
        documents = [doc["content"] for doc in product_docs]
        metadata = [doc["metadata"] for doc in product_docs]
        
        vector_store.add_documents(documents, metadata)
        vector_store.save(settings.vector_store_path)
        
        print(f"Indexed {len(documents)} documents and saved to {settings.vector_store_path}")
    
    return vector_store