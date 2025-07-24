import pytest
import tempfile
import os
from app.vector_store import VectorStore, load_product_documents


class TestVectorStore:
    
    def test_vector_store_initialization(self):
        vector_store = VectorStore()
        assert vector_store.index is None
        assert vector_store.documents == []
        assert vector_store.document_metadata == []
        assert vector_store.dimension == 384
    
    def test_create_index(self):
        vector_store = VectorStore()
        vector_store.create_index()
        assert vector_store.index is not None
        assert vector_store.index.ntotal == 0
    
    def test_add_documents(self):
        vector_store = VectorStore()
        
        documents = [
            "This is a test document about smartphones",
            "This is another document about laptops"
        ]
        
        metadata = [
            {"source": "doc1.txt"},
            {"source": "doc2.txt"}
        ]
        
        vector_store.add_documents(documents, metadata)
        
        assert len(vector_store.documents) == 2
        assert len(vector_store.document_metadata) == 2
        assert vector_store.index.ntotal == 2
        assert vector_store.documents[0] == documents[0]
        assert vector_store.document_metadata[0] == metadata[0]
    
    def test_add_documents_without_metadata(self):
        vector_store = VectorStore()
        
        documents = ["Test document"]
        vector_store.add_documents(documents)
        
        assert len(vector_store.documents) == 1
        assert len(vector_store.document_metadata) == 1
        assert "source" in vector_store.document_metadata[0]
    
    def test_search_empty_store(self):
        vector_store = VectorStore()
        results = vector_store.search("test query")
        assert results == []
    
    def test_search_with_documents(self):
        vector_store = VectorStore()
        
        documents = [
            "iPhone 15 Pro Max with advanced camera system",
            "MacBook Air with M2 chip and long battery life",
            "AirPods Pro with noise cancellation"
        ]
        
        vector_store.add_documents(documents)
        
        results = vector_store.search("iPhone camera", top_k=2)
        
        assert len(results) <= 2
        assert all("content" in result for result in results)
        assert all("metadata" in result for result in results)
        assert all("score" in result for result in results)
        assert all("rank" in result for result in results)
        
        # Check that results are ranked by relevance
        if len(results) > 1:
            assert results[0]["rank"] == 1
            assert results[1]["rank"] == 2
    
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore()
            
            documents = ["Test document for save/load"]
            metadata = [{"source": "test.txt"}]
            
            vector_store.add_documents(documents, metadata)
            
            # Save
            save_path = os.path.join(temp_dir, "test_store")
            vector_store.save(save_path)
            
            # Load into new instance
            new_vector_store = VectorStore()
            loaded = new_vector_store.load(save_path)
            
            assert loaded is True
            assert len(new_vector_store.documents) == 1
            assert new_vector_store.documents[0] == documents[0]
            assert new_vector_store.document_metadata[0] == metadata[0]
            assert new_vector_store.index.ntotal == 1
    
    def test_load_nonexistent_store(self):
        vector_store = VectorStore()
        loaded = vector_store.load("/nonexistent/path")
        assert loaded is False


class TestProductDocumentLoader:
    
    def test_load_product_documents_empty_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            documents = load_product_documents(temp_dir)
            assert documents == []
    
    def test_load_product_documents_with_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_content = "Product: Test Product\nPrice: $100"
            
            with open(os.path.join(temp_dir, "product1.txt"), "w") as f:
                f.write(test_content)
            
            with open(os.path.join(temp_dir, "product2.txt"), "w") as f:
                f.write("Another product description")
            
            # Create a non-txt file that should be ignored
            with open(os.path.join(temp_dir, "readme.md"), "w") as f:
                f.write("This should be ignored")
            
            documents = load_product_documents(temp_dir)
            
            assert len(documents) == 2
            
            # Check structure
            for doc in documents:
                assert "content" in doc
                assert "metadata" in doc
                assert "source" in doc["metadata"]
                assert "filepath" in doc["metadata"]
                assert doc["metadata"]["source"].endswith(".txt")