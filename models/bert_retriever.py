"""
BERT-based retriever model.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Dict, Tuple

class BERTRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize BERT-based retrieval system
        Using sentence-transformers model optimized for semantic similarity
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
            """Encode list of texts into embeddings"""
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded)
                    sentence_embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
                    all_embeddings.append(sentence_embeddings.cpu().numpy())
            
            return np.vstack(all_embeddings)

    
    def build_index(self, documents: List[str], metadata: List[Dict] = None):
        """Build vector index from documents"""
        print(f"Encoding {len(documents)} documents...")
        self.documents = documents
        self.metadata = metadata or []
        self.embeddings = self.encode_texts(documents)
        print(f"Built index with {self.embeddings.shape[0]} documents")
        
    def calculate_cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and all documents"""
        return cosine_similarity(query_embedding, self.embeddings)[0]
    
    def calculate_euclidean_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate Euclidean similarity (inverse of distance) between query and all documents"""
        distances = euclidean_distances(query_embedding, self.embeddings)[0]
        # Convert distances to similarities (higher is better)
        # Use negative distance so higher values indicate better matches
        return -distances
    
    def calculate_yat_similarity(self, query_embedding: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        Calculate YAT similarity: (x·w)² / (||x-w||² + ε)
        Where x is query, w is document embedding
        """
        similarities = []
        query_vec = query_embedding[0]  # Remove batch dimension
        
        for doc_vec in self.embeddings:
            # Calculate dot product
            dot_product = np.dot(query_vec, doc_vec)
            
            # Calculate squared Euclidean distance
            diff = query_vec - doc_vec
            squared_distance = np.dot(diff, diff)
            
            # Calculate YAT similarity
            yat_sim = (dot_product ** 2) / (squared_distance + epsilon)
            similarities.append(yat_sim)
        
        return np.array(similarities)
    
    def calculate_manhattan_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate Manhattan similarity (inverse of L1 distance)"""
        query_vec = query_embedding[0]
        similarities = []
        
        for doc_vec in self.embeddings:
            manhattan_distance = np.sum(np.abs(query_vec - doc_vec))
            similarities.append(-manhattan_distance) # Negative distance, so higher is better
        
        return np.array(similarities)
    
    def calculate_dot_product_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate dot product similarity"""
        return np.dot(query_embedding[0], self.embeddings.T)
    
    def search(self, query: str, top_k: int = 5, similarity_metric: str = "cosine") -> List[Tuple[str, float, int]]:
        """
        Search for most similar documents using specified similarity metric
        
        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_metric: One of 'cosine', 'euclidean', 'yat', 'manhattan', 'dot_product'
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encode_texts([query])
        
        # Calculate similarities based on chosen metric
        if similarity_metric == "cosine":
            similarities = self.calculate_cosine_similarity(query_embedding)
        elif similarity_metric == "euclidean":
            similarities = self.calculate_euclidean_similarity(query_embedding)
        elif similarity_metric == "yat":
            similarities = self.calculate_yat_similarity(query_embedding)
        elif similarity_metric == "manhattan":
            similarities = self.calculate_manhattan_similarity(query_embedding)
        elif similarity_metric == "dot_product":
            similarities = self.calculate_dot_product_similarity(query_embedding)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        # Get top-k results
        # Ensure k is not larger than the number of documents
        effective_top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:effective_top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = float(similarities[idx])
            results.append((doc, score, idx)) # Original index 'idx' is preserved
            
        return results
