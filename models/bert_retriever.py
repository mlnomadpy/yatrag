"""
BERT-based retriever model.
"""
import torch
import torch.nn.functional as F  # Added for F.cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModel
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances # Removed as PyTorch alternatives are used
from typing import List, Dict, Tuple

class BERTRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize BERT-based retrieval system
        Using sentence-transformers model optimized for semantic similarity
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.documents: List[str] = []
        self.embeddings: torch.Tensor | None = None  # Changed from np.ndarray
        self.metadata: List[Dict] = []
        
    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:  # Return type changed
            """Encode list of texts into embeddings as PyTorch tensors on the active device"""
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
                    all_embeddings.append(sentence_embeddings)  # Store as tensor
            
            return torch.cat(all_embeddings, dim=0)  # Concatenate tensors

    
    def build_index(self, documents: List[str], metadata: List[Dict] = None):
        """Build vector index from documents"""
        print(f"Encoding {len(documents)} documents...")
        self.documents = documents
        self.metadata = metadata or []
        self.embeddings = self.encode_texts(documents)  # self.embeddings is now a tensor
        print(f"Built index with {self.embeddings.shape[0]} documents, tensor on device: {self.embeddings.device}")
        
    def calculate_cosine_similarity(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Calculate cosine similarity between query and all documents using PyTorch"""
        # query_embedding: (1, D), self.embeddings: (N, D)
        # F.cosine_similarity broadcasts query_embedding and computes similarity for each document.
        return F.cosine_similarity(query_embedding, self.embeddings, dim=1)  # Returns 1D tensor of size N
    
    def calculate_euclidean_similarity(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Calculate Euclidean similarity (inverse of distance) between query and all documents using PyTorch"""
        # query_embedding: (1, D), self.embeddings: (N, D)
        # torch.cdist computes pairwise distances. Result for (1,D) vs (N,D) is (1,N).
        distances = torch.cdist(query_embedding, self.embeddings)[0]  # Get 1D tensor of size N
        # We use negative distance because higher similarity should be better (like cosine similarity).
        # Alternatively, similarity = 1 / (1 + distance)
        return -distances
    
    def calculate_yat_similarity(self, query_embedding: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Calculate YAT similarity: (x·w)² / (||x-w||² + ε) using PyTorch
        Where x is query (1,D), w is document embeddings (N,D)
        """
        # query_embedding (1,D) is broadcast against self.embeddings (N,D)
        dot_products = torch.sum(query_embedding * self.embeddings, dim=1)  # Shape: (N)
        
        diff = query_embedding - self.embeddings  # Shape: (N, D)
        squared_distances = torch.sum(diff * diff, dim=1)  # Shape: (N)
        
        yat_sims = (dot_products ** 2) / (squared_distances + epsilon)
        return yat_sims
    
    def calculate_manhattan_similarity(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Calculate Manhattan similarity (inverse of L1 distance) using PyTorch"""
        # query_embedding (1,D) is broadcast against self.embeddings (N,D)
        manhattan_distances = torch.sum(torch.abs(query_embedding - self.embeddings), dim=1) # Shape (N)
        # We use negative distance because higher similarity should be better.
        # Alternatively, similarity = 1 / (1 + distance)
        return -manhattan_distances
    
    def calculate_dot_product_similarity(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Calculate dot product similarity using PyTorch"""
        # query_embedding: (1, D), self.embeddings: (N, D)
        # (1,D) @ (D,N) -> (1,N). Squeeze to (N).
        return torch.matmul(query_embedding, self.embeddings.T).squeeze()
    
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
        
        # Encode query, result is a tensor on self.device, shape (1, D)
        query_embedding = self.encode_texts([query])
        
        # Calculate similarities based on chosen metric. Result is a 1D tensor on self.device.
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
        effective_top_k = min(top_k, len(self.documents))
        
        # Use torch.topk to get scores and indices directly
        top_scores, top_indices = torch.topk(similarities, k=effective_top_k, largest=True)
        
        results = []
        for i in range(effective_top_k):
            idx = top_indices[i].item()          # Convert tensor index to Python int
            score = top_scores[i].item()         # Convert tensor score to Python float
            doc = self.documents[idx]
            results.append((doc, score, idx)) # Original index 'idx' is preserved
            
        return results
