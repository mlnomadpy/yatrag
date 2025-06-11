import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
from typing import List, Tuple, Dict
import warnings
import math
from collections import defaultdict
warnings.filterwarnings('ignore')

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
                    embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
                    # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) # <--- COMMENT OUT OR REMOVE THIS LINE
                    all_embeddings.append(embeddings.cpu().numpy())
            
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
            # Calculate Manhattan distance
            manhattan_distance = np.sum(np.abs(query_vec - doc_vec))
            # Convert to similarity (higher is better)
            similarity = 1.0 / (1.0 + manhattan_distance)
            similarities.append(similarity)
        
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
            raise ValueError("No index built. Call build_index() first.")
        
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
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx]), int(idx)))
            
        return results

class RetrievalEvaluator:
    def __init__(self):
        self.categories = ['World', 'Sports', 'Business', 'Sci/Tech']
        self.similarity_metrics = ['cosine', 'euclidean', 'yat', 'manhattan', 'dot_product']
        
    def calculate_precision_at_k(self, retrieved_labels: List[int], target_label: int, k: int) -> float:
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        retrieved_k = retrieved_labels[:k]
        relevant_count = sum(1 for label in retrieved_k if label == target_label)
        return relevant_count / k
    
    def calculate_recall_at_k(self, retrieved_labels: List[int], target_label: int, k: int, total_relevant: int) -> float:
        """Calculate Recall@K"""
        if total_relevant == 0:
            return 0.0
        retrieved_k = retrieved_labels[:k]
        relevant_count = sum(1 for label in retrieved_k if label == target_label)
        return relevant_count / total_relevant
    
    def calculate_f1_at_k(self, precision: float, recall: float) -> float:
        """Calculate F1@K"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_map(self, retrieved_labels: List[int], target_label: int) -> float:
        """Calculate Mean Average Precision (MAP)"""
        relevant_positions = []
        for i, label in enumerate(retrieved_labels):
            if label == target_label:
                relevant_positions.append(i + 1)
        
        if not relevant_positions:
            return 0.0
        
        ap = 0.0
        for i, pos in enumerate(relevant_positions):
            precision_at_pos = (i + 1) / pos
            ap += precision_at_pos
        
        return ap / len(relevant_positions)
    
    def calculate_mrr(self, retrieved_labels: List[int], target_label: int) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        for i, label in enumerate(retrieved_labels):
            if label == target_label:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_ndcg_at_k(self, retrieved_labels: List[int], target_label: int, k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K)"""
        def dcg(labels, target):
            dcg_score = 0.0
            for i, label in enumerate(labels[:k]):
                relevance = 1 if label == target else 0
                dcg_score += relevance / math.log2(i + 2)
            return dcg_score
        
        # Calculate DCG
        dcg_score = dcg(retrieved_labels, target_label)
        
        # Calculate IDCG (ideal DCG)
        ideal_labels = [target_label] * min(k, sum(1 for l in retrieved_labels if l == target_label))
        ideal_labels.extend([l for l in retrieved_labels if l != target_label][:k - len(ideal_labels)])
        idcg_score = dcg(ideal_labels, target_label)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0
    
    def evaluate_query(self, retriever: BERTRetriever, query: str, target_category: str, 
                      similarity_metric: str, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Evaluate a single query with a specific similarity metric"""
        # Map category to label
        category_to_label = {cat: i for i, cat in enumerate(self.categories)}
        target_label = category_to_label[target_category]
        
        # Get retrieval results
        results = retriever.search(query, top_k=max(k_values), similarity_metric=similarity_metric)
        
        # Extract labels from results
        retrieved_labels = []
        for doc, score, idx in results:
            if idx < len(retriever.metadata):
                retrieved_labels.append(retriever.metadata[idx]['label'])
            else:
                retrieved_labels.append(-1)  # Unknown label
        
        # Count total relevant documents
        total_relevant = sum(1 for meta in retriever.metadata if meta['label'] == target_label)
        
        # Calculate metrics
        metrics = {}
        
        # Precision, Recall, F1 at different K values
        for k in k_values:
            precision = self.calculate_precision_at_k(retrieved_labels, target_label, k)
            recall = self.calculate_recall_at_k(retrieved_labels, target_label, k, total_relevant)
            f1 = self.calculate_f1_at_k(precision, recall)
            
            metrics[f'precision@{k}'] = precision
            metrics[f'recall@{k}'] = recall
            metrics[f'f1@{k}'] = f1
            metrics[f'ndcg@{k}'] = self.calculate_ndcg_at_k(retrieved_labels, target_label, k)
        
        # MAP and MRR
        metrics['map'] = self.calculate_map(retrieved_labels, target_label)
        metrics['mrr'] = self.calculate_mrr(retrieved_labels, target_label)
        
        # Success rate (whether any relevant document is retrieved)
        metrics['success_rate'] = 1.0 if any(label == target_label for label in retrieved_labels) else 0.0
        
        return metrics
    
    def debug_similarity_calculations(self, retriever: BERTRetriever, query: str, debug_docs: int = 5):
        """Debug similarity calculations to verify they're different"""
        print(f"\nDEBUG: Similarity calculations for query: '{query}'")
        print("=" * 80)
        
        # Encode query
        query_embedding = retriever.encode_texts([query])
        
        # Calculate all similarities
        cosine_sims = retriever.calculate_cosine_similarity(query_embedding)
        euclidean_sims = retriever.calculate_euclidean_similarity(query_embedding)
        yat_sims = retriever.calculate_yat_similarity(query_embedding)
        manhattan_sims = retriever.calculate_manhattan_similarity(query_embedding)
        dot_product_sims = retriever.calculate_dot_product_similarity(query_embedding)
        
        print(f"Showing similarity scores for first {debug_docs} documents:")
        print("-" * 80)
        
        for i in range(min(debug_docs, len(retriever.documents))):
            print(f"\nDoc {i}: {retriever.documents[i][:100]}...")
            print(f"  Cosine:      {cosine_sims[i]:.6f}")
            print(f"  Euclidean:   {euclidean_sims[i]:.6f}")
            print(f"  YAT:         {yat_sims[i]:.6f}")
            print(f"  Manhattan:   {manhattan_sims[i]:.6f}")
            print(f"  Dot Product: {dot_product_sims[i]:.6f}")
        
        # Check if similarities are identical
        print(f"\nSimilarity uniqueness check:")
        print(f"  Cosine == Euclidean: {np.allclose(cosine_sims, euclidean_sims)}")
        print(f"  Cosine == YAT: {np.allclose(cosine_sims, yat_sims)}")
        print(f"  Euclidean == YAT: {np.allclose(euclidean_sims, yat_sims)}")
        
        # Show top documents for each metric
        print(f"\nTop 3 documents by each metric:")
        metrics = {
            'cosine': cosine_sims,
            'euclidean': euclidean_sims,
            'yat': yat_sims,
            'manhattan': manhattan_sims,
            'dot_product': dot_product_sims
        }
        
        for metric_name, scores in metrics.items():
            top_indices = np.argsort(scores)[::-1][:3]
            print(f"\n{metric_name.upper()}:")
            for j, idx in enumerate(top_indices, 1):
                category = "Unknown"
                if idx < len(retriever.metadata):
                    category = retriever.metadata[idx]['category']
                print(f"  {j}. Score: {scores[idx]:.6f} | Category: {category}")
    
    def compare_similarity_metrics(self, retriever: BERTRetriever, test_queries: Dict[str, List[str]], 
                                 k_values: List[int] = [1, 3, 5, 10], debug: bool = False) -> pd.DataFrame:
        """Compare all similarity metrics across all queries"""
        results = []
        
        print("Comparing similarity metrics...")
        print("=" * 80)
        
        if debug:
            # Run debug for first query
            first_category = list(test_queries.keys())[0]
            first_query = test_queries[first_category][0]
            self.debug_similarity_calculations(retriever, first_query)
        
        total_queries = sum(len(queries) for queries in test_queries.values())
        query_count = 0
        
        for category, queries in test_queries.items():
            print(f"\nEvaluating {category} queries...")
            
            for i, query in enumerate(queries):
                query_count += 1
                print(f"  Query {query_count}/{total_queries}: {query[:50]}...")
                
                # Test each similarity metric
                for metric in self.similarity_metrics:
                    metrics = self.evaluate_query(retriever, query, category, metric, k_values)
                    metrics['category'] = category
                    metrics['query'] = query
                    metrics['similarity_metric'] = metric
                    
                    results.append(metrics)
        
        return pd.DataFrame(results)
    
    def analyze_metric_comparison(self, results_df: pd.DataFrame) -> None:
        """Analyze and display comparison results between similarity metrics"""
        print("\n" + "="*100)
        print("SIMILARITY METRICS COMPARISON ANALYSIS")
        print("="*100)
        
        # Overall performance by metric
        print("\n1. OVERALL PERFORMANCE BY SIMILARITY METRIC:")
        print("-" * 60)
        
        numeric_cols = [col for col in results_df.columns 
                       if col not in ['category', 'query', 'similarity_metric']]
        
        overall_by_metric = results_df.groupby('similarity_metric')[numeric_cols].mean()
        
        # Display key metrics
        key_metrics = ['precision@1', 'precision@5', 'recall@5', 'f1@5', 'map', 'mrr', 'ndcg@5']
        comparison_table = overall_by_metric[key_metrics].round(4)
        
        print(comparison_table.to_string())
        
        # Rank metrics by performance
        print("\n2. METRIC RANKINGS (by F1@5):")
        print("-" * 40)
        f1_ranking = overall_by_metric['f1@5'].sort_values(ascending=False)
        for i, (metric, score) in enumerate(f1_ranking.items(), 1):
            print(f"{i}. {metric:12s}: {score:.4f}")
        
        # Statistical significance tests (basic)
        print("\n3. PERFORMANCE DIFFERENCES:")
        print("-" * 40)
        
        metrics_list = list(self.similarity_metrics)
        for i in range(len(metrics_list)):
            for j in range(i+1, len(metrics_list)):
                metric1, metric2 = metrics_list[i], metrics_list[j]
                
                data1 = results_df[results_df['similarity_metric'] == metric1]['f1@5']
                data2 = results_df[results_df['similarity_metric'] == metric2]['f1@5']
                
                diff = data1.mean() - data2.mean()
                print(f"{metric1} vs {metric2}: {diff:+.4f} (F1@5)")
        
        # Per-category analysis
        print("\n4. PER-CATEGORY PERFORMANCE:")
        print("-" * 60)
        
        category_comparison = results_df.groupby(['category', 'similarity_metric'])[key_metrics].mean()
        
        for category in self.categories:
            print(f"\n{category}:")
            cat_data = category_comparison.loc[category][['precision@5', 'recall@5', 'f1@5', 'map']]
            print(cat_data.round(4).to_string())
        
        # Best metric per category
        print("\n5. BEST METRIC PER CATEGORY (by F1@5):")
        print("-" * 50)
        
        for category in self.categories:
            cat_data = results_df[results_df['category'] == category]
            best_metric = cat_data.groupby('similarity_metric')['f1@5'].mean().idxmax()
            best_score = cat_data.groupby('similarity_metric')['f1@5'].mean().max()
            print(f"{category:10s}: {best_metric:12s} ({best_score:.4f})")
        
        # Variance analysis
        print("\n6. PERFORMANCE VARIANCE ANALYSIS:")
        print("-" * 50)
        
        variance_analysis = results_df.groupby('similarity_metric')['f1@5'].agg(['mean', 'std', 'min', 'max'])
        print(variance_analysis.round(4).to_string())
        
        # Save detailed comparison
        results_df.to_csv('similarity_metrics_comparison.csv', index=False)
        overall_by_metric.to_csv('metrics_summary_comparison.csv')
        
        print(f"\nDetailed results saved to 'similarity_metrics_comparison.csv'")
        print(f"Summary comparison saved to 'metrics_summary_comparison.csv'")

def load_toy_dataset():
    """Load and prepare AG News dataset from Hugging Face"""
    print("Loading AG News dataset...")
    
    # Load AG News dataset
    dataset = load_dataset("ag_news")
    
    # Get the training split
    train_data = dataset['train']
    
    # Convert to lists and sample first 1000 examples for better evaluation
    documents = []
    labels = []
    
    # Take first 1000 examples
    for i in range(min(1000, len(train_data))):
        example = train_data[i]
        documents.append(example['text'])
        labels.append(example['label'])
    
    print(f"Loaded {len(documents)} documents")
    
    # Category mapping
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    # Create metadata
    metadata = []
    for i, label in enumerate(labels):
        category = categories[label]
        metadata.append({
            'id': i,
            'label': label,
            'category': category
        })
    
    # Create comprehensive test queries
# Create comprehensive test queries
    test_queries = {
        'World': [
            "international news and world events",
            "global politics and world affairs",
            "international conflicts and diplomacy",
            "foreign policy and international relations",
            "global economy and trade wars",
            "international terrorism and security",
            "United Nations meetings and resolutions",
            "major earthquakes and natural disasters globally",
            "summit talks between world leaders",
            "border disputes and international tensions",
            "refugee crisis and migration patterns",
            "climate change international agreements",
            "humanitarian aid and global health issues",
            "political unrest and civil conflicts in different countries",
            "European Union developments and policies",
            "NATO operations and security concerns",
            "news from the Middle East",
            "African political and economic news",
            "Asian geopolitical events",
            "news from Latin America"
        ],
        'Sports': [
            "sports news and athletic competitions",
            "football basketball and other sports",
            "athletes and sporting events",
            "Olympic games and championships",
            "professional sports leagues",
            "soccer and football matches",
            "major league baseball news",
            "NBA scores and highlights",
            "NFL updates and game analysis",
            "tennis tournaments and player rankings",
            "golf championships and tour news",
            "motorsports and racing events",
            "winter sports like skiing and snowboarding",
            "track and field events",
            "swimming competitions",
            "college sports news",
            "sports injuries and athlete health",
            "transfer news in football",
            "basketball team standings",
            "highlights from yesterday's games"
        ],
        'Business': [
            "business news and economic updates",
            "corporate earnings and financial markets",
            "stock market and investment news",
            "company mergers and acquisitions",
            "economic indicators and GDP growth",
            "banking and financial services",
            "Wall Street news and trends",
            "technology stock performance",
            "energy prices and oil markets",
            "real estate market trends",
            "unemployment rates and job market data",
            "inflation and monetary policy",
            "venture capital and startup funding",
            "cryptocurrency news and bitcoin price",
            "consumer spending and retail sales",
            "international trade agreements",
            "corporate governance and ethics",
            "industry analysis and sector reports",
            "small business news and resources",
            "economic forecasts and predictions"
        ],
        'Sci/Tech': [
            "technology news and scientific discoveries",
            "computer science and technology updates",
            "scientific research and innovations",
            "artificial intelligence and machine learning",
            "biotechnology and medical breakthroughs",
            "space exploration and astronomy",
            "latest in quantum computing",
            "developments in renewable energy technology",
            "genetics research and gene editing",
            "advances in robotics",
            " cybersecurity news and data breaches",
            "software development and programming trends",
            "latest smartphone releases and reviews",
            "internet of things (IoT) applications",
            "virtual reality and augmented reality news",
            "neuroscience research and brain studies",
            "new findings in physics",
            "environmental science and conservation efforts",
            "medical technology and treatments",
            "innovations in transportation technology"
        ]
    }
    
    print(f"Successfully loaded AG News dataset with {len(documents)} documents")
    print(f"Categories: {categories}")
    
    # Show category distribution
    category_counts = {}
    for meta in metadata:
        cat = meta['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("Document distribution:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} documents")
    
    return documents, test_queries, metadata

def demo_similarity_metrics():
    """Demonstrate different similarity metrics with sample queries"""
    documents, test_queries, metadata = load_toy_dataset()
    
    retriever = BERTRetriever()
    retriever.build_index(documents, metadata)
    
    print("\n" + "="*100)
    print("SIMILARITY METRICS DEMONSTRATION")
    print("="*100)
    
    # Test with sample queries
    sample_queries = [
        ("artificial intelligence and machine learning", "Sci/Tech"),
        ("Olympic games and championships", "Sports"),
        ("stock market and investment news", "Business")
    ]
    
    similarity_metrics = ['cosine', 'euclidean', 'yat', 'manhattan', 'dot_product']
    
    for query, expected_category in sample_queries:
        print(f"\nQuery: '{query}' (Expected: {expected_category})")
        print("=" * 80)
        
        for metric in similarity_metrics:
            print(f"\n{metric.upper()} SIMILARITY:")
            print("-" * 40)
            
            results = retriever.search(query, top_k=3, similarity_metric=metric)
            
            for i, (doc, score, idx) in enumerate(results, 1):
                category = "Unknown"
                if idx < len(metadata):
                    category = metadata[idx]['category']
                
                print(f"{i}. Score: {score:.6f} | Category: {category}")
                print(f"   Text: {doc[:100]}...")
                print()

def run_comprehensive_comparison():
    """Run comprehensive comparison of all similarity metrics"""
    # Load dataset
    documents, test_queries, metadata = load_toy_dataset()
    
    # Initialize retriever
    retriever = BERTRetriever()
    retriever.build_index(documents, metadata)
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator()
    
    # Run comparison with debug enabled
    results_df = evaluator.compare_similarity_metrics(retriever, test_queries, k_values=[1, 3, 5, 10], debug=True)
    
    # Analyze results
    evaluator.analyze_metric_comparison(results_df)
    
    return results_df

if __name__ == "__main__":
    print("BERT-based Text Retrieval System - Similarity Metrics Comparison")
    print("=" * 80)
    
    # Run demonstration
    # demo_similarity_metrics()
    
    # Run comprehensive comparison
    results_df = run_comprehensive_comparison()