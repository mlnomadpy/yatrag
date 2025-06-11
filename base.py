"""
Main script for running BERT-based text retrieval system.
"""
import warnings
from models import BERTRetriever
from evaluation import RetrievalEvaluator
from utils import load_toy_dataset

warnings.filterwarnings('ignore')

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