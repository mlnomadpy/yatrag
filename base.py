"""
Main script for running BERT-based text retrieval system.

This script demonstrates and evaluates a BERT-based retrieval system using various
similarity metrics. The evaluation focuses on understanding how different metrics
perform in retrieving relevant documents from a toy dataset.

Key Evaluation Metrics Used:
-----------------------------
The following metrics are calculated by `evaluation.retrieval_evaluator.py` to assess
the performance of the retrieval system:

1.  **Precision@K (P@K)**:
    *   What it is: Of the top K documents retrieved for a query, what fraction is relevant?
    *   Importance: Measures the accuracy of the top K results. High P@K means users quickly see
        relevant documents. This is crucial as users often only look at the first few results.

2.  **Recall@K (R@K)**:
    *   What it is: Of all the relevant documents that exist in the entire collection for a query,
        what fraction is found within the top K retrieved documents?
    *   Importance: Measures how many of the total relevant documents are retrieved. High R@K means
        the system is good at finding most of the relevant information. This is important when
        finding all (or most) relevant documents is critical.

3.  **F1-Score@K (F1@K)**:
    *   What it is: The harmonic mean of Precision@K and Recall@K (2 * (P@K * R@K) / (P@K + R@K)).
    *   Importance: Provides a single measure that balances precision and recall. It's useful when
        both finding relevant documents (recall) and ensuring retrieved documents are relevant
        (precision) are equally important.

4.  **Accuracy (Precision@1)**:
    *   What it is: Is the very first document retrieved relevant? (Equivalent to P@1).
    *   Importance: Critical for tasks where the user expects the single best answer or document
        at the top position (e.g., question answering, "I'm feeling lucky" features).

5.  **Mean Average Precision (MAP)**:
    *   What it is: The mean of Average Precision (AP) scores over a set of queries. AP for a single
        query is the average of precision scores calculated after each relevant document is retrieved
        in the ranked list.
    *   Importance: Provides a single-figure measure of quality across recall levels and is a standard
        benchmark in Information Retrieval. It's sensitive to the rank of every relevant item,
        rewarding systems that rank relevant documents higher.

6.  **Mean Reciprocal Rank (MRR)**:
    *   What it is: The average of the reciprocal ranks of the first relevant document retrieved for
        each query. If the first relevant document is ranked 1st, the reciprocal rank is 1; if 2nd,
        it's 1/2; if 3rd, 1/3, and so on. If no relevant document is retrieved, the rank is 0.
    *   Importance: Measures how quickly the system finds the *first* relevant result. It's very
        important for tasks like known-item search or question answering where users want one
        good answer quickly.

7.  **Normalized Discounted Cumulative Gain @K (NDCG@K)**:
    *   What it is: Measures the quality of ranking by considering the position of relevant documents
        (higher is better) and using a graded relevance scale (though in this system, binary
        relevance is used). The gain is accumulated from the top, with a discount factor for
        lower-ranked documents. It's normalized by the ideal ranking (IDCG).
    *   Importance: A sophisticated metric that accounts for the position of relevant items. Higher
        ranked relevant items contribute more to the score. It's particularly useful when the
        degree of relevance matters (not just binary) and the order of results is crucial.

8.  **Success Rate**:
    *   What it is: The proportion of queries for which at least one relevant document was retrieved
        (within the top_k results considered by the search function).
    *   Importance: A basic measure indicating if the system can find *any* relevant information
        for a query, reflecting its ability to satisfy user needs at a fundamental level.

Other Potentially Important Metrics (Not explicitly implemented but good to consider):
------------------------------------------------------------------------------------
*   **Coverage**: Similar to Success Rate, this is the proportion of queries for which the system
    can return at least one relevant document.
*   **Diversity**: Measures how varied the retrieved results are. This is important if a query
    has multiple facets or interpretations, and the user benefits from seeing different aspects.
*   **Novelty**: Assesses whether the retrieved documents provide new information to the user,
    rather than redundant content they might have already seen.
*   **Latency/Query Time**: The actual time taken (e.g., in milliseconds or seconds) to process
    a query and return results. Critical for user experience.
*   **Throughput**: The number of queries the system can handle effectively per unit of time
    (e.g., queries per second). Important for system scalability and performance under load.
"""
import warnings
from models import BERTRetriever
from evaluation import RetrievalEvaluator
from utils import load_toy_dataset
import wandb # Add wandb import

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
    # Initialize wandb run
    wandb.init(project="bert-retrieval-comparison", name="comprehensive_metrics_run")

    # Load dataset
    documents, test_queries, metadata = load_toy_dataset()
    
    # Initialize retriever
    retriever = BERTRetriever()
    retriever.build_index(documents, metadata)
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator()
    
    # Run comparison with debug enabled
    results_df = evaluator.compare_similarity_metrics(retriever, test_queries, k_values=[1, 3, 5, 10], debug=True)
    
    # Log results_df to wandb as a table
    wandb.log({{"results_table": wandb.Table(dataframe=results_df)}})
    
    # Analyze results
    evaluator.analyze_metric_comparison(results_df)
    
    return results_df

# Example of using the parameterized load_toy_dataset function
if __name__ == "__main__":
    # Load AG News (default)
    documents, test_queries, metadata = load_toy_dataset()
    print(f"Loaded {len(documents)} documents and {sum(len(q) for q in test_queries.values())} queries from AG News.")

    # Example: Load a different dataset (e.g., "imdb")
    # You might need to specify text_field, label_field, and category_mapping for other datasets
    # For IMDB, label 0 is negative, 1 is positive.
    # documents_imdb, test_queries_imdb, metadata_imdb = load_toy_dataset(
    #     dataset_name="imdb",
    #     num_documents=500,
    #     num_queries_per_category=10,
    #     text_field="text",
    #     label_field="label",
    #     category_mapping={0: "Negative Review", 1: "Positive Review"}
    # )
    # print(f"Loaded {len(documents_imdb)} documents and {sum(len(q) for q in test_queries_imdb.values())} queries from IMDB.")

    # Initialize retriever and evaluator (assuming these are defined elsewhere)
    # retriever = BERTRetriever(documents, metadata)
    # evaluator = RetrievalEvaluator()
    # results_df = evaluator.compare_similarity_metrics(retriever, test_queries)
    # print(results_df)
    pass # Placeholder for further script logic

    # Finish wandb run
    wandb.finish()