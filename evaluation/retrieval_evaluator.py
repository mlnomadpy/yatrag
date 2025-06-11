"""
Retrieval evaluator module.
"""
import pandas as pd # Added import
import matplotlib.pyplot as plt # Added import
import math # Added import
from typing import List, Dict, Any

class RetrievalEvaluator:
    def __init__(self):
        self.categories = ['World', 'Sports', 'Business', 'Sci/Tech']
        # Ensure this list is consistent with BERTRetriever or configurable
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
                relevant_positions.append(i + 1) # rank is i+1

        if not relevant_positions:
            return 0.0

        ap = 0.0
        for i, rank in enumerate(relevant_positions):
            precision_at_rank = (i + 1) / rank # (number of relevant docs found so far) / rank of current relevant doc
            ap += precision_at_rank

        return ap / len(relevant_positions)

    def calculate_mrr(self, retrieved_labels: List[int], target_label: int) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        for i, label in enumerate(retrieved_labels):
            if label == target_label:
                return 1.0 / (i + 1) # rank is i+1
        return 0.0

    def calculate_ndcg_at_k(self, retrieved_labels: List[int], target_label: int, k: int, total_relevant_in_collection: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K)"""
        # Calculate DCG@k
        dcg = 0.0
        for i in range(min(k, len(retrieved_labels))):
            relevance = 1 if retrieved_labels[i] == target_label else 0
            if relevance > 0:
                dcg += relevance / math.log2(i + 2) # rank is i+1, so denominator is log2(rank+1) -> log2((i+1)+1)

        # Calculate IDCG@k
        idcg = 0.0
        # Ideal ranking would have all relevant items (relevance=1) at the top.
        # Number of items to consider for IDCG is min(k, total_relevant_in_collection)
        for i in range(min(k, total_relevant_in_collection)):
            idcg += 1.0 / math.log2(i + 2) # relevance is 1 for ideal items

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_query(self, retriever: Any, query: str, target_category: str,
                      similarity_metric: str, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Evaluate a single query with a specific similarity metric"""
        category_to_label = {cat: i for i, cat in enumerate(self.categories)}
        target_label = category_to_label[target_category]

        results = retriever.search(query, top_k=max(k_values), similarity_metric=similarity_metric)

        retrieved_labels = []
        for doc, score, idx in results:
            if retriever.metadata and 0 <= idx < len(retriever.metadata):
                retrieved_labels.append(retriever.metadata[idx]['label'])
            else:
                # If metadata is missing for a valid index, it's treated as non-relevant as its label cannot be determined.
                # This might indicate an issue with data preparation if idx should always have metadata.
                print(f"Warning: Metadata not found for document index {idx} while evaluating query '{query}'. Document will be treated as non-relevant.")
                # Optionally, append a special non-matching label, or simply skip. Skipping means it's not in retrieved_labels.

        total_relevant_in_collection = sum(1 for meta in retriever.metadata if meta['label'] == target_label)

        metrics = {'query': query, 'target_category': target_category, 'similarity_metric': similarity_metric}

        for k in k_values:
            precision_k = self.calculate_precision_at_k(retrieved_labels, target_label, k)
            recall_k = self.calculate_recall_at_k(retrieved_labels, target_label, k, total_relevant_in_collection)
            f1_k = self.calculate_f1_at_k(precision_k, recall_k)
            # Pass total_relevant_in_collection for NDCG calculation
            ndcg_k = self.calculate_ndcg_at_k(retrieved_labels, target_label, k, total_relevant_in_collection)

            metrics[f'precision@{k}'] = precision_k
            metrics[f'recall@{k}'] = recall_k
            metrics[f'f1@{k}'] = f1_k
            metrics[f'ndcg@{k}'] = ndcg_k

        metrics['map'] = self.calculate_map(retrieved_labels, target_label)
        metrics['mrr'] = self.calculate_mrr(retrieved_labels, target_label)
        metrics['success_rate'] = 1.0 if any(label == target_label for label in retrieved_labels) else 0.0

        return metrics

    def debug_similarity_calculations(self, retriever: Any, query: str, debug_docs: int = 5):
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
        print(f"{'Doc ID':<8} {'Category':<10} {'Cosine':<12} {'Euclidean':<12} {'YAT':<12} {'Manhattan':<12} {'Dot Product':<12}")
        for i in range(min(debug_docs, len(retriever.documents))):
            category = "N/A"
            if retriever.metadata and i < len(retriever.metadata) and 'category' in retriever.metadata[i]:
                category = retriever.metadata[i]['category']
            print(f"{i:<8} {category:<10} {cosine_sims[i]:<12.4f} {euclidean_sims[i]:<12.4f} {yat_sims[i]:<12.4f} {manhattan_sims[i]:<12.4f} {dot_product_sims[i]:<12.4f}")
        print("-" * 80)


    def compare_similarity_metrics(self, retriever: Any, test_queries: Dict[str, List[str]],
                                 k_values: List[int] = [1, 3, 5, 10], debug: bool = False) -> pd.DataFrame:
        all_results = []

        first_query_debugged_overall = False # To debug only the very first query overall if debug is True

        for category, queries in test_queries.items():
            print(f"Evaluating category: {category} ({len(queries)} queries)")
            for i, query in enumerate(queries):
                if debug and not first_query_debugged_overall:
                    self.debug_similarity_calculations(retriever, query)
                    first_query_debugged_overall = True # Set flag after debugging one query

                for metric_name in self.similarity_metrics:
                    query_eval_results = self.evaluate_query(retriever, query, category, metric_name, k_values)
                    all_results.append(query_eval_results)
                if (i + 1) % 10 == 0: # Print progress every 10 queries per category
                    print(f"  Processed {i+1}/{len(queries)} queries for {category}...")


        results_df = pd.DataFrame(all_results)
        return results_df

    def analyze_metric_comparison(self, results_df: pd.DataFrame, k_values: List[int] = [1, 3, 5, 10]) -> None:
        print("\\n" + "="*100)
        print("COMPREHENSIVE METRIC COMPARISON ANALYSIS")
        print("="*100)

        if results_df.empty:
            print("No results to analyze.")
            return

        # Overall average performance per metric
        print("\\nOverall Average Performance Across All Queries and Categories:")
        avg_metrics_cols = [f'{m}@{k}' for k in k_values for m in ['precision', 'recall', 'f1', 'ndcg']] + \
                           ['map', 'mrr', 'success_rate']

        # Ensure all expected metric columns exist for aggregation, fill with NaN if not (though groupby().mean() handles this)
        for metric_col in avg_metrics_cols:
            if metric_col not in results_df.columns:
                results_df[metric_col] = pd.NA # Or float('nan')

        overall_summary = results_df.groupby('similarity_metric')[avg_metrics_cols].mean(numeric_only=True)
        print(overall_summary.to_string()) # Print full summary

        # Plotting key metrics
        plot_metrics = {
            'Precision': [f'precision@{k}' for k in k_values],
            'Recall': [f'recall@{k}' for k in k_values],
            'F1-Score': [f'f1@{k}' for k in k_values],
            'NDCG': [f'ndcg@{k}' for k in k_values],
            'MAP': ['map'],
            'MRR': ['mrr'],
            'Success Rate': ['success_rate']
        }

        for main_metric_name, specific_metric_cols in plot_metrics.items():
            for metric_col in specific_metric_cols:
                if metric_col in overall_summary.columns:
                    plt.figure(figsize=(10, 6)) # Adjusted figure size
                    overall_summary[metric_col].plot(kind='bar')
                    plt.title(f'Average {metric_col.replace("@", "@K=")} by Similarity Metric')
                    plt.ylabel(metric_col)
                    plt.xlabel('Similarity Metric')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(axis='y', linestyle='--')
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"Warning: Metric column '{metric_col}' not found in summary for plotting.")

        # Performance by Category (e.g., MAP by category for each metric)
        if 'map' in results_df.columns and 'target_category' in results_df.columns:
            map_by_category = results_df.groupby(['similarity_metric', 'target_category'])['map'].mean(numeric_only=True).unstack()
            if not map_by_category.empty:
                map_by_category.plot(kind='bar', figsize=(14, 8), width=0.8) # Adjusted figure size
                plt.title('Average MAP by Category and Similarity Metric')
                plt.ylabel('MAP')
                plt.xlabel('Similarity Metric')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Category')
                plt.grid(axis='y', linestyle='--')
                plt.tight_layout()
                plt.show()
            else:
                print("Warning: MAP by category data is empty, skipping plot.")
        else:
            print("Warning: 'map' or 'target_category' column not found, skipping MAP by category plot.")

        print("\\n" + "="*100)
        print("ANALYSIS COMPLETE")
        print("="*100)
