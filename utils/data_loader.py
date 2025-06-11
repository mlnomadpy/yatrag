"""
Data loading utilities.
"""
from datasets import load_dataset
from typing import List, Dict, Tuple

def load_toy_dataset() -> Tuple[List[str], Dict[str, List[str]], List[Dict]]:
    """Load and prepare AG News dataset from Hugging Face"""
    print("Loading AG News dataset...")
    
    # Load AG News dataset
    dataset = load_dataset("ag_news")
    
    # Get the training split for documents
    train_data = dataset['train']
    
    # Convert to lists and sample first 1000 examples for documents
    documents = []
    doc_labels = [] # Renamed from labels to avoid confusion with test set labels
    
    # Take first 1000 examples for documents
    for i in range(min(1000, len(train_data))):
        example = train_data[i]
        documents.append(example['text'])
        doc_labels.append(example['label'])
    
    print(f"Loaded {len(documents)} documents for indexing")
    
    # Category mapping
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    # Create metadata for the indexed documents
    metadata = []
    for i, label_id in enumerate(doc_labels):
        category = categories[label_id]
        metadata.append({
            'id': i, # This id refers to the index in the 'documents' list
            'label': label_id,
            'category': category
        })
    
    # Load the test split for test queries
    print("Loading AG News test split for queries...")
    test_data = dataset['test']
    
    # Create test_queries from the test split
    # We can take a sample from the test data to form queries
    # For instance, take up to N queries per category from the test set
    
    test_queries = {category: [] for category in categories}
    # Let's aim for a similar number of queries as before, e.g., 20 per category
    # We'll iterate through the test set and collect texts until we have enough queries per category
    
    # To make it more robust, let's collect a certain number of test examples
    # and then distribute them. Or, more simply, iterate and fill.
    
    # For simplicity, let's take the first few examples from the test set as queries,
    # ensuring they are distributed among categories.
    # We can limit the number of queries per category.
    
    num_queries_per_category = 20 # Desired number of queries per category
    # Temp storage for queries from test set
    raw_test_samples = {label_id: [] for label_id in range(len(categories))}

    # Collect samples from the test set
    for example in test_data:
        label_id = example['label']
        if len(raw_test_samples[label_id]) < num_queries_per_category:
            raw_test_samples[label_id].append(example['text'])

    # Populate test_queries
    for label_id, texts in raw_test_samples.items():
        category_name = categories[label_id]
        test_queries[category_name].extend(texts)

    # Ensure all categories have some queries, even if fewer than num_queries_per_category
    # if the test set is small or skewed for those categories.
    # The current AG News test set is large enough.

    print(f"Generated test queries from the test split:")
    for cat, qs in test_queries.items():
        print(f"  {cat}: {len(qs)} queries")

    print(f"Successfully loaded AG News dataset with {len(documents)} documents for indexing.")
    print(f"Generated test queries from the AG News test split.")
    print(f"Categories: {categories}")
    
    # Show category distribution for the indexed documents
    category_counts = {}
    for meta_item in metadata:
        cat = meta_item['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("Indexed document distribution:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} documents")
    
    return documents, test_queries, metadata
