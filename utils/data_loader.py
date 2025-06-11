"""
Data loading utilities.
"""
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional

def load_toy_dataset(
    dataset_name: str = "ag_news",
    num_documents: int = 1000,
    num_queries_per_category: int = 20,
    category_mapping: Optional[Dict[int, str]] = None,
    text_field: str = "text",
    label_field: str = "label"
) -> Tuple[List[str], Dict[str, List[str]], List[Dict]]:
    """Load and prepare a dataset from Hugging Face"""
    print(f"Loading {dataset_name} dataset...")
    
    # Load AG News dataset
    dataset = load_dataset(dataset_name)
    
    # Get the training split for documents
    train_data = dataset['train']
    
    # Convert to lists and sample first N examples for documents
    documents = []
    doc_labels = [] # Renamed from labels to avoid confusion with test set labels
    
    # Take first num_documents examples for documents
    for i in range(min(num_documents, len(train_data))):
        example = train_data[i]
        documents.append(example[text_field])
        doc_labels.append(example[label_field])
    
    print(f"Loaded {len(documents)} documents for indexing")
    
    # Category mapping
    if category_mapping:
        categories_list = sorted(list(set(category_mapping.values())))
    else:
        # Attempt to infer categories if not provided, assuming labels are integers from 0 to N-1
        unique_labels = sorted(list(set(doc_labels)))
        if all(isinstance(lbl, int) for lbl in unique_labels):
            categories_list = [f"Category {lbl}" for lbl in unique_labels]
            category_mapping = {lbl: f"Category {lbl}" for lbl in unique_labels}
            print(f"Inferred categories: {categories_list}")
        else:
            # Fallback if labels are not simple integers or no mapping provided
            raise ValueError("Category mapping must be provided if labels are not integers or cannot be inferred.")

    # Create metadata for the indexed documents
    metadata = []
    for i, label_id in enumerate(doc_labels):
        category = category_mapping.get(label_id, f"Unknown Label {label_id}")
        metadata.append({
            'id': i, # This id refers to the index in the 'documents' list
            'label': label_id,
            'category': category
        })
    
    # Load the test split for test queries
    print(f"Loading {dataset_name} test split for queries...")
    test_data = dataset['test']
    
    # Create test_queries from the test split
    test_queries = {category: [] for category in categories_list}
    
    # Temp storage for queries from test set
    # Ensure raw_test_samples keys cover all possible label_ids present in the test set
    # that are also in our defined category_mapping
    
    # Get all unique label ids from the provided or inferred category_mapping
    valid_label_ids = set(category_mapping.keys())
    
    raw_test_samples = {label_id: [] for label_id in valid_label_ids}

    # Collect samples from the test set
    for example in test_data:
        label_id = example[label_field]
        if label_id in raw_test_samples and len(raw_test_samples[label_id]) < num_queries_per_category:
            raw_test_samples[label_id].append(example[text_field])

    # Populate test_queries
    for label_id, texts in raw_test_samples.items():
        category_name = category_mapping.get(label_id)
        if category_name: # Ensure category_name is valid
            if category_name not in test_queries:
                test_queries[category_name] = [] # Initialize if somehow missed
            test_queries[category_name].extend(texts)
        else:
            print(f"Warning: Label ID {label_id} from test set not found in category_mapping. Skipping these queries.")


    # Ensure all categories have some queries, even if fewer than num_queries_per_category
    # if the test set is small or skewed for those categories.
    # The current AG News test set is large enough.

    print(f"Generated test queries from the test split:")
    for cat, qs in test_queries.items():
        print(f"  {cat}: {len(qs)} queries")

    print(f"Successfully loaded {dataset_name} dataset with {len(documents)} documents for indexing.")
    print(f"Generated test queries from the {dataset_name} test split.")
    print(f"Categories: {categories_list}")
    
    # Show category distribution for the indexed documents
    category_counts = {}
    for meta_item in metadata:
        cat = meta_item['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("Indexed document distribution:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} documents")
    
    return documents, test_queries, metadata
