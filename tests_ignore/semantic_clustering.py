import os
import re
import json
import numpy as np
from collections import defaultdict
from Levenshtein import distance
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Set
from tqdm import tqdm

def cluster_semantic_words(spelling_threshold: float = 0.8, semantic_threshold: float = 0.7):
    """
    Cluster words based on both spelling and semantic similarity.
    
    Args:
        spelling_threshold: Minimum spelling similarity ratio (0-1)
        semantic_threshold: Minimum semantic similarity ratio (0-1)
    
    Returns:
        Dictionary mapping each word to its cluster of similar words
    """
    # Get all words from files
    print("Reading files and extracting words...")
    words = set()
    chat_dir = '../data/only_text_chats'
    
    for filename in os.listdir(chat_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(chat_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read().lower()
                file_words = re.findall(r'\b\w+\b', text)
                words.update(file_words)
    
    words = list(words)
    print(f"Found {len(words)} unique words")
    
    # Load multilingual model
    print("Loading multilingual model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Get semantic embeddings
    print("Getting semantic embeddings...")
    embeddings = model.encode(words, show_progress_bar=True)
    
    # Calculate semantic similarity matrix
    print("Calculating semantic similarities...")
    semantic_sim = cosine_similarity(embeddings)
    
    # Initialize clusters
    clusters = defaultdict(set)
    
    # Compare each word with every other word
    print("Clustering words...")
    total_pairs = len(words) * (len(words) - 1) // 2
    processed_pairs = 0
    
    with tqdm(total=total_pairs, desc="Processing word pairs") as pbar:
        for i, word1 in enumerate(words):
            if word1 in clusters:  # Skip if already clustered
                pbar.update(len(words) - i - 1)
                continue
                
            clusters[word1].add(word1)  # Add word to its own cluster
            
            for j, word2 in enumerate(words[i+1:], i+1):
                if word2 in clusters:  # Skip if already clustered
                    continue
                    
                # Calculate spelling similarity
                max_len = max(len(word1), len(word2))
                if max_len == 0:
                    continue
                    
                spelling_sim = 1 - (distance(word1, word2) / max_len)
                
                # Get semantic similarity
                semantic_sim_ratio = semantic_sim[i, j]
                
                # Combine both similarities (you can adjust weights)
                combined_sim = (spelling_sim + semantic_sim_ratio) / 2
                
                if combined_sim >= semantic_threshold:
                    clusters[word1].add(word2)
                    clusters[word2].add(word1)
                
                processed_pairs += 1
                pbar.update(1)
    
    return clusters

def save_clusters_to_json(clusters: Dict[str, Set[str]], min_cluster_size: int = 2):
    """Save clusters with at least min_cluster_size words to JSON"""
    # Convert sets to lists for JSON serialization
    filtered_clusters = {
        word: sorted(list(cluster))
        for word, cluster in clusters.items()
        if len(cluster) >= min_cluster_size
    }
    
    output_file = 'semantic_clusters.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_clusters, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(filtered_clusters)} clusters to {output_file}")

if __name__ == "__main__":
    print("Starting semantic clustering process...")
    clusters = cluster_semantic_words(spelling_threshold=0.8, semantic_threshold=0.7)
    save_clusters_to_json(clusters, min_cluster_size=2)