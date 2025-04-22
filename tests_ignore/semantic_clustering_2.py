import os
import re
import json
import numpy as np
from collections import defaultdict, Counter
from Levenshtein import distance
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Set, List, Tuple
from tqdm import tqdm

def extract_patterns(texts: List[str], min_freq: int = 5) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Automatically extract Franco-Arabic patterns and abbreviations from text.
    
    Args:
        texts: List of text messages
        min_freq: Minimum frequency for a pattern to be considered
    
    Returns:
        Tuple of (arabizi_map, abbreviations)
    """
    # Extract all words containing numbers
    number_words = []
    for text in texts:
        words = re.findall(r'\b\w*\d+\w*\b', text)
        number_words.extend(words)
    
    # Count frequency of each number pattern
    number_patterns = Counter()
    for word in number_words:
        # Extract the number and surrounding letters
        pattern = re.sub(r'[^0-9]', '', word)
        if pattern:
            number_patterns[pattern] += 1
    
    # Create arabizi map from frequent patterns
    arabizi_map = {}
    for num, freq in number_patterns.most_common():
        if freq >= min_freq:
            # Find the most common word containing this number
            words_with_num = [w for w in number_words if num in w]
            if words_with_num:
                most_common = Counter(words_with_num).most_common(1)[0][0]
                # Extract the letter that might correspond to the number
                letter = re.sub(r'[0-9]', '', most_common)
                if letter:
                    arabizi_map[num] = letter
    
    # Extract potential abbreviations
    abbreviations = {}
    short_words = [w for w in re.findall(r'\b\w{2,4}\b', ' '.join(texts))]
    word_freq = Counter(short_words)
    
    # Look for words that might be abbreviations
    for word, freq in word_freq.most_common():
        if freq >= min_freq and any(c.isdigit() for c in word):
            # Find similar words without numbers
            similar_words = [w for w in word_freq if 
                           len(w) > len(word) and 
                           distance(word, w) <= 2 and 
                           not any(c.isdigit() for c in w)]
            if similar_words:
                most_similar = similar_words[0]
                abbreviations[word] = most_similar
    
    return arabizi_map, abbreviations

def learn_patterns_from_data(chat_dir: str, min_freq: int = 5) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Learn patterns from all chat files"""
    print("Learning patterns from chat data...")
    all_texts = []
    
    for filename in os.listdir(chat_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(chat_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read().lower()
                all_texts.append(text)
    
    return extract_patterns(all_texts, min_freq)

def normalize_text(text: str, arabizi_map: Dict[str, str], abbreviations: Dict[str, str]) -> str:
    """Normalize text using learned patterns"""
    # Normalize repeated letters (max 2 repetitions)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Replace numbers with letters
    for num, letter in arabizi_map.items():
        text = text.replace(num, letter)
    
    # Replace abbreviations
    for abbr, full in abbreviations.items():
        text = re.sub(r'\b' + abbr + r'\b', full, text)
    
    return text

def preprocess_word(word: str, arabizi_map: Dict[str, str], abbreviations: Dict[str, str]) -> str:
    """Preprocess individual word using learned patterns"""
    word = re.sub(r'[^\w\s]', '', word)
    word = word.lower()
    word = normalize_text(word, arabizi_map, abbreviations)
    return word

def cluster_semantic_words(spelling_threshold: float = 0.8, semantic_threshold: float = 0.7, min_pattern_freq: int = 5):
    """Cluster words based on both spelling and semantic similarity"""
    print("Reading files and extracting words...")
    chat_dir = '../data/only_text_chats'
    
    # Learn patterns from data
    arabizi_map, abbreviations = learn_patterns_from_data(chat_dir, min_pattern_freq)
    print(f"Learned {len(arabizi_map)} Franco-Arabic patterns and {len(abbreviations)} abbreviations")
    
    words = set()
    for filename in os.listdir(chat_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(chat_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read().lower()
                file_words = re.findall(r'\b\w+\b', text)
                processed_words = [preprocess_word(w, arabizi_map, abbreviations) for w in file_words]
                words.update(processed_words)
    
    words = list(words)
    print(f"Found {len(words)} unique words after preprocessing")
    
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
    
    with tqdm(total=total_pairs, desc="Processing word pairs") as pbar:
        for i, word1 in enumerate(words):
            if word1 in clusters:  # Skip if already clustered
                pbar.update(len(words) - i - 1)
                continue
                
            clusters[word1].add(word1)
            
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
                
                # Combine both similarities with adjusted weights
                # Give more weight to semantic similarity for multilingual text
                combined_sim = (spelling_sim * 0.4 + semantic_sim_ratio * 0.6)
                
                if combined_sim >= semantic_threshold:
                    clusters[word1].add(word2)
                    clusters[word2].add(word1)
                
                pbar.update(1)
    
    return clusters, arabizi_map, abbreviations

def save_clusters_to_json(clusters: Dict[str, Set[str]], arabizi_map: Dict[str, str], 
                         abbreviations: Dict[str, str], min_cluster_size: int = 2):
    """Save clusters and learned patterns to JSON"""
    filtered_clusters = {
        word: sorted(list(cluster))
        for word, cluster in clusters.items()
        if len(cluster) >= min_cluster_size
    }
    
    output = {
        'clusters': filtered_clusters,
        'learned_patterns': {
            'arabizi_map': arabizi_map,
            'abbreviations': abbreviations
        }
    }
    
    output_file = 'semantic_clusters_2.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(filtered_clusters)} clusters and learned patterns to {output_file}")

if __name__ == "__main__":
    print("Starting semantic clustering process...")
    clusters, arabizi_map, abbreviations = cluster_semantic_words(
        spelling_threshold=0.7, 
        semantic_threshold=0.6,
        min_pattern_freq=5
    )
    save_clusters_to_json(clusters, arabizi_map, abbreviations, min_cluster_size=2)