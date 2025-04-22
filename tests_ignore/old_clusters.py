from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import json

# Step 0: Load your clusters
with open('clusters.json', 'r') as f:
    clusters = json.load(f)

# Flatten words + generate InputExamples for training
examples = []
all_words = []
cluster_ids = list(clusters.keys())

for cluster_id in cluster_ids:
    words = clusters[cluster_id]
    all_words.extend(words)
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            anchor, positive = words[i], words[j]
            # Sample a negative word from a different cluster
            negative_cluster = random.choice([c for c in cluster_ids if c != cluster_id])
            negative = random.choice(clusters[negative_cluster])
            examples.append(InputExample(texts=[anchor, positive, negative]))

# Step 1: Load and fine-tune a multilingual embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(model=model)
model.fit([(train_dataloader, train_loss)], epochs=3)

# Step 2: Build cluster centroids
def compute_centroids(clusters, model):
    centroids = {}
    for cluster_id, words in clusters.items():
        embeddings = model.encode(words)
        centroid = np.mean(embeddings, axis=0)
        centroids[cluster_id] = centroid
    return centroids

centroids = compute_centroids(clusters, model)

# Step 3: Classify new word
def classify_word(word, centroids, model, threshold=0.6):
    word_embedding = model.encode([word])[0]
    similarities = {
        cid: cosine_similarity([word_embedding], [centroid])[0][0]
        for cid, centroid in centroids.items()
    }
    best_match = max(similarities, key=similarities.get)
    best_score = similarities[best_match]

    print("best_match", best_match, "best_score", best_score)
    
    if best_score >= threshold:
        return best_match
    else:
        return "new_cluster"


# TODO: make this better
def learn_new_cluster(word: str, cluster_id: str, clusters: dict, model: SentenceTransformer, centroids: dict) -> dict:
    """
    Add a new word to an existing cluster or create a new cluster
    
    Args:
        word: The word to add
        cluster_id: The cluster ID to add to (or 'new_cluster' to create new)
        clusters: Current clusters dictionary
        model: The embedding model
        centroids: Current centroids dictionary
        
    Returns:
        Updated clusters dictionary
    """
    if cluster_id == "new_cluster":
        # Create new cluster ID
        new_id = str(max([int(k) for k in clusters.keys()]) + 1)
        clusters[new_id] = [word]
        # Update centroid
        centroids[new_id] = model.encode([word])[0]
    else:
        # Check if word already exists in cluster
        if word in clusters[cluster_id]:
            print(f"Word '{word}' already exists in cluster {cluster_id}")
            return clusters
            
        # Add to existing cluster
        clusters[cluster_id].append(word)
        # Update centroid
        embeddings = model.encode(clusters[cluster_id])
        centroids[cluster_id] = np.mean(embeddings, axis=0)
    
    # Save updated clusters
    with open('clusters.json', 'w') as f:
        json.dump(clusters, f, indent=2)
    
    return clusters

# Example usage
if __name__ == "__main__":
    # Load initial clusters
    with open('clusters.json', 'r') as f:
        clusters = json.load(f)
    
    # Initialize model and centroids
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    centroids = compute_centroids(clusters, model)
    
    # Test word
    word = "lae"
    assigned_cluster = classify_word(word, centroids, model)
    
    # Learn the new word
    clusters = learn_new_cluster(word, assigned_cluster, clusters, model, centroids)
    print(f"Updated clusters: {clusters}")