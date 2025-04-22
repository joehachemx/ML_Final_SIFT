from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import umap
import networkx as nx
import numpy as np
import random
import json
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from keras import layers, Model, optimizers
import keras
import tensorflow as tf

class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)
        
    def call(self, features):
        attention = keras.activations.tanh(self.W(features))
        attention_weights = keras.activations.softmax(self.V(attention), axis=1)
        context = attention_weights * features
        context = keras.backend.sum(context, axis=1)
        return context, attention_weights

class FrancoArabicClusterer:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                 k_neighbors: int = 5, embedding_dim: int = 384):
        self.model = SentenceTransformer(model_name)
        self.clusters: Dict[str, List[str]] = {}
        self.centroids: Dict[str, np.ndarray] = {}
        self.word_embeddings: Dict[str, np.ndarray] = {}
        self.cluster_graph = nx.Graph()
        self.knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='cosine')
        self.knn_trained = False
        self.embedding_dim = embedding_dim
        self.keras_model = None
        self.cluster_to_idx = {}
        self.idx_to_cluster = {}
        
    def _build_keras_model(self):
        """Build Keras model with attention mechanism"""
        input_layer = layers.Input(shape=(self.embedding_dim,))
        
        # Attention mechanism
        attention_output, attention_weights = AttentionLayer(64)(tf.expand_dims(input_layer, 1))
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(attention_output)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Output layer
        output_layer = layers.Dense(len(self.clusters) + 1, activation='softmax')(x)
        
        # Create model
        self.keras_model = Model(inputs=input_layer, outputs=output_layer)
        self.keras_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def _prepare_keras_data(self):
        """Prepare data for Keras model training"""
        X = []
        y = []
        
        # Create cluster mappings
        self.cluster_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(self.clusters.keys())}
        self.cluster_to_idx['new_cluster'] = len(self.clusters)
        self.idx_to_cluster = {v: k for k, v in self.cluster_to_idx.items()}
        
        # Prepare training data
        for cluster_id, words in self.clusters.items():
            for word in words:
                X.append(self.word_embeddings[word])
                y_one_hot = np.zeros(len(self.clusters) + 1)
                y_one_hot[self.cluster_to_idx[cluster_id]] = 1
                y.append(y_one_hot)
                
        return np.array(X), np.array(y)
        
    def _train_keras_model(self, epochs=50, batch_size=32):
        """Train Keras model"""
        X, y = self._prepare_keras_data()
        
        # Add some noise to prevent overfitting
        X_noisy = X + np.random.normal(0, 0.1, X.shape)
        X = np.vstack([X, X_noisy])
        y = np.vstack([y, y])
        
        # Train model
        self.keras_model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
    def load_clusters(self, filepath: str):
        """Load existing clusters from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.clusters = json.load(f)
        self._update_embeddings()
        self._build_cluster_graph()
        self._train_knn()
        self._build_keras_model()
        self._train_keras_model()
        
    def _train_knn(self):
        """Train KNN classifier on existing clusters"""
        if not self.word_embeddings:
            return
            
        # Prepare training data
        X = []
        y = []
        for cluster_id, words in self.clusters.items():
            for word in words:
                X.append(self.word_embeddings[word])
                y.append(cluster_id)
                
        if X and y:
            self.knn.fit(X, y)
            self.knn_trained = True
            
    def _update_embeddings(self):
        """Update word embeddings and centroids"""
        all_words = []
        for cluster_id, words in self.clusters.items():
            all_words.extend(words)
        
        # Get embeddings for all words
        embeddings = self.model.encode(all_words, show_progress_bar=True)
        
        # Store word embeddings
        idx = 0
        for cluster_id, words in self.clusters.items():
            for word in words:
                self.word_embeddings[word] = embeddings[idx]
                idx += 1
        
        # Update centroids
        self._update_centroids()
        
    def _update_centroids(self):
        """Update cluster centroids"""
        for cluster_id, words in self.clusters.items():
            embeddings = np.array([self.word_embeddings[word] for word in words])
            self.centroids[cluster_id] = np.mean(embeddings, axis=0)
            
    def _build_cluster_graph(self):
        """Build graph representation of clusters for community detection"""
        self.cluster_graph.clear()
        
        # Add all words as nodes
        for cluster_id, words in self.clusters.items():
            for word in words:
                self.cluster_graph.add_node(word, cluster=cluster_id)
        
        # Add edges based on similarity
        words = list(self.word_embeddings.keys())
        for i, word1 in enumerate(words):
            for word2 in words[i+1:]:
                sim = cosine_similarity(
                    [self.word_embeddings[word1]], 
                    [self.word_embeddings[word2]]
                )[0][0]
                if sim > 0.7:  # Threshold for edge creation
                    self.cluster_graph.add_edge(word1, word2, weight=sim)
                    
    def visualize_clusters(self, output_path: str = "cluster_visualization.png"):
        """Visualize clusters using UMAP dimensionality reduction"""
        # Get all embeddings
        words = list(self.word_embeddings.keys())
        embeddings = np.array([self.word_embeddings[word] for word in words])
        
        # Reduce dimensionality
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        for cluster_id in self.clusters:
            cluster_words = self.clusters[cluster_id]
            indices = [words.index(word) for word in cluster_words]
            plt.scatter(
                reduced_embeddings[indices, 0],
                reduced_embeddings[indices, 1],
                label=f"Cluster {cluster_id}",
                alpha=0.6
            )
        
        plt.legend()
        plt.title("Cluster Visualization")
        plt.savefig(output_path)
        plt.close()
        
    def hierarchical_cluster(self, words: List[str], threshold: float = 0.7) -> Dict[str, List[str]]:
        """Perform hierarchical clustering on new words"""
        # Get embeddings
        embeddings = self.model.encode(words, show_progress_bar=True)
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity='cosine',
            linkage='average',
            distance_threshold=1 - threshold
        )
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Create new clusters
        new_clusters = defaultdict(list)
        for word, label in zip(words, cluster_labels):
            new_clusters[str(label)].append(word)
            
        return dict(new_clusters)
        
    def classify_word(self, word: str, threshold: float = 0.6) -> Tuple[str, float]:
        """Classify a new word using Keras model, KNN, and similarity-based approaches"""
        word_embedding = self.model.encode([word])[0]
        
        # Try Keras model first
        if self.keras_model is not None:
            pred_probs = self.keras_model.predict(np.array([word_embedding]), verbose=0)[0]
            pred_cluster_idx = np.argmax(pred_probs)
            pred_score = pred_probs[pred_cluster_idx]
            
            if pred_score >= threshold:
                cluster_id = self.idx_to_cluster[pred_cluster_idx]
                return cluster_id, float(pred_score)
        
        # Try KNN classification if trained
        if self.knn_trained:
            knn_cluster = self.knn.predict([word_embedding])[0]
            knn_proba = self.knn.predict_proba([word_embedding])[0]
            knn_score = max(knn_proba)
            
            if knn_score >= threshold:
                return knn_cluster, knn_score
        
        # Fallback to similarity-based classification
        similarities = {
            cid: cosine_similarity([word_embedding], [centroid])[0][0]
            for cid, centroid in self.centroids.items()
        }
        
        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]
        
        if best_score >= threshold:
            return best_match, best_score
        else:
            return "new_cluster", best_score
            
    def learn_new_cluster(self, word: str, cluster_id: str) -> None:
        """Add a new word to a cluster or create new cluster"""
        word_embedding = self.model.encode([word])[0]
        self.word_embeddings[word] = word_embedding
        
        if cluster_id == "new_cluster":
            # Create new cluster
            new_id = str(max([int(k) for k in self.clusters.keys()]) + 1)
            self.clusters[new_id] = [word]
            self.centroids[new_id] = word_embedding
        else:
            # Add to existing cluster
            self.clusters[cluster_id].append(word)
            self._update_centroids()
            
        # Update graph and retrain models
        self._build_cluster_graph()
        self._train_knn()
        self._build_keras_model()
        self._train_keras_model()
        
    def evaluate_transliteration(self, test_data: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate transliteration accuracy"""
        results = {
            "exact_match": 0,
            "cluster_match": 0,
            "total": len(test_data)
        }
        
        for word, expected in test_data:
            cluster_id, _ = self.classify_word(word)
            if cluster_id != "new_cluster":
                cluster_words = self.clusters[cluster_id]
                if expected in cluster_words:
                    results["cluster_match"] += 1
                    if word == expected:
                        results["exact_match"] += 1
                        
        results["cluster_accuracy"] = results["cluster_match"] / results["total"]
        results["exact_accuracy"] = results["exact_match"] / results["total"]
        
        return results
        
    def save_clusters(self, filepath: str):
        """Save clusters to JSON file"""
        output = {
            "clusters": self.clusters,
            "metadata": {
                "num_clusters": len(self.clusters),
                "total_words": sum(len(words) for words in self.clusters.values())
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

# Example usage
if __name__ == "__main__":
    # Initialize clusterer
    clusterer = FrancoArabicClusterer(k_neighbors=5)
    
    # Load existing clusters
    clusterer.load_clusters('normal_clusters.json')
    
    # Visualize clusters
    clusterer.visualize_clusters()
    
    # Test new word
    test_word = "lae"
    cluster_id, score = clusterer.classify_word(test_word)
    print(f"Word '{test_word}' assigned to cluster {cluster_id} with score {score}")
    
    # Learn new word
    clusterer.learn_new_cluster(test_word, cluster_id)
    
    # Save updated clusters
    clusterer.save_clusters('clusters_2.json')
    
    # Example evaluation
    test_data = [
        ("lae", "lae"),
        ("m3a", "مع"),
        ("7bb", "حب")
    ]
    results = clusterer.evaluate_transliteration(test_data)
    print(f"Evaluation results: {results}")