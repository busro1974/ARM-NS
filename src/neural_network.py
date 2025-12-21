"""
Neural Network Component untuk Link Prediction dan Entity Embedding
Menggunakan Graph Neural Network untuk representasi entity dan relation
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class EntityEmbedding:
    """Entity embedding menggunakan neural approach"""

    def __init__(self, entity_id: str, embedding_dim: int = 100):
        self.entity_id = entity_id
        self.embedding_dim = embedding_dim
        self.embedding = np.random.randn(embedding_dim) * 0.01
        self.grad = np.zeros(embedding_dim)

    def update(self, grad: np.ndarray, learning_rate: float = 0.01):
        """Update embedding dengan gradient descent"""
        self.embedding -= learning_rate * grad


class RelationEmbedding:
    """Relation embedding untuk semantic representation"""

    def __init__(self, relation_id: str, embedding_dim: int = 100):
        self.relation_id = relation_id
        self.embedding_dim = embedding_dim
        self.embedding = np.random.randn(embedding_dim) * 0.01


class TransEModel:
    """
    TransE: Knowledge Graph Embedding Model
    h + r ≈ t (head + relation ≈ tail)
    """

    def __init__(self, embedding_dim: int = 100, margin: float = 1.0):
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.entity_embeddings: Dict[str, EntityEmbedding] = {}
        self.relation_embeddings: Dict[str, RelationEmbedding] = {}

    def add_entity(self, entity_id: str):
        """Tambah entity embedding"""
        if entity_id not in self.entity_embeddings:
            self.entity_embeddings[entity_id] = EntityEmbedding(entity_id, self.embedding_dim)

    def add_relation(self, relation_id: str):
        """Tambah relation embedding"""
        if relation_id not in self.relation_embeddings:
            self.relation_embeddings[relation_id] = RelationEmbedding(relation_id, self.embedding_dim)

    def predict_score(self, head: str, relation: str, tail: str) -> float:
        """
        Hitung score untuk triple (h, r, t)
        Lebih rendah score = lebih likely untuk true
        """
        if head not in self.entity_embeddings or tail not in self.entity_embeddings:
            return float('inf')
        if relation not in self.relation_embeddings:
            return float('inf')

        h = self.entity_embeddings[head].embedding
        r = self.relation_embeddings[relation].embedding
        t = self.entity_embeddings[tail].embedding

        # TransE: ||h + r - t||
        return np.linalg.norm(h + r - t)

    def normalize_embeddings(self):
        """Normalize semua embeddings ke unit sphere"""
        for entity in self.entity_embeddings.values():
            norm = np.linalg.norm(entity.embedding)
            if norm > 0:
                entity.embedding /= norm


class RelationalGraphNeuralNetwork:
    """Graph Neural Network untuk link prediction dalam RDF graphs"""

    def __init__(self, graph, embedding_dim: int = 64, num_layers: int = 2):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.transe_model = TransEModel(embedding_dim)
        self.entity_neighbors_cache = {}
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embeddings untuk semua entities dan relations"""
        for entity in self.graph.get_entities():
            self.transe_model.add_entity(entity)

        for predicate in self.graph.get_predicates():
            self.transe_model.add_relation(predicate)

    def _aggregate_neighbor_info(self, entity: str, layer: int) -> np.ndarray:
        """Aggregate informasi dari neighbors"""
        neighbors = self.graph.get_neighbors(entity)

        if not neighbors:
            return self.transe_model.entity_embeddings[entity].embedding.copy()

        neighbor_embeddings = []
        for neighbor in neighbors:
            if neighbor in self.transe_model.entity_embeddings:
                neighbor_embeddings.append(
                    self.transe_model.entity_embeddings[neighbor].embedding
                )

        if not neighbor_embeddings:
            return self.transe_model.entity_embeddings[entity].embedding.copy()

        # Mean aggregation
        aggregated = np.mean(neighbor_embeddings, axis=0)
        return aggregated

    def train_on_triple(
        self,
        head: str,
        relation: str,
        tail: str,
        learning_rate: float = 0.01,
    ):
        """Train model pada satu triple"""
        pos_score = self.transe_model.predict_score(head, relation, tail)

        # Generate negative sample (corruption)
        entities = list(self.graph.get_entities())
        neg_tail = np.random.choice(entities)

        neg_score = self.transe_model.predict_score(head, relation, neg_tail)

        # Margin-based loss: max(0, margin + pos - neg)
        loss = max(0, self.transe_model.margin + pos_score - neg_score)

        # Simplified gradient update
        if loss > 0:
            h_emb = self.transe_model.entity_embeddings[head]
            r_emb = self.transe_model.relation_embeddings[relation]
            t_emb = self.transe_model.entity_embeddings[tail]

            # Update embeddings dengan simple gradient
            gradient = 0.01
            h_emb.embedding -= learning_rate * gradient
            t_emb.embedding -= learning_rate * gradient
            r_emb.embedding -= learning_rate * gradient

        return loss

    def train_on_triples(self, num_epochs: int = 10, learning_rate: float = 0.01):
        """Train model pada semua triples"""
        triples = list(self.graph.triples)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for triple in triples:
                loss = self.train_on_triple(
                    triple.subject,
                    triple.predicate,
                    triple.object,
                    learning_rate,
                )
                total_loss += loss

            self.transe_model.normalize_embeddings()

            if epoch % max(1, num_epochs // 10) == 0:
                avg_loss = total_loss / len(triples) if triples else 0
                print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

    def predict_link(self, head: str, relation: str, tail: str) -> float:
        """
        Predict probability untuk link
        Menggunakan sigmoid dari negative score
        """
        score = self.transe_model.predict_score(head, relation, tail)

        # Convert distance to probability (lower distance = higher probability)
        probability = 1.0 / (1.0 + score)
        return probability

    def rank_candidates(
        self,
        head: str,
        relation: str,
        candidates: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Rank candidate entities untuk link prediction"""
        scores = []

        for candidate in candidates:
            prob = self.predict_link(head, relation, candidate)
            scores.append((candidate, prob))

        # Sort by probability descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_entity_embedding(self, entity: str) -> Optional[np.ndarray]:
        """Dapatkan embedding untuk entity"""
        if entity in self.transe_model.entity_embeddings:
            return self.transe_model.entity_embeddings[entity].embedding.copy()
        return None

    def get_similar_entities(self, entity: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Dapatkan entities yang paling similar dengan embedding cosine similarity"""
        if entity not in self.transe_model.entity_embeddings:
            return []

        entity_emb = self.get_entity_embedding(entity)
        similarities = []

        for other_entity in self.transe_model.entity_embeddings:
            if other_entity == entity:
                continue

            other_emb = self.get_entity_embedding(other_entity)
            similarity = self._cosine_similarity(entity_emb, other_emb)
            similarities.append((other_entity, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Hitung cosine similarity antara dua vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def get_relation_strength(self, relation: str) -> float:
        """Dapatkan norm dari relation embedding (strength indicator)"""
        if relation in self.transe_model.relation_embeddings:
            rel_emb = self.transe_model.relation_embeddings[relation].embedding
            return float(np.linalg.norm(rel_emb))
        return 0.0
