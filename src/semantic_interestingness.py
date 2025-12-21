"""
Semantic-based Interestingness Measure untuk Rule Mining
Menggabungkan berbagai metrik untuk evaluasi kualitas rule
"""

from typing import List, Tuple, Dict, Optional, Set
import math
import numpy as np


class SemanticInterestingnessMeasure:
    """Comprehensive semantic-based interestingness measures untuk rules"""

    def __init__(self, graph, use_embedding: bool = True):
        self.graph = graph
        self.use_embedding = use_embedding
        self.entity_similarity_cache = {}

    @staticmethod
    def support(body_count: int, total_count: int) -> float:
        """Support: P(body)"""
        if total_count == 0:
            return 0.0
        return body_count / total_count

    @staticmethod
    def confidence(body_head_count: int, body_count: int) -> float:
        """Confidence: P(head|body)"""
        if body_count == 0:
            return 0.0
        return body_head_count / body_count

    @staticmethod
    def lift(confidence: float, head_support: float) -> float:
        """Lift: conf / support(head)"""
        if head_support == 0:
            return 0.0
        return confidence / head_support

    @staticmethod
    def conviction(confidence: float, head_support: float) -> float:
        """Conviction: (1 - head_support) / (1 - confidence)"""
        if confidence >= 1.0:
            return float('inf')
        denominator = 1.0 - confidence
        if denominator == 0:
            return float('inf')
        return (1.0 - head_support) / denominator

    def coherence(self, rule: Tuple, path_metrics: Dict) -> float:
        """Coherence: semantic relatedness berdasarkan path properties"""
        path_length = path_metrics.get('avg_path_length', 1)
        common_neighbors = path_metrics.get('common_neighbors', 0)

        path_score = 1.0 / (1.0 + math.log(path_length + 1))
        neighbor_score = min(common_neighbors / 10.0, 1.0)

        return 0.6 * path_score + 0.4 * neighbor_score

    def informativeness(self, confidence: float, support: float) -> float:
        """Informativeness: information content dari rule"""
        if support <= 0 or confidence <= 0:
            return 0.0

        info = confidence * math.log(confidence / support)
        return min(info, 1.0)

    def specificity(self, rule_body_predicate_count: int, total_predicate_count: int) -> float:
        """Specificity: seberapa specific body dari rule"""
        if total_predicate_count == 0:
            return 0.0
        return rule_body_predicate_count / total_predicate_count

    def relevance(self, rule: Tuple, domain_context: Dict) -> float:
        """Relevance: seberapa relevan rule dengan domain"""
        domain_predicates = domain_context.get('predicates', set())
        rule_predicates = self._extract_predicates(rule)

        if not domain_predicates:
            return 0.5

        matches = len(rule_predicates.intersection(domain_predicates))
        return matches / max(len(rule_predicates), 1)

    def consistency(self, rule: Tuple, exceptions: List[Tuple]) -> float:
        """Consistency: consistency dari rule"""
        if len(exceptions) == 0:
            return 1.0

        consistency_score = 1.0 / (1.0 + len(exceptions))
        return consistency_score

    def novelty(self, rule: Tuple, existing_rules: List[Tuple]) -> float:
        """Novelty: seberapa novel/unique rule dibanding existing rules"""
        if not existing_rules:
            return 1.0

        similarities = [self._rule_similarity(rule, existing) for existing in existing_rules]
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_similarity

    def semantic_cohesion(self, rule_entities: Set[str], embedding_model=None) -> float:
        """Semantic Cohesion: semantic relatedness entities dalam rule"""
        if len(rule_entities) < 2 or embedding_model is None:
            return 0.5

        entities_list = list(rule_entities)
        similarities = []

        for i in range(len(entities_list)):
            for j in range(i + 1, len(entities_list)):
                sim = self._entity_embedding_similarity(entities_list[i], entities_list[j], embedding_model)
                similarities.append(sim)

        if not similarities:
            return 0.5

        return np.mean(similarities)

    def combined_interestingness(self, rule: Tuple, metrics: Dict, weights: Optional[Dict] = None) -> float:
        """Combined score menggunakan weighted sum dari berbagai measures"""
        if weights is None:
            weights = {
                'confidence': 0.25,
                'lift': 0.15,
                'coherence': 0.20,
                'informativeness': 0.15,
                'novelty': 0.10,
                'semantic_cohesion': 0.15
            }

        total_score = 0.0
        total_weight = 0.0

        for measure_name, weight in weights.items():
            if measure_name in metrics:
                total_score += weight * metrics[measure_name]
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    def _extract_predicates(self, rule: Tuple) -> Set[str]:
        """Extract semua predicates dari rule"""
        predicates = set()
        prefixes = (
            'p_', 'has_', 'is_', 'buys', 'related', 'instance',
            'transaction', 'viewed', 'purchased', 'rating'
        )
        for item in rule:
            if isinstance(item, str) and any(item.startswith(p) for p in prefixes):
                predicates.add(item)
        return predicates

    def _rule_similarity(self, rule1: Tuple, rule2: Tuple) -> float:
        """Calculate similarity antara dua rules"""
        pred1 = self._extract_predicates(rule1)
        pred2 = self._extract_predicates(rule2)

        if not pred1 and not pred2:
            return 1.0

        intersection = len(pred1.intersection(pred2))
        union = len(pred1.union(pred2))

        if union == 0:
            return 0.0

        return intersection / union

    def _entity_embedding_similarity(self, entity1: str, entity2: str, embedding_model) -> float:
        """Calculate embedding similarity antara entities"""
        cache_key = (entity1, entity2)
        if cache_key in self.entity_similarity_cache:
            return self.entity_similarity_cache[cache_key]

        emb1 = embedding_model.get_entity_embedding(entity1)
        emb2 = embedding_model.get_entity_embedding(entity2)

        if emb1 is None or emb2 is None:
            return 0.0

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            sim = 0.0
        else:
            sim = np.dot(emb1, emb2) / (norm1 * norm2)

        self.entity_similarity_cache[cache_key] = sim
        return sim


class PathAnalyzer:
    """Analyze paths dalam graph untuk semantic measures"""

    def __init__(self, graph):
        self.graph = graph

    def get_path_metrics(self, source: str, target: str) -> Dict:
        """Get metrics tentang path antara source dan target"""
        try:
            path_length = self.graph.get_path_length(source, target)
        except Exception:
            path_length = None

        common_neighbors = len(self.graph.get_common_neighbors(source, target))

        return {
            'path_length': path_length,
            'common_neighbors': common_neighbors,
            'avg_path_length': path_length if path_length else 999
        }
