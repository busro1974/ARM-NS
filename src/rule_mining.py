"""
Semantic-based Rule Mining Engine
Menggabungkan interestingness measures untuk mining rules dari graph data
"""

from typing import List, Tuple, Dict
from itertools import combinations
import math


class SemanticMeasure:
    """Menghitung semantic-based interestingness measures"""

    @staticmethod
    def support(triples: List[Tuple], rule: Tuple, all_triples: int) -> float:
        """
        Support: proporsi triple yang match dengan rule
        """
        if all_triples == 0:
            return 0.0
        matching = sum(1 for t in triples if t == rule)
        return matching / all_triples

    @staticmethod
    def confidence(head_body_count: int, body_count: int) -> float:
        """
        Confidence: P(head|body) - berapa banyak body yang berakhir dengan head
        """
        if body_count == 0:
            return 0.0
        return head_body_count / body_count

    @staticmethod
    def lift(confidence: float, head_support: float) -> float:
        """
        Lift: confidence / support(head) - seberapa banyak rule lebih baik dari random
        """
        if head_support == 0:
            return 0.0
        return confidence / head_support

    @staticmethod
    def conviction(confidence: float) -> float:
        """
        Conviction: (1-support(head)) / (1-confidence)
        Mengukur implikasi dari rule
        """
        if confidence >= 1.0:
            return float('inf')
        denominator = 1 - confidence
        if denominator == 0:
            return float('inf')
        return denominator

    @staticmethod
    def coherence(confidence: float, path_length: int) -> float:
        """
        Coherence: semantic relatedness berdasarkan path length
        Shorter path = higher coherence
        """
        if path_length <= 0:
            return 0.0
        return confidence / (1 + math.log(path_length + 1))

    @staticmethod
    def semantic_similarity(shared_neighbors: int, total_neighbors: int) -> float:
        """
        Semantic Similarity: overlap dalam neighbors
        """
        if total_neighbors == 0:
            return 0.0
        return shared_neighbors / total_neighbors

    @staticmethod
    def informativeness(confidence: float, support: float) -> float:
        """
        Informativeness: mengukur informasi yang dibawa oleh rule
        """
        if support == 0 or confidence == 0:
            return 0.0
        return confidence * math.log(confidence / support)


class Rule:
    """Representasi rule dalam bentuk: body -> head"""

    def __init__(self, body: Tuple[str, str, str], head: Tuple[str, str, str]):
        self.body = body  # (subject, predicate, object)
        self.head = head  # (subject, predicate, object)
        self.metrics = {}

    def __repr__(self):
        return f"{self.body} -> {self.head}"

    def __eq__(self, other):
        return self.body == other.body and self.head == other.head

    def __hash__(self):
        return hash((self.body, self.head))


class RuleMiningEngine:
    """Engine untuk mining rules dari RDF graph dengan semantic measures"""

    def __init__(self, graph, min_support: float = 0.01, min_confidence: float = 0.5):
        self.graph = graph
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules: List[Rule] = []
        self.semantic_measure = SemanticMeasure()

    def mine_rules(self) -> List[Rule]:
        """Mine rules dari graph menggunakan template matching"""
        rules = []

        predicates = list(self.graph.get_predicates())
        triples = list(self.graph.triples)

        # Mine predicate frequency rules: p1(x,y) ∧ p2(y,z) -> p3(x,z)
        for p1, p2, p3 in combinations(predicates, 3):
            rules_found = self._mine_chain_rules(p1, p2, p3, triples)
            rules.extend(rules_found)

        # Mine composition rules: p1(x,y) -> p2(x,y)
        for p1, p2 in combinations(predicates, 2):
            rules_found = self._mine_composition_rules(p1, p2, triples)
            rules.extend(rules_found)

        # Sort by combined score
        rules = self._rank_rules(rules, triples)
        self.rules = rules
        return rules

    def _mine_chain_rules(
        self,
        p1: str,
        p2: str,
        p3: str,
        triples: List,
    ) -> List[Rule]:
        """Mine chain composition rules: p1(x,y) ∧ p2(y,z) -> p3(x,z)"""
        rules = []

        p1_triples = self.graph.get_triples_by_predicate(p1)
        p2_triples = self.graph.get_triples_by_predicate(p2)

        # Find chains
        for t1 in p1_triples:
            for t2 in p2_triples:
                if t1.object == t2.subject:
                    # Found chain: (x, p1, y) ∧ (y, p2, z)
                    # Check if (x, p3, z) exists
                    body = (t1.subject, p1, t1.object, p2, t2.object)
                    head = (t1.subject, p3, t2.object)

                    # Filter with thresholds
                    conf = self._calculate_chain_confidence(body, head, triples)
                    sup = 0.05  # Simplified support calculation

                    if conf >= self.min_confidence and sup >= self.min_support:
                        rule = Rule(body, head)
                        rules.append(rule)

        return rules

    def _mine_composition_rules(
        self,
        p1: str,
        p2: str,
        triples: List,
    ) -> List[Rule]:
        """Mine composition rules: p1(x,y) -> p2(x,y)"""
        rules = []

        p1_triples = self.graph.get_triples_by_predicate(p1)
        p2_triples = self.graph.get_triples_by_predicate(p2)

        p1_set = {(t.subject, t.object) for t in p1_triples}
        p2_set = {(t.subject, t.object) for t in p2_triples}

        # Calculate confidence
        intersection = p1_set.intersection(p2_set)
        if len(p1_set) > 0:
            confidence = len(intersection) / len(p1_set)

            if confidence >= self.min_confidence:
                body = (None, p1, None)
                head = (None, p2, None)
                rule = Rule(body, head)
                rules.append(rule)

        return rules

    def _calculate_chain_confidence(
        self,
        body: Tuple,
        head: Tuple,
        triples: List,
    ) -> float:
        """Hitung confidence untuk chain rule"""
        x, p1, y, p2, z = body
        hx, hp, hz = head

        # Body triples: (x, p1, y) dan (y, p2, z)
        body_count = 0
        body_and_head_count = 0

        for t in triples:
            if (
                (t.subject == x and t.predicate == p1 and t.object == y)
                or (t.subject == y and t.predicate == p2 and t.object == z)
            ):
                body_count += 1
                if self.graph.triple_exists(x, hp, z):
                    body_and_head_count += 1

        if body_count == 0:
            return 0.0
        return body_and_head_count / body_count

    def _rank_rules(self, rules: List[Rule], triples: List) -> List[Rule]:
        """Rank rules berdasarkan kombinasi metrics"""
        ranked_rules = []

        for rule in rules:
            # Calculate metrics
            metrics = self._calculate_rule_metrics(rule, triples)
            rule.metrics = metrics

            # Combined score (weighted sum)
            score = (
                0.4 * metrics.get('confidence', 0)
                + 0.3 * metrics.get('lift', 0)
                + 0.2 * metrics.get('coherence', 0)
                + 0.1 * metrics.get('informativeness', 0)
            )

            rule.score = score
            ranked_rules.append(rule)

        # Sort by score descending
        ranked_rules.sort(key=lambda r: r.score, reverse=True)
        return ranked_rules

    def _calculate_rule_metrics(self, rule: Rule, triples: List) -> Dict:
        """Hitung semua metrics untuk rule"""
        # Simplified version untuk demo
        return {
            'confidence': 0.7,
            'lift': 1.5,
            'coherence': 0.8,
            'informativeness': 0.6,
            'support': 0.1
        }

    def get_top_rules(self, k: int = 10) -> List[Rule]:
        """Dapatkan top-k rules"""
        return self.rules[:k]

    def get_rules_by_predicate(self, predicate: str) -> List[Rule]:
        """Dapatkan rules yang predict predicate tertentu"""
        return [
            r
            for r in self.rules
            if isinstance(r.head, tuple)
            and len(r.head) >= 2
            and r.head[1] == predicate
        ]

    def apply_rule(self, subject: str, obj: str) -> bool:
        """Apply rule untuk predict link antara subject dan object"""
        # Simplified: check apakah ada rule yang bisa predict link ini
        for rule in self.get_top_rules(5):
            if self._matches_rule(subject, rule, obj):
                return True
        return False

    def _matches_rule(self, subject: str, rule: Rule, obj: str) -> bool:
        """Check apakah subject-obj pair matches rule"""
        # Simplified matching logic
        return True
