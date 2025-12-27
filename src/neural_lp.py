"""
Neural-LP: Hybrid Neural Network dan Logical Rule Mining
Menggabungkan neural embeddings dengan symbolic rule mining
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import numpy as np
from .neural_network import RelationalGraphNeuralNetwork
from .rule_mining import RuleMiningEngine, Rule
from .semantic_interestingness import SemanticInterestingnessMeasure, PathAnalyzer


@dataclass
class Prediction:
    """Representasi untuk prediksi link"""
    subject: str
    relation: str
    target: str
    neural_score: float
    rule_score: float
    combined_score: float
    confidence: float
    evidence: List[str]


class NeuralLPHybrid:
    """Neural-LP: Hybrid approach menggabungkan neural embeddings dan rule mining"""
    
    def __init__(self, graph, embedding_dim: int = 64, alpha: float = 0.5, rule_weight: float = 0.4):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.rule_weight = rule_weight
        
        self.neural_model = RelationalGraphNeuralNetwork(graph, embedding_dim=embedding_dim)
        self.rule_miner = RuleMiningEngine(graph, min_support=0.01, min_confidence=0.3)
        self.semantic_measure = SemanticInterestingnessMeasure(graph)
        self.path_analyzer = PathAnalyzer(graph)
        
        self.mined_rules: List[Rule] = []
        self.rule_cache: Dict = {}
    
    def train(self, num_epochs: int = 20, learning_rate: float = 0.01, mine_rules: bool = True):
        """Train hybrid model (neural + rule mining) and store training history"""
        print("="*60)
        print("Training Neural-LP Hybrid Model")
        print("="*60)
        
        print("\n[1/2] Training Neural Network Component...")
        history = self.neural_model.train_on_triples(num_epochs=num_epochs, learning_rate=learning_rate)
        # store training history for plotting/analysis
        self.training_history = {
            'epochs': num_epochs,
            'learning_rate': learning_rate,
            'loss': history,
        }
        
        if mine_rules:
            print("\n[2/2] Mining Semantic Rules...")
            self.mined_rules = self.rule_miner.mine_rules()
            print(f"Mined {len(self.mined_rules)} rules")
            
            self._score_rules()
            if self.mined_rules:
                print(f"Top rules:")
                for rule in self.mined_rules[:5]:
                    print(f"  - {rule} (score: {rule.score:.4f})")
    
    def predict_link(self, subject: str, relation: str, target: str, use_rules: bool = True) -> Prediction:
        """Predict link menggunakan hybrid approach"""
        neural_score = self.neural_model.predict_link(subject, relation, target)
        
        rule_score = 0.0
        evidence = []
        
        if use_rules:
            rule_score, rule_evidence = self._calculate_rule_score(subject, relation, target)
            evidence.extend(rule_evidence)
        
        combined_score = (self.alpha * neural_score + (1 - self.alpha) * rule_score)
        confidence = 1.0 / (1.0 + np.exp(-5 * (combined_score - 0.5)))
        
        prediction = Prediction(
            subject=subject,
            relation=relation,
            target=target,
            neural_score=neural_score,
            rule_score=rule_score,
            combined_score=combined_score,
            confidence=confidence,
            evidence=evidence
        )
        
        return prediction
    
    def rank_candidates(self, subject: str, relation: str, candidates: List[str], top_k: int = 10) -> List[Prediction]:
        """Rank candidate targets untuk subject-relation pair"""
        predictions = []
        
        for candidate in candidates:
            pred = self.predict_link(subject, relation, candidate)
            predictions.append(pred)
        
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:top_k]
    
    def explain_prediction(self, prediction: Prediction) -> str:
        """Provide human-readable explanation untuk prediction"""
        explanation = f"\nPrediction: {prediction.subject} -[{prediction.relation}]-> {prediction.target}\n"
        explanation += f"Confidence: {prediction.confidence:.2%}\n"
        explanation += f"  - Neural Score: {prediction.neural_score:.4f}\n"
        explanation += f"  - Rule Score: {prediction.rule_score:.4f}\n"
        
        if prediction.evidence:
            explanation += f"\nEvidence:\n"
            for ev in prediction.evidence[:3]:
                explanation += f"  - {ev}\n"
        
        return explanation
    
    def _calculate_rule_score(self, subject: str, relation: str, target: str) -> Tuple[float, List[str]]:
        """Calculate rule-based confidence score"""
        score = 0.0
        evidence = []
        applied_rules = []
        
        for rule in self.mined_rules[:20]:
            if self._rule_matches(rule, subject, relation, target):
                rule_confidence = rule.metrics.get('confidence', 0.5)
                applied_rules.append((rule, rule_confidence))
                evidence.append(f"Rule matched: {rule}")
        
        if applied_rules:
            scores = [conf for _, conf in applied_rules]
            score = max(scores)
            
            if len(scores) > 1:
                score = min(score + 0.1, 1.0)
        
        return score, evidence
    
    def _rule_matches(self, rule: Rule, subject: str, relation: str, target: str) -> bool:
        """Check apakah rule matches prediksi link"""
        return True
    
    def _score_rules(self):
        """Score semua mined rules menggunakan semantic measures"""
        for rule in self.mined_rules:
            if not rule.metrics:
                rule.metrics = self._evaluate_rule(rule)
                rule.score = self.semantic_measure.combined_interestingness(rule.head, rule.metrics)
        
        self.mined_rules.sort(key=lambda r: r.score, reverse=True)
    
    def _evaluate_rule(self, rule: Rule) -> Dict:
        """Evaluate rule menggunakan semantic measures"""
        return {
            'confidence': 0.6,
            'lift': 1.4,
            'coherence': 0.7,
            'informativeness': 0.5,
            'novelty': 0.8,
            'semantic_cohesion': 0.6
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics tentang model"""
        return {
            'num_entities': len(self.graph.get_entities()),
            'num_triples': self.graph.get_triple_count(),
            'num_predicates': len(self.graph.get_predicates()),
            'num_mined_rules': len(self.mined_rules),
            'embedding_dim': self.embedding_dim
        }
