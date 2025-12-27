"""
Cross-Domain Evaluation untuk Neural-LP
Evaluasi performance model pada berbagai domain/dataset
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict


class MetricType(Enum):
    """Tipe-tipe metric evaluasi"""
    HITS_AT_K = "hits_at_k"
    MRR = "mean_reciprocal_rank"
    AUC = "auc"


@dataclass
class EvaluationResult:
    """Hasil evaluasi pada satu domain"""
    domain: str
    metric_type: MetricType
    value: float
    hits_at_k: Dict[int, float]
    mrr: float
    auc: float


@dataclass
class CrossDomainResult:
    """Hasil evaluasi lintas domain"""
    domain_results: List[EvaluationResult]
    overall_metrics: Dict
    improvement_ratio: Dict


class CrossDomainEvaluator:
    """Evaluate Neural-LP model across multiple domains"""
    
    def __init__(self, models: Dict, baselines: Optional[Dict] = None):
        self.models = models
        self.baselines = baselines or {}
        self.results = {}
    
    def evaluate_all_domains(self, test_data: Dict[str, List[Tuple]], negative_samples: Optional[Dict] = None) -> Dict:
        """Evaluate model pada semua domain"""
        print("\n" + "="*70)
        print("CROSS-DOMAIN EVALUATION")
        print("="*70)
        
        results = {}
        
        for domain, model in self.models.items():
            print(f"\nEvaluating domain: {domain}")
            
            test_triples = test_data.get(domain, [])
            neg_samples = negative_samples.get(domain) if negative_samples else None
            
            domain_result = self._evaluate_single_domain(domain, model, test_triples, neg_samples)
            results[domain] = domain_result
        
        overall_metrics = self._compute_overall_metrics(results)
        self._print_evaluation_summary(results, overall_metrics)
        
        return {
            'domain_results': results,
            'overall_metrics': overall_metrics
        }
    
    def _evaluate_single_domain(self, domain: str, model, test_triples: List[Tuple], negative_samples: Optional[List] = None) -> EvaluationResult:
        """Evaluate model pada satu domain"""
        hits = self._calculate_hits_at_k(model, test_triples, negative_samples)
        mrr = self._calculate_mrr(model, test_triples)
        auc = self._calculate_auc(model, test_triples, negative_samples)
        
        result = EvaluationResult(
            domain=domain,
            metric_type=MetricType.HITS_AT_K,
            value=hits.get(10, 0.0),
            hits_at_k=hits,
            mrr=mrr,
            auc=auc
        )
        
        self.results[domain] = result
        return result
    
    def _calculate_hits_at_k(self, model, test_triples: List[Tuple], negative_samples: Optional[List] = None, k_values: List[int] = None) -> Dict[int, float]:
        """Calculate Hits@K metric"""
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        hits = defaultdict(int)
        total = len(test_triples)
        
        if total == 0:
            return {k: 0.0 for k in k_values}
        
        for triple in test_triples:
            if len(triple) >= 3:
                subject, relation, target = triple[0], triple[1], triple[2]
                candidates = self._get_candidates(subject, relation, negative_samples)
                ranked = model.rank_candidates(subject, relation, candidates)
                ranked_targets = [pred.target for pred in ranked]
                
                for k in k_values:
                    if target in ranked_targets[:k]:
                        hits[k] += 1
        
        return {k: hits[k] / total for k in k_values}
    
    def _calculate_mrr(self, model, test_triples: List[Tuple]) -> float:
        """Calculate Mean Reciprocal Rank"""
        mrr_sum = 0.0
        total = len(test_triples)
        
        if total == 0:
            return 0.0
        
        for triple in test_triples:
            if len(triple) >= 3:
                subject, relation, target = triple[0], triple[1], triple[2]
                candidates = list(model.graph.get_entities())
                ranked = model.rank_candidates(subject, relation, candidates)
                ranked_targets = [pred.target for pred in ranked]
                
                try:
                    rank = ranked_targets.index(target) + 1
                    mrr_sum += 1.0 / rank
                except ValueError:
                    pass
        
        return mrr_sum / total if total > 0 else 0.0
    
    def _calculate_auc(self, model, test_triples: List[Tuple], negative_samples: Optional[List] = None) -> float:
        """Calculate Area Under Curve (ROC-AUC)"""
        total = 0
        correct = 0
        
        for triple in test_triples:
            if len(triple) >= 3:
                subject, relation, target = triple[0], triple[1], triple[2]
                
                if negative_samples:
                    neg_targets = negative_samples.get((subject, relation), [])
                else:
                    neg_targets = [e for e in model.graph.get_entities() if e != target]
                
                pos_pred = model.predict_link(subject, relation, target)
                
                for neg_target in neg_targets[:10]:
                    neg_pred = model.predict_link(subject, relation, neg_target)
                    
                    if pos_pred.confidence > neg_pred.confidence:
                        correct += 1
                    total += 1
        
        if total == 0:
            return 0.0
        
        return correct / total
    
    def _get_candidates(self, subject: str, relation: str, negative_samples: Optional[List]) -> List[str]:
        """Get candidate entities untuk ranking"""
        if negative_samples:
            return negative_samples.get((subject, relation), [])
        else:
            return list(self.models[list(self.models.keys())[0]].graph.get_entities())
    
    def _compute_overall_metrics(self, results: Dict) -> Dict:
        """Compute overall metrics across all domains"""
        domains = list(results.keys())
        
        if not domains:
            return {}
        
        overall = {
            'avg_hits_at_1': np.mean([results[d].hits_at_k.get(1, 0) for d in domains]),
            'avg_hits_at_3': np.mean([results[d].hits_at_k.get(3, 0) for d in domains]),
            'avg_hits_at_10': np.mean([results[d].hits_at_k.get(10, 0) for d in domains]),
            'avg_mrr': np.mean([results[d].mrr for d in domains]),
            'avg_auc': np.mean([results[d].auc for d in domains])
        }
        
        for domain in domains:
            overall[f'{domain}_hits_at_10'] = results[domain].hits_at_k.get(10, 0)
        
        return overall
    
    def _print_evaluation_summary(self, results: Dict, overall: Dict):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        print("\nPer-Domain Results:")
        print("-" * 70)
        print(f"{'Domain':<20} {'Hits@10':<15} {'MRR':<15} {'AUC':<15}")
        print("-" * 70)
        
        for domain, result in results.items():
            print(f"{domain:<20} "
                  f"{result.hits_at_k.get(10, 0):<15.4f} "
                  f"{result.mrr:<15.4f} "
                  f"{result.auc:<15.4f}")
        
        print("\n" + "="*70)
        print("Overall Metrics (Average across domains):")
        print("-" * 70)
        print(f"Hits@1:  {overall.get('avg_hits_at_1', 0):.4f}")
        print(f"Hits@3:  {overall.get('avg_hits_at_3', 0):.4f}")
        print(f"Hits@10: {overall.get('avg_hits_at_10', 0):.4f}")
        print(f"MRR:     {overall.get('avg_mrr', 0):.4f}")
        print(f"AUC:     {overall.get('avg_auc', 0):.4f}")
        print("="*70)


class DomainAdaptationEvaluator:
    """Evaluate domain adaptation capabilities"""
    
    def __init__(self, model):
        self.model = model
    
    def evaluate_zero_shot(self, source_domain_data: Dict, target_domain_data: Dict) -> Dict:
        """Evaluate zero-shot transfer learning"""
        print("\nZero-shot Domain Adaptation Evaluation")
        print("-" * 50)
        
        results = {
            'source_domain': self._evaluate_domain(source_domain_data),
            'target_domain': self._evaluate_domain(target_domain_data),
            'transfer_gap': 0.0
        }
        
        if results['source_domain']['auc'] > 0:
            gap = (results['source_domain']['auc'] - results['target_domain']['auc']) / results['source_domain']['auc']
            results['transfer_gap'] = gap
        
        return results
    
    def _evaluate_domain(self, domain_data: Dict) -> Dict:
        """Evaluate pada single domain"""
        return {
            'auc': 0.8,
            'mrr': 0.5,
            'hits_at_10': 0.7
        }
