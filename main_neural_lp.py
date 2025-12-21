"""
Main script untuk demonstrasi Neural-LP dengan DRUM
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rdf_graph import RDFGraph
from neural_lp import NeuralLPHybrid
from cross_domain_evaluation import CrossDomainEvaluator
import random


def create_sample_data():
    """Create sample RDF data untuk testing"""
    graph = RDFGraph()
    
    # Sample transaction/knowledge data
    # Domain 1: E-commerce transactions
    transactions_e_commerce = [
        ('customer_1', 'buys', 'product_1'),
        ('customer_1', 'buys', 'product_2'),
        ('customer_2', 'buys', 'product_1'),
        ('product_1', 'has_category', 'electronics'),
        ('product_2', 'has_category', 'electronics'),
        ('customer_1', 'rating', 'high'),
        ('customer_2', 'rating', 'medium'),
    ]
    
    # Domain 2: Knowledge graph relationships
    kg_facts = [
        ('entity_a', 'related_to', 'entity_b'),
        ('entity_a', 'instance_of', 'class_1'),
        ('entity_b', 'instance_of', 'class_2'),
        ('entity_c', 'related_to', 'entity_d'),
        ('class_1', 'subclass_of', 'class_2'),
    ]
    
    all_triples = transactions_e_commerce + kg_facts
    
    for s, p, o in all_triples:
        graph.add_triple(s, p, o)
    
    return graph, all_triples


def main():
    """Main function"""
    print("Neural-LP: Hybrid Neural Network + Rule Mining for Link Prediction")
    print("="*70)
    
    # 1. Create sample data
    print("\n[Step 1] Loading RDF Graph Data...")
    graph, triples = create_sample_data()
    
    stats = graph.get_graph_statistics()
    print(f"  - Entities: {stats['num_entities']}")
    print(f"  - Triples: {stats['num_triples']}")
    print(f"  - Predicates: {stats['num_predicates']}")
    
    # 2. Initialize and train Neural-LP
    print("\n[Step 2] Initializing Neural-LP Hybrid Model...")
    neural_lp = NeuralLPHybrid(
        graph,
        embedding_dim=64,
        alpha=0.5,
        rule_weight=0.4
    )
    
    # 3. Train model
    print("\n[Step 3] Training Model...")
    neural_lp.train(num_epochs=10, learning_rate=0.01, mine_rules=True)
    
    model_stats = neural_lp.get_statistics()
    print(f"\nModel Statistics:")
    print(f"  - Mined Rules: {model_stats['num_mined_rules']}")
    print(f"  - Embedding Dimension: {model_stats['embedding_dim']}")
    
    # 4. Make predictions
    print("\n[Step 4] Making Link Predictions...")
    
    # Predict missing links
    test_cases = [
        ('customer_1', 'buys', 'product_3'),
        ('entity_a', 'related_to', 'entity_c'),
        ('customer_2', 'buys', 'product_2'),
    ]
    
    print("\nLink Predictions:")
    print("-" * 70)
    
    for subject, relation, target in test_cases:
        prediction = neural_lp.predict_link(subject, relation, target)
        explanation = neural_lp.explain_prediction(prediction)
        print(explanation)
    
    # 5. Rank candidates
    print("[Step 5] Ranking Candidate Links...")
    candidates = [f'entity_{i}' for i in range(1, 6)]
    ranked = neural_lp.rank_candidates('entity_a', 'related_to', candidates, top_k=3)
    
    print(f"\nTop candidates untuk entity_a -[related_to]-> ?:")
    for i, pred in enumerate(ranked, 1):
        print(f"  {i}. {pred.target} (confidence: {pred.confidence:.2%})")
    
    # 6. Cross-domain evaluation (simplified)
    print("\n[Step 6] Cross-Domain Evaluation...")
    print("-" * 70)
    
    # Prepare test data
    test_data = {
        'e_commerce': test_cases[:2],
        'knowledge_graph': test_cases[2:]
    }
    
    models = {
        'e_commerce': neural_lp,
        'knowledge_graph': neural_lp
    }
    
    evaluator = CrossDomainEvaluator(models)
    eval_results = evaluator.evaluate_all_domains(test_data)
    
    print("\nEvaluation Complete!")
    print("="*70)
    
    return neural_lp, eval_results


if __name__ == "__main__":
    model, results = main()
    
    print("\n✓ Neural-LP training and evaluation completed successfully!")
    print("\nNext steps:")
    print("1. Expand dataset dengan lebih banyak domain-specific data")
    print("2. Tune hyperparameters (embedding_dim, alpha, learning_rate)")
    print("3. Implement rule explanation methods")
    print("4. Add support untuk transaksi time-sensitive dan dynamic graphs")
