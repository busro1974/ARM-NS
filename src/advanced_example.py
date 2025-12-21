"""
Advanced example dengan semantic-based evaluation
"""

import sys
import os

# Ensure local `src` imports work when running script directly
try:
    from semantic_interestingness import SemanticInterestingnessMeasure, PathAnalyzer
    from neural_lp import NeuralLPHybrid
    from rdf_graph import RDFGraph
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from semantic_interestingness import SemanticInterestingnessMeasure, PathAnalyzer
    from neural_lp import NeuralLPHybrid
    from rdf_graph import RDFGraph


def example_semantic_evaluation():
    """
    Contoh evaluasi dengan semantic interestingness measures
    """
    print("Advanced Semantic Interestingness Measure Example")
    print("=" * 70)

    # Create graph
    graph = RDFGraph()

    # Add domain-specific triples
    domain_data = [
        # Banking domain
        ('account_1', 'has_owner', 'person_1'),
        ('person_1', 'has_type', 'premium_customer'),
        ('account_1', 'transaction_with', 'account_2'),
        ('account_2', 'has_owner', 'person_2'),
        ('person_2', 'has_type', 'regular_customer'),

        # E-commerce domain
        ('user_1', 'viewed', 'product_1'),
        ('product_1', 'has_category', 'electronics'),
        ('user_1', 'purchased', 'product_2'),
        ('product_2', 'has_category', 'electronics'),
        ('user_2', 'viewed', 'product_3'),
    ]

    for s, p, o in domain_data:
        graph.add_triple(s, p, o)

    # Initialize semantic measure
    semantic = SemanticInterestingnessMeasure(graph, use_embedding=False)
    path_analyzer = PathAnalyzer(graph)

    # Test semantic measures
    print("\nSemantic Interestingness Measures:")
    print("=" * 70)

    # Example rule
    rule = (('account_1', 'transaction_with', 'account_2'),
            ('person_1', 'has_type', 'premium_customer'))

    metrics = {
        'confidence': 0.85,
        'lift': 1.2,
        'coherence': semantic.coherence(rule, {'avg_path_length': 2, 'common_neighbors': 3}),
        'informativeness': semantic.informativeness(0.85, 0.1),
        'specificity': semantic.specificity(2, 5),
    }

    combined = semantic.combined_interestingness(rule, metrics)

    print(f"\nRule: {rule}")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric:<20}: {value:.4f}")
    print(f"\nCombined Interestingness Score: {combined:.4f}")

    print("\n" + "=" * 70)
    print("Path Analysis:")
    print("-" * 70)

    path_metrics = path_analyzer.get_path_metrics('person_1', 'person_2')
    print("Path metrics between person_1 and person_2:")
    for key, value in path_metrics.items():
        print(f"  {key}: {value}")

    # Train Neural-LP on this domain
    print("\n" + "=" * 70)
    print("Training Neural-LP on Banking Domain:")
    print("-" * 70)

    neural_lp = NeuralLPHybrid(
        graph,
        embedding_dim=32,
        alpha=0.6
    )

    neural_lp.train(num_epochs=5, mine_rules=True)

    # Make predictions
    print("\n" + "=" * 70)
    print("Link Predictions for Banking Domain:")
    print("-" * 70)

    predictions = [
        ('person_1', 'has_type', 'vip_customer'),
        ('account_2', 'transaction_with', 'account_1'),
    ]

    for subject, relation, target in predictions:
        pred = neural_lp.predict_link(subject, relation, target)
        print(f"\nPrediction: {subject} -[{relation}]-> {target}")
        print(f"Confidence: {pred.confidence:.2%}")
        print(f"Neural Score: {pred.neural_score:.4f}")
        print(f"Rule Score: {pred.rule_score:.4f}")


if __name__ == "__main__":
    example_semantic_evaluation()

    print("\n" + "=" * 70)
    print("✓ Advanced example completed successfully!")
    print("=" * 70)
