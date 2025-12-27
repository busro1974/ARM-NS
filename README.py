"""
README: Neural-LP System - Hybrid Neural Network + Rule Mining for Link Prediction

[![CI](https://github.com/busro1974/ARM-NS/actions/workflows/ci.yml/badge.svg)](https://github.com/busro1974/ARM-NS/actions/workflows/ci.yml)

# Catatan: Badge ini diisi otomatis dari remote `origin` (git@github.com:busro1974/ARM-NS.git). Jika perlu, ganti manual `busro1974/ARM-NS` dengan repo Anda.

DESKRIPSI SISTEM
================
Sistem Neural-LP adalah implementasi hybrid framework yang menggabungkan:
1. Neural Network (TransE Model) untuk entity dan relation embeddings
2. Rule Mining Engine dengan semantic-based interestingness measures
3. Cross-Domain Evaluation framework untuk evaluasi lintas domain
4. Link Prediction system yang menggabungkan neural dan symbolic approaches

KOMPONEN UTAMA
==============

1. RDF GRAPH HANDLER (rdf_graph.py)
   - Menangani triple RDF (Subject, Predicate, Object)
   - Graph operations: neighbors, paths, statistics
   - Support untuk RDF data, transaction data, dan knowledge graphs
   
   Contoh penggunaan:
   ```python
   from rdf_graph import RDFGraph
   graph = RDFGraph()
   graph.add_triple('customer_1', 'buys', 'product_1')
   ```

2. NEURAL NETWORK COMPONENT (neural_network.py)
   - TransE Model: h + r ≈ t (head + relation ≈ tail)
   - Entity dan Relation Embeddings
   - Link prediction menggunakan distance-based scoring
   - Entity similarity calculation
   
   Contoh penggunaan:
   ```python
   from neural_network import RelationalGraphNeuralNetwork
   neural_model = RelationalGraphNeuralNetwork(graph, embedding_dim=64)
   neural_model.train_on_triples(num_epochs=10)
   score = neural_model.predict_link('customer_1', 'buys', 'product_2')
   ```

3. RULE MINING ENGINE (rule_mining.py)
   - Mining chain rules: p1(x,y) ∧ p2(y,z) -> p3(x,z)
   - Mining composition rules: p1(x,y) -> p2(x,y)
   - Semantic Measure class dengan berbagai interestingness metrics:
     * Support, Confidence, Lift, Conviction
     * Coherence, Informativeness, Specificity
     * Relevance, Consistency, Novelty
   
   Contoh penggunaan:
   ```python
   from rule_mining import RuleMiningEngine
   rule_miner = RuleMiningEngine(graph)
   rules = rule_miner.mine_rules()
   ```

4. SEMANTIC INTERESTINGNESS MEASURE (semantic_interestingness.py)
   - Traditional measures: Support, Confidence, Lift, Conviction
   - Semantic measures: Coherence, Informativeness, Specificity, Relevance
   - Path-based analysis untuk semantic relatedness
   - Combined interestingness scoring dengan weighted aggregation
   
   Contoh penggunaan:
   ```python
   from semantic_interestingness import SemanticInterestingnessMeasure
   semantic = SemanticInterestingnessMeasure(graph)
   metrics = {
       'confidence': 0.85,
       'coherence': 0.75,
       'informativeness': 0.9
   }
   combined_score = semantic.combined_interestingness(rule, metrics)
   ```

5. NEURAL-LP HYBRID SYSTEM (neural_lp.py)
   - Menggabungkan neural predictions dengan rule-based scoring
   - Hybrid approach dengan configurable weights (alpha parameter)
   - Prediksi dengan penjelasan (evidence)
   - Ranking candidates untuk link prediction
   
   Contoh penggunaan:
   ```python
   from neural_lp import NeuralLPHybrid
   model = NeuralLPHybrid(graph, embedding_dim=64, alpha=0.5)
   model.train(num_epochs=20, mine_rules=True)
   prediction = model.predict_link('customer_1', 'buys', 'product_3')
   print(model.explain_prediction(prediction))
   ```

6. CROSS-DOMAIN EVALUATION (cross_domain_evaluation.py)
   - Evaluasi performance pada multiple domains
   - Metrics: Hits@K, Mean Reciprocal Rank (MRR), AUC
   - Domain adaptation evaluation
   - Perbandingan dengan baseline models
   
   Contoh penggunaan:
   ```python
   from cross_domain_evaluation import CrossDomainEvaluator
   evaluator = CrossDomainEvaluator(models={'domain1': model1, 'domain2': model2})
   results = evaluator.evaluate_all_domains(test_data)
   ```

WORKFLOW SISTEM
===============

1. DATA LOADING
   - Load RDF triples atau transaction data
   - Build graph structure dengan RDFGraph

2. INITIALIZATION
   - Initialize NeuralLPHybrid dengan graph
   - Configure hyperparameters (embedding_dim, alpha, learning_rate)

3. TRAINING
   - Train neural network component (TransE model)
   - Mine rules dari graph menggunakan RuleMiningEngine
   - Score rules menggunakan semantic measures
   - Rank rules berdasarkan interestingness score

4. PREDICTION
   - Untuk setiap (subject, relation, target):
     * Hitung neural score menggunakan trained embeddings
     * Hitung rule-based score menggunakan matched rules
     * Combine scores dengan weighted averaging
     * Convert ke confidence probability

5. EVALUATION
   - Evaluate pada test set menggunakan Hits@K, MRR, AUC
   - Support cross-domain evaluation
   - Compare dengan baselines

PENGGUNAAN
==========

Main Script (main_neural_lp.py):
```bash
python main_neural_lp.py
```

Advanced Example (src/advanced_example.py):
```bash
cd src
python advanced_example.py
```

FITUR KUNCI
===========

1. HYBRID APPROACH
   - Menggabungkan kekuatan neural networks dan symbolic rules
   - Neural: scalability, generalization
   - Symbolic: interpretability, explainability

2. SEMANTIC-BASED EVALUATION
   - 10+ semantic measures untuk rule quality
   - Path-based analysis untuk semantic relatedness
   - Domain-aware relevance scoring

3. CROSS-DOMAIN EVALUATION
   - Evaluate performance pada multiple domains
   - Zero-shot transfer learning evaluation
   - Domain adaptation analysis

4. INTERPRETABILITY
   - Explain predictions dengan evidence dari rules
   - Path metrics untuk semantic analysis
   - Rule statistics dan visualization

HYPERPARAMETERS
===============

embedding_dim (default: 64)
  - Dimensionality dari entity dan relation embeddings
  - Larger values: more expressive, slower training
  - Recommended range: 32-128

alpha (default: 0.5)
  - Weight untuk neural vs rule scores
  - alpha=1.0: pure neural approach
  - alpha=0.0: pure rule-based approach
  - Recommended: 0.4-0.6 untuk balance

rule_weight (default: 0.4)
  - Importance weight untuk rule-based components
  - Used dalam hybrid scoring

learning_rate (default: 0.01)
  - Training rate untuk embedding updates
  - Smaller values: more stable, slower convergence
  - Recommended: 0.001-0.1

num_epochs (default: 10-20)
  - Number of training epochs
  - More epochs: better convergence, risk of overfitting

PERFORMA
========

Benchmark pada sample dataset:
- Entities: 13
- Triples: 12
- Predicates: 6

Main Script Results:
- Hits@1:  0.0000
- Hits@3:  0.7500
- Hits@10: 0.7500
- MRR:     0.3750
- AUC:     0.6750

NEXT STEPS
==========

1. Expand dataset dengan domain-specific data
2. Tune hyperparameters untuk optimal performance
3. Implement rule explanation methods
4. Add support untuk time-sensitive transactions
5. Support dynamic graphs dengan incremental learning
6. Visualization tools untuk rules dan predictions
7. Integration dengan external knowledge bases

REQUIREMENTS
============

- Python 3.7+
- NumPy
- NetworkX
- (Optional) TensorFlow/PyTorch untuk advanced models

INSTALASI
=========

1. Create dataset folder:
   mkdir data

2. Create model checkpoint folder:
   mkdir models

3. Jalankan main script:
   python main_neural_lp.py

REFERENCES
==========

- Neural-LP (He et al., 2016)
- DRUM (Sadeghian et al., 2019)
- TransE (Bordes et al., 2013)
- Knowledge Graph Embedding
- Rule Mining for Knowledge Graphs

AUTHOR & CONTACT
================

Neural-LP System Implementation
Last Updated: December 2025
"""

if __name__ == "__main__":
    print(__doc__)
