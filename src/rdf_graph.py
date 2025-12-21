"""
RDF Graph Handler untuk penanganan triple data dan operasi graph
"""

import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional


class RDFTriple:
    """Representasi triple RDF (Subject, Predicate, Object)"""

    def __init__(self, subject: str, predicate: str, obj: str):
        self.subject = subject
        self.predicate = predicate
        self.object = obj

    def __repr__(self):
        return f"({self.subject}, {self.predicate}, {self.object})"

    def __eq__(self, other):
        return (
            self.subject == other.subject
            and self.predicate == other.predicate
            and self.object == other.object
        )

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))


class RDFGraph:
    """Graph struktur untuk menangani RDF triples dan link prediction"""

    def __init__(self):
        self.triples: Set[RDFTriple] = set()
        self.graph = nx.DiGraph()
        self.predicate_index = defaultdict(set)
        self.entity_pairs = defaultdict(set)

    def add_triple(self, subject: str, predicate: str, obj: str):
        """Tambah triple ke graph"""
        triple = RDFTriple(subject, predicate, obj)

        if triple not in self.triples:
            self.triples.add(triple)
            self.graph.add_edge(subject, obj, predicate=predicate)
            self.predicate_index[predicate].add(triple)
            self.entity_pairs[(subject, obj)].add(predicate)

    def add_triples_batch(self, triples: List[Tuple[str, str, str]]):
        """Tambah multiple triples sekaligus"""
        for s, p, o in triples:
            self.add_triple(s, p, o)

    def get_triples_by_predicate(self, predicate: str) -> Set[RDFTriple]:
        """Dapatkan semua triple dengan predicate tertentu"""
        return self.predicate_index.get(predicate, set())

    def get_outgoing_edges(self, entity: str) -> List[Tuple[str, str]]:
        """Dapatkan semua edge outgoing dari entity (neighbors dan predicates)"""
        edges = []
        for successor in self.graph.successors(entity):
            predicates = self.graph[entity][successor]
            edges.append((successor, predicates.get('predicate', '')))
        return edges

    def get_incoming_edges(self, entity: str) -> List[Tuple[str, str]]:
        """Dapatkan semua edge incoming ke entity"""
        edges = []
        for predecessor in self.graph.predecessors(entity):
            predicates = self.graph[predecessor][entity]
            edges.append((predecessor, predicates.get('predicate', '')))
        return edges

    def get_neighbors(self, entity: str, direction: str = 'both') -> Set[str]:
        """Dapatkan semua neighbors dari entity"""
        neighbors = set()
        if direction in ['out', 'both']:
            neighbors.update(self.graph.successors(entity))
        if direction in ['in', 'both']:
            neighbors.update(self.graph.predecessors(entity))
        return neighbors

    def get_common_neighbors(self, entity1: str, entity2: str) -> Set[str]:
        """Dapatkan common neighbors antara dua entity"""
        neighbors1 = self.get_neighbors(entity1)
        neighbors2 = self.get_neighbors(entity2)
        return neighbors1.intersection(neighbors2)

    def get_path_length(self, source: str, target: str, max_length: int = 3) -> Optional[int]:
        """Dapatkan shortest path length antara dua entity"""
        try:
            return nx.shortest_path_length(self.graph, source, target, cutoff=max_length)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_entities(self) -> Set[str]:
        """Dapatkan semua entities dalam graph"""
        return set(self.graph.nodes())

    def get_predicates(self) -> Set[str]:
        """Dapatkan semua unique predicates"""
        return set(self.predicate_index.keys())

    def get_triple_count(self) -> int:
        """Total jumlah triples"""
        return len(self.triples)

    def entity_exists(self, entity: str) -> bool:
        """Check apakah entity ada dalam graph"""
        return entity in self.graph.nodes()

    def triple_exists(self, subject: str, predicate: str, obj: str) -> bool:
        """Check apakah triple sudah ada"""
        return RDFTriple(subject, predicate, obj) in self.triples

    def get_graph_statistics(self) -> Dict:
        """Dapatkan statistik graph"""
        return {
            'num_entities': len(self.get_entities()),
            'num_triples': self.get_triple_count(),
            'num_predicates': len(self.get_predicates()),
            'density': nx.density(self.graph.to_undirected()),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(len(self.graph), 1),
            'num_weakly_connected_components': nx.number_weakly_connected_components(self.graph)
        }

    def get_triples_with_entity(self, entity: str) -> Set[RDFTriple]:
        """Dapatkan semua triples yang melibatkan entity tertentu"""
        relevant_triples = set()
        for triple in self.triples:
            if triple.subject == entity or triple.object == entity:
                relevant_triples.add(triple)
        return relevant_triples

    def sample_triples(self, predicate: str, num_samples: int = 10) -> List[RDFTriple]:
        """Sample triples dari predicate tertentu"""
        triples = list(self.get_triples_by_predicate(predicate))
        return triples[:min(num_samples, len(triples))]

    def remove_triple(self, subject: str, predicate: str, obj: str):
        """Hapus triple dari graph"""
        triple = RDFTriple(subject, predicate, obj)
        if triple in self.triples:
            self.triples.remove(triple)
            # Update graph jika tidak ada edge lain dengan predicates berbeda
            if self.graph.has_edge(subject, obj):
                del self.graph[subject][obj]
            if triple in self.predicate_index[predicate]:
                self.predicate_index[predicate].remove(triple)
