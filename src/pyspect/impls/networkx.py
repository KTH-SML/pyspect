import heapq
from collections import defaultdict

import networkx as nx

class NetworkXImpl:

    def __init__(self, graph: nx.DiGraph, weights: str | list[str] = 'weight', budgets: float | list[float] = float('inf')):
        self.graph = graph
        self.state_space = set(graph.nodes)

        if not self.graph.is_directed():
            raise ValueError("Graph must be directed")

        self.weights = (weights,) if isinstance(weights, str) else tuple(weights)
        self.budgets = ((float(budgets),) if isinstance(budgets, (int, float)) else
                        tuple(float(b) for b in budgets))

        if len(self.weights) != len(self.budgets):
            raise ValueError("len(weights) must equal len(budgets)")

    def filter(self, pred) -> set:
        return {n for n in self.state_space if pred(n)}

    ## Set Interfaces ##

    def empty(self) -> bool:
        return set()
    
    def complement(self, s: set) -> set:
        return self.state_space - s
    
    def intersect(self, s1: set, s2: set) -> set:
        return s1 & s2

    def union(self, s1: set, s2: set) -> set:
        return s1 | s2
    
    def reach(self, target: set, constraints: None | set = None) -> set:

        if not target:
            return set()

        def dominates(a, b):
            return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

        def edge_vec(u, v):
            if G.is_multigraph():
                # pick one parallel edge (cheapest by first weight, tie by sum)
                vecs = [
                    tuple(float(ed.get(w, 1.0)) for w in self.weights)
                    for ed in G.get_edge_data(u, v).values()
                ]
                return min(vecs, key=lambda c: (c[0], sum(c)))
            ed = G.get_edge_data(u, v)
            return tuple(float(ed.get(w, 1.0)) for w in self.weights)

        labels = defaultdict(list)  # node -> list of nondominated cost vectors
        pq = []

        start = tuple([0.0] * len(self.weights))
        for s in target:
            labels[s].append(start)
            heapq.heappush(pq, (start[0], s, start))

        G = self.graph.reverse(copy=False)
        budgets = list(self.budgets)
        while pq:
            _, u, cu = heapq.heappop(pq)
            if cu not in labels[u]:             # stale label
                continue

            for v in G.neighbors(u):
                if v not in constraints:
                    continue

                cv = tuple(x + y for x, y in zip(cu, edge_vec(u, v)))
                if any(c > B for c, B in zip(cv, budgets)):
                    continue
                rem = tuple(B - c for B, c in zip(budgets, cv))
                if any(r < 0 for r in rem):     # budget exhausted
                    continue

                # dominance check at v
                if any(dominates(c_old, cv) or c_old == cv for c_old in labels[v]):
                    continue
                labels[v] = [c_old for c_old in labels[v] if not dominates(cv, c_old)]
                labels[v].append(cv)
                heapq.heappush(pq, (cv[0], v, cv))

        return set(labels)
