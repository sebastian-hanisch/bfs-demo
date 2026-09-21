"""Breitensuche (BFS) und die Vergleichsverfahren: kostenoptimale Referenz (Dijkstra, Stück 2 der Linie), beste Route unter den kantenkürzesten, Tiefensuche (nur als Gegenprobe).

Alles ist eigene Umsetzung auf dem CSR-Graphen aus bf_graph.py; networkx kommt nur in den Tests vor (Kreuzprobe)."""

import heapq
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from bf_graph import route_cost


@dataclass
class Bfs:
    dist: np.ndarray                          # Kanten vom nächsten Start, -1 = nicht entdeckt
    parent: np.ndarray                        # Vorgänger im Suchbaum, -1 = Start oder nicht entdeckt
    expanded: list = field(default_factory=list)      # Knoten in der Reihenfolge, in der ihre Nachbarn geprüft wurden
    discovered: list = field(default_factory=list)    # Knoten in der Reihenfolge ihrer Entdeckung (Starts zuerst)
    found: int = -1                           # zuerst entdecktes Ziel, -1 = keines
    max_front: int = 0                        # größte Warteschlange (entdeckt, aber noch nicht abgearbeitet): der Speicherbedarf der Suche
    scanned: int = 0                          # geprüfte Kanten

    def layer_sizes(self):
        """Entdeckte Knoten je Schicht (Kantenzahl vom Start)."""
        d = self.dist[self.dist >= 0]
        return np.bincount(d) if len(d) else np.zeros(0, dtype=int)

    def route(self, target):
        """Knotenfolge vom Start zum Knoten `target` entlang des Suchbaums (leer, wenn nicht entdeckt)."""
        if self.dist[target] < 0:
            return []
        path = [int(target)]
        while self.parent[path[-1]] >= 0:
            path.append(int(self.parent[path[-1]]))
        return path[::-1]


def bfs(g, sources, targets=(), finish_level=False):
    """Schichtenweise Breitensuche ab einem oder mehreren Starts. Sobald ein Ziel entdeckt wird, ist seine Kantenzahl endgültig (jede Kante zählt eins) - die Suche endet dort
    (`finish_level=True`: erst die Schicht davor fertig abarbeiten, damit alle kantenkürzesten Vorgänger des Ziels bekannt sind). Die Nachbarn kommen in der Reihenfolge der
    Knotennummern; bei mehreren gleich kurzen Routen entscheidet allein diese Reihenfolge."""
    ip, ix = g.indptr.tolist(), g.indices.tolist()
    dist, parent = [-1] * g.n, [-1] * g.n
    targets = set(int(t) for t in targets)
    queue, discovered, expanded = deque(), [], []
    found, stop_level, scanned = -1, None, 0
    for s in sources:
        s = int(s)
        if dist[s] < 0:
            dist[s] = 0
            queue.append(s)
            discovered.append(s)
            if found < 0 and s in targets:
                found, stop_level = s, 0
    max_front = len(queue)
    done = found >= 0 and not finish_level
    while queue and not done:
        u = queue.popleft()
        if stop_level is not None and dist[u] >= stop_level:
            break
        expanded.append(u)
        du = dist[u]
        for k in range(ip[u], ip[u + 1]):
            v = ix[k]
            scanned += 1
            if dist[v] < 0:
                dist[v], parent[v] = du + 1, u
                queue.append(v)
                discovered.append(v)
                if found < 0 and v in targets:
                    found, stop_level = v, du + 1
                    if not finish_level:
                        done = True
                        break
        max_front = max(max_front, len(queue))
    return Bfs(np.array(dist), np.array(parent), expanded, discovered, found, max_front, scanned)


@dataclass
class Shortest:
    dist: np.ndarray                          # Kosten vom nächsten Start, inf = nicht erreicht
    parent: np.ndarray
    found: int = -1
    settled: int = 0                          # Knoten, deren Kosten endgültig festgelegt wurden

    def route(self, target):
        if not np.isfinite(self.dist[target]):
            return []
        path = [int(target)]
        while self.parent[path[-1]] >= 0:
            path.append(int(self.parent[path[-1]]))
        return path[::-1]


def dijkstra_reference(g, sources, targets=()):
    """Kostenoptimale Referenz (Dijkstra) - nur zum Messen, wie weit die kantenkürzeste Route von der kostenkürzesten entfernt ist. Das Verfahren selbst ist das nächste Stück der Linie."""
    ip, ix, w = g.indptr.tolist(), g.indices.tolist(), g.weight.tolist()
    inf = float("inf")
    dist, parent = [inf] * g.n, [-1] * g.n
    targets = set(int(t) for t in targets)
    heap = []
    for s in sources:
        s = int(s)
        dist[s] = 0.0
        heap.append((0.0, s))
    heapq.heapify(heap)
    settled, found, done = 0, -1, [False] * g.n
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        settled += 1
        if u in targets:
            found = u
            break
        for k in range(ip[u], ip[u + 1]):
            v = ix[k]
            nd = d + w[k]
            if nd < dist[v]:
                dist[v], parent[v] = nd, u
                heapq.heappush(heap, (nd, v))
    return Shortest(np.array(dist), np.array(parent), found, settled)


def cheapest_among_fewest_edges(g, sources, target):
    """Die billigste Route unter allen Routen mit der kleinsten Kantenzahl (Programmierung über den Schichten-Graphen). Eine BFS-Route ist eine beliebige davon (Nachbar-Reihenfolge);
    diese hier ist die bestmögliche - was danach an Umweg bleibt, liegt nicht am Gleichstand, sondern daran, dass BFS Kosten ignoriert. Leere Liste, wenn das Ziel nicht erreichbar ist."""
    res = bfs(g, sources, (target,), finish_level=True)
    if res.found < 0:
        return []
    last = int(res.dist[res.found])
    ip, ix, w = g.indptr.tolist(), g.indices.tolist(), g.weight.tolist()
    dist = res.dist.tolist()
    inf = float("inf")
    best, parent = [inf] * g.n, [-1] * g.n
    for s in sources:
        best[int(s)] = 0.0
    for u in res.expanded:                                   # Schichten aufsteigend
        if best[u] == inf:
            continue
        for k in range(ip[u], ip[u + 1]):
            v = ix[k]
            if dist[v] == dist[u] + 1 and dist[v] <= last and best[u] + w[k] < best[v]:
                best[v], parent[v] = best[u] + w[k], u
    path = [int(res.found)]
    while parent[path[-1]] >= 0:
        path.append(parent[path[-1]])
    return path[::-1]


def dfs_route(g, source, targets):
    """Erste Route, die eine Tiefensuche findet (Nachbarn in Knotennummern-Reihenfolge): (Route, besuchte Knoten). Keine Kürzeste-Weg-Garantie, nur die Gegenprobe."""
    ip, ix = g.indptr.tolist(), g.indices.tolist()
    targets = set(int(t) for t in targets)
    seen = [False] * g.n
    seen[int(source)] = True
    stack, nxt = [int(source)], [ip[int(source)]]
    visited = 1
    if int(source) in targets:
        return [int(source)], visited
    while stack:
        u = stack[-1]
        if nxt[-1] >= ip[u + 1]:
            stack.pop()
            nxt.pop()
            continue
        v = ix[nxt[-1]]
        nxt[-1] += 1
        if seen[v]:
            continue
        seen[v] = True
        visited += 1
        stack.append(v)
        nxt.append(ip[v])
        if v in targets:
            return list(stack), visited
    return [], visited


def route_summary(g, route):
    """(Kanten, Kosten) einer Route; (0, 0.0) bei leerer Route."""
    if not route:
        return 0, 0.0
    return len(route) - 1, route_cost(g, route)
