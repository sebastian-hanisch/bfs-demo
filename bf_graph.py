"""Gerichteter Graph in CSR-Form (Kompaktzeilen): Nachbarn eines Knotens u stehen in indices[indptr[u]:indptr[u+1]], die Kosten der Kanten in weight."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Graph:
    n: int
    indptr: np.ndarray            # (n + 1,) int
    indices: np.ndarray           # (m,) int   Zielknoten je gerichteter Kante, je Knoten aufsteigend sortiert
    weight: np.ndarray            # (m,) float Kosten je gerichteter Kante
    xy: np.ndarray                # (n, 2) Lage für die Zeichnung
    names: tuple = ()             # optionale Beschriftungen
    directed: bool = False

    @property
    def m(self):
        return len(self.indices)

    def out(self, u):
        return self.indices[self.indptr[u]:self.indptr[u + 1]]

    def out_weights(self, u):
        return self.weight[self.indptr[u]:self.indptr[u + 1]]

    def degree(self):
        return np.diff(self.indptr)

    def arc(self, u, v):
        """Index der Kante u -> v in indices (oder -1)."""
        lo, hi = self.indptr[u], self.indptr[u + 1]
        k = lo + int(np.searchsorted(self.indices[lo:hi], v))
        return k if k < hi and self.indices[k] == v else -1

    def edge_list(self):
        """Jede Kante einmal: bei ungerichteten Graphen u < v, bei gerichteten alle Kanten. Rückgabe (u, v, w)."""
        src = np.repeat(np.arange(self.n), self.degree())
        if self.directed:
            return src, self.indices, self.weight
        keep = src < self.indices
        return src[keep], self.indices[keep], self.weight[keep]


def from_arcs(n, arcs, xy, names=(), directed=False):
    """Baut den Graphen aus (u, v, Kosten)-Tupeln. Ungerichtet: jede Kante einmal angeben, die Rückrichtung entsteht hier. Selbstschleifen fallen weg,
    von mehreren parallelen Kanten bleibt die billigste (so empfiehlt es das Buch für OSM-Netze: die Länge einer Route hinge sonst davon ab, welche Parallelkante gewählt wird)."""
    best = {}
    for u, v, w in arcs:
        u, v, w = int(u), int(v), float(w)
        if u == v:
            continue
        pairs = ((u, v),) if directed else ((u, v), (v, u))
        for a, b in pairs:
            if (a, b) not in best or w < best[(a, b)]:
                best[(a, b)] = w
    keys = sorted(best)
    src = np.fromiter((k[0] for k in keys), dtype=np.int64, count=len(keys))
    dst = np.fromiter((k[1] for k in keys), dtype=np.int64, count=len(keys))
    wts = np.fromiter((best[k] for k in keys), dtype=float, count=len(keys))
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return Graph(n, indptr, dst, wts, np.asarray(xy, dtype=float), tuple(names), directed)


def route_cost(g, route):
    """Summe der Kantenkosten entlang einer Knotenfolge; Kanten, die es nicht gibt, sind ein Fehler."""
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        k = g.arc(u, v)
        if k < 0:
            raise ValueError(f"keine Kante {u} -> {v}")
        total += float(g.weight[k])
    return total
