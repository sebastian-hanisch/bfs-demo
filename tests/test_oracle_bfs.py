"""Unabhängige Orakel für BFS, kostenoptimale Referenz, beste Route unter den kantenkürzesten und Tiefensuche:
eigene Warteschlangen-Simulation (Zähler: geprüfte Kanten, größte Front, Reihenfolge), networkx mit Super-Quelle (Mehrfach-Start),
Aufzählung ALLER kantenkürzesten Routen (nx.all_shortest_paths) für das Gleichstands-Minimum. Gleichstände (ganzzahlige Kosten),
Mehrfach-Start/-Ziel und unerreichbare Ziele sind absichtlich häufig. Große Fassung (400 Graphen) im Scratchpad."""

import random

import numpy as np
import pytest

import bf_algorithm as alg
from bf_graph import from_arcs, route_cost

nx = pytest.importorskip("networkx")


def _to_nx(g):
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()):
            G.add_edge(u, v, weight=w)
    return G


def _queue_reference(g, sources):
    seen = {s: 0 for s in sources}
    q = list(dict.fromkeys(sources))
    order, expanded, head, scanned, max_front = list(q), [], 0, 0, len(q)
    while head < len(q):
        u = q[head]
        head += 1
        expanded.append(u)
        for v in g.out(u).tolist():
            scanned += 1
            if v not in seen:
                seen[v] = seen[u] + 1
                q.append(v)
                order.append(v)
        max_front = max(max_front, len(q) - head)
    return seen, max_front, scanned, expanded, order


def _instances(count, seed):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.choice([1, 2, 3, 5, 8, 12, 20])
        directed = rng.random() < 0.5
        integer = rng.random() < 0.6
        arcs = [(rng.randrange(n), rng.randrange(n), float(rng.randint(1, 4)) if integer else 0.5 + 5 * rng.random())
                for _ in range(rng.randrange(0, 3 * n + 1))]
        g = from_arcs(n, arcs, np.zeros((n, 2)), directed=directed)
        yield rng, g, rng.sample(range(n), rng.randint(1, min(3, n))), rng.sample(range(n), rng.randint(1, min(3, n)))


def test_bfs_matches_queue_simulation_and_networkx():
    for rng, g, S, T in _instances(60, 2024):
        seen, max_front, scanned, expanded, order = _queue_reference(g, S)
        r = alg.bfs(g, S)
        assert all(r.dist[v] == seen.get(v, -1) for v in range(g.n))
        assert (r.max_front, r.scanned, r.expanded, r.discovered) == (max_front, scanned, expanded, order)
        G = _to_nx(g)
        G.add_node("src")
        for s in S:
            G.add_edge("src", s, weight=1)
        ref = nx.single_source_shortest_path_length(G, "src")
        assert all(r.dist[v] == (ref[v] - 1 if v in ref else -1) for v in range(g.n))
        for finish_level in (False, True):
            b = alg.bfs(g, S, T, finish_level=finish_level)
            reach = [t for t in T if t in seen]
            if not reach:
                assert b.found == -1
                continue
            dmin = min(seen[t] for t in reach)
            assert b.found in T and seen[b.found] == dmin  # das zuerst entdeckte Ziel ist ein nächstes
            rt = b.route(b.found)
            assert rt[0] in S and len(rt) - 1 == dmin and all(g.arc(a, c) >= 0 for a, c in zip(rt, rt[1:]))


def test_reference_dijkstra_cheapest_tie_route_and_dfs():
    for rng, g, S, T in _instances(60, 11):
        G = _to_nx(g)
        dr = alg.dijkstra_reference(g, S, T)
        H = G.copy()
        H.add_node("src")
        for s in S:
            H.add_edge("src", s, weight=0)
        dd = nx.single_source_dijkstra_path_length(H, "src")
        reach = [t for t in T if t in dd]
        if reach:
            assert dr.found >= 0 and dr.dist[dr.found] == pytest.approx(min(dd[t] for t in reach))
            assert route_cost(g, dr.route(dr.found)) == pytest.approx(dr.dist[dr.found])
        else:
            assert dr.found == -1
        s, t = S[0], T[0]
        route = alg.cheapest_among_fewest_edges(g, (s,), t)
        dfs_route, visited = alg.dfs_route(g, s, (t,))
        if not nx.has_path(G, s, t):
            assert route == [] and dfs_route == []
            continue
        paths = list(nx.all_shortest_paths(G, s, t))  # alle Routen mit den wenigsten Kanten
        assert len(route) == len(paths[0]) and route[0] == s and route[-1] == t
        assert route_cost(g, route) == pytest.approx(min(route_cost(g, p) for p in paths))
        assert dfs_route[0] == s and dfs_route[-1] == t and len(set(dfs_route)) == len(dfs_route) and visited >= len(dfs_route)
        assert all(g.arc(a, c) >= 0 for a, c in zip(dfs_route, dfs_route[1:]))
