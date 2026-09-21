"""BFS, kostenoptimale Referenz, beste Route unter den kantenkürzesten und Tiefensuche - gegen networkx und gegen Handrechnung."""

import networkx as nx
import numpy as np
import pytest

import bf_algorithm as alg
from bf_graph import from_arcs, route_cost


def random_graph(n, m, seed, directed=False, integer_weights=False):
    rng = np.random.default_rng(seed)
    arcs = []
    for _ in range(m):
        u, v = rng.integers(0, n, 2)
        arcs.append((u, v, float(rng.integers(1, 9)) if integer_weights else 0.5 + 5 * rng.random()))
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=directed)


def to_nx(g):
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v, w in zip(g.out(u), g.out_weights(u)):
            G.add_edge(u, int(v), weight=float(w))
    return G


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("directed", [False, True])
def test_bfs_distances_equal_networkx_hop_counts(seed, directed):
    g = random_graph(60, 110, seed, directed)
    res = alg.bfs(g, [0])
    ref = nx.single_source_shortest_path_length(to_nx(g), 0)
    for v in range(g.n):
        assert res.dist[v] == ref.get(v, -1)


@pytest.mark.parametrize("seed", range(6))
def test_bfs_route_has_the_fewest_edges_and_is_a_real_path(seed):
    g = random_graph(50, 90, seed)
    G = to_nx(g)
    for t in range(1, 50):
        res = alg.bfs(g, [0], [t])
        if not nx.has_path(G, 0, t):
            assert res.found == -1
            continue
        route = res.route(t)
        assert route[0] == 0 and route[-1] == t and len(route) - 1 == nx.shortest_path_length(G, 0, t)
        route_cost(g, route)                                     # jede Kante existiert


@pytest.mark.parametrize("seed", range(6))
def test_reference_costs_equal_networkx_dijkstra(seed):
    g = random_graph(60, 120, seed)
    G = to_nx(g)
    for t in range(1, 60, 3):
        ref = alg.dijkstra_reference(g, [0], [t])
        if nx.has_path(G, 0, t):
            assert ref.dist[t] == pytest.approx(nx.dijkstra_path_length(G, 0, t))
            assert route_cost(g, ref.route(t)) == pytest.approx(ref.dist[t])
        else:
            assert ref.found == -1 and ref.route(t) == []


@pytest.mark.parametrize("seed", range(6))
def test_cost_order_bfs_route_tie_route_reference(seed):
    """Kosten: Referenz <= beste Gleichstands-Route <= BFS-Route; Kantenzahl: die beiden BFS-Routen gleich, die Referenz nie kleiner."""
    g = random_graph(70, 160, seed)
    G = to_nx(g)
    for t in range(1, 70, 2):
        if not nx.has_path(G, 0, t):
            continue
        b = alg.bfs(g, [0], [t]).route(t)
        tie = alg.cheapest_among_fewest_edges(g, [0], t)
        ref = alg.dijkstra_reference(g, [0], [t]).route(t)
        assert len(tie) == len(b) <= len(ref)
        assert route_cost(g, ref) <= route_cost(g, tie) + 1e-9 <= route_cost(g, b) + 2e-9


def test_tie_route_is_the_cheapest_among_all_fewest_edge_routes():
    g = random_graph(40, 70, 3)
    G = to_nx(g)
    for t in range(1, 40):
        if not nx.has_path(G, 0, t):
            continue
        hops = nx.shortest_path_length(G, 0, t)
        cheapest = min(sum(G[a][b]["weight"] for a, b in zip(p[:-1], p[1:])) for p in nx.all_shortest_paths(G, 0, t))
        assert route_cost(g, alg.cheapest_among_fewest_edges(g, [0], t)) == pytest.approx(cheapest)
        assert len(alg.cheapest_among_fewest_edges(g, [0], t)) - 1 == hops


def test_a_hand_built_graph_where_fewer_edges_cost_more():
    # 0 -> 3 direkt für 10, oder 0 -> 1 -> 2 -> 3 für 1 + 1 + 1
    g = from_arcs(4, [(0, 3, 10), (0, 1, 1), (1, 2, 1), (2, 3, 1)], np.zeros((4, 2)))
    b = alg.bfs(g, [0], [3]).route(3)
    ref = alg.dijkstra_reference(g, [0], [3]).route(3)
    assert b == [0, 3] and route_cost(g, b) == 10
    assert ref == [0, 1, 2, 3] and route_cost(g, ref) == 3


def test_unweighted_bfs_is_exactly_optimal():
    rng = np.random.default_rng(5)
    arcs = [(int(u), int(v), 1.0) for u, v in rng.integers(0, 50, (120, 2))]
    g = from_arcs(50, arcs, np.zeros((50, 2)))
    for t in range(1, 50):
        b = alg.bfs(g, [0], [t])
        ref = alg.dijkstra_reference(g, [0], [t])
        if b.found >= 0:
            assert len(b.route(t)) - 1 == ref.dist[t]


def test_multiple_targets_return_the_nearest_and_multiple_sources_the_nearest_start():
    g = from_arcs(7, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1)], np.zeros((7, 2)))
    assert alg.bfs(g, [0], [3, 5]).found == 3
    res = alg.bfs(g, [0, 6], [3, 4])
    assert res.found == 4 and res.dist[4] == 2                    # von Start 6 aus ist Knoten 4 zwei Kanten entfernt, von Start 0 aus vier
    assert alg.bfs(g, [2], [2]).found == 2 and alg.bfs(g, [2], [2]).expanded == []


def test_unreachable_target_is_reported_not_faked():
    g = from_arcs(4, [(0, 1, 1), (2, 3, 1)], np.zeros((4, 2)))
    res = alg.bfs(g, [0], [3])
    assert res.found == -1 and res.route(3) == [] and res.dist[3] == -1
    assert alg.dijkstra_reference(g, [0], [3]).found == -1
    assert alg.cheapest_among_fewest_edges(g, [0], 3) == []
    assert alg.dfs_route(g, 0, [3])[0] == []


def test_one_way_edges_are_respected():
    g = from_arcs(3, [(0, 1, 1), (1, 2, 1)], np.zeros((3, 2)), directed=True)
    assert alg.bfs(g, [0], [2]).found == 2 and alg.bfs(g, [2], [0]).found == -1


def test_parallel_edges_keep_the_cheapest_and_self_loops_vanish():
    g = from_arcs(3, [(0, 1, 5), (0, 1, 2), (1, 1, 1), (1, 2, 1)], np.zeros((3, 2)))
    assert g.weight[g.arc(0, 1)] == 2 and g.arc(1, 1) == -1 and g.m == 4


def test_finish_level_expands_the_whole_previous_layer():
    g = random_graph(80, 160, 2)
    early = alg.bfs(g, [0], [40])
    full = alg.bfs(g, [0], [40], finish_level=True)
    if early.found >= 0:
        assert len(full.expanded) >= len(early.expanded)
        assert all(full.dist[u] < full.dist[full.found] for u in full.expanded)


def test_dfs_finds_a_valid_route_but_not_a_short_one():
    g = random_graph(200, 450, 1)
    G = to_nx(g)
    longer = 0
    for t in range(1, 200, 5):
        if not nx.has_path(G, 0, t):
            continue
        route, visited = alg.dfs_route(g, 0, [t])
        assert route[0] == 0 and route[-1] == t and len(set(route)) == len(route) and visited >= len(route)
        route_cost(g, route)
        longer += len(route) > nx.shortest_path_length(G, 0, t) + 1
    assert longer > 0


def test_max_front_and_layer_sizes_are_consistent():
    g = random_graph(100, 220, 4)
    res = alg.bfs(g, [0])
    assert res.layer_sizes().sum() == len(res.discovered) == (res.dist >= 0).sum()
    assert res.max_front >= res.layer_sizes().max() - 1 and res.scanned == sum(len(g.out(u)) for u in res.expanded)
