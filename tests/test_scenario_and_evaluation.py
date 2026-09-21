"""Netze (Stadtnetz, Labyrinth, Toronto, Lehrbuch-Graphen), Kennzahlen, Verteilung über Paare, Sweeps."""

import numpy as np
import pytest

import bf_algorithm as alg
import bf_constants as C
import bf_evaluation as ev
import bf_scenario as sc
from bf_graph import route_cost


def _connected(g):
    return (alg.bfs(g, [0]).dist >= 0).all()


# --- Stadtnetz -------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("reach,blocked", [(1.0, 60), (1.5, 40), (2.3, 20), (3.2, 60)])
def test_city_is_always_connected_and_symmetric(seed, reach, blocked):
    g = sc.build_city(12, reach, 1.0, blocked, seed)
    assert g.n == 144 and _connected(g)
    for u in range(g.n):
        for v in g.out(u):
            assert g.arc(int(v), u) >= 0 and g.weight[g.arc(u, int(v))] == g.weight[g.arc(int(v), u)]


def test_city_is_deterministic_per_seed_and_differs_between_seeds():
    a, b, c = (sc.build_city(10, 2.3, 1.0, 20, s) for s in (1, 1, 2))
    assert np.array_equal(a.indices, b.indices) and np.array_equal(a.weight, b.weight) and not np.array_equal(a.weight, c.weight)


def test_reach_one_is_a_plain_grid_and_larger_reach_adds_edges():
    g1 = sc.build_city(10, 1.0, 0.0, 0, 3)
    assert g1.degree().max() == 4 and g1.m == 2 * 2 * 10 * 9
    counts = [sc.build_city(10, r, 0.0, 0, 3).m for r in (1.0, 1.5, 2.3, 3.2)]
    assert counts == sorted(counts) and len(set(counts)) == 4


def test_offsets_are_primitive_and_one_per_pair():
    off = sc._primitive_offsets(3.2)
    assert all(np.gcd(abs(dx), dy) == 1 for dy, dx in off) and len(set(off)) == len(off)
    assert (0, 1) in off and (1, 0) in off and (1, 1) in off and (2, 1) in off and (3, 1) in off and (0, 2) not in off and (2, 2) not in off


def test_zero_spread_costs_equal_lengths_and_spread_only_raises_them():
    g0, g1 = sc.build_city(10, 2.3, 0.0, 0, 5), sc.build_city(10, 2.3, 2.0, 0, 5)
    src = np.repeat(np.arange(g0.n), g0.degree())
    length = np.hypot(*(g0.xy[src] - g0.xy[g0.indices]).T)
    assert np.allclose(g0.weight, length) and (g1.weight >= g0.weight - 1e-9).all() and g1.weight.sum() > 1.5 * g0.weight.sum()


def test_blocking_removes_edges_but_keeps_the_network_connected():
    counts = [sc.build_city(15, 2.3, 1.0, b, 4).m for b in (0, 20, 40, 60)]
    assert counts == sorted(counts, reverse=True) and counts[-1] < 0.8 * counts[0]


# --- Labyrinth -------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("side", [6, 20, 40])
@pytest.mark.parametrize("walls", [10, 35, 45])
@pytest.mark.parametrize("seed", range(4))
def test_maze_start_and_goal_are_always_connected(side, walls, seed):
    g, s, t = sc.build_maze(side, walls, seed)
    assert alg.bfs(g, [s], [t]).found == t and (g.weight == 1.0).all()
    assert g.n <= side * side and g.n >= side * 2 - 1


def test_more_walls_leave_fewer_open_cells():
    counts = [np.mean([sc.build_maze(20, w, s)[0].n for s in range(8)]) for w in (10, 35, 45)]
    assert counts[0] > counts[1] > counts[2]


# --- Toronto ---------------------------------------------------------------------------------------------------------------------------------

def test_toronto_data_is_the_expected_openstreetmap_excerpt():
    net = sc.toronto_network()
    g = net.graph
    assert (g.n, g.m) == (5072, 14503) and g.directed and net.weighted and net.unit == "m"
    assert (g.weight > 0).all() and np.abs(g.xy).max() < 1300
    src = np.repeat(np.arange(g.n), g.degree())
    assert sum(g.arc(int(v), int(u)) < 0 for u, v in zip(src, g.indices)) > 0            # es gibt Einbahnstraßen
    assert alg.bfs(g, net.sources, net.targets).found == net.targets[0]


def test_toronto_has_no_self_loops_or_parallel_edges():
    g = sc.toronto_network().graph
    src = np.repeat(np.arange(g.n), g.degree())
    assert (src != g.indices).all() and len(set(zip(src.tolist(), g.indices.tolist()))) == g.m


# --- Lehrbuch-Graphen ------------------------------------------------------------------------------------------------------------------------

def test_contacts_graph_matches_its_definition_and_the_nearer_holder_is_found():
    net = sc.contacts_network()
    g = net.graph
    for name, outs in sc.CONTACT_LINKS.items():
        assert sorted(g.names[v] for v in g.out(g.names.index(name))) == sorted(outs)
    res = alg.bfs(g, net.sources, net.targets)
    assert [g.names[i] for i in res.route(res.found)] == ["ich", "Anna", "David", "Gero"] and res.dist[res.found] == 3
    full = alg.bfs(g, net.sources)
    assert full.dist[g.names.index("Ida")] == 4 and len(net.targets) == 2                # die zweite Person mit Staplerschein ist weiter weg


def test_bus_network_has_a_five_ride_route_from_twin_peaks_to_the_bridge():
    net = sc.bus_network()
    g = net.graph
    res = alg.bfs(g, net.sources, net.targets)
    assert res.dist[res.found] == 5 and not net.weighted
    names = [g.names[i] for i in res.route(res.found)]
    assert names[0] == "Twin Peaks" and names[-1] == "Golden Gate Bridge"
    assert alg.bfs(g, net.sources, net.targets, finish_level=True).dist[res.found] == 5


# --- Kennzahlen ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", C.NETS)
def test_analysis_invariants(key):
    a = ev.analyse(sc.make_network(key))
    m, net = a.metrics, a.net
    assert m["reachable"] and m["hops_bfs"] <= m["hops_optimal"] and m["cost_optimal"] <= m["cost_tie"] + 1e-9 <= m["cost_bfs"] + 2e-9
    assert m["detour"] >= -1e-12 and m["tie_detour"] <= m["detour"] + 1e-9 and m["max_front"] >= 1 and m["discovered"] <= m["n"]
    assert a.routes["bfs"][0] in net.sources and a.routes["bfs"][-1] in net.targets
    if not net.weighted:
        assert m["detour"] == 0 and ev.verdict(a) == "unweighted"
    assert m["scanned"] <= m["m"]


def test_verdict_codes():
    assert ev.verdict(ev.analyse(sc.make_network("toronto"))) == "large"
    assert ev.verdict(ev.analyse(sc.make_network("maze"))) == "unweighted"
    net = sc.make_network("city", 6, 1.0, 0.0, 0, 35, 3)
    assert ev.verdict(ev.analyse(net)) in ("same_route", "small", "large")


def test_pair_stats_fields_and_ordering():
    ps = ev.pair_stats(sc.make_network("city"), 50, 3)
    assert ps["n_pairs"] == 50 and len(ps["detour"]) == len(ps["tie"]) == len(ps["extra_hops"]) == 50
    assert 0 <= ps["share_zero"] <= 1 and ps["median"] <= ps["p90"] <= ps["max"] and ps["tie_median"] <= ps["median"] + 1e-12
    assert (ps["tie"] <= ps["detour"] + 1e-9).all() and (ps["detour"] >= -1e-12).all()


def test_pair_stats_are_deterministic():
    net = sc.make_network("city", 10)
    a, b = ev.pair_stats(net, 30, 1), ev.pair_stats(net, 30, 1)
    assert np.array_equal(a["detour"], b["detour"]) and not np.array_equal(a["detour"], ev.pair_stats(net, 30, 2)["detour"])


def test_unweighted_networks_have_no_detour_over_random_pairs():
    for key in ("maze", "bus", "contacts"):
        ps = ev.pair_stats(sc.make_network(key), 60, 1)
        assert ps["share_zero"] == 1.0 and ps["max"] == 0.0


def test_city_sweep_rows_carry_all_fields():
    rows = ev.city_sweep("reach", (1.0, 2.3), dict(side=8, reach=2.3, spread=1.0, blocked=20), pairs=10, seeds=C.SWEEP_SEEDS[:2])
    assert [r["value"] for r in rows] == [1.0, 2.3] and all({"median", "share_zero", "tie_median", "p90"} <= set(r) for r in rows)
    assert rows[0]["median"] < rows[1]["median"]


def test_scaling_table_counts_every_edge_exactly_once():
    rows = ev.scaling_table(sides=(8, 16))
    for r in rows:
        assert r["scanned"] == r["m"] and r["expanded"] == r["n"] and r["max_front"] >= r["side"]


def test_dfs_table_has_one_row_per_net_with_ratios_at_least_one():
    rows = ev.dfs_table(nets=("maze", "bus"), pairs=15)
    assert [r["net"] for r in rows] == ["maze", "bus"] and all(r["hop_ratio"] >= 1.0 and r["cost_ratio"] >= 1.0 for r in rows)


def test_route_costs_agree_with_the_graph():
    a = ev.analyse(sc.make_network("city"))
    assert a.metrics["cost_bfs"] == pytest.approx(route_cost(a.net.graph, a.routes["bfs"]))
