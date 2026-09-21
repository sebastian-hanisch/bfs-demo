"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier belegt (Toleranz ±0.02 = Rundung auf zwei Stellen plus Luft).
Positive UND negative Aussagen: wo BFS gewinnt (Labyrinth, Bus, Kontaktnetz: Umweg genau 0), steht hier ebenso wie dort, wo es verliert (Toronto, Stadtnetz)."""

from functools import lru_cache

import numpy as np
import pytest

import bf_constants as C
import bf_evaluation as ev
import bf_scenario as sc

TOL = 0.02
CITY = dict(side=C.DEFAULT_SIDE, reach=C.DEFAULT_REACH, spread=C.DEFAULT_SPREAD, blocked=C.DEFAULT_BLOCKED)


def near(value, expected, tol=TOL):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


@lru_cache(maxsize=None)
def analysis(key, *args):
    return ev.analyse(sc.make_network(key, *args))


@lru_cache(maxsize=None)
def sweep(parameter, values):
    return ev.city_sweep(parameter, values, CITY)


# --- Toronto Campus ----------------------------------------------------------------------------------------------------------------------------

def test_toronto_start_to_goal_route_has_fewer_edges_but_is_longer():
    m = analysis("toronto").metrics
    assert (m["hops_bfs"], m["hops_optimal"]) == (29, 57) and m["extra_hops"] == 28
    near(m["detour"], 0.27)                                                        # Preset-Hilfe, Grenzen-Tabelle: 27 %
    assert round(m["cost_optimal"]) == 892 and 1100 < m["cost_bfs"] < 1150
    assert m["tie_detour"] > 0.20                                                  # auch mit bestem Gleichstand bleibt der Umweg: Kostenblindheit, nicht Zufall
    assert (m["n"], m["m"], m["discovered"]) == (5072, 14503, 3043)


def test_toronto_distribution_over_random_pairs():
    ps = ev.pair_stats(sc.toronto_network(), C.PAIRS, C.DEFAULT_SEED)
    assert ps["n_pairs"] == C.PAIRS
    near(ps["median"], 0.07)
    near(ps["p90"], 0.25)
    near(ps["max"], 2.37, 0.1)
    near(ps["share_zero"], 0.035)
    assert ps["mean"] > ps["median"]                                               # sehr ungleich verteilt: der Mittelwert liegt über dem Median


# --- Stadtnetz ---------------------------------------------------------------------------------------------------------------------------------

def test_default_city_corner_route():
    m = analysis("city").metrics
    assert (m["hops_bfs"], m["hops_optimal"]) == (13, 15)
    near(m["detour"], 0.27)                                                        # Preset-Hilfe: 27 %
    near(m["tie_detour"], 0.15)


def test_default_city_distribution_over_random_pairs():
    ps = ev.pair_stats(sc.make_network("city"), C.PAIRS, C.DEFAULT_SEED)
    near(ps["share_zero"], 0.05)
    near(ps["median"], 0.28)
    near(ps["tie_median"], 0.09)
    near(ps["p90"], 0.53)
    near(ps["max"], 1.22, 0.05)


@pytest.mark.parametrize("value,expected", [(6, 0.20), (10, 0.24), (20, 0.23), (30, 0.28), (40, 0.26)])
def test_side_sweep_median_detour(value, expected):
    near(next(r for r in sweep("side", (6, 10, 20, 30, 40)) if r["value"] == value)["median"], expected)


def test_side_sweep_tie_part_grows_from_2_to_13_percent():
    rows = sweep("side", (6, 10, 20, 30, 40))
    near(next(r for r in rows if r["value"] == 20)["tie_median"], 0.08, 0.02)                # README: bei Standardgröße bleiben etwa 8 Prozentpunkte übrig
    near(rows[0]["tie_median"], 0.02, 0.01)
    near(rows[-1]["tie_median"], 0.13, 0.02)


@pytest.mark.parametrize("value,expected", [(1.0, 0.08), (1.5, 0.20), (2.3, 0.23), (3.2, 0.35)])
def test_reach_sweep_median_detour(value, expected):
    near(next(r for r in sweep("reach", (1.0, 1.5, 2.3, 3.2)) if r["value"] == value)["median"], expected)


@pytest.mark.parametrize("value,expected", [(0.0, 0.07), (0.5, 0.14), (1.0, 0.23), (2.0, 0.39), (3.0, 0.53)])
def test_spread_sweep_median_detour(value, expected):
    near(next(r for r in sweep("spread", (0.0, 0.5, 1.0, 2.0, 3.0)) if r["value"] == value)["median"], expected)


@pytest.mark.parametrize("value,expected", [(0, 0.28), (20, 0.23), (40, 0.22), (60, 0.15)])
def test_blocked_sweep_median_detour(value, expected):
    near(next(r for r in sweep("blocked", (0, 20, 40, 60)) if r["value"] == value)["median"], expected)


def test_spread_zero_still_has_a_detour_because_lengths_differ():
    assert sweep("spread", (0.0, 0.5, 1.0, 2.0, 3.0))[0]["median"] > 0.03


def test_detour_grows_with_reach_and_with_spread():
    r = [x["median"] for x in sweep("reach", (1.0, 1.5, 2.3, 3.2))]
    s = [x["median"] for x in sweep("spread", (0.0, 0.5, 1.0, 2.0, 3.0))]
    assert r == sorted(r) and s == sorted(s)


# --- Wo BFS exakt richtig ist ----------------------------------------------------------------------------------------------------------------

def test_maze_route_has_42_steps_and_zero_detour_for_every_wall_share():
    m = analysis("maze").metrics
    assert m["hops_bfs"] == m["hops_optimal"] == 42 and m["detour"] == 0.0
    for walls in (10, 35, 45):
        for seed in C.SWEEP_SEEDS:
            ps = ev.pair_stats(sc.make_network("maze", 20, 2.3, 1.0, 20, walls, seed), 20, seed)
            assert ps["share_zero"] == 1.0 and ps["max"] == 0.0


def test_bus_and_contacts_examples():
    assert analysis("bus").metrics["hops_bfs"] == 5
    m = analysis("contacts").metrics
    assert m["hops_bfs"] == 3 and m["expanded"] == 5 and m["discovered"] == 8 and m["n"] == 10


# --- Aufwand und Gegenprobe ----------------------------------------------------------------------------------------------------------------

def test_scaling_every_edge_once_and_front_grows_like_the_square_root():
    rows = ev.scaling_table()
    assert all(r["scanned"] == r["m"] and r["expanded"] == r["n"] for r in rows)
    assert (rows[-1]["n"], rows[-1]["m"]) == (6400, 50244)                          # Grenzen-Tabelle: 50 244 geprüfte Kanten
    slope = np.polyfit(np.log([r["n"] for r in rows]), np.log([r["max_front"] for r in rows]), 1)[0]
    assert 0.4 < slope < 0.6                                                       # Front wächst wie die Wurzel der Knotenzahl
    assert rows[-1]["m"] / rows[0]["m"] > 5 * rows[-1]["max_front"] / rows[0]["max_front"]       # die Kanten wachsen um das 73-Fache, die Front nur um das 8-Fache


def test_dfs_route_has_many_times_the_edges_of_bfs():
    rows = {r["net"]: r["hop_ratio"] for r in ev.dfs_table()}
    near(rows["city"], 42, 3)                                                       # Bildunterschrift: 42- und 30-Fache, 4-Fache, 2-Fache
    near(rows["toronto"], 30, 3)
    near(rows["maze"], 4, 0.5)
    near(rows["bus"], 2, 0.5)
    assert all(v > 1.5 for v in rows.values())
