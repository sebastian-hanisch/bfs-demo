"""Läufe, Kennzahlen, Verteilung über zufällige Start-Ziel-Paare, Sweeps und das Urteil für die App."""

import time
from dataclasses import dataclass

import numpy as np

import bf_algorithm as alg
import bf_constants as C
from bf_graph import route_cost
from bf_scenario import make_network


@dataclass(frozen=True)
class Analysis:
    net: object
    bfs: alg.Bfs
    routes: dict                   # "bfs", "optimal", "tie", "dfs" -> Knotenfolge
    metrics: dict
    seconds: dict


def _cost(net, route):
    return route_cost(net.graph, route) if route else float("nan")


def analyse(net):
    """BFS (mit Abbruch beim Ziel) gegen kostenoptimale Referenz, beste Gleichstands-Route und Tiefensuche für den Start und das Ziel des Netzes."""
    g, S, T = net.graph, net.sources, net.targets
    t0 = time.perf_counter()
    b = alg.bfs(g, S, T)
    t_bfs = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = alg.dijkstra_reference(g, S, T)
    t_ref = time.perf_counter() - t0
    reachable = b.found >= 0
    routes = {"bfs": b.route(b.found) if reachable else [], "optimal": ref.route(ref.found) if reachable else [],
              "tie": alg.cheapest_among_fewest_edges(g, S, b.found) if reachable else [], "dfs": []}
    dfs_visited = 0
    t0 = time.perf_counter()
    if reachable:
        routes["dfs"], dfs_visited = alg.dfs_route(g, S[0], T) if len(S) == 1 else ([], 0)
    t_dfs = time.perf_counter() - t0
    hops = {k: len(r) - 1 if r else 0 for k, r in routes.items()}
    cost = {k: _cost(net, r) for k, r in routes.items()}
    detour = cost["bfs"] / cost["optimal"] - 1.0 if reachable and cost["optimal"] > 0 else float("nan")
    tie_detour = cost["tie"] / cost["optimal"] - 1.0 if reachable and cost["optimal"] > 0 else float("nan")
    dfs_detour = cost["dfs"] / cost["optimal"] - 1.0 if routes["dfs"] and cost["optimal"] > 0 else float("nan")
    metrics = {"reachable": reachable, "n": g.n, "m": g.m, "hops_bfs": hops["bfs"], "hops_optimal": hops["optimal"], "extra_hops": hops["optimal"] - hops["bfs"],
               "cost_bfs": cost["bfs"], "cost_optimal": cost["optimal"], "cost_tie": cost["tie"], "cost_dfs": cost["dfs"], "hops_dfs": hops["dfs"],
               "detour": detour, "tie_detour": tie_detour, "dfs_detour": dfs_detour, "expanded": len(b.expanded), "discovered": len(b.discovered), "max_front": b.max_front,
               "scanned": b.scanned, "dfs_visited": dfs_visited, "settled": ref.settled}
    return Analysis(net, b, routes, metrics, {"bfs": t_bfs, "optimal": t_ref, "dfs": t_dfs})



# --- Verteilung über zufällige Start-Ziel-Paare ----------------------------------------------------------------------------------------------

def pair_stats(net, pairs=C.PAIRS, seed=0):
    """Umweg der BFS-Route (Kosten gegen die kostenoptimale Route) über zufällige erreichbare Start-Ziel-Paare, dazu der Umweg mit dem besten Gleichstand und die Mehrkanten der Referenz.
    Rückgabe: dict mit Feldern (Arrays) und Kennzahlen (Anteil ohne Umweg, Median, 90 %-Quantil, Maximum, Mittel)."""
    rng = np.random.default_rng([int(seed), 303])
    g = net.graph
    detour, tie, extra = [], [], []
    tries = 0
    while len(detour) < pairs and tries < pairs * 20:
        tries += 1
        s, t = (int(x) for x in rng.integers(0, g.n, 2))
        if s == t:
            continue
        b = alg.bfs(g, (s,), (t,))
        if b.found < 0:
            continue
        ref = alg.dijkstra_reference(g, (s,), (t,))
        r_ref = ref.route(t)
        c_ref = route_cost(g, r_ref)
        r_b = b.route(t)
        detour.append(route_cost(g, r_b) / c_ref - 1.0)
        tie.append(route_cost(g, alg.cheapest_among_fewest_edges(g, (s,), t)) / c_ref - 1.0)
        extra.append(len(r_ref) - len(r_b))
    d, tt, e = np.array(detour), np.array(tie), np.array(extra)
    if not len(d):
        return {"detour": d, "tie": tt, "extra_hops": e, "n_pairs": 0, "share_zero": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan"), "mean": float("nan"),
                "tie_median": float("nan"), "tie_share_zero": float("nan")}
    return {"detour": d, "tie": tt, "extra_hops": e, "n_pairs": len(d), "share_zero": float((d < 1e-9).mean()), "median": float(np.median(d)), "p90": float(np.quantile(d, 0.9)),
            "max": float(d.max()), "mean": float(d.mean()), "tie_median": float(np.median(tt)), "tie_share_zero": float((tt < 1e-9).mean())}


# --- Sweeps und Experimente ------------------------------------------------------------------------------------------------------------------

def city_sweep(parameter, values, base, pairs=40, seeds=C.SWEEP_SEEDS):
    """Ein Regler des Stadtnetzes durchgefahren, alle anderen wie in `base` (dict mit side, reach, spread, blocked): Median-Umweg der BFS-Route, Anteil der Paare ohne Umweg und
    Median-Umweg mit dem besten Gleichstand, jeweils Mittel über die fünf festen Sweep-Datensätze (getrennt vom Seed der Seitenleiste)."""
    rows = []
    for v in values:
        kw = {**base, parameter: v}
        st = [pair_stats(make_network("city", kw["side"], kw["reach"], kw["spread"], kw["blocked"], C.DEFAULT_WALLS, sd), pairs, sd) for sd in seeds]
        rows.append({"value": v, "median": float(np.mean([x["median"] for x in st])), "share_zero": float(np.mean([x["share_zero"] for x in st])),
                     "tie_median": float(np.mean([x["tie_median"] for x in st])), "p90": float(np.mean([x["p90"] for x in st]))})
    return rows


def scaling_table(sides=(10, 20, 30, 40, 60, 80), reach=1.5, seed=C.SWEEP_SEEDS[0]):
    """BFS ohne Ziel (gesamtes Netz) gegen die Größe: jede Kante wird genau einmal geprüft, die Front wächst langsamer als das Netz."""
    rows = []
    for side in sides:
        g = make_network("city", side, reach, C.DEFAULT_SPREAD, 0, C.DEFAULT_WALLS, seed).graph
        t0 = time.perf_counter()
        res = alg.bfs(g, (0,))
        sec = time.perf_counter() - t0
        rows.append({"side": side, "n": g.n, "m": g.m, "expanded": len(res.expanded), "scanned": res.scanned, "max_front": res.max_front, "levels": len(res.layer_sizes()), "seconds": sec})
    return rows


def dfs_table(nets=("city", "toronto", "maze", "bus"), pairs=60, seed=0):
    """Vergleich BFS gegen Tiefensuche: Kanten der gefundenen Route und Kosten über zufällige Paare (Mittel des Verhältnisses DFS/BFS)."""
    rows = []
    for key in nets:
        net = make_network(key)
        g = net.graph
        rng = np.random.default_rng([seed, 404])
        hop_ratio, cost_ratio = [], []
        tries = 0
        while len(hop_ratio) < pairs and tries < pairs * 20:
            tries += 1
            s, t = (int(x) for x in rng.integers(0, g.n, 2))
            if s == t:
                continue
            b = alg.bfs(g, (s,), (t,))
            if b.found < 0:
                continue
            r_dfs, _ = alg.dfs_route(g, s, (t,))
            r_b = b.route(t)
            hop_ratio.append((len(r_dfs) - 1) / (len(r_b) - 1))
            cost_ratio.append(route_cost(g, r_dfs) / route_cost(g, r_b))
        rows.append({"net": key, "hop_ratio": float(np.median(hop_ratio)), "cost_ratio": float(np.median(cost_ratio)), "n_pairs": len(hop_ratio)})
    return rows


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

def verdict(a):
    """Code für die App: unreachable / unweighted / same_route / small / large."""
    m = a.metrics
    if not m["reachable"]:
        return "unreachable"
    if not a.net.weighted:
        return "unweighted"
    if m["detour"] < 1e-9:
        return "same_route"
    return "small" if m["detour"] < 0.05 else "large"
