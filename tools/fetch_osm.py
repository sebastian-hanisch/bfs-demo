"""Einmal-Skript: holt den Straßenausschnitt um den St. George Campus (Toronto) aus OpenStreetMap und legt ihn als data/toronto_campus.json ab.

Läuft NICHT in der App (die liest nur die JSON-Datei) und braucht osmnx, das nicht in requirements.txt steht:
    pip install osmnx && python tools/fetch_osm.py

Ausschnitt und Start/Ziel wie im Beispiel des Buchs "Optimization Algorithms" (Kap. 3): Mittelpunkt des Campus, 1000 m Umkreis, alle Wegearten;
Start = Reiterstandbild König Eduards VII. am Queen's Park, Ziel = Bahen Centre for Information Technology.
Daten: (c) OpenStreetMap-Mitwirkende, ODbL 1.0 (https://www.openstreetmap.org/copyright)."""

import json
import math
from pathlib import Path

import osmnx as ox

CENTER = (43.662643, -79.395689)
STATUE = (43.664527, -79.392442)
BAHEN = (43.659659, -79.397669)
DIST = 1000
OUT = Path(__file__).resolve().parent.parent / "data" / "toronto_campus.json"


def main():
    G = ox.graph_from_point(CENTER, dist=DIST, network_type="all", simplify=True)
    ids = list(G.nodes)
    index = {osm: i for i, osm in enumerate(ids)}
    lat0, lon0 = CENTER
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    nodes = [[round((G.nodes[o]["x"] - lon0) * kx, 1), round((G.nodes[o]["y"] - lat0) * ky, 1)] for o in ids]
    arcs = {}
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        key = (index[u], index[v])
        length = round(float(data["length"]), 1)
        if key not in arcs or length < arcs[key]:
            arcs[key] = length
    src = ox.distance.nearest_nodes(G, STATUE[1], STATUE[0])
    dst = ox.distance.nearest_nodes(G, BAHEN[1], BAHEN[0])
    payload = {
        "source": "OpenStreetMap, ODbL 1.0", "center": CENTER, "dist_m": DIST, "network_type": "all",
        "start": {"node": index[src], "label": "Reiterstandbild König Eduards VII."}, "goal": {"node": index[dst], "label": "Bahen Centre"},
        "nodes_xy_m": nodes, "arcs": [[u, v, w] for (u, v), w in sorted(arcs.items())],
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(len(nodes), "Knoten,", len(arcs), "Kanten ->", OUT, OUT.stat().st_size // 1024, "kB")


if __name__ == "__main__":
    main()
