"""Konstanten, Grenzen der Regler und Presets. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

SPACING = 100.0                    # Meter zwischen benachbarten Kreuzungen im erzeugten Stadtnetz
JITTER = 0.25                      # Lageabweichung der Kreuzungen in Blocklängen

NETS = ("city", "maze", "toronto", "bus", "contacts")
NET_LABELS = {
    "city": "🏙️ Stadtnetz (erzeugt)",
    "maze": "🧱 Labyrinth (erzeugt)",
    "toronto": "🍁 Toronto Campus (OpenStreetMap)",
    "bus": "🚌 Bus-Umstiege (San Francisco)",
    "contacts": "🤝 Kontaktnetz (Staplerschein)",
}
GRID_NETS = ("city", "maze")       # Netze mit Größenregler

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 6, 40, 20
REACH_MIN, REACH_MAX, DEFAULT_REACH = 1.0, 3.2, 2.3
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 0.0, 3.0, 1.0
BLOCKED_MIN, BLOCKED_MAX, DEFAULT_BLOCKED = 0, 60, 20          # Prozent der Straßen
WALLS_MIN, WALLS_MAX, DEFAULT_WALLS = 10, 45, 35               # Prozent der Zellen
DEFAULT_SEED = 7
DEFAULT_NET = "city"

SWEEP_SEEDS = tuple(range(100000, 100005))
PAIRS = 200                        # zufällige Start-Ziel-Paare je Netz für die Verteilung des Umwegs

COLORS = {"bfs": "#1f77b4", "optimal": "#d62728", "dfs": "#7f7f7f", "tie": "#2ca02c", "start": "#111111", "goal": "#ff7f0e"}

# Jedes Preset setzt alle Regler; bei den festen Netzen (Bus, Kontakte, Toronto) sind Größe, Reichweite, Streuung, Sperrungen und Wände ohne Wirkung und bleiben unverändert (die Regler sind dort ausgeblendet).
_BASE = dict(side=DEFAULT_SIDE, reach=DEFAULT_REACH, spread=DEFAULT_SPREAD, blocked=DEFAULT_BLOCKED, walls=DEFAULT_WALLS, seed=DEFAULT_SEED)
PRESETS = {
    "🚌 Bus-Umstiege": {**_BASE, "net": "bus"},
    "🤝 Kontaktnetz": {**_BASE, "net": "contacts"},
    "🧱 Labyrinth": {**_BASE, "net": "maze"},
    "🍁 Toronto Campus": {**_BASE, "net": "toronto"},
    "🏙️ Stadtnetz": {**_BASE, "net": "city"},
}
PRESET_HELP = {
    "🚌 Bus-Umstiege": "Kleines Liniennetz in der Art des Beispiels aus Grokking Algorithms: von Twin Peaks zur Golden Gate Bridge mit so wenigen Fahrten wie möglich. Alle Fahrten zählen gleich - BFS ist hier genau richtig.",
    "🤝 Kontaktnetz": "Ein eigenes Kontaktnetz in der Art des Freundesnetzes aus Grokking Algorithms: wer kennt jemanden mit Staplerschein, und wie viele Bekanntschaften sind es bis zur nächsten Person? Zwei Personen kommen infrage, die Suche endet bei der näheren.",
    "🧱 Labyrinth": "Erzeugtes Labyrinth (35 % Wände, 20 × 20): jeder Schritt kostet dasselbe, also ist die BFS-Route die kürzeste.",
    "🍁 Toronto Campus": "Echte OpenStreetMap-Daten aus dem Beispiel in Optimization Algorithms: vom Reiterstandbild am Queen's Park zum Bahen Centre. Die Route mit den wenigsten Kanten ist 27 % länger als die kürzeste.",
    "🏙️ Stadtnetz": "Erzeugtes Stadtnetz (20 × 20 Kreuzungen, Straßen bis 2.3 Blocklängen, Kosten mit Streuung 1.0): die Route mit den wenigsten Kanten ist hier 27 % länger als die kürzeste.",
}
