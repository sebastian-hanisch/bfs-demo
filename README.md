# Breitensuche – die Route mit den wenigsten Kanten – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-bfs-demo.streamlit.app/)**

Erstes Stück (**Wurzel**) der **Kürzeste-Wege-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – die **Breitensuche (BFS)** – an einem wachsenden Beispiel.
Vom Start aus werden erst alle Nachbarn besucht, dann deren Nachbarn – **Schicht für Schicht**; das Ziel ist gefunden, sobald es zum ersten Mal entdeckt wird, und seine Route hat garantiert die **wenigsten Kanten**.
Der Haken steckt im Wort: BFS **zählt Kanten, nicht Kosten**.

**Einordnung in die Reihe (die Kanten des Graphen):** die Wurzel ist bewusst die einfachste Suche im Netz. Ihre Schwäche – Kosten werden ignoriert – ist der Ansatzpunkt des nächsten Stücks (**Dijkstra**: Kosten korrekt, aber blind in alle Richtungen);
die Suche in alle Richtungen und die Anfrage von vorn sind die Ansatzpunkte von Bidirektionaler Suche und Contraction Hierarchies. Bisher gebaut: nur die Wurzel.
```
bfs-demo (Wurzel: Kanten zählen, nicht Kosten)
  └─ Dijkstra (Kosten korrekt, blind in alle Richtungen)                       [nicht gebaut]
       ├─ Bidirektionale Suche → Contraction Hierarchies                       [nicht gebaut]
       ├─ Bellman-Ford + Floyd-Warshall → Johnson (Konvergenz: Umgewichtung)   [nicht gebaut]
       └─ Mehrkriterien-Routing (Zeit gegen CO₂, Pareto)                       [nicht gebaut]
A* steht einmal in der Baumsuche-Linie und wird von hier aus nur verlinkt.
```

## Beispiele aus den Lehrbüchern

| Netz in der Demo | Quelle |
|---|---|
| **Bus-Umstiege** (Twin Peaks → Golden Gate Bridge, so wenige Fahrten wie möglich) | in der Art des Beispiels aus *Grokking Algorithms* (A. Bhargava), Kap. 6 – das Liniennetz ist selbst gezeichnet, nicht das Bild aus dem Buch |
| **Kontaktnetz** (wer kennt jemanden mit Staplerschein? zehn Personen, zwei mit Schein) | in der Art des Freundesnetzes aus *Grokking Algorithms*, Kap. 6 – eigener Graph mit eigenen Namen, nicht der aus dem Buch |
| **Labyrinth** (Raster mit Wänden, kürzester Weg) | in der Art des Labyrinths aus *Grokking AI Algorithms* (R. Hurbans), Kap. 2 – erzeugt, mit Größen- und Wandregler |
| **Toronto Campus** (Reiterstandbild am Queen's Park → Bahen Centre, St. George Campus, 1 km Umkreis) | *Optimization Algorithms* (A. Khamis), Kap. 3 – **echte OpenStreetMap-Daten**, einmalig geholt (`tools/fetch_osm.py`), fest in `data/toronto_campus.json` |
| **Stadtnetz** (erzeugt, mit Reichweite, Streuung der Kosten, Sperrungen) | eigene Erzeugung für die Größenregler |

Aus den Büchern stammt nur die Idee der Beispiele (das Szenario in Toronto mit Start und Ziel); Text, Abbildungen, Code und Graphen der Bücher sind nicht übernommen, die kleinen Netze sind eigene Zeichnungen.

**Daten und Lizenz:** Kartendaten © [OpenStreetMap-Mitwirkende](https://www.openstreetmap.org/copyright), Open Database License (ODbL) 1.0. `data/toronto_campus.json` ist ein Auszug daraus und steht deshalb ebenfalls unter der ODbL – siehe [data/LICENSE-ODbL.md](data/LICENSE-ODbL.md).

## Ergebnis (Zahlen aus den Tests)

| Frage | Ergebnis |
|---|---|
| Wo BFS exakt richtig ist | ✅ **Labyrinth, Bus-Umstiege, Kontaktnetz**: jede Kante kostet dasselbe, der Umweg ist über alle gezogenen Paare **genau 0** (Labyrinth bei 10 / 35 / 45 % Wänden, 5 feste Netze). Kontaktnetz: 3 Bekanntschaften bis zur näheren der beiden Personen mit Staplerschein, 5 Knoten abgearbeitet, 8 von 10 entdeckt |
| Toronto Campus, Standbild → Bahen Centre | ❌ BFS-Route mit **29 Kanten**, die kürzeste hat **57 Kanten** und 892 m; die BFS-Route ist **27 % länger**. Auch die beste unter den kantenkürzesten Routen ist noch 25 % länger: das liegt nicht an der Reihenfolge der Nachbarn, sondern an der Kostenblindheit. Die Kantenzahl hängt bei OSM-Daten davon ab, wie dicht digitalisiert wurde |
| Toronto, 200 zufällige Paare | ❌ Median-Umweg **7 %**, 90 %-Quantil **25 %**, schlimmster Fall **237 %**; nur 3,5 % der Paare ohne Umweg – der Mittelwert liegt über dem Median, der Umweg ist sehr ungleich verteilt |
| Stadtnetz (Standard: 20 × 20, Reichweite 2.3, Streuung 1.0) | ❌ Ecke zu Ecke 13 statt 15 Kanten, aber **27 % länger**; Median über 5 feste Netze **23 %**. Davon wären mit bestem Gleichstand noch etwa 8 Prozentpunkte übrig – der Rest ist zufällige Wahl unter gleich kurzen Routen |
| Reichweite der Straßen | ❌ Median-Umweg 8 % / 20 % / 23 % / 35 % bei 1.0 / 1.5 / 2.3 / 3.2 – BFS liebt lange Verbindungen, weil sie die Kantenzahl senken |
| Streuung der Kosten | ❌ 7 % / 14 % / 23 % / 39 % / 53 % bei 0 / 0.5 / 1 / 2 / 3; auch bei 0 bleibt ein Umweg, weil sich die Länge der Straßen unterscheidet |
| Sperrungen | ⚠️ 28 % / 23 % / 22 % / 15 % bei 0 / 20 / 40 / 60 % gesperrten Straßen (weniger Alternativen, weniger Spielraum – Deutung, nicht bewiesen) |
| Größe | ⚠️ 20 % / 24 % / 23 % / 28 % / 26 % bei 6 / 10 / 20 / 30 / 40 Kreuzungen je Seite; der Teil, der auch bei bestem Gleichstand bleibt, wächst von 2 % auf 13 % |
| Aufwand | ✅ jede Kante wird **genau einmal** geprüft (6 400 Kreuzungen: 50 244 Kanten), die **Front** – der Speicherbedarf – wächst nur wie die **Wurzel** der Knotenzahl (Steigung zwischen 0.4 und 0.6 im Log-Log) |
| Tiefensuche als Gegenprobe | ✅ die Schichten-Reihenfolge trägt die Garantie: Routen mit dem **42-Fachen** (Stadtnetz), **30-Fachen** (Toronto), 4-Fachen (Labyrinth), 2-Fachen (Bus) der Kanten von BFS (Median über 60 Paare) |
| Blind in alle Richtungen | ❌ Toronto: 3 043 von 5 072 Knoten sind entdeckt, bevor das Ziel (892 m entfernt) gefunden ist – der Ansatzpunkt von Bidirektionaler Suche und A* |

Die kostenoptimale Route zum Messen kommt aus einer kleinen Dijkstra-Referenz (`dijkstra_reference`); das Verfahren selbst ist das nächste Stück der Linie.

## Was die Demo zeigt

1. **Breitensuche in Aktion** (Schicht-Regler + Abspielen): die Welle im Netz (Farbe = Kanten vom Start), daneben die Front je Schicht; zuschaltbare Vergleichsrouten: kostenoptimal, beste unter den kantenkürzesten, Tiefensuche.
2. **Zählen Kanten dasselbe wie Kosten?** – Kennzahlen des gezeigten Paars (Kanten, Länge, Umweg, entdeckte Knoten) mit Urteil, dazu die **Verteilung des Umwegs über 200 zufällige Paare** (Anteil ohne Umweg, Median, 90 %-Quantil, Maximum, Histogramm).
3. **Vergleich** (Expander) der Verfahren für dasselbe Paar; **Experimente auf Knopfdruck**: ein Regler des Stadtnetzes über fünf feste Netze durchfahren, Aufwand und Front gegen die Netzgröße, Tiefensuche gegen BFS.
4. **Wo die Annahmen enden** (Tabelle mit den Ansatzpunkten der nächsten Stücke) und **Mathematische Formulierung** (Schichten, Laufzeit O(|V|+|E|), Korrektheit, Umweg, bester Gleichstand).

Bedienung: Beispielnetz per Schnellstart-Knopf laden oder in der Seitenleiste Netz und Regler wählen; die Adresszeile spiegelt die Konfiguration (Permalink). Regler, die zum gewählten Netz nicht gehören, sind ausgeblendet.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `bf_graph.py` | gerichteter Graph in CSR-Form (Parallelkanten: die billigste bleibt, Selbstschleifen fallen weg) |
| `bf_algorithm.py` | BFS (Mehrfach-Start und -Ziel, Abbruch beim ersten Ziel), kostenoptimale Referenz, beste Route unter den kantenkürzesten, Tiefensuche |
| `bf_scenario.py` | Netze: Stadtnetz, Labyrinth, Toronto, Bus, Kontaktnetz |
| `bf_evaluation.py` | Kennzahlen, Verteilung über Paare, Sweeps, Urteil |
| `bf_visualization.py`, `bf_presets.py`, `bf_constants.py` | Abbildungen, Presets und Permalink, Konstanten |
| `tools/fetch_osm.py` | Einmal-Skript: holt den Toronto-Ausschnitt (braucht `osmnx`, nicht in `requirements.txt`) |
| `data/toronto_campus.json` | der Ausschnitt (5 072 Knoten, 14 503 gerichtete Kanten) |

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/`. Jede Zahl in Hilfetexten, Presets und Tabellen ist in `tests/test_claims.py` belegt; die Kreuzprobe von BFS und Referenz läuft gegen networkx.
