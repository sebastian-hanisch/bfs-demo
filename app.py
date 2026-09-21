"""Breitensuche (BFS) - die Route mit den wenigsten Kanten - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - die Breitensuche - und lässt stattdessen das Beispiel wachsen.
Wurzel der Kürzeste-Wege-Linie der "Konzepte"-Reihe: die einfachste Suche im Netz, an deren Schwäche (sie zählt Kanten, nicht Kosten) die späteren Stücke ansetzen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import bf_constants as C
import bf_evaluation as ev
from bf_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from bf_scenario import make_network
from bf_visualization import ROUTE_NAMES, build_detour_hist, build_dfs, build_layers, build_network, build_scaling, build_sweep

st.set_page_config(page_title="Breitensuche – Sebastian Hanisch", layout="wide")

FIXED_NETS = ("toronto", "bus", "contacts")


def _pct(x):
    return "–" if x is None or np.isnan(x) else f"{x:.0%}"


def _cost(net, x):
    if net.unit == "m":
        return f"{x:,.0f} m".replace(",", ".")
    return f"{x:.0f} {net.unit}"


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(params):
    return ev.analyse(make_network(*params))


@st.cache_data(show_spinner=False)
def _pair_stats(params, pairs):
    return ev.pair_stats(make_network(*params), pairs, params[-1])


@st.cache_data(show_spinner=False)
def _sweep(parameter, values, base):
    return ev.city_sweep(parameter, values, dict(base))


@st.cache_data(show_spinner=False)
def _scaling():
    return ev.scaling_table()


@st.cache_data(show_spinner=False)
def _dfs():
    return ev.dfs_table()


st.title("🧭 Breitensuche – die Route mit den wenigsten Kanten")
st.markdown(
    """
Wie kommt man in einem Netz von A nach B? Die **Breitensuche (BFS)** ist die einfachste Antwort: vom Start aus erst alle Nachbarn besuchen, dann deren Nachbarn, dann wieder deren Nachbarn - **Schicht für Schicht**, wie eine Welle.
Das Ziel wird in der Schicht entdeckt, die seiner **Kantenzahl** vom Start entspricht, und die Route dorthin hat garantiert die wenigsten Kanten. Der Haken steckt im Wort: BFS **zählt Kanten, nicht Kosten**.
Sobald Kanten verschieden lang, teuer oder langsam sind, ist die Route mit den wenigsten Kanten nicht mehr die kürzeste. Diese Demo zeigt beides: wo BFS genau richtig ist (Labyrinth, Bus-Umstiege, Kontaktnetz)
und wie weit es auf einem echten Straßennetz danebenliegt (Toronto Campus).
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - erstes Stück der Kürzeste-Wege-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Das nächste Stück, **Dijkstra**, behebt genau die Schwäche dieser Wurzel: Kosten korrekt, aber blind in alle Richtungen. "
    "Die kleinen Netze sind eigene Graphen in der Art der Beispiele aus *Grokking Algorithms* (A. Bhargava, Kap. 6) und *Grokking AI Algorithms* (R. Hurbans, Kap. 2), das Toronto-Szenario folgt dem Beispiel in *Optimization Algorithms* (A. Khamis, Kap. 3)."
)

with st.expander("So funktioniert die Breitensuche", expanded=True):
    st.markdown(
        """
1. **Schicht 0** ist der Start. Er kommt in eine Warteschlange und bekommt die Kantenzahl 0.
2. **Abarbeiten:** der vorderste Knoten wird aus der Warteschlange genommen, jeder seiner noch unbekannten Nachbarn bekommt seine Kantenzahl + 1, merkt sich den Vorgänger und geht hinten in die Warteschlange.
   Weil die Warteschlange first-in-first-out arbeitet, sind erst alle Knoten der Schicht *k* dran, bevor einer der Schicht *k + 1* drankommt.
3. **Das Ziel** ist gefunden, sobald es zum ersten Mal entdeckt wird: seine Kantenzahl ist endgültig, die Route ergibt sich rückwärts über die Vorgänger. Es gibt keine Route mit weniger Kanten.
4. **Was BFS nicht weiß:** wie lang, teuer oder langsam eine Kante ist. Unter mehreren Routen mit gleich vielen Kanten entscheidet nur die Reihenfolge der Nachbarn. Und die Welle läuft in alle Richtungen gleich weit - auch vom Ziel weg.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", C.NETS, key="net_select", format_func=lambda k: C.NET_LABELS[k],
        help="Erzeugt (Stadtnetz, Labyrinth) oder fest: Toronto Campus sind echte OpenStreetMap-Daten (5 072 Kreuzungen und Wegpunkte, 14 503 gerichtete Kanten), Bus-Umstiege und Kontaktnetz sind kleine selbst gezeichnete Graphen.",
    )
    if net_key in C.GRID_NETS:
        side = st.slider(
            "Kreuzungen je Seite" if net_key == "city" else "Zellen je Seite", *bounds("side_slider"), key="side_slider",
            help="Größe des Netzes (Seite × Seite). Beim Stadtnetz (Reichweite 2.3, Streuung 1.0) ist der Median-Umweg der BFS-Route bei 6 / 10 / 20 / 30 / 40 Kreuzungen je Seite 20 % / 24 % / 23 % / 28 % / 26 % - "
                 "mit der Größe wächst er kaum, aber der Teil, der auch bei bester Wahl unter gleich kurzen Routen bleibt, wächst von 2 % auf 13 %.",
        )
        st.session_state[KEPT["side_slider"]] = side
    else:
        side = int(st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))
    if net_key == "city":
        reach = st.slider(
            "Reichweite der Straßen [Blocklängen]", *bounds("reach_slider"), key="reach_slider", step=0.1,
            help="Wie weit eine Straße zwischen zwei Kreuzungen reichen darf (1 = nur Nachbarn im Raster, größer = auch längere Verbindungen). Median-Umweg der BFS-Route bei 1.0 / 1.5 / 2.3 / 3.2: 8 % / 20 % / 23 % / 35 % - "
                 "BFS liebt die langen Verbindungen, weil sie die Kantenzahl senken.",
        )
        st.session_state[KEPT["reach_slider"]] = reach
        spread = st.slider(
            "Streuung der Kosten", *bounds("spread_slider"), key="spread_slider", step=0.25,
            help="Kosten einer Straße = Länge × (1 + Streuung × Zufall): Ampeln, Steigung, Belag. Median-Umweg bei 0 / 0.5 / 1 / 2 / 3: 7 % / 14 % / 23 % / 39 % / 53 %. "
                 "Bei 0 sind alle Kosten gleich der Länge - auch dann bleibt ein Umweg, weil sich die Länge der Straßen unterscheidet, nicht nur ihre Zahl.",
        )
        st.session_state[KEPT["spread_slider"]] = spread
        blocked = st.slider(
            "Gesperrte Straßen [%]", *bounds("blocked_slider"), key="blocked_slider",
            help="Anteil der Straßen, die gesperrt sind (das Netz bleibt zusammenhängend). Median-Umweg bei 0 / 20 / 40 / 60 %: 28 % / 23 % / 22 % / 15 % - vermutlich, weil mit weniger Alternativen beide Routen weniger Spielraum haben.",
        )
        st.session_state[KEPT["blocked_slider"]] = blocked
    else:
        reach = float(st.session_state.get(KEPT["reach_slider"], C.DEFAULT_REACH))
        spread = float(st.session_state.get(KEPT["spread_slider"], C.DEFAULT_SPREAD))
        blocked = int(st.session_state.get(KEPT["blocked_slider"], C.DEFAULT_BLOCKED))
    if net_key == "maze":
        walls = st.slider(
            "Wände [%]", *bounds("walls_slider"), key="walls_slider",
            help="Anteil der Zellen, die Wand sind (bei zu vielen Wänden werden einzelne wieder geöffnet, damit Start und Ziel verbunden bleiben). Der Umweg der BFS-Route ist bei 10 / 35 / 45 % immer 0 - jeder Schritt kostet dasselbe.",
        )
        st.session_state[KEPT["walls_slider"]] = walls
    else:
        walls = int(st.session_state.get(KEPT["walls_slider"], C.DEFAULT_WALLS))
    if net_key in C.GRID_NETS:
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für das Netz.")
    else:
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen. Größe, Reichweite und Streuung gehören zum Stadtnetz.")

sync_query_params({"net_select": net_key, "side_slider": int(side), "reach_slider": round(float(reach), 1), "spread_slider": round(float(spread), 2), "blocked_slider": int(blocked),
                   "walls_slider": int(walls), "seed_input": int(seed)})

# fest gewählte Netze ignorieren die Regler des Rasters: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
params = (net_key, int(side), round(float(reach), 1), round(float(spread), 2), int(blocked), int(walls), int(seed))
if net_key in FIXED_NETS:
    params = (net_key, C.DEFAULT_SIDE, C.DEFAULT_REACH, C.DEFAULT_SPREAD, C.DEFAULT_BLOCKED, C.DEFAULT_WALLS, C.DEFAULT_SEED)
elif net_key == "city":
    params = params[:5] + (C.DEFAULT_WALLS, params[6])
else:
    params = (net_key, params[1], C.DEFAULT_REACH, C.DEFAULT_SPREAD, C.DEFAULT_BLOCKED, params[5], params[6])
with st.spinner("Rechne..."):
    a = _analysis(params)
net, res, m = a.net, a.bfs, a.metrics
g = net.graph

# --- Breitensuche in Aktion ------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Breitensuche in Aktion")
last_level = int(res.dist[res.found]) if res.found >= 0 else int(res.dist.max())
if st.session_state.get("bf_level_owner") != params:
    st.session_state["bf_level"] = last_level
    st.session_state["bf_level_owner"] = params
step_col, play_col = st.columns([5, 2])
with step_col:
    if last_level > 0:
        level = st.slider("Schicht (Kanten vom Start)", 0, last_level, key="bf_level", help="Wie viele Schichten der Welle gezeigt werden: 0 = nur der Start, ganz rechts = das Ziel ist entdeckt und die Routen erscheinen.")
    else:
        level = 0
        st.caption("Start und Ziel sind derselbe Knoten - es gibt nur Schicht 0.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
routes_shown = st.multiselect("Vergleichsrouten einblenden", ["optimal", "tie", "dfs"], default=["optimal"], key="routes_select", format_func=lambda k: ROUTE_NAMES[k],
                              help="Blau ist immer die BFS-Route. Rot gestrichelt: die kostenoptimale Route (Referenz, Dijkstra folgt im nächsten Stück). Grün gepunktet: die billigste Route unter allen mit der kleinsten Kantenzahl. "
                                   "Grau: was eine Tiefensuche findet. Wo BFS optimal ist, liegen Blau und Rot übereinander.")
view_slot = st.empty()
sizes = res.layer_sizes()


def _render(current_level):
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(build_network(net, a, current_level, ("bfs",) + tuple(routes_shown)), width="stretch", key="net_chart")
        c2.markdown("**Entdeckte Knoten je Schicht** (die Front der Welle)")
        c2.plotly_chart(build_layers(sizes, current_level), width="stretch", key="layer_chart")
        seen = int(((res.dist >= 0) & (res.dist <= current_level)).sum())
        c2.caption(f"Bis Schicht {current_level}: {seen} von {g.n} Knoten entdeckt. Die Front - die Knoten, die entdeckt, aber noch nicht abgearbeitet sind - war höchstens {m['max_front']} groß.")


if auto_play:
    for lv in range(last_level + 1):
        _render(lv)
        time.sleep(min(0.6, 6.0 / max(last_level, 1)))
    level = last_level
else:
    _render(level)

st.caption(net.note + (" Karte: © [OpenStreetMap-Mitwirkende](https://www.openstreetmap.org/copyright), Daten unter der [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/)." if net.key == "toronto" else ""))

st.markdown("---")

# --- Kanten zählen gegen Kosten --------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Zählen Kanten dasselbe wie Kosten?")
st.caption(
    "**Umweg** = Länge der BFS-Route geteilt durch die der kostenoptimalen Route, minus 1. Die kostenoptimale Route kommt aus einer kleinen Dijkstra-Referenz, die hier nur zum Messen dient; das Verfahren selbst ist das nächste Stück der Linie. "
    "**Bester Gleichstand:** BFS nimmt unter mehreren gleich kurzen Routen (nach Kanten) irgendeine, je nach Reihenfolge der Nachbarn - die billigste davon zeigt, wie viel vom Umweg an dieser Zufallswahl hängt und wie viel daran, dass BFS Kosten gar nicht kennt."
)
if not m["reachable"]:
    st.warning("⚠️ Das Ziel ist vom Start aus nicht erreichbar - BFS meldet das ausdrücklich, statt eine Route zu erfinden.")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kanten der BFS-Route", f"{m['hops_bfs']}", delta=f"{m['hops_bfs'] - m['hops_optimal']:+d} gegenüber der kostenoptimalen Route", delta_color="off",
              help="BFS findet immer die Route mit den wenigsten Kanten; die kostenoptimale Route kann mehr Kanten haben.")
    m2.metric("Länge der BFS-Route", _cost(net, m["cost_bfs"]), delta=f"{_cost(net, m['cost_bfs'] - m['cost_optimal'])} gegenüber der kostenoptimalen" if m["cost_bfs"] > m["cost_optimal"] + 1e-9 else "so kurz wie die kostenoptimale",
              delta_color="inverse", help=f"Länge der kostenoptimalen Route: {_cost(net, m['cost_optimal'])}.")
    m3.metric("Umweg der BFS-Route", _pct(m["detour"]), delta=f"beste kantenkürzeste Route: {_pct(m['tie_detour'])}", delta_color="off",
              help="Umweg gegenüber der kostenoptimalen Route; im Delta der Umweg der billigsten unter allen Routen mit der kleinsten Kantenzahl.")
    m4.metric("Entdeckte Knoten", f"{m['discovered']}", delta=f"von {m['n']} · größte Front {m['max_front']}", delta_color="off",
              help="Die Suche endet, sobald das Ziel entdeckt ist - der Rest des Netzes bleibt unberührt.")
    code = ev.verdict(a)
    if code == "unweighted":
        st.success(f"✅ Hier zählen Kanten genau dasselbe wie Kosten: jede Kante kostet gleich viel ({net.unit}). Die BFS-Route hat {m['hops_bfs']} Kanten, die kostenoptimale Referenz ebenfalls - BFS ist hier die richtige Wahl "
                   f"und musste nur {m['discovered']} von {m['n']} Knoten entdecken.")
    elif code == "same_route":
        st.info(f"Zufällig gleich: die BFS-Route ist auch die kostenoptimale ({_cost(net, m['cost_bfs'])}). Das ist die Ausnahme - die Verteilung unten zeigt, wie selten.")
    else:
        why = (f"Davon gehen {(m['detour'] - m['tie_detour']) * 100:.0f} Prozentpunkte auf die zufällige Wahl unter gleich kurzen Routen zurück (die beste kantenkürzeste Route hätte {_pct(m['tie_detour'])} Umweg); "
               "der Rest ist die Kostenblindheit von BFS." if m["tie_detour"] < m["detour"] - 0.02
               else f"Auch die beste unter den kantenkürzesten Routen hätte {_pct(m['tie_detour'])} Umweg: das liegt nicht an der Reihenfolge der Nachbarn, sondern daran, dass BFS Kosten gar nicht kennt.")
        (st.warning if code == "large" else st.info)(
            f"{'⚠️' if code == 'large' else 'ℹ️'} Die BFS-Route hat {m['hops_bfs']} Kanten, die kostenoptimale {m['hops_optimal']} - und ist trotzdem {_cost(net, m['cost_optimal'])} statt {_cost(net, m['cost_bfs'])} lang: "
            f"**{_pct(m['detour'])} Umweg**. {why}"
            + (" Bei OpenStreetMap-Daten kommt hinzu: die Kantenzahl hängt davon ab, wie dicht ein Weg digitalisiert ist - mit Länge hat sie nur lose zu tun." if net.key == "toronto" else ""))

    st.markdown("**Nicht nur dieses eine Paar**")
    ps = _pair_stats(params, C.PAIRS)
    if ps["n_pairs"]:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Paare ohne Umweg", _pct(ps["share_zero"]), help=f"Anteil der {ps['n_pairs']} zufälligen erreichbaren Start-Ziel-Paare, bei denen die BFS-Route so kurz ist wie die kostenoptimale.")
        p2.metric("Median-Umweg", _pct(ps["median"]), delta=f"mit bestem Gleichstand {_pct(ps['tie_median'])}", delta_color="off", help="Die Hälfte der Paare hat höchstens diesen Umweg.")
        p3.metric("90 %-Quantil", _pct(ps["p90"]), help="Bei einem Zehntel der Paare ist der Umweg größer.")
        p4.metric("Maximum", _pct(ps["max"]), help="Der schlimmste Umweg unter den gezogenen Paaren.")
        if ps["max"] > 1e-9:
            st.plotly_chart(build_detour_hist(ps["detour"], ps["tie"]), width="stretch", key="detour_hist")
        st.caption(f"{ps['n_pairs']} zufällige Start-Ziel-Paare im gewählten Netz (nicht nur das oben gezeigte). Der Mittelwert ({_pct(ps['mean'])}) sagt weniger als Median und 90 %-Quantil: "
                   "der Umweg ist sehr ungleich verteilt, ein paar Paare erwischt es hart. " + ("Im Netz mit gleichen Kosten je Kante ist der Umweg immer 0 - hier ist die Kantenzahl selbst das Maß." if not net.weighted else ""))

st.markdown("---")

# --- Vergleich -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – Breitensuche im Vergleich"):
    st.markdown("**Was jedes Verfahren für Start und Ziel oben findet**")
    rows = [("BFS (wenigste Kanten)", "bfs", m["hops_bfs"], m["cost_bfs"], m["discovered"]), ("beste unter den kantenkürzesten", "tie", m["hops_bfs"], m["cost_tie"], "–"),
            ("kostenoptimal (Referenz)", "optimal", m["hops_optimal"], m["cost_optimal"], m["settled"]), ("Tiefensuche", "dfs", m["hops_dfs"], m["cost_dfs"], m["dfs_visited"])]
    if m["reachable"]:
        st.table({"Verfahren": [r[0] for r in rows], "Kanten": [r[2] if a.routes[r[1]] else "–" for r in rows],
                  f"Länge [{net.unit}]": [f"{r[3]:,.0f}".replace(",", ".") if a.routes[r[1]] else "–" for r in rows], "Knoten besucht": [str(r[4]) for r in rows]})
    st.caption("Die Tiefensuche geht so tief wie möglich, bevor sie zurückkehrt: sie findet irgendeine Route, meist eine sehr lange - die Gegenprobe, dass die Schichten-Reihenfolge von BFS kein Zufall ist, sondern die Garantie liefert. "
               "Bei der kostenoptimalen Referenz zählt die Spalte die Knoten, deren Kosten am Ende feststanden.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt der Umweg von den Reglern des Stadtnetzes ab?")
SWEEPS = {"reach": ("Reichweite der Straßen", (1.0, 1.5, 2.3, 3.2)), "spread": ("Streuung der Kosten", (0.0, 0.5, 1.0, 2.0, 3.0)), "blocked": ("Gesperrte Straßen [%]", (0, 20, 40, 60)),
          "side": ("Kreuzungen je Seite", (6, 10, 20, 30, 40))}
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEPS), format_func=lambda k: SWEEPS[k][0], key="sweep_select")
base_city = {"side": int(side), "reach": round(float(reach), 1), "spread": round(float(spread), 2), "blocked": int(blocked)}
if net_key != "city":
    base_city = {"side": C.DEFAULT_SIDE, "reach": C.DEFAULT_REACH, "spread": C.DEFAULT_SPREAD, "blocked": C.DEFAULT_BLOCKED}
if st.button("Regler über 5 feste Stadtnetze durchfahren (dauert einige Sekunden)", key="sweep_start"):
    st.session_state["sweep_on"] = (sweep_param, tuple(sorted(base_city.items())))
if st.session_state.get("sweep_on") == (sweep_param, tuple(sorted(base_city.items()))):
    with st.spinner("Rechne den Sweep über 5 feste Stadtnetze × 40 Paare..."):
        sweep_rows = _sweep(sweep_param, SWEEPS[sweep_param][1], tuple(sorted(base_city.items())))
    st.plotly_chart(build_sweep(sweep_rows, SWEEPS[sweep_param][0], base_city[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Median-Umweg der BFS-Route über 40 zufällige Paare, Mittel über 5 feste Stadtnetze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste (bei einem anderen Netz als dem Stadtnetz: Standardwerte). "
               "Gepunktet: der Umweg mit bestem Gleichstand - der Abstand zwischen beiden Linien ist die Zufallswahl unter gleich kurzen Routen, der Rest die Kostenblindheit.")

st.markdown("---")

st.subheader("🔬 Aufwand: wächst BFS mit dem Netz - und wie schnell die Front?")
if st.button("Stadtnetze von 100 bis 6 400 Kreuzungen durchlaufen (dauert wenige Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Durchlaufe 6 Netze vollständig..."):
        sc_rows = _scaling()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_scaling(sc_rows), width="stretch", key="scaling_chart")
    c2.table({"Kreuzungen": [f"{r['n']:,}".replace(",", ".") for r in sc_rows], "geprüfte Kanten": [f"{r['scanned']:,}".replace(",", ".") for r in sc_rows], "größte Front": [r["max_front"] for r in sc_rows],
              "Schichten": [r["levels"] for r in sc_rows]})
    st.caption("Vollständiger Durchlauf ohne Ziel, Reichweite 1.5. Jede Kante wird **genau einmal** geprüft (die Zahl der geprüften Kanten ist die der Kanten des Netzes): der Aufwand ist linear in Knoten plus Kanten. "
               "Die **Front** - der Speicherbedarf der Suche - wächst dagegen nur wie die Wurzel der Knotenzahl, weil die Welle auf der Fläche einen Ring bildet.")

st.markdown("---")

st.subheader("🔬 Gegenprobe: Tiefensuche")
if st.button("BFS gegen Tiefensuche über zufällige Paare (dauert wenige Sekunden)", key="dfs_start"):
    st.session_state["dfs_on"] = True
if st.session_state.get("dfs_on"):
    with st.spinner("Vergleiche vier Netze × 60 Paare..."):
        dfs_rows = _dfs()
    st.plotly_chart(build_dfs(dfs_rows), width="stretch", key="dfs_chart")
    st.caption("Median über 60 zufällige Paare je Netz (Nachbarn in Reihenfolge der Knotennummern). Die Tiefensuche findet eine Route, aber ohne jede Garantie: im Stadtnetz und in Toronto hat sie das 42- beziehungsweise 30-Fache der Kanten von BFS, "
               "im Labyrinth das 4-Fache, im Bus-Netz das 2-Fache. Die Schichten-Reihenfolge ist es, die BFS die Garantie gibt.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Jede Kante kostet dasselbe** | Toronto Campus: die Route mit den wenigsten Kanten (29) ist 27 % länger als die kürzeste (57 Kanten); über 200 zufällige Paare Median-Umweg 7 %, 90 %-Quantil 25 %, schlimmster Fall 237 %; nur 3,5 % der Paare kommen ohne Umweg davon. Im Stadtnetz mit Standardwerten liegt der Median-Umweg im Mittel über fünf Netze bei 23 %. | **Dijkstra** (nächstes Stück): Kosten korrekt |
| **Die Welle darf in alle Richtungen gleich weit laufen** | BFS weiß nichts über die Lage des Ziels: für Toronto (kürzeste Route 892 m) entdeckt sie 3 043 von 5 072 Knoten, bevor sie das Ziel findet. Die Front wächst mit der Wurzel der Knotenzahl. | **Bidirektionale Suche** (von beiden Enden), **A\\*** (Baumsuche-Linie) |
| **Jede Anfrage beginnt von vorn** | Jede Anfrage kostet bis zu Knoten plus Kanten: im größten Netz des Experiments 50 244 geprüfte Kanten - für **eine** Anfrage. | **Contraction Hierarchies**: erst vorrechnen, dann blitzschnell fragen |
| **Kantenzahl ist ein Maß für Entfernung** | Bei echten Kartendaten hängt sie davon ab, wie dicht digitalisiert wurde: im Toronto-Beispiel hat die kürzere Route 57 statt 29 Kanten. | (Kosten in das Modell) |
"""
)
st.caption("Die Nachbarn der Kürzeste-Wege-Linie (noch nicht gebaut): Dijkstra, Bidirektionale Suche, Contraction Hierarchies, Bellman-Ford, Floyd-Warshall, Johnson und Mehrkriterien-Routing. A\\* steht in der Baumsuche-Linie.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Gerichteter Graph $G=(V,E)$ mit Kosten $c_e \ge 0$; Kantenzahl $\delta(s,v)$ = kleinste Zahl von Kanten einer Route von $s$ nach $v$ (unabhängig von den Kosten).

**BFS.** Schicht $L_0=\{s\}$, $L_{k+1}=\{v \notin L_0\cup\dots\cup L_k : \exists u\in L_k,\ (u,v)\in E\}$. Die Warteschlange arbeitet first-in-first-out; jeder Knoten wird genau einmal eingereiht und abgearbeitet, jede Kante genau einmal geprüft: Laufzeit $O(|V|+|E|)$, Speicher $O(|V|)$.
**Korrektheit:** ein Knoten in $L_k$ hat $\delta(s,v)=k$ (Induktion über $k$: jeder Vorgänger einer Route mit $k$ Kanten liegt in $L_{k-1}$, und die Warteschlange ordnet die Schichten). Abbruch beim ersten entdeckten Ziel ist erlaubt, weil sein $\delta$ dann endgültig ist.

**Umweg.** Für die BFS-Route $P_{\mathrm{BFS}}$ und die kostenoptimale Route $P^*$: $\ \mathrm{Umweg}=c(P_{\mathrm{BFS}})/c(P^*)-1\ \ge 0$. Bei $c_e\equiv 1$ ist $P_{\mathrm{BFS}}$ selbst kostenoptimal, der Umweg ist 0 - das ist die einzige Bedingung, unter der BFS Kosten und Kanten gleichsetzen darf.
**Bester Gleichstand:** unter allen Routen mit $\delta(s,t)$ Kanten die billigste, per Programmierung über den Schichten-Graphen: $b(v)=\min_{u\in L_{k-1},\,(u,v)\in E}\ b(u)+c_{uv}$. Sie ist die bestmögliche Antwort, die ein BFS mit beliebiger Nachbar-Reihenfolge geben könnte.

**Mehrfach-Start und -Ziel.** Alle Starts kommen in Schicht 0; endet die Suche beim ersten entdeckten Ziel aus einer Menge, ist es das nächste (Beispiel: das Kontaktnetz mit zwei Personen mit Staplerschein).

**Kennzahlen.** Front = größte Länge der Warteschlange (Speicherbedarf); entdeckte Knoten; geprüfte Kanten. Im Stadtnetz mit Seitenlänge $m$ hat ein vollständiger Durchlauf Schichten in der Größenordnung $m$ und eine Front der Größenordnung $m=\sqrt{|V|}$.

**Grenzen.** (1) Kosten werden ignoriert (Umweg). (2) Die Welle kennt die Richtung des Ziels nicht (besuchte Fläche wächst quadratisch mit der Entfernung). (3) Jede Anfrage beginnt von vorn.

Implementiert in `bf_graph.py` (CSR-Graph), `bf_algorithm.py` (BFS, kostenoptimale Referenz, beste Gleichstands-Route, Tiefensuche), `bf_scenario.py` (Netze), `bf_evaluation.py` (Kennzahlen, Verteilung, Sweeps).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
