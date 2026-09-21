"""Plotly-Abbildungen: Netz mit Schichten und Routen, Schichtgrößen, Verteilung des Umwegs, Sweeps. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import bf_constants as C

ROUTE_NAMES = {"bfs": "BFS (wenigste Kanten)", "optimal": "kostenoptimal (Referenz)", "tie": "beste unter den kantenkürzesten", "dfs": "Tiefensuche"}
NET_NAMES = {"city": "Stadtnetz", "maze": "Labyrinth", "toronto": "Toronto Campus", "bus": "Bus-Umstiege", "contacts": "Kontaktnetz"}


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.08), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _edge_segments(g):
    """Alle Kanten als eine Linienspur (None trennt die Segmente); Hin- und Rückrichtung nur einmal."""
    src = np.repeat(np.arange(g.n), g.degree())
    dst = g.indices
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    keep = np.zeros(len(src), dtype=bool)
    _, first = np.unique(lo * g.n + hi, return_index=True)
    keep[first] = True
    u, v = lo[keep], hi[keep]
    x = np.full(3 * len(u), None, dtype=object)
    y = np.full(3 * len(u), None, dtype=object)
    x[0::3], x[1::3] = g.xy[u, 0], g.xy[v, 0]
    y[0::3], y[1::3] = g.xy[u, 1], g.xy[v, 1]
    return x, y


def build_network(net, analysis, level, show_routes=("bfs", "optimal"), height=520):
    """Das Netz mit den bis zur Schicht `level` entdeckten Knoten (Farbe = Kantenzahl vom Start); Routen erscheinen, sobald das Ziel entdeckt ist."""
    g, res = net.graph, analysis.bfs
    fig = go.Figure()
    ex, ey = _edge_segments(g)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.45)", width=1), hoverinfo="skip", showlegend=False))
    is_seen = (res.dist >= 0) & (res.dist <= level)
    seen = np.where(is_seen)[0]
    small = g.n <= 40
    if len(seen):
        fig.add_trace(go.Scatter(
            x=g.xy[seen, 0], y=g.xy[seen, 1], mode="markers+text" if small else "markers", showlegend=False,
            text=[g.names[i] for i in seen] if small and g.names else None, textposition="top center",
            marker=dict(size=14 if small else (3 if g.n > 1500 else 7), opacity=1.0 if small else 0.85, color=res.dist[seen], colorscale="Viridis", cmin=0, cmax=max(int(res.dist.max()), 1),
                        colorbar=dict(title="Kanten<br>vom Start", thickness=12, len=0.6, dtick=1 if res.dist.max() <= 8 else None)),
            customdata=res.dist[seen], hovertemplate="%{customdata} Kanten vom Start<extra></extra>"))
    if small:
        unseen = np.where(~is_seen)[0]
        if len(unseen):
            fig.add_trace(go.Scatter(x=g.xy[unseen, 0], y=g.xy[unseen, 1], mode="markers+text" if g.names else "markers", showlegend=False,
                                     text=[g.names[i] for i in unseen] if g.names else None, textposition="top center",
                                     marker=dict(size=12, color="white", line=dict(color="gray", width=1.5)), hoverinfo="skip"))
        if g.directed:
            src = np.repeat(np.arange(g.n), g.degree())
            for u, v in zip(src, g.indices):
                fig.add_annotation(x=g.xy[v, 0], y=g.xy[v, 1], ax=g.xy[u, 0], ay=g.xy[u, 1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2,
                                   arrowwidth=1.2, arrowcolor="rgba(120,120,120,0.8)", standoff=9, startstandoff=9)
    found = res.found >= 0 and res.dist[res.found] <= level
    if found:
        styles = {"optimal": dict(color=C.COLORS["optimal"], width=3, dash="dash"), "tie": dict(color=C.COLORS["tie"], width=3, dash="dot"),
                  "dfs": dict(color=C.COLORS["dfs"], width=2), "bfs": dict(color=C.COLORS["bfs"], width=5)}
        for key in ("dfs", "tie", "optimal", "bfs"):
            route = analysis.routes.get(key)
            if key in show_routes and route:
                pts = g.xy[route]
                fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=styles[key], name=ROUTE_NAMES[key], hoverinfo="skip"))
    for nodes, name, label, color in ((net.sources, "Start", net.start_label, C.COLORS["start"]), (net.targets, "Ziel", net.goal_label, C.COLORS["goal"])):
        pts = g.xy[list(nodes)]
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name=f"{name}: {label}", hoverinfo="skip",
                                 marker=dict(size=16, color=color, symbol="star" if name == "Ziel" else "diamond", line=dict(color="white", width=1.5))))
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    if small:                                                                    # Platz für die Beschriftungen am Rand
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        pad = 0.22 * (hi - lo)
        fig.update_xaxes(range=[lo[0] - pad[0], hi[0] + pad[0]])
        fig.update_yaxes(range=[lo[1] - 0.12 * (hi[1] - lo[1]), hi[1] + 0.12 * (hi[1] - lo[1])])
    return _base(fig, height)


def build_layers(sizes, level=None, height=260):
    """Entdeckte Knoten je Schicht (Kantenzahl vom Start): die Front."""
    x = np.arange(len(sizes))
    colors = ["#1f77b4" if level is None or i <= level else "#c8d6e5" for i in x]
    fig = go.Figure(go.Bar(x=x, y=sizes, marker_color=colors, hovertemplate="Schicht %{x}: %{y} Knoten<extra></extra>"))
    fig.update_layout(xaxis_title="Kanten vom Start", yaxis_title="entdeckte Knoten")
    return _base(fig, height)


def build_detour_hist(detour, tie, height=300):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=detour * 100, name="BFS-Route", marker_color=C.COLORS["bfs"], opacity=0.75, xbins=dict(start=0, size=5)))
    fig.add_trace(go.Histogram(x=tie * 100, name="beste unter den kantenkürzesten", marker_color=C.COLORS["tie"], opacity=0.6, xbins=dict(start=0, size=5)))
    fig.add_vline(x=float(np.median(detour)) * 100, line=dict(color=C.COLORS["bfs"], dash="dash"), annotation_text="Median BFS", annotation_position="top")
    fig.update_layout(barmode="overlay", xaxis_title="Umweg gegenüber der kostenoptimalen Route [%]", yaxis_title="Start-Ziel-Paare")
    return _base(fig, height)


def build_sweep(rows, xlabel, current=None, height=320):
    """Median-Umweg der BFS-Route gegen einen Regler des Stadtnetzes; gepunktet: mit dem besten Gleichstand (der Rest ist die Kostenblindheit von BFS)."""
    x = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[r["median"] * 100 for r in rows], mode="lines+markers", name="BFS-Route", line=dict(color=C.COLORS["bfs"]),
                             hovertemplate="Median-Umweg %{y:.0f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=[r["tie_median"] * 100 for r in rows], mode="lines+markers", name="beste unter den kantenkürzesten", line=dict(color=C.COLORS["tie"], dash="dot"),
                             hovertemplate="Median-Umweg %{y:.0f} %<extra></extra>"))
    if current is not None and min(x) <= current <= max(x):
        fig.add_vline(x=current, line=dict(color="gray", dash="dash"), annotation_text="aktuell", annotation_position="top")
    fig.update_layout(xaxis_title=xlabel, yaxis_title="Median-Umweg [%]", yaxis_rangemode="tozero")
    return _base(fig, height)


def build_scaling(rows, height=320):
    n = np.array([r["n"] for r in rows], dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n, y=[r["scanned"] for r in rows], mode="lines+markers", name="geprüfte Kanten", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=n, y=[r["max_front"] for r in rows], mode="lines+markers", name="größte Front", line=dict(color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=n, y=n, mode="lines", name="Steigung 1", line=dict(color="rgba(120,120,120,0.6)", dash="dot")))
    fig.add_trace(go.Scatter(x=n, y=np.sqrt(n) * rows[0]["max_front"] / np.sqrt(n[0]), mode="lines", name="Steigung ½", line=dict(color="rgba(120,120,120,0.6)", dash="dash")))
    fig.update_layout(xaxis=dict(title="Knoten im Netz", type="log"), yaxis=dict(title="Anzahl", type="log"))
    return _base(fig, height)


def build_dfs(rows, height=280):
    fig = go.Figure(go.Bar(x=[NET_NAMES[r["net"]] for r in rows], y=[r["hop_ratio"] for r in rows], marker_color=C.COLORS["dfs"], text=[f"{r['hop_ratio']:.1f}×" for r in rows],
                           textposition="outside", hovertemplate="%{x}: %{y:.1f}-fache Kantenzahl<extra></extra>"))
    fig.update_layout(yaxis=dict(title="Kanten der Tiefensuche-Route / BFS-Route (Median)", type="log"))
    return _base(fig, height)
