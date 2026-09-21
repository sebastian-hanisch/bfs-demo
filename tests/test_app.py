"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, ausgeblendete Regler, Permalink, Experimente auf Abruf, Schlüssel und Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import bf_constants as C
from bf_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
EXPECTED_KIND = {"🚌 Bus-Umstiege": "success", "🤝 Kontaktnetz": "success", "🧱 Labyrinth": "success", "🍁 Toronto Campus": "warning", "🏙️ Stadtnetz": "warning"}


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input)}


def test_default_renders_without_exception():
    at = _run()
    assert any("Breitensuche in Aktion" in m.value for m in at.markdown)
    assert len(at.warning) == 1 and not at.error and not at.success                   # Stadtnetz mit Standardwerten: BFS liegt daneben


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict_kind(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    kind = EXPECTED_KIND[name]
    assert (len(at.success) == 1 and not at.warning) if kind == "success" else (len(at.warning) == 1 and not at.success)


def test_extreme_grid_sizes_render():
    for net, extra in (("city", {"reach_slider": C.REACH_MAX, "spread_slider": C.SPREAD_MAX, "blocked_slider": C.BLOCKED_MAX}), ("city", {"reach_slider": C.REACH_MIN, "spread_slider": C.SPREAD_MIN}),
                       ("maze", {"walls_slider": C.WALLS_MAX})):
        for side in (C.SIDE_MIN, C.SIDE_MAX):
            def setup(at, net=net, extra=extra, side=side):
                at.session_state["net_select"] = net
                at.session_state["side_slider"] = side
                for k, v in extra.items():
                    at.session_state[k] = v
            at = _run(setup)
            assert at.slider(key="bf_level").value == at.slider(key="bf_level").max


def test_hidden_controls_follow_the_net():
    def labels_for(net):
        return _labels(_run(lambda a: a.session_state.__setitem__("net_select", net)))
    city, maze, fixed = labels_for("city"), labels_for("maze"), labels_for("toronto")
    assert {"Kreuzungen je Seite", "Reichweite der Straßen [Blocklängen]", "Streuung der Kosten", "Gesperrte Straßen [%]", "Zufalls-Seed"} <= city and "Wände [%]" not in city
    assert {"Zellen je Seite", "Wände [%]", "Zufalls-Seed"} <= maze and "Streuung der Kosten" not in maze
    assert fixed == {"Netz"}                                                             # keine toten Regler bei festen Netzen


def test_hidden_slider_values_come_back_when_the_net_is_shown_again():
    at = _run(lambda a: a.session_state.__setitem__("spread_slider", 2.5))
    at.session_state["net_select"] = "toronto"
    at.run()
    at.session_state["net_select"] = "city"
    at.run()
    assert not at.exception and at.slider(key="spread_slider").value == 2.5


def test_layer_slider_returns_to_the_last_layer_when_the_net_changes():
    at = _run()
    at.slider(key="bf_level").set_value(3)
    at.run()
    assert at.slider(key="bf_level").value == 3
    at.session_state["net_select"] = "bus"
    at.run()
    assert not at.exception and at.slider(key="bf_level").value == 5 == at.slider(key="bf_level").max


def test_layer_slider_at_zero_does_not_break_the_route_view():
    at = _run()
    at.slider(key="bf_level").set_value(0)
    at.run()
    assert not at.exception


def test_permalink_parameters_select_the_net_and_are_clamped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "maze"
    at.query_params["side"] = "9999"
    at.query_params["walls"] = "abc"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == "maze"
    assert at.slider(key="side_slider").value == C.SIDE_MAX and at.slider(key="walls_slider").value == C.DEFAULT_WALLS


def test_unknown_net_in_the_permalink_falls_back_to_the_default():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET


def test_experiments_run_on_demand():
    at = _run()
    assert not any("Median-Umweg der BFS-Route über 40 zufällige Paare" in c.value for c in at.caption)
    for key in ("sweep_start", "scaling_start", "dfs_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert "Median-Umweg der BFS-Route über 40 zufällige Paare" in text and "**genau einmal**" in text and "Nachbarn in Reihenfolge der Knotennummern" in text


@pytest.mark.parametrize("param", ["reach", "spread", "blocked", "side"])
def test_every_sweep_parameter_runs(param):
    at = _run(lambda a: a.session_state.__setitem__("sweep_select", param))
    at.button(key="sweep_start").click()
    at.run()
    assert not at.exception


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    assert len(calls) == 6 and all(re.search(r'key=f?"[a-z_]+(_\{\w+\})?"', c) for c in calls), calls
    assert len({re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls}) == 6            # jeder Schlüssel nur einmal
    viz = (ROOT / "bf_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("_base(fig") >= 6


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))


def test_play_runs_through_all_frames_without_duplicate_chart_keys():
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    at = _run()
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
