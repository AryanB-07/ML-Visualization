"""Plotly Dash dashboard: a live, explorable map of one run.

The dashboard only reads the run directory (and writes ``control.json`` for
the worker), so it behaves the same for a run that is still training and for
one reopened later.
"""
from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

from .basis import MODES, BasisManager
from .demos import PRESETS, list_presets, list_runs, start_demo, unique_run_name
from .landscape import RESOLUTIONS, Evaluator, GridCache, SurfaceModel, grid_bounds, unit
from .store import TRUNK, Run, atomic_write_json
from .worker import ensure_worker, worker_status

# Dark wireframe theme: near-black navy ground, electric-blue mesh, white trunk.
INK, MUTED, GRID, SURFACE_BG = "#e6ecff", "#8793b3", "rgba(140,165,255,0.13)", "rgba(0,0,0,0)"
PANEL = "#0a1022"                       # opaque stand-in for the card colour (hover labels, legends)
ACCENT, ACCENT_SOFT = "#3d7bff", "#8fb4ff"
TRUNK_COLOR = "#ffffff"
BRANCH_COLORS = ["#d95926", "#9085e9", "#d55181"]  # validated on the dark surface; no greens
EXPLORE_COLOR = "#c98500"                           # every exploration shares one colour
LOSS_SCALE = [[0.0, "#030712"], [0.55, "#0a1a4a"], [1.0, "#16348a"]]   # one hue, dark -> bright with loss
# One surface trace, two colour ramps: evaluated cells use the loss ramp, interpolated cells a dim
# desaturated ramp.  ``surfacecolor`` is the loss mapped to [0.5, 1] or [0, 0.5) accordingly.
SPLIT_SCALE = ([[0.0, "#05070e"], [0.4999, "#151c2e"]] +
               [[0.5 + 0.5 * f, c] for f, c in LOSS_SCALE])
WIRE = "rgba(118,162,255,0.9)"           # lighter tint of the accent: reads against the dark fill
MODE_LABELS = {"pca_trunk": "PCA, trunk only", "pca_joint": "PCA, trunk + branches",
               "random": "Random, filter-normalized"}
FONT = dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif", size=12, color=INK)

CSS = """
:root {
  --bg:#04060c; --panel:rgba(10,16,34,.78); --line:rgba(125,165,255,.16); --line-strong:rgba(125,165,255,.34);
  --text:#e6ecff; --muted:#8793b3; --accent:#3d7bff; --accent-soft:#8fb4ff;
  --Dash-Fill-Interactive-Strong:#3d7bff; --Dash-Fill-Interactive-Weak:rgba(61,123,255,.10);
  --Dash-Fill-Inverse-Strong:#0a1022; --Dash-Fill-Primary-Hover:rgba(125,165,255,.10);
  --Dash-Fill-Primary-Active:rgba(125,165,255,.18); --Dash-Fill-Disabled:rgba(125,165,255,.12);
  --Dash-Stroke-Strong:rgba(125,165,255,.42); --Dash-Stroke-Weak:rgba(125,165,255,.16);
  --Dash-Text-Primary:#e6ecff; --Dash-Text-Strong:#ffffff; --Dash-Text-Weak:#8793b3; --Dash-Text-Disabled:#5b6788;
  --Dash-Shading-Strong:rgba(0,0,0,.6); --Dash-Shading-Weak:rgba(0,0,0,.45);
  color-scheme:dark;
}
html { background:var(--bg); }
body { margin:0; min-height:100vh; color:var(--text);
  font:14px 'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  background:
    radial-gradient(900px 520px at 50% -8%, rgba(52,104,255,.30), transparent 62%),
    radial-gradient(700px 500px at 100% 30%, rgba(30,60,170,.14), transparent 60%),
    radial-gradient(700px 500px at 0% 55%, rgba(24,44,140,.12), transparent 60%),
    var(--bg);
  background-attachment:fixed; }
.pages { padding:10px 14px 14px; }
.two-col { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(280px,1fr); gap:12px; align-items:start; }
.two-col .card + .card { margin-top:0; }
.navs { display:flex; gap:2px; background:rgba(10,16,34,.7); border:1px solid var(--line); border-radius:8px; padding:3px; }
button.nav { background:transparent; border:0; color:var(--muted); font:inherit; font-size:12.5px; padding:5px 12px; border-radius:6px; cursor:pointer; }
button.nav:hover { color:var(--text); }
button.nav.on { background:rgba(61,123,255,.22); color:#fff; }
button.ghost { background:transparent; border:1px solid var(--line-strong); color:var(--accent-soft); font:inherit; font-size:12px; padding:5px 11px; border-radius:6px; cursor:pointer; }
button.ghost:hover { background:rgba(61,123,255,.14); color:#fff; }
.drawer { position:fixed; top:0; right:0; bottom:0; width:310px; z-index:30; padding:16px 18px; overflow-y:auto;
  background:rgba(7,11,24,.96); border-left:1px solid var(--line-strong); box-shadow:-30px 0 60px -30px rgba(40,90,255,.45);
  transform:translateX(105%); transition:transform .18s ease; }
.drawer.open { transform:none; }
.drawer h2 { margin:0; font-size:13px; letter-spacing:.14em; text-transform:uppercase; font-weight:600; }
.drawer h3 { margin:20px 0 8px; font-size:11px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--accent-soft); }
.drawer .note { margin-top:8px; }
.drawer-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:8px; }
.map-graph { height:calc(100vh - 150px); min-height:440px; }
.chip { position:absolute; top:12px; right:12px; z-index:5; background:rgba(6,10,22,.6); color:var(--text); font:inherit; font-size:12px;
  border:1px solid var(--line-strong); border-radius:999px; padding:5px 12px; cursor:pointer; }
.chip:hover { border-color:var(--accent-soft); }
.chip.low { border-color:rgba(217,137,38,.7); color:#f1d3a6; }
.popup { position:absolute; right:12px; bottom:64px; z-index:6; width:300px; display:none; padding:12px 14px;
  background:rgba(7,11,24,.94); border:1px solid var(--line-strong); border-radius:10px; box-shadow:0 20px 50px -20px rgba(40,90,255,.6); }
.popup.open { display:block; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px;
  box-shadow:0 0 0 1px rgba(0,0,0,.4), 0 18px 40px -24px rgba(40,90,255,.35); backdrop-filter:blur(6px); }
.card h3 { margin:0 0 10px; font-size:11px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--accent-soft); }
.card + .card { margin-top:12px; }
.map-card { padding:0; overflow:hidden; position:relative;
  background:
    radial-gradient(ellipse 70% 55% at 50% 30%, rgba(60,115,255,.26), rgba(24,56,170,.12) 45%, rgba(6,9,19,0) 80%),
    linear-gradient(#070b18, #04060c); }
.map-card .below { padding:4px 14px 12px; }
.top { padding:12px 16px 0; display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
.top .status { flex:1; text-align:right; }
.top h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.14em; text-transform:uppercase; }
.top h1 span { color:var(--accent-soft); font-weight:500; }
.top { gap:10px 14px; }
.run-select { width:270px; font-size:13px; }
.top .status { flex:1 1 260px; min-width:0; font-size:11.5px; line-height:1.35; }
.run-select .Select-control, .run-select .dash-dropdown { background:rgba(10,16,34,.7); }
.drawer.wide { width:400px; }
.presets input[type=radio] { margin-top:3px; }
.preset-title { font-weight:600; font-size:13px; color:var(--text); }
.presets .note { margin-top:3px; }
.preset-cost { font-size:11.5px; color:var(--accent-soft); margin-top:4px; }
.status { color:var(--muted); font-size:12.5px; font-variant-numeric:tabular-nums; }
label.ctl { display:block; font-size:12px; color:var(--muted); margin:12px 0 4px; }
.note { font-size:12px; color:var(--muted); line-height:1.5; }
.warn { background:rgba(217,137,38,.10); border:1px solid rgba(217,137,38,.45); color:#f1d3a6; border-radius:6px; padding:8px 10px; font-size:12.5px; margin-top:8px; }
.tag { display:inline-block; background:rgba(61,123,255,.16); color:var(--accent-soft); border:1px solid var(--line-strong); border-radius:4px; padding:2px 7px; font-size:10.5px; font-weight:600; letter-spacing:.1em; }
table.kv { border-collapse:collapse; width:100%; font-size:12.5px; }
table.kv td { padding:4px 0; border-bottom:1px solid rgba(125,165,255,.08); color:var(--muted); }
table.kv td:last-child { text-align:right; font-variant-numeric:tabular-nums; color:var(--text); }
button.btn { margin-top:10px; background:var(--accent); color:#fff; border:0; border-radius:6px; padding:7px 12px; font-size:12.5px; font-weight:600; cursor:pointer; box-shadow:0 0 18px -4px rgba(61,123,255,.8); }
button.btn:hover { background:#5a8fff; }
input[type=number] { background:#070b18; color:var(--text); border:1px solid var(--line-strong); border-radius:6px; }
input[type=radio], input[type=checkbox] { accent-color:var(--accent); }
.modebar { background:transparent !important; } .modebar-btn path { fill:rgba(143,180,255,.55) !important; }
@media (max-width:900px) { .two-col { grid-template-columns:1fr; } .top .status { flex-basis:100%; text-align:left; } }
"""


def _fmt(x: float | None, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}g}"


class DashboardState:
    """Reads run state for the callbacks and caches the fitted surface between refreshes."""

    def __init__(self, run_dir: str | Path, device: str = "cpu"):
        self.run = Run(run_dir)
        self.bm = BasisManager(self.run)
        self.device = device
        self._surface: tuple[Any, SurfaceModel] | None = None
        self._evaluator: Evaluator | None = None
        self._lock = threading.Lock()

    def branch_colors(self) -> dict[str, str]:
        bs = sorted(self.run.branches(), key=lambda t: t.info().get("created", 0))
        out = {t.name: EXPLORE_COLOR for t in bs if t.kind == "explore"}
        out.update({t.name: BRANCH_COLORS[i % len(BRANCH_COLORS)]
                    for i, t in enumerate(t for t in bs if t.kind != "explore")})
        return out

    def view(self, mode: str) -> dict[str, Any]:
        info = self.bm.latest_info(mode)
        if info is None:
            return {"info": None}
        proj = self.bm.read_projections(info["id"])
        points = GridCache.read(self.run.dir, info["id"])
        return {"info": info, "proj": proj, "points": points}

    def surface(self, info: dict[str, Any], points: dict) -> SurfaceModel:
        key = (info["id"], len(points))
        if self._surface is None or self._surface[0] != key:
            self._surface = (key, SurfaceModel(points, info["coarse_h"]))
        return self._surface[1]

    def evaluator(self) -> Evaluator:
        if self._evaluator is None:
            self._evaluator = Evaluator(self.run, self.device)
        return self._evaluator

    def in_plane(self, basis_id: str, a: float, b: float) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Evaluate the in-plane weights at (a, b) on the whole eval subset; also return their state_dict."""
        mode, _, v = basis_id.rpartition("-v")
        info = self.bm.latest_info(mode)
        if info is None or info["id"] != basis_id:
            raise LookupError("the basis was refit since this point was clicked; click again")
        basis = self.bm.load(mode, int(v))
        bn_mode = self.run.control()["bn_mode"]
        with self._lock:
            ev = self.evaluator()
            try:
                snap = self.run.trunk.load(info["center"][1])
                fb, ib = snap.buffers, snap.int_buffers
            except FileNotFoundError:
                fb = ib = None
            w = basis.point(a, b)
            res = ev.evaluate(w, fb, ib, eval_batches=None, bn_mode=bn_mode)
            sd = {k: t.detach().cpu().clone() for k, t in ev.model.state_dict().items()}
        res.update(a=a, b=b, basis_id=basis_id, bn_mode=bn_mode if ev.bn_layers else "none")
        return res, sd


# ---------------------------------------------------------------------- figures
def _empty(msg: str, height: int = 620) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(height=height, paper_bgcolor=SURFACE_BG, plot_bgcolor=SURFACE_BG, font=FONT, template=None,
                      xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10),
                      annotations=[dict(text=msg, showarrow=False, font=dict(size=14, color=MUTED))])
    return fig


def map_figure(st: DashboardState, view: dict[str, Any], sel_step: int | None, level: int,
               zlog: bool, clip: bool) -> go.Figure:
    info = view["info"]
    if info is None:
        return _empty("Waiting for the first basis (needs at least two snapshots and a running worker).")
    proj, points = view["proj"], view["points"]
    if TRUNK not in proj or not proj[TRUNK]["steps"]:
        return _empty("Projecting snapshots…")
    colors = st.branch_colors()
    trunk = proj[TRUNK]
    recs = {t.name: {r["step"]: r.get("loss") for r in t.records()} for t in st.run.trajectories()}
    sel_step = trunk["steps"][-1] if sel_step is None else sel_step
    all_ab = np.vstack([np.column_stack([p["a"], p["b"]]) for p in proj.values() if p["steps"]])
    bounds = grid_bounds(all_ab, info["coarse_h"], info.get("min_extent", 0.0))
    model = st.surface(info, points) if points else None
    have_surface = model is not None and model.ok

    def path_z(p: dict[str, Any], name: str) -> np.ndarray:
        if have_surface:
            return model(np.column_stack([p["a"], p["b"]]))
        return np.array([recs[name].get(s) or 0.0 for s in p["steps"]], dtype=float)

    path_zs = {name: path_z(p, name) for name, p in proj.items() if p["steps"]}
    fig = go.Figure()
    lift = 0.0
    if have_surface:
        mesh = model.mesh(bounds, info["coarse_h"], level)
        z = mesh["z"]
        zcap = None
        if clip:
            zcap = 2.0 * max(float(v.max()) for v in path_zs.values())
            z = np.minimum(z, zcap)
        zlo, zhi = float(z.min()), float(z.max())
        lift = 0.03 * (zhi - zlo)   # keeps paths above the mesh between grid nodes
        sup = mesh["supported"]

        def wire(color: str) -> dict[str, Any]:
            def ax(v: np.ndarray) -> dict[str, Any]:
                return dict(show=True, start=float(v[0]), end=float(v[-1]), size=float(v[1] - v[0]),
                            color=color, width=1, highlight=False)
            return dict(x=ax(mesh["a"]), y=ax(mesh["b"]), z=dict(show=False, highlight=False))
        # A single trace: two overlapping surfaces z-fight and leave stair-stepped holes at their seam.
        span = max(zhi - zlo, 1e-12)
        frac = (z - zlo) / span
        sc = np.where(sup, 0.5 + 0.5 * frac, 0.4999 * frac)
        fig.add_trace(go.Surface(
            x=mesh["a"], y=mesh["b"], z=z, surfacecolor=sc, customdata=np.where(sup, 1, 0),
            colorscale=SPLIT_SCALE, cmin=0.0, cmax=1.0, showscale=False, name="Surface (dim = interpolated)",
            showlegend=True, contours=wire(WIRE),
            hovertemplate="a %{x:.3g}, b %{y:.3g}<br>loss %{z:.4g}<extra></extra>",
            lighting=dict(ambient=0.85, diffuse=0.45, specular=0.05, roughness=0.9, fresnel=0.1),
            lightposition=dict(x=-2000, y=-5000, z=10000)))
        # Colour bar for the loss ramp alone (the surface's own bar would show the split scale).
        fig.add_trace(go.Scatter3d(
            x=[mesh["a"][0]], y=[mesh["b"][0]], z=[zlo], mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(size=0.1, opacity=0, color=[zlo], colorscale=LOSS_SCALE, cmin=zlo, cmax=zhi, showscale=True,
                        colorbar=dict(title=dict(text="loss", side="right"), thickness=8, len=0.5, x=0.97,
                                      outlinewidth=0, tickfont=dict(color=MUTED, size=10)))))
        u = unit(info["coarse_h"])
        keys = [k for k, v in points.items() if v[0] is not None]
        pz = np.array([points[k][0] for k in keys])
        if zcap is not None:
            pz = np.minimum(pz, zcap)
        fig.add_trace(go.Scatter3d(
            x=[k[0] * u for k in keys], y=[k[1] * u for k in keys], z=pz, mode="markers",
            marker=dict(size=1.2, color="#cfe0ff", opacity=0.35), name=f"Evaluated grid points ({len(keys)})",
            hovertemplate="a %{x:.3g}, b %{y:.3g}<br>loss %{z:.4g}<extra>evaluated</extra>"))

    def hover_text(name: str, p: dict[str, Any]) -> list[str]:
        return [f"{name} step {s}<br>recorded loss {_fmt(recs[name].get(s))}"
                f"<br>captured by plane {c:.1%}<br>residual norm {r:.3g}"
                for s, c, r in zip(p["steps"], p["captured"], p["residual"])]

    tz = path_zs[TRUNK] + lift
    ta, tb, ts = np.array(trunk["a"]), np.array(trunk["b"]), np.array(trunk["steps"])
    upto = ts <= sel_step
    if (~upto).any():
        fig.add_trace(go.Scatter3d(x=ta, y=tb, z=tz, mode="lines", line=dict(color="rgba(255,255,255,0.35)", width=4),
                                   name="Trunk (after selected step)", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=ta[upto], y=tb[upto], z=tz[upto], mode="lines+markers", line=dict(color=TRUNK_COLOR, width=8),
        marker=dict(size=3.5, color=TRUNK_COLOR), name="Trunk (training)",
        text=[t for t, k in zip(hover_text(TRUNK, trunk), upto) if k], hovertemplate="%{text}<extra></extra>"))
    explores = [n for n, c in colors.items() if c == EXPLORE_COLOR and n in proj and proj[n]["steps"]]
    for name, p in proj.items():
        if name == TRUNK or not p["steps"] or p.get("from_step", 0) > sel_step:
            continue
        c = colors.get(name, BRANCH_COLORS[0])
        is_exp = name in explores
        sizes = [3] * len(p["steps"])
        if is_exp:
            sizes[0] = 7  # where the exploration started
        fig.add_trace(go.Scatter3d(
            x=p["a"], y=p["b"], z=path_zs[name] + lift, mode="lines+markers", line=dict(color=c, width=5 if is_exp else 7),
            marker=dict(size=sizes, color=c, symbol="diamond" if is_exp else "circle"),
            name=f"Explorations ({len(explores)})" if is_exp else f"Branch {name}",
            legendgroup="explore" if is_exp else name, showlegend=(not is_exp) or name == explores[0],
            text=hover_text(name, p), hovertemplate="%{text}<extra></extra>"))
    k = int(np.searchsorted(ts, sel_step, side="right") - 1)
    fig.add_trace(go.Scatter3d(x=[ta[k]], y=[tb[k]], z=[tz[k]], mode="markers",
                               marker=dict(size=7, color=ACCENT, line=dict(color="#ffffff", width=3)),
                               name=f"Selected snapshot (step {int(ts[k])})", hoverinfo="skip"))
    axis = dict(showbackground=False, gridcolor="rgba(140,165,255,0.07)", zerolinecolor="rgba(140,165,255,0.18)",
                showspikes=False, tickfont=dict(color=MUTED, size=10), title_font=dict(color=MUTED, size=11),
                linecolor="rgba(140,165,255,0.25)")
    fig.update_layout(
        autosize=True, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=SURFACE_BG, font=FONT, template=None,
        hoverlabel=dict(bgcolor=PANEL, bordercolor="rgba(125,165,255,0.5)", font=dict(color=INK, size=12)),
        uirevision=info["mode"], legend=dict(x=0.012, y=0.985, bgcolor="rgba(6,10,22,0.45)",
                                             bordercolor="rgba(125,165,255,0.18)", borderwidth=1,
                                             font=dict(size=11, color=INK)),
        scene=dict(xaxis=dict(title="a (direction 1)", **axis), yaxis=dict(title="b (direction 2)", **axis),
                   zaxis=dict(title="loss", type="log" if zlog else "linear", **axis),
                   aspectmode="manual", aspectratio=dict(x=1.2, y=1.2, z=0.7),
                   camera=dict(eye=dict(x=0.95, y=-1.1, z=1.2), center=dict(x=0, y=0, z=-0.12))))
    return fig


def fidelity_figure(st: DashboardState, view: dict[str, Any], sel_step: int | None) -> go.Figure:
    if view["info"] is None or not view["proj"]:
        return _empty("No projection yet.", 230)
    colors = st.branch_colors()
    fig = go.Figure()
    for name, p in view["proj"].items():
        if not p["steps"]:
            continue
        x = np.array(p["steps"]) + (0 if name == TRUNK else p.get("from_step", 0))
        c = TRUNK_COLOR if name == TRUNK else colors.get(name, BRANCH_COLORS[0])
        fig.add_trace(go.Scatter(x=x, y=p["captured"], mode="lines+markers", name=name,
                                 line=dict(color=c, width=2), marker=dict(size=5, color=c),
                                 customdata=np.column_stack([p["steps"], p["residual"]]),
                                 hovertemplate=name + " step %{customdata[0]}<br>captured %{y:.1%}"
                                 "<br>residual norm %{customdata[1]:.3g}<extra></extra>"))
    if sel_step is not None:
        fig.add_vline(x=sel_step, line=dict(color=MUTED, width=1, dash="dot"))
    multi = len(fig.data) > 1
    fig.update_layout(height=400, margin=dict(l=48, r=10, t=44 if multi else 8, b=40),
                      paper_bgcolor=SURFACE_BG, plot_bgcolor=SURFACE_BG, font=FONT, showlegend=multi, template=None,
                      hoverlabel=dict(bgcolor=PANEL, bordercolor="rgba(125,165,255,0.5)", font=dict(color=INK)),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=10)),
                      xaxis=dict(title="step (branches offset by their start step)", gridcolor=GRID, zeroline=False, linecolor=GRID),
                      yaxis=dict(title="captured", range=[-0.03, 1.03], tickformat=".0%", gridcolor=GRID, zeroline=False, linecolor=GRID))
    return fig


def fidelity_panel(view: dict[str, Any], sel_step: int | None) -> list[Any]:
    info = view["info"]
    if info is None or TRUNK not in view.get("proj", {}):
        return [html.Div("No basis yet.", className="note")]
    proj = view["proj"]
    rows: list[tuple[str, str]] = [("Basis", info["id"]),
                                   ("Fit on", f"{len(info.get('fit_keys') or [])} snapshots" if info["mode"] != "random" else "random directions")]
    evr = info.get("explained_variance_ratio")
    if evr:
        rows.append(("Displacement variance on axes 1 + 2", f"{evr[0]:.1%} + {evr[1]:.1%}"))
    t = proj[TRUNK]
    if sel_step is not None and t["steps"]:
        k = int(np.searchsorted(t["steps"], sel_step, side="right") - 1)
        rows += [(f"Selected step {t['steps'][k]}: captured", f"{t['captured'][k]:.1%}"),
                 ("Residual norm / displacement norm", f"{t['residual'][k]:.3g} / {t['norm'][k]:.3g}")]
    low = []
    for name, p in proj.items():
        if p["steps"]:
            med, mn = float(np.median(p["captured"])), float(np.min(p["captured"]))
            rows.append((f"{name}: median / min captured", f"{med:.1%} / {mn:.1%}"))
            if med < 0.5:
                low.append(name)
    out: list[Any] = [html.Table([html.Tr([html.Td(k), html.Td(v)]) for k, v in rows], className="kv")]
    if low:
        out.append(html.Div(f"The plane captures less than half of the displacement for: {', '.join(low)}. "
                            "Positions and distances on the map are misleading there; rely on the full-space metrics.",
                            className="warn"))
    out.append(html.Div("Captured = share of a snapshot's squared distance from the map center that lies in the "
                        "2D plane. Residual norm is the part the map cannot show.", className="note",
                        style={"marginTop": "8px"}))
    return out


def fidelity_chip(view: dict[str, Any]) -> tuple[str, str]:
    """One-line fidelity summary shown on the map; turns amber when the picture is misleading."""
    proj = view.get("proj") or {}
    if view["info"] is None or TRUNK not in proj or not proj[TRUNK]["steps"]:
        return "Fidelity: no basis yet", "chip"
    med = {n: float(np.median(p["captured"])) for n, p in proj.items() if p["steps"]}
    worst = min(med, key=med.get)
    text = f"Plane captures {med[TRUNK]:.0%} of the trunk (median)"
    if worst != TRUNK:
        text += f" · lowest other path {med[worst]:.0%}"
    return text, "chip low" if med[worst] < 0.5 else "chip"


def metric_options(st: DashboardState) -> list[str]:
    seen: list[str] = []
    for b in st.run.branches():
        for name, m in b.metrics().items():
            for s in m["series"]:
                if f"{name} / {s}" not in seen:
                    seen.append(f"{name} / {s}")
    return seen


def metrics_figure(st: DashboardState, selected: list[str]) -> go.Figure:
    branches = st.run.branches()
    data = {b.name: b.metrics() for b in branches}
    selected = [s for s in (selected or []) if any(s.split(" / ")[0] in m for m in data.values())]
    if not selected:
        return _empty("No branch metrics yet. Run `tracer unlearn` to create a branch.", 260)
    colors = st.branch_colors()
    fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=[s.split(" / ")[1].replace("_", " ") for s in selected])
    for r, sel in enumerate(selected, start=1):
        metric, series = sel.split(" / ")
        for name, ms in data.items():
            if metric in ms and series in ms[metric]["series"]:
                fig.add_trace(go.Scatter(x=ms[metric]["branch_steps"], y=ms[metric]["series"][series],
                                         mode="lines+markers", name=name, legendgroup=name, showlegend=(r == 1),
                                         line=dict(color=colors.get(name), width=2), marker=dict(size=5),
                                         hovertemplate=name + " step %{x}<br>%{y:.4g}<extra></extra>"), row=r, col=1)
        fig.update_yaxes(gridcolor=GRID, zeroline=False, row=r, col=1)
        fig.update_xaxes(gridcolor=GRID, zeroline=False, row=r, col=1)
    fig.update_xaxes(title_text="branch step", row=len(selected), col=1)
    fig.update_annotations(font=dict(size=12, color=INK), x=0, xanchor="left")
    fig.update_layout(height=150 + 150 * len(selected), margin=dict(l=48, r=10, t=30, b=40), paper_bgcolor=SURFACE_BG,
                      plot_bgcolor=SURFACE_BG, font=FONT, showlegend=True, template=None,
                      hoverlabel=dict(bgcolor=PANEL, bordercolor="rgba(125,165,255,0.5)", font=dict(color=INK)),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=1, xanchor="right", font=dict(size=10)))
    return fig


# -------------------------------------------------------------------------- app
class _Runs:
    """The runs under one root folder and which one the dashboard shows (one DashboardState each)."""

    def __init__(self, path: str | Path, device: str):
        path = Path(path)
        if (path / "meta.json").exists():
            self.root, first = path.parent, path.name
        else:
            self.root = path
            self.root.mkdir(parents=True, exist_ok=True)
            runs = list_runs(self.root)
            first = runs[0]["name"] if runs else None
        self.device = device
        self.states: dict[str, DashboardState] = {}
        self.name: str | None = None
        if first:
            self.switch(first)

    def switch(self, name: str) -> None:
        if name not in self.states:
            self.states[name] = DashboardState(self.root / name, self.device)
        self.name = name

    @property
    def st(self) -> DashboardState:
        if self.name is None:
            raise LookupError("no run selected")
        return self.states[self.name]


class _StateProxy:
    """Stands in for the current run's DashboardState inside callbacks, so switching runs needs no rewiring."""

    def __init__(self, runs: _Runs):
        self._runs = runs

    def __getattr__(self, k: str) -> Any:
        return getattr(self._runs.st, k)


def _run_options(runs: _Runs) -> list[dict[str, str]]:
    out = []
    for r in list_runs(runs.root):
        title = PRESETS[r["demo"]].title if r["demo"] in PRESETS else None
        label = r["name"] + (f" · {title}" if title and title.lower() != r["name"] else "")
        out.append({"label": label + ("" if r["finished"] else " · training"), "value": r["name"]})
    return out


def _new_run_drawer() -> html.Div:
    presets = list_presets()
    return html.Div([
        html.Div([html.H2("New run"), html.Button("Close", id="newrun-close", className="ghost", n_clicks=0)],
                 className="drawer-head"),
        html.Div("Train a ready-made dataset and model in the background and watch its map form. "
                 "Each run is saved next to the others and stays in the run menu.", className="note"),
        html.H3("Setup"),
        dcc.RadioItems(id="newrun-preset", value=presets[0].name, className="presets", options=[
            {"value": p.name, "label": html.Div([
                html.Div(p.title, className="preset-title"),
                html.Div(p.description, className="note"),
                html.Div(" · ".join(x for x in (p.cost, (f"downloads {p.download} first" if p.download
                                                          and not p.data_ready() else "")) if x),
                         className="preset-cost")])} for p in presets],
            labelStyle={"display": "flex", "gap": "8px", "alignItems": "flex-start", "margin": "0 0 12px"}),
        html.H3("Options"),
        dcc.Checklist(id="newrun-opts", value=["watch", "extras"], options=[
            {"label": " Slow training down so the map forms while you watch", "value": "watch"},
            {"label": " Afterwards add an unlearning branch and two explorations", "value": "extras"}],
            labelStyle={"display": "block", "margin": "3px 0"}),
        html.Label("Run name (optional)", className="ctl"),
        dcc.Input(id="newrun-name", type="text", placeholder="defaults to the setup name",
                  style={"width": "100%", "boxSizing": "border-box", "padding": "6px"}),
        html.Button("Train", id="newrun-go", className="btn", n_clicks=0),
        html.Div(id="newrun-status", className="note", style={"marginTop": "10px", "wordBreak": "break-word"}),
    ], id="newrun", className="drawer wide")


def build_app(run_dir: str | Path = "runs", interval_ms: int = 2000, device: str = "cpu",
              start_worker: bool = True) -> Dash:
    runs = _Runs(run_dir, device)
    st = _StateProxy(runs)
    app = Dash(__name__, title="trajmap", update_title=None, suppress_callback_exceptions=True)
    app.index_string = app.index_string.replace("{%css%}", "{%css%}<style>" + CSS + "</style>")
    exports: dict[str, Any] = {}

    def top_bar(*extra: Any) -> html.Div:
        return html.Div([
            html.H1([html.Span("trajmap /")]),
            dcc.Dropdown(id="run-select", value=runs.name, options=_run_options(runs), clearable=False,
                         searchable=False, placeholder="no runs yet", className="run-select"),
            html.Button("New run", id="newrun-btn", className="ghost", n_clicks=0),
            *extra,
        ], className="top")

    def chrome() -> list[Any]:
        """Pieces shared by the run layout and the empty one."""
        return [dcc.Interval(id="clock", interval=1500), dcc.Store(id="pending"), dcc.Store(id="reload"),
                _new_run_drawer()]

    def layout() -> html.Div:
        if runs.name is None:
            return html.Div([*chrome(), top_bar(html.Div(className="status")), html.Div([
                html.Div([html.H3("No runs yet"),
                          html.Div("Start one with New run (top right), or record your own training loop with "
                                   "Tracer and point the dashboard at its folder.", className="note")],
                         className="card", style={"maxWidth": "520px", "margin": "60px auto"})], className="pages")])
        ctl = st.run.control()
        opts = metric_options(st)
        default_series = [o for o in opts if o in ("nearest_trunk / nearest_trunk_step", "off_path / off_path_distance")]
        settings = html.Div([
            html.Div([html.H2("Settings"), html.Button("Close", id="drawer-close", className="ghost", n_clicks=0)],
                     className="drawer-head"),
            html.H3("Projection basis"),
            dcc.RadioItems(id="mode", value=ctl["mode"], options=[{"label": MODE_LABELS[m], "value": m} for m in MODES],
                           labelStyle={"display": "block", "margin": "3px 0"}),
            html.Div("Each basis is its own coordinate system with its own cached surface.", className="note"),
            html.H3("Accuracy vs. latency"),
            html.Label("Finest grid (points per axis)", className="ctl"),
            dcc.Dropdown(id="resolution", value=int(ctl["resolution"]), clearable=False,
                         options=[{"label": f"{r} × {r}", "value": r} for r in RESOLUTIONS]),
            html.Label("Eval batches per grid point", className="ctl"),
            dcc.Dropdown(id="eval-batches", value=int(ctl["eval_batches"]), clearable=False,
                         options=[{"label": str(n), "value": n} for n in (1, 2, 4, 8)] + [{"label": "whole eval subset", "value": 0}]),
            html.Label("Refinement budget (grid points per basis)", className="ctl"),
            dcc.Input(id="budget", type="number", value=int(ctl["budget"]), min=50, max=20000, step=50,
                      debounce=True, style={"width": "100%", "boxSizing": "border-box", "padding": "6px"}),
            html.Div("Changing eval batches re-evaluates the surface; resolution and budget keep what is cached.",
                     className="note"),
            html.H3("Display"),
            dcc.Checklist(id="display", value=["clip", "follow"], options=[
                {"label": " Log loss axis", "value": "zlog"},
                {"label": " Clip tall walls (2× max path loss)", "value": "clip"},
                {"label": " Follow the latest snapshot", "value": "follow"}],
                labelStyle={"display": "block", "margin": "3px 0"}),
        ], id="drawer", className="drawer")

        page_map = html.Div([
            html.Div([
                dcc.Graph(id="map", figure=_empty("Loading…", None), responsive=True,
                          style={"height": "calc(100vh - 150px)", "minHeight": "440px"},
                          config={"displaylogo": False, "displayModeBar": False}),
                html.Button(id="fidelity-chip", className="chip", n_clicks=0, title="Open projection fidelity"),
                html.Div([
                    html.Div([html.Span("In-plane approximation", className="tag"),
                              html.Button("Close", id="popup-close", className="ghost", n_clicks=0)], className="drawer-head"),
                    html.Div(id="click-panel"),
                    html.Button("Export state_dict (.pt)", id="export", className="btn", n_clicks=0, style={"display": "none"}),
                ], id="popup", className="popup"),
                html.Div([html.Div([html.Label(id="step-label", className="ctl", style={"margin": "0"}),
                                    html.Span("Click the surface to evaluate a point · hover a snapshot for details",
                                              className="note")], className="drawer-head", style={"marginBottom": "2px"}),
                          dcc.Slider(id="step", min=0, max=1, step=1, value=1, marks=None, updatemode="drag")],
                         className="below"),
            ], className="card map-card"),
        ], id="page-map")

        page_fidelity = html.Div([
            html.Div([
                html.Div([html.H3("Captured by the plane, per snapshot"),
                          dcc.Graph(id="fidelity", figure=_empty("…", 380), responsive=True, config={"displayModeBar": False})],
                         className="card"),
                html.Div([html.H3("Numbers"), html.Div(id="fidelity-text")], className="card"),
            ], className="two-col"),
        ], id="page-fidelity", style={"display": "none"})

        page_metrics = html.Div([
            html.Div([
                html.H3("Branch metrics (full parameter space)"),
                dcc.Dropdown(id="series", multi=True, value=default_series, options=opts, placeholder="Metric series to plot"),
                dcc.Graph(id="metrics", figure=_empty("…", 260), responsive=True, config={"displayModeBar": False}),
            ], className="card"),
        ], id="page-metrics", style={"display": "none"})

        return html.Div([
            dcc.Interval(id="tick", interval=interval_ms),
            dcc.Store(id="sig"), dcc.Store(id="ctl-sink"), dcc.Store(id="clicked"), dcc.Download(id="download"),
            *chrome(),
            top_bar(html.Div([html.Button("Map", id="nav-map", className="nav on", n_clicks=0),
                              html.Button("Fidelity", id="nav-fidelity", className="nav", n_clicks=0),
                              html.Button("Branch metrics", id="nav-metrics", className="nav", n_clicks=0)],
                             className="navs"),
                    html.Div(id="status", className="status"),
                    html.Button("Improve map", id="improve-btn", className="ghost", n_clicks=0),
                    html.Button("Settings", id="settings-btn", className="ghost", n_clicks=0)),
            settings,
            html.Div([
                html.Div([html.H2("Improve map"), html.Button("Close", id="improve-close", className="ghost", n_clicks=0)],
                         className="drawer-head"),
                html.Div("Trains from new start points far from every recorded path, down to a minimum, and adds "
                         "each descent to the map. This costs training compute; choose how much.", className="note"),
                html.H3("How much"),
                dcc.RadioItems(id="improve-mode", value="one", options=[
                    {"label": "One exploration", "value": "one"},
                    {"label": "Auto: until the map is covered", "value": "auto"},
                    {"label": "For a fixed time", "value": "time"}],
                    labelStyle={"display": "block", "margin": "3px 0"}),
                html.Div([html.Div([html.Label("Minutes (fixed time)", className="ctl"),
                                    dcc.Input(id="improve-minutes", type="number", value=2, min=0.5, max=600, step=0.5,
                                              style={"width": "100%", "boxSizing": "border-box", "padding": "6px"})]),
                          html.Div([html.Label("Max explorations (auto)", className="ctl"),
                                    dcc.Input(id="improve-max", type="number", value=6, min=1, max=50, step=1,
                                              style={"width": "100%", "boxSizing": "border-box", "padding": "6px"})])],
                         style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"}),
                html.H3("Start from"),
                dcc.RadioItems(id="improve-start", value="plane", options=[
                    {"label": "A point on the map, far from all paths", "value": "plane"},
                    {"label": "A fresh random initialization", "value": "init"}],
                    labelStyle={"display": "block", "margin": "3px 0"}),
                html.Div([html.Button("Start", id="improve-go", className="btn", n_clicks=0),
                          html.Button("Stop", id="improve-stop", className="ghost", n_clicks=0)],
                         style={"display": "flex", "gap": "8px", "alignItems": "end", "marginTop": "4px"}),
                html.Div(id="improve-status", className="note", style={"marginTop": "10px"}),
            ], id="improve", className="drawer"),
            html.Div([page_map, page_fidelity, page_metrics], className="pages"),
        ])

    app.layout = layout

    @app.callback(Output("ctl-sink", "data"), Input("mode", "value"), Input("resolution", "value"),
                  Input("eval-batches", "value"), Input("budget", "value"))
    def write_control(mode, resolution, eval_batches, budget):
        if None in (mode, resolution, eval_batches, budget):
            return no_update
        st.run.set_control(mode=mode, resolution=int(resolution), eval_batches=int(eval_batches), budget=int(budget))
        if start_worker:
            ensure_worker(st.run.dir)
        return time.time()

    @app.callback(Output("step", "max"), Output("step", "value"), Output("series", "options"), Output("series", "value"),
                  Input("tick", "n_intervals"), Input("display", "value"), State("step", "value"), State("step", "max"),
                  State("series", "options"), State("series", "value"))
    def heartbeat(_, display, value, old_max, old_opts, old_series):
        atomic_write_json(st.run.dir / "dashboard.json", {"heartbeat": time.time()})
        if start_worker:
            ensure_worker(st.run.dir)
        n = max(len(st.run.trunk.steps()) - 1, 1)
        at_end = value is None or value >= (old_max or 0)  # scrubbing back pauses following
        new_value = n if (("follow" in (display or []) and at_end) or value is None or value > n) else value
        opts = metric_options(st)
        series = no_update
        if opts != old_opts and not old_series:
            series = [o for o in opts if o in ("nearest_trunk / nearest_trunk_step", "off_path / off_path_distance")]
        return (n if n != old_max else no_update, new_value if new_value != value else no_update,
                opts if opts != old_opts else no_update, series)

    @app.callback(Output("map", "figure"), Output("fidelity", "figure"), Output("fidelity-text", "children"),
                  Output("metrics", "figure"), Output("status", "children"), Output("step-label", "children"),
                  Output("fidelity-chip", "children"), Output("fidelity-chip", "className"), Output("sig", "data"),
                  Input("tick", "n_intervals"), Input("step", "value"), Input("mode", "value"),
                  Input("resolution", "value"), Input("display", "value"), Input("series", "value"),
                  Input("ctl-sink", "data"), State("sig", "data"))
    def refresh(_, step_idx, mode, resolution, display, series, _ctl, old_sig):
        display = display or []
        view = st.view(mode)
        steps = st.run.trunk.steps()
        sel = steps[min(int(step_idx or 0), len(steps) - 1)] if steps else None
        info = view["info"]
        ws = worker_status(st.run.dir)
        ctl = st.run.control()
        n_pts = len(view.get("points", {}))
        kinds = [t.kind for t in st.run.branches()]
        n_exp = kinds.count("explore")
        parts = [f"Training {'finished' if st.run.trunk.finished else 'in progress'}",
                 f"{len(steps)} snapshots", f"{len(kinds) - n_exp} branches",
                 *([f"{n_exp} explorations"] if n_exp else []),
                 f"{n_pts} / {ctl['budget']} grid points" if info else "no basis yet",
                 f"worker {ws.get('state', 'not running') if ws['alive'] else 'not running'}"]
        if ws.get("error"):
            parts.append(f"landscape disabled: {ws['error']}")
        status = " · ".join(parts)
        label = f"Trunk snapshot {min(int(step_idx or 0), max(len(steps) - 1, 0)) + 1} of {len(steps)}" + (f", step {sel}" if sel is not None else "")
        metric_sig = [[b.name, sorted(b.metrics())] for b in st.run.branches()]
        sig = [info["id"] if info else None, n_pts, {k: len(v["steps"]) for k, v in view.get("proj", {}).items()},
               sel, mode, resolution, sorted(display), series, metric_sig]
        if sig == old_sig:
            return no_update, no_update, no_update, no_update, status, label, no_update, no_update, no_update
        level = RESOLUTIONS.get(int(resolution or 33), 2)
        return (map_figure(st, view, sel, level, "zlog" in display, "clip" in display),
                fidelity_figure(st, view, sel), fidelity_panel(view, sel), metrics_figure(st, series or []),
                status, label, *fidelity_chip(view), sig)

    @app.callback(Output("click-panel", "children"), Output("export", "style"), Output("clicked", "data"),
                  Input("map", "clickData"), State("mode", "value"), prevent_initial_call=True)
    def on_click(click, mode):
        if not click or not click.get("points"):
            return no_update, no_update, no_update
        pt = click["points"][0]
        info = st.bm.latest_info(mode)
        if info is None or "x" not in pt:
            return no_update, no_update, no_update
        a, b = float(pt["x"]), float(pt["y"])
        try:
            res, sd = st.in_plane(info["id"], a, b)
        except Exception as e:
            return html.Div(f"Could not evaluate this point: {e}", className="warn"), {"display": "none"}, None
        exports.clear()
        exports[f"{info['id']}:{a}:{b}"] = sd
        rows = [("a, b", f"{a:.4g}, {b:.4g}"), ("Basis", info["id"]), ("Loss", _fmt(res["loss"], 5)),
                ("Accuracy", "n/a" if res["acc"] is None else f"{res['acc']:.2%}"),
                ("Eval samples", str(res["n"])), ("BatchNorm statistics", res["bn_mode"])]
        if pt.get("z") is not None:
            rows.append(("Surface value at click", _fmt(float(pt["z"]), 5)))
        panel = [html.Table([html.Tr([html.Td(k), html.Td(v)]) for k, v in rows], className="kv", style={"marginTop": "8px"}),
                 html.Div("These weights are center + a·d1 + b·d2, reconstructed from the 2D basis. They are not a "
                          "recorded snapshot, and a snapshot drawn at this spot generally has different weights and loss.",
                          className="note", style={"marginTop": "8px"}),
]
        return panel, {"display": "inline-block"}, {"key": f"{info['id']}:{a}:{b}", "a": a, "b": b, "basis": info["id"]}

    @app.callback(Output("popup", "className"), Input("click-panel", "children"), Input("popup-close", "n_clicks"),
                  prevent_initial_call=True)
    def toggle_popup(children, _close):
        return "popup" if ctx.triggered_id == "popup-close" or not children else "popup open"

    @app.callback(Output("improve", "className"), Input("improve-btn", "n_clicks"), Input("improve-close", "n_clicks"),
                  State("improve", "className"), prevent_initial_call=True)
    def toggle_improve(_open, _close, cls):
        return "drawer" if ctx.triggered_id == "improve-close" or "open" in (cls or "") else "drawer open"

    @app.callback(Output("improve-status", "children"), Input("tick", "n_intervals"), Input("improve-go", "n_clicks"),
                  Input("improve-stop", "n_clicks"), State("improve-mode", "value"), State("improve-minutes", "value"),
                  State("improve-max", "value"), State("improve-start", "value"))
    def improve(_tick, _go, _stop, mode, minutes, max_runs, start):
        from .explore import improve_status, request_stop, start_improve
        if ctx.triggered_id == "improve-go":
            start_improve(st.run.dir, mode, float(minutes or 2) if mode == "time" else None, int(max_runs or 6),
                          start, device)
        elif ctx.triggered_id == "improve-stop" and improve_status(st.run.dir)["alive"]:
            request_stop(st.run.dir)
        s = improve_status(st.run.dir)
        n_exp = sum(t.kind == "explore" for t in st.run.branches())
        if s["alive"]:
            cur = s.get("current")
            left = f" · {max(0, s['deadline'] - time.time()) / 60:.1f} min left" if s.get("deadline") else ""
            doing = f"{cur['name']} at step {cur['step']}, loss {cur['loss']:.3g}" if cur else "choosing a start point"
            return f"Running ({s.get('mode')}): {doing} · {s.get('done', 0)} finished this session{left}"
        tail = f" Last session: {s['result']}." if s.get("result") else ""
        return f"Idle. {n_exp} exploration{'s' if n_exp != 1 else ''} on the map.{tail}"

    @app.callback(Output("drawer", "className"), Input("settings-btn", "n_clicks"), Input("drawer-close", "n_clicks"),
                  State("drawer", "className"), prevent_initial_call=True)
    def toggle_drawer(_open, _close, cls):
        return "drawer" if ctx.triggered_id == "drawer-close" or "open" in (cls or "") else "drawer open"

    @app.callback(Output("page-map", "style"), Output("page-fidelity", "style"), Output("page-metrics", "style"),
                  Output("nav-map", "className"), Output("nav-fidelity", "className"), Output("nav-metrics", "className"),
                  Input("nav-map", "n_clicks"), Input("nav-fidelity", "n_clicks"), Input("nav-metrics", "n_clicks"),
                  Input("fidelity-chip", "n_clicks"), prevent_initial_call=True)
    def navigate(*_):
        page = {"nav-fidelity": "fidelity", "fidelity-chip": "fidelity", "nav-metrics": "metrics"}.get(ctx.triggered_id, "map")
        shown = [{"display": "block" if page == p else "none"} for p in ("map", "fidelity", "metrics")]
        return (*shown, *("nav on" if page == p else "nav" for p in ("map", "fidelity", "metrics")))

    # Plots on a page that was hidden when they rendered need a resize once the page is shown.
    app.clientside_callback(
        "function(_s){setTimeout(function(){window.dispatchEvent(new Event('resize'));},60);"
        "return window.dash_clientside.no_update;}",
        Output("page-map", "title"), Input("page-map", "style"), prevent_initial_call=True)

    # ---- runs: switch, list, start new ones
    @app.callback(Output("reload", "data", allow_duplicate=True), Input("run-select", "value"), prevent_initial_call=True)
    def select_run(name):
        if not name or name == runs.name or not (runs.root / name / "meta.json").exists():
            return no_update
        runs.switch(name)
        exports.clear()
        return time.time()

    app.clientside_callback("function(r){ if (r) { window.location.reload(); } return window.dash_clientside.no_update; }",
                            Output("run-select", "title"), Input("reload", "data"), prevent_initial_call=True)

    @app.callback(Output("newrun", "className"), Input("newrun-btn", "n_clicks"), Input("newrun-close", "n_clicks"),
                  State("newrun", "className"), prevent_initial_call=True)
    def toggle_newrun(_open, _close, cls):
        return "drawer wide" if ctx.triggered_id == "newrun-close" or "open" in (cls or "") else "drawer wide open"

    @app.callback(Output("pending", "data"), Output("newrun-status", "children"), Output("run-select", "options"),
                  Output("reload", "data", allow_duplicate=True),
                  Input("clock", "n_intervals"), Input("newrun-go", "n_clicks"),
                  State("newrun-preset", "value"), State("newrun-opts", "value"), State("newrun-name", "value"),
                  State("pending", "data"), State("run-select", "options"), prevent_initial_call=True)
    def new_run(_tick, _go, preset, opts, name, pending, old_opts):
        opts_now = _run_options(runs)
        options = opts_now if opts_now != old_opts else no_update
        if ctx.triggered_id == "newrun-go" and preset:
            base = "".join(c for c in (name or preset).strip() if c.isalnum() or c in "-_.") or preset
            name = unique_run_name(runs.root, base)
            log = start_demo(preset, runs.root / name, watch="watch" in (opts or []),
                             extras="extras" in (opts or []), device=device)
            return {"name": name, "log": str(log), "t": time.time()}, f"Starting {name}…", options, no_update
        if not pending:
            return no_update, no_update, options, no_update
        if (runs.root / pending["name"] / "trunk" / "trajectory.json").exists():
            runs.switch(pending["name"])
            exports.clear()
            return None, f"Started {pending['name']}.", options, time.time()
        try:
            tail = Path(pending["log"]).read_text()[-4000:]
        except OSError:
            tail = ""
        if "Traceback" in tail:
            return None, html.Pre(tail[-1200:], className="warn", style={"whiteSpace": "pre-wrap"}), options, no_update
        pct = tail.rsplit("%", 1)[0].rsplit("%", 1)[-1][-5:] if tail.rstrip().endswith("%") else None
        msg = f"Downloading data… {pct.strip()}%" if pct else "Loading data and building the model…"
        return no_update, f"Starting {pending['name']}: {msg} ({time.time() - pending['t']:.0f}s)", options, no_update

    @app.callback(Output("download", "data"), Input("export", "n_clicks"), State("clicked", "data"),
                  prevent_initial_call=True)
    def export(n, clicked):
        if not n or not clicked or clicked["key"] not in exports:
            return no_update
        name = f"inplane_{clicked['basis']}_a{clicked['a']:.4g}_b{clicked['b']:.4g}.pt"
        out = st.run.dir / "exports"
        out.mkdir(exist_ok=True)
        torch.save(exports[clicked["key"]], out / name)
        buf = io.BytesIO()
        torch.save(exports[clicked["key"]], buf)
        return dcc.send_bytes(buf.getvalue(), name)

    return app


def serve(run_dir: str | Path = "runs", host: str = "127.0.0.1", port: int = 8050, interval_ms: int = 2000,
          device: str = "cpu", debug: bool = False) -> None:
    app = build_app(run_dir, interval_ms=interval_ms, device=device)
    print(f"trajmap dashboard for {run_dir}: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
