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
from dash import Dash, Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots
from scipy.ndimage import binary_dilation

from .basis import MODES, BasisManager
from .landscape import RESOLUTIONS, Evaluator, GridCache, SurfaceModel, grid_bounds, unit
from .store import TRUNK, Run, atomic_write_json
from .worker import ensure_worker, worker_status

INK, MUTED, GRID, SURFACE_BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BRANCH_COLORS = ["#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#eda100", "#008300", "#e34948"]
LOSS_SCALE = [[0.0, "#e3eefb"], [0.5, "#6ba6e6"], [1.0, "#12407c"]]
GRAY_SCALE = [[0.0, "#d9d8d3"], [1.0, "#a9a8a2"]]
MODE_LABELS = {"pca_trunk": "PCA, trunk only", "pca_joint": "PCA, trunk + branches",
               "random": "Random, filter-normalized"}
FONT = dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif", size=12, color=INK)

CSS = """
body { margin:0; background:#f4f3f0; color:#0b0b0b; font:14px -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; }
.wrap { display:grid; grid-template-columns:270px minmax(0,1fr) 360px; gap:12px; padding:12px; align-items:start; }
.card { background:#fcfcfb; border:1px solid #e4e3df; border-radius:8px; padding:12px 14px; }
.card h3 { margin:0 0 8px; font-size:13px; font-weight:600; }
.card + .card { margin-top:12px; }
.top { padding:12px 16px 0; display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
.top h1 { font-size:16px; margin:0; font-weight:600; }
.status { color:#52514e; font-size:12.5px; }
label.ctl { display:block; font-size:12px; color:#52514e; margin:12px 0 4px; }
.note { font-size:12px; color:#52514e; line-height:1.45; }
.warn { background:#fdf1e3; border:1px solid #f0c98f; border-radius:6px; padding:8px 10px; font-size:12.5px; margin-top:8px; }
.tag { display:inline-block; background:#12407c; color:#fff; border-radius:4px; padding:2px 7px; font-size:11px; font-weight:600; letter-spacing:.03em; }
table.kv { border-collapse:collapse; width:100%; font-size:12.5px; }
table.kv td { padding:3px 0; } table.kv td:last-child { text-align:right; font-variant-numeric:tabular-nums; }
button.btn { margin-top:10px; background:#0b0b0b; color:#fff; border:0; border-radius:6px; padding:7px 12px; font-size:12.5px; cursor:pointer; }
@media (max-width:1250px) { .wrap { grid-template-columns:250px minmax(0,1fr); } .right { grid-column:1 / -1; } }
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
        return {t.name: BRANCH_COLORS[i % len(BRANCH_COLORS)] for i, t in enumerate(bs)}

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
    fig.update_layout(height=height, paper_bgcolor=SURFACE_BG, plot_bgcolor=SURFACE_BG, font=FONT,
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
        lift = 0.012 * (zhi - zlo)
        sup = mesh["supported"]
        hover = "a %{x:.3g}, b %{y:.3g}<br>loss %{z:.4g}<extra>" 
        fig.add_trace(go.Surface(
            x=mesh["a"], y=mesh["b"], z=np.where(sup, z, np.nan), colorscale=LOSS_SCALE, cmin=zlo, cmax=zhi,
            name="Surface, evaluated", showlegend=True, hovertemplate=hover + "near an evaluated point</extra>",
            colorbar=dict(title=dict(text="loss", side="right"), thickness=10, len=0.55, x=1.0, outlinewidth=0),
            lighting=dict(ambient=0.75, diffuse=0.5, specular=0.05, roughness=0.9)))
        interp = binary_dilation(~sup)
        if interp.any():
            fig.add_trace(go.Surface(
                x=mesh["a"], y=mesh["b"], z=np.where(interp, z, np.nan), colorscale=GRAY_SCALE, showscale=False,
                opacity=0.55, name="Surface, interpolated", showlegend=True,
                hovertemplate=hover + "interpolated, not evaluated</extra>",
                lighting=dict(ambient=0.9, diffuse=0.2, specular=0.0)))
        u = unit(info["coarse_h"])
        keys = [k for k, v in points.items() if v[0] is not None]
        pz = np.array([points[k][0] for k in keys])
        if zcap is not None:
            pz = np.minimum(pz, zcap)
        fig.add_trace(go.Scatter3d(
            x=[k[0] * u for k in keys], y=[k[1] * u for k in keys], z=pz, mode="markers",
            marker=dict(size=1.6, color=MUTED, opacity=0.7), name=f"Evaluated grid points ({len(keys)})",
            hovertemplate="a %{x:.3g}, b %{y:.3g}<br>loss %{z:.4g}<extra>evaluated</extra>"))

    def hover_text(name: str, p: dict[str, Any]) -> list[str]:
        return [f"{name} step {s}<br>recorded loss {_fmt(recs[name].get(s))}"
                f"<br>captured by plane {c:.1%}<br>residual norm {r:.3g}"
                for s, c, r in zip(p["steps"], p["captured"], p["residual"])]

    tz = path_zs[TRUNK] + lift
    ta, tb, ts = np.array(trunk["a"]), np.array(trunk["b"]), np.array(trunk["steps"])
    upto = ts <= sel_step
    if (~upto).any():
        fig.add_trace(go.Scatter3d(x=ta, y=tb, z=tz, mode="lines", line=dict(color="rgba(11,11,11,0.25)", width=3),
                                   name="Trunk (after selected step)", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=ta[upto], y=tb[upto], z=tz[upto], mode="lines+markers", line=dict(color=INK, width=5),
        marker=dict(size=3, color=INK), name="Trunk (training)",
        text=[t for t, k in zip(hover_text(TRUNK, trunk), upto) if k], hovertemplate="%{text}<extra></extra>"))
    for name, p in proj.items():
        if name == TRUNK or not p["steps"] or p.get("from_step", 0) > sel_step:
            continue
        c = colors.get(name, BRANCH_COLORS[0])
        fig.add_trace(go.Scatter3d(
            x=p["a"], y=p["b"], z=path_zs[name] + lift, mode="lines+markers", line=dict(color=c, width=5),
            marker=dict(size=3, color=c), name=f"Branch {name}", text=hover_text(name, p),
            hovertemplate="%{text}<extra></extra>"))
    k = int(np.searchsorted(ts, sel_step, side="right") - 1)
    fig.add_trace(go.Scatter3d(x=[ta[k]], y=[tb[k]], z=[tz[k]], mode="markers",
                               marker=dict(size=7, color="#ffffff", line=dict(color=INK, width=3)),
                               name=f"Selected snapshot (step {int(ts[k])})", hoverinfo="skip"))
    axis = dict(backgroundcolor=SURFACE_BG, gridcolor=GRID, zerolinecolor=GRID, showspikes=False)
    fig.update_layout(
        height=640, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=SURFACE_BG, font=FONT,
        uirevision=info["mode"], legend=dict(x=0.01, y=0.99, bgcolor="rgba(252,252,251,0.85)", font=dict(size=11)),
        scene=dict(xaxis=dict(title="a (direction 1)", **axis), yaxis=dict(title="b (direction 2)", **axis),
                   zaxis=dict(title="loss", type="log" if zlog else "linear", **axis),
                   aspectmode="manual", aspectratio=dict(x=1.2, y=1.2, z=0.7),
                   camera=dict(eye=dict(x=1.35, y=-1.45, z=0.95))))
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
        c = INK if name == TRUNK else colors.get(name, BRANCH_COLORS[0])
        fig.add_trace(go.Scatter(x=x, y=p["captured"], mode="lines+markers", name=name,
                                 line=dict(color=c, width=2), marker=dict(size=5, color=c),
                                 customdata=np.column_stack([p["steps"], p["residual"]]),
                                 hovertemplate=name + " step %{customdata[0]}<br>captured %{y:.1%}"
                                 "<br>residual norm %{customdata[1]:.3g}<extra></extra>"))
    if sel_step is not None:
        fig.add_vline(x=sel_step, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(height=230, margin=dict(l=44, r=10, t=8, b=36), paper_bgcolor=SURFACE_BG, plot_bgcolor=SURFACE_BG,
                      font=FONT, showlegend=len(fig.data) > 1, legend=dict(orientation="h", y=-0.28, font=dict(size=10)),
                      xaxis=dict(title="step (branches offset by their start step)", gridcolor=GRID, zeroline=False),
                      yaxis=dict(title="captured", range=[-0.03, 1.03], tickformat=".0%", gridcolor=GRID, zeroline=False))
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
                      plot_bgcolor=SURFACE_BG, font=FONT, showlegend=True,
                      legend=dict(orientation="h", y=-0.18 if len(selected) > 1 else -0.35, font=dict(size=10)))
    return fig


# -------------------------------------------------------------------------- app
def build_app(run_dir: str | Path, interval_ms: int = 2000, device: str = "cpu", start_worker: bool = True) -> Dash:
    st = DashboardState(run_dir, device)
    app = Dash(__name__, title=f"trajmap · {Path(run_dir).name}", update_title=None)
    app.index_string = app.index_string.replace("{%css%}", "{%css%}<style>" + CSS + "</style>")
    exports: dict[str, Any] = {}

    def layout() -> html.Div:
        ctl = st.run.control()
        opts = metric_options(st)
        default_series = [o for o in opts if o in ("nearest_trunk / nearest_trunk_step", "off_path / off_path_distance")]
        return html.Div([
            dcc.Interval(id="tick", interval=interval_ms),
            dcc.Store(id="sig"), dcc.Store(id="ctl-sink"), dcc.Store(id="clicked"), dcc.Download(id="download"),
            html.Div([html.H1(f"Run {Path(run_dir).name}"), html.Div(id="status", className="status")], className="top"),
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Projection basis"),
                        dcc.RadioItems(id="mode", value=ctl["mode"], options=[{"label": MODE_LABELS[m], "value": m} for m in MODES],
                                       labelStyle={"display": "block", "margin": "3px 0"}),
                        html.Div("Trunk-only and joint are separate coordinate systems, each with its own cached surface.",
                                 className="note", style={"marginTop": "6px"}),
                    ], className="card"),
                    html.Div([
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
                                 className="note", style={"marginTop": "8px"}),
                    ], className="card"),
                    html.Div([
                        html.H3("Display"),
                        dcc.Checklist(id="display", value=["clip", "follow"], options=[
                            {"label": " Log loss axis", "value": "zlog"},
                            {"label": " Clip tall walls (2× max path loss)", "value": "clip"},
                            {"label": " Follow the latest snapshot", "value": "follow"}],
                            labelStyle={"display": "block", "margin": "3px 0"}),
                    ], className="card"),
                ]),
                html.Div([
                    html.Div([
                        dcc.Graph(id="map", figure=_empty("Loading…"), config={"displaylogo": False}),
                        html.Div([html.Label(id="step-label", className="ctl", style={"margin": "4px 0"}),
                                  dcc.Slider(id="step", min=0, max=1, step=1, value=1, marks=None, updatemode="drag")]),
                        html.Div("Paths are drawn on the interpolated surface at each snapshot's in-plane position. "
                                 "Hover a snapshot for its recorded loss and how much of it the plane captures. "
                                 "Click anywhere on the surface to evaluate the weights at that point.",
                                 className="note", style={"marginTop": "6px"}),
                    ], className="card"),
                    html.Div([
                        html.H3("Branch metrics (full parameter space)"),
                        dcc.Dropdown(id="series", multi=True, value=default_series, options=opts,
                                     placeholder="Metric series to plot"),
                        dcc.Graph(id="metrics", figure=_empty("…", 260), config={"displayModeBar": False}),
                    ], className="card"),
                ]),
                html.Div([
                    html.Div([html.H3("Projection fidelity"),
                              dcc.Graph(id="fidelity", figure=_empty("…", 230), config={"displayModeBar": False}),
                              html.Div(id="fidelity-text")], className="card"),
                    html.Div([html.H3("Clicked point"),
                              html.Div(id="click-panel", children=html.Div("Click a point on the surface.", className="note"))],
                             className="card"),
                ], className="right"),
            ], className="wrap"),
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
        new_value = n if ("follow" in (display or []) or value is None or value > n) else value
        opts = metric_options(st)
        series = no_update
        if opts != old_opts and not old_series:
            series = [o for o in opts if o in ("nearest_trunk / nearest_trunk_step", "off_path / off_path_distance")]
        return (n if n != old_max else no_update, new_value if new_value != value else no_update,
                opts if opts != old_opts else no_update, series)

    @app.callback(Output("map", "figure"), Output("fidelity", "figure"), Output("fidelity-text", "children"),
                  Output("metrics", "figure"), Output("status", "children"), Output("step-label", "children"),
                  Output("sig", "data"),
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
        parts = [f"Training {'finished' if st.run.trunk.finished else 'in progress'}",
                 f"{len(steps)} trunk snapshots", f"{len(st.run.branch_names())} branches",
                 f"basis {info['id']}" if info else "no basis yet",
                 f"{n_pts} / {ctl['budget']} grid points evaluated",
                 f"worker {ws.get('state', 'not running') if ws['alive'] else 'not running'}"]
        if ws.get("error"):
            parts.append(f"landscape disabled: {ws['error']}")
        status = " · ".join(parts)
        label = f"Trunk snapshot {min(int(step_idx or 0), max(len(steps) - 1, 0)) + 1} of {len(steps)}" + (f", step {sel}" if sel is not None else "")
        metric_sig = [[b.name, sorted(b.metrics())] for b in st.run.branches()]
        sig = [info["id"] if info else None, n_pts, {k: len(v["steps"]) for k, v in view.get("proj", {}).items()},
               sel, mode, resolution, sorted(display), series, metric_sig]
        if sig == old_sig:
            return no_update, no_update, no_update, no_update, status, label, no_update
        level = RESOLUTIONS.get(int(resolution or 33), 2)
        return (map_figure(st, view, sel, level, "zlog" in display, "clip" in display),
                fidelity_figure(st, view, sel), fidelity_panel(view, sel), metrics_figure(st, series or []),
                status, label, sig)

    @app.callback(Output("click-panel", "children"), Output("clicked", "data"),
                  Input("map", "clickData"), State("mode", "value"), prevent_initial_call=True)
    def on_click(click, mode):
        if not click or not click.get("points"):
            return no_update, no_update
        pt = click["points"][0]
        info = st.bm.latest_info(mode)
        if info is None or "x" not in pt:
            return no_update, no_update
        a, b = float(pt["x"]), float(pt["y"])
        try:
            res, sd = st.in_plane(info["id"], a, b)
        except Exception as e:
            return html.Div(f"Could not evaluate this point: {e}", className="warn"), None
        exports.clear()
        exports[f"{info['id']}:{a}:{b}"] = sd
        rows = [("a, b", f"{a:.4g}, {b:.4g}"), ("Basis", info["id"]), ("Loss", _fmt(res["loss"], 5)),
                ("Accuracy", "n/a" if res["acc"] is None else f"{res['acc']:.2%}"),
                ("Eval samples", str(res["n"])), ("BatchNorm statistics", res["bn_mode"])]
        if pt.get("z") is not None:
            rows.append(("Surface value at click", _fmt(float(pt["z"]), 5)))
        panel = [html.Span("IN-PLANE APPROXIMATION", className="tag"),
                 html.Table([html.Tr([html.Td(k), html.Td(v)]) for k, v in rows], className="kv", style={"marginTop": "8px"}),
                 html.Div("These weights are center + a·d1 + b·d2, reconstructed from the 2D basis. They are not a "
                          "recorded snapshot, and a snapshot drawn at this spot generally has different weights and loss.",
                          className="note", style={"marginTop": "8px"}),
                 html.Button("Export state_dict (.pt)", id="export", className="btn", n_clicks=0)]
        return panel, {"key": f"{info['id']}:{a}:{b}", "a": a, "b": b, "basis": info["id"]}

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


def serve(run_dir: str | Path, host: str = "127.0.0.1", port: int = 8050, interval_ms: int = 2000,
          device: str = "cpu", debug: bool = False) -> None:
    app = build_app(run_dir, interval_ms=interval_ms, device=device)
    print(f"trajmap dashboard for {run_dir}: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
