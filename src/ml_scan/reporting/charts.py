"""Plotly helpers. Pure functions; notebooks call fig.show()."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def equity_figure(equity: pd.Series | pd.DataFrame) -> go.Figure:
    if isinstance(equity, pd.DataFrame):
        series = equity.set_index("ts")["equity"] if "equity" in equity.columns else equity.iloc[:, 0]
    else:
        series = pd.Series(equity).astype(float).sort_index()
    peak = series.cummax()
    dd = series / peak - 1.0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=series.index, y=series.to_numpy(), name="Equity", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.to_numpy(), name="Drawdown", mode="lines", fill="tozeroy"), row=2, col=1)
    fig.update_layout(title="Equity and underwater", height=640, template="plotly_white")
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
    return fig


def importance_figure(importances: pd.Series, *, title: str = "Feature importance") -> go.Figure:
    s = importances.sort_values(ascending=True).tail(30)
    fig = go.Figure(go.Bar(x=s.to_numpy(), y=s.index.astype(str), orientation="h"))
    fig.update_layout(title=title, height=max(360, 18 * len(s)), template="plotly_white")
    return fig


def write_html(fig: go.Figure, path: Path | str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(dest, include_plotlyjs="cdn")
    return dest
