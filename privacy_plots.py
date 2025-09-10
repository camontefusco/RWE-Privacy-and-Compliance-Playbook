"""
visuals/privacy_plots.py — Lightweight plotting utilities for the
RWE Privacy & Compliance Playbook.

No seaborn. Each function:
- uses matplotlib only
- returns the created Figure (caller decides to show/save)
- avoids hard-coded colors unless passed explicitly

Charts:
- plot_identifier_heatmap(): direct vs quasi identifier presence by column
- plot_k_equivalence_hist(): distribution of equivalence class sizes (k-anonymity)
- plot_compliance_bar(): HIPAA/GDPR/DUA/compliance index bar chart
- plot_roi_waterfall(): ROI components waterfall (benefits, costs, net)
- plot_sensitivity_tornado(): one-way sensitivity “tornado” style

Helpers:
- save_fig(fig, path, dpi=200, tight=True)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _annotate_bars(ax, fmt="{:.2f}", rotation=0, fontsize=9, offset=0.01):
    for p in ax.patches:
        try:
            value = p.get_height()
            x = p.get_x() + p.get_width() / 2
            y = value
            ax.text(x, y + (ax.get_ylim()[1] * offset), fmt.format(value),
                    ha="center", va="bottom", rotation=rotation, fontsize=fontsize)
        except Exception:
            pass

def save_fig(fig, path: str, dpi: int = 200, tight: bool = True):
    """Safe save with optional tight_layout."""
    import pathlib
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(p, dpi=dpi)
    return p


# ---------------------------------------------------------------------
# 1) Identifier heatmap
# ---------------------------------------------------------------------

def plot_identifier_heatmap(
    columns: Iterable[str],
    direct_identifiers: Iterable[str],
    quasi_identifiers: Iterable[str],
    *,
    title: str = "Identifier Map (Direct vs Quasi)",
    figsize: Tuple[float, float] = (8, 0.35),
    colors: Optional[Dict[str, str]] = None,
):
    """
    Shows a 1D heatmap-like bar per column:
      - 2 if in direct_identifiers
      - 1 if in quasi_identifiers
      - 0 otherwise

    Note: To keep dependencies minimal, we use a horizontal bar chart with categorical y.
    """
    cols = [str(c) for c in columns]
    direct = set(map(str, direct_identifiers or []))
    quasi = set(map(str, quasi_identifiers or []))

    vals = [2 if c in direct else 1 if c in quasi else 0 for c in cols]

    fig, ax = plt.subplots(figsize=(max(figsize[0], 8), max(figsize[1], 0.35*len(cols)+1)))
    y_pos = np.arange(len(cols))
    ax.barh(y_pos, vals)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cols)
    ax.set_xlabel("Identifier Level (2=Direct, 1=Quasi, 0=None)")
    ax.set_title(title)
    ax.set_xlim(0, 2.1)

    # Legend (no explicit colors set, uses default)
    ax.plot([], [], label="Direct", linewidth=0, marker="s")
    ax.plot([], [], label="Quasi", linewidth=0, marker="s")
    ax.plot([], [], label="None", linewidth=0, marker="s")
    ax.legend(["Direct (2)", "Quasi (1)", "None (0)"], loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# 2) k-anonymity equivalence class distribution
# ---------------------------------------------------------------------

def plot_k_equivalence_hist(
    df: pd.DataFrame,
    quasi_cols: Iterable[str],
    *,
    bins: int = 20,
    title: str = "Equivalence Class Size Distribution (k-anonymity)",
    figsize: Tuple[float, float] = (6, 4),
):
    """
    Group by quasi_cols, compute sizes, and show a histogram of equivalence class sizes.
    """
    q = list(quasi_cols or [])
    if not q:
        sizes = np.array([len(df)])
    else:
        try:
            sizes = df.groupby(q, dropna=False).size().values
        except Exception:
            sizes = np.array([len(df)])

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(sizes, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Equivalence class size (k)")
    ax.set_ylabel("Count of classes")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# 3) Compliance bar (HIPAA/GDPR/DUA/Compliance Index)
# ---------------------------------------------------------------------

def plot_compliance_bar(
    compliance_report: Dict,
    *,
    title: str = "Compliance Readiness",
    figsize: Tuple[float, float] = (6, 4),
):
    """
    Expects the object returned by build_compliance_report(...):
      {
        "hipaa": {"pass": bool, ...},
        "gdpr": {"pass": bool, "risk_index": float},
        "jurisdictions": {"DUA": {"score": float}, "compliance_index": float}
      }
    """
    hipaa = (compliance_report.get("hipaa") or {})
    gdpr  = (compliance_report.get("gdpr") or {})
    juris = (compliance_report.get("jurisdictions") or {})

    # Map to normalized bars (1=pass/good, 0=fail/bad; risk_index inverted)
    hipaa_pass = 1.0 if hipaa.get("pass") else 0.0
    gdpr_pass  = 1.0 if gdpr.get("pass") else max(0.0, 1.0 - float(gdpr.get("risk_index") or 0.0))
    dua_score  = float(((juris.get("DUA") or {}).get("score") or 1.0))
    comp_idx   = float(juris.get("compliance_index") or 0.0)

    labels = ["HIPAA", "GDPR", "DUA", "Index"]
    vals = [hipaa_pass, gdpr_pass, dua_score, comp_idx]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, vals)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Readiness (0–1)")
    ax.set_title(title)
    _annotate_bars(ax, fmt="{:.2f}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# 4) ROI waterfall (benefits vs costs)
# ---------------------------------------------------------------------

def plot_roi_waterfall(
    roi: Dict[str, float],
    *,
    title: str = "ROI Breakdown (Benefits vs Costs)",
    figsize: Tuple[float, float] = (7, 4),
):
    """
    Accepts dict from roi_privacy.roi_summary(...).
    Shows stacked steps: avoided_loss + time_benefit - safeguard_cost_npv = net_benefit.
    """
    avoided = float(roi.get("avoided_loss") or 0.0)
    time_b  = float(roi.get("time_benefit") or 0.0)
    cost    = -abs(float(roi.get("safeguard_cost_npv") or 0.0))
    net     = float(roi.get("net_benefit") or (avoided + time_b + cost))

    steps = [("Avoided loss", avoided), ("Time benefit", time_b), ("Safeguard cost", cost), ("Net", net)]
    vals = [s[1] for s in steps]
    cum = np.cumsum([0.0] + vals[:-1])

    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, value) in enumerate(steps):
        ax.bar(i, value, bottom=cum[i])
        ax.text(i, cum[i] + value + (abs(max(vals)) * 0.02), f"{value:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps])
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# 5) Sensitivity tornado (one-way)
# ---------------------------------------------------------------------

def plot_sensitivity_tornado(
    rows: List[Dict],
    metric: str = "net_benefit",
    *,
    title: str = "One-way Sensitivity (Tornado)",
    figsize: Tuple[float, float] = (7, 5),
):
    """
    rows: output of roi_privacy.sensitivity(...), list of dicts with keys:
          'param', 'value', and metrics including `metric` (e.g., net_benefit).
    We compute the min/max of `metric` for each parameter and plot horizontal bars.

    Note: Assumes at least 2 values per parameter.
    """
    if not rows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title); ax.text(0.5, 0.5, "No data", ha="center"); return fig

    df = pd.DataFrame(rows)
    params = []
    mins = []
    maxs = []
    for p, g in df.groupby("param"):
        if metric not in g.columns:
            continue
        params.append(p)
        mins.append(float(g[metric].min()))
        maxs.append(float(g[metric].max()))

    if not params:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title); ax.text(0.5, 0.5, "No data", ha="center"); return fig

    ranges = np.array(maxs) - np.array(mins)
    order = np.argsort(ranges)  # ascending; tornado often uses descending
    params = [params[i] for i in order]
    mins = [mins[i] for i in order]
    maxs = [maxs[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(params))
    for i, p in enumerate(params):
        ax.barh(y[i], maxs[i] - mins[i], left=mins[i])
        ax.text(maxs[i], y[i], f" {maxs[i]:,.0f}", va="center", ha="left", fontsize=8)
        ax.text(mins[i], y[i], f"{mins[i]:,.0f} ", va="center", ha="right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(params)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()
    return fig
