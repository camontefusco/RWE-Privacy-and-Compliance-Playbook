"""
report_builder.py — Generate a leadership-friendly Privacy & Compliance PDF report
with auto-embedded charts from visuals/privacy_plots.py.

Inputs expected in ./data (configurable via ReportPaths):
- privacy_report.json              (from privacy_checks.build_privacy_report)
- privacy_compliance_report.json   (from compliance_matrix.build_compliance_report)
- privacy_roi_summary.json         (optional; from roi_privacy.roi_summary)
- privacy_sensitivity.json         (optional; list of rows from roi_privacy.sensitivity)

Outputs:
- ./reports/privacy_compliance_report.pdf  (ReportLab) or .md fallback
- ./reports/assets/*.png                   (charts)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths & helpers
# -----------------------------

@dataclass
class ReportPaths:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    assets_dir: Path = Path("reports/assets")

    privacy_json: Path = Path("data/privacy_report.json")
    compliance_json: Path = Path("data/privacy_compliance_report.json")
    roi_json: Path = Path("data/privacy_roi_summary.json")          # optional
    sensitivity_json: Path = Path("data/privacy_sensitivity.json")  # optional

    out_pdf: Path = Path("reports/privacy_compliance_report.pdf")
    out_md: Path = Path("reports/privacy_compliance_report.md")

def _mkdirs(paths: ReportPaths):
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.assets_dir.mkdir(parents=True, exist_ok=True)

def _load_json(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None

def _short(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return x


# -----------------------------
# Plot builders (hydrate visuals/* if available)
# -----------------------------

def _plots(privacy: Dict[str, Any], compliance: Dict[str, Any], roi: Dict[str, Any], sens_rows: Optional[List[Dict[str, Any]]], paths: ReportPaths) -> Dict[str, Optional[Path]]:
    """
    Create charts via visuals/privacy_plots.py (if import succeeds).
    Returns dict of label -> path (or None on failure).
    """
    out = {"identifier_map": None, "k_hist": None, "compliance_bar": None, "roi_waterfall": None, "tornado": None}

    try:
        from visuals.privacy_plots import (
            plot_identifier_heatmap, plot_k_equivalence_hist, plot_compliance_bar,
            plot_roi_waterfall, plot_sensitivity_tornado, save_fig
        )
    except Exception:
        # visuals module not present; skip charts
        return out

    # 1) Identifier heatmap
    try:
        direct = privacy.get("direct_identifiers") or []
        quasi = privacy.get("quasi_identifiers") or []
        # Use union of roles for column ordering (fallback)
        roles = privacy.get("roles") or {}
        cols = roles.get("id_cols") or []
        cols = list(dict.fromkeys(cols + (roles.get("date_cols") or []) + (roles.get("numeric_cols") or []) + (roles.get("categorical_cols") or [])))
        if not cols:
            # fallback to union of detected identifiers
            cols = list(dict.fromkeys(list(direct) + list(quasi)))
        if cols:
            fig = plot_identifier_heatmap(cols, direct, quasi)
            out["identifier_map"] = save_fig(fig, paths.assets_dir / "identifier_map.png")
            plt.close(fig)
    except Exception:
        pass

    # 2) k-anonymity histogram
    try:
        quasi = privacy.get("quasi_identifiers") or []
        # If we saved a preview flat CSV, you could load it here. For privacy report only,
        # we simulate class sizes from summary (n_rows) and k value to sketch a simple hist.
        # Prefer real df if you have it; otherwise skip gracefully.
        # Example (optional): df = pd.read_csv("data/sample_synthetic.csv")
        # fig = plot_k_equivalence_hist(df, quasi)
        # Here we skip if no real df:
        pass
    except Exception:
        pass

    # 3) Compliance bar
    try:
        if compliance:
            fig = plot_compliance_bar(compliance)
            out["compliance_bar"] = save_fig(fig, paths.assets_dir / "compliance_bar.png")
            plt.close(fig)
    except Exception:
        pass

    # 4) ROI waterfall
    try:
        if roi:
            fig = plot_roi_waterfall(roi)
            out["roi_waterfall"] = save_fig(fig, paths.assets_dir / "roi_waterfall.png")
            plt.close(fig)
    except Exception:
        pass

    # 5) Sensitivity tornado
    try:
        if sens_rows and isinstance(sens_rows, list) and len(sens_rows) >= 2:
            fig = plot_sensitivity_tornado(sens_rows, metric="net_benefit")
            out["tornado"] = save_fig(fig, paths.assets_dir / "sensitivity_tornado.png")
            plt.close(fig)
    except Exception:
        pass

    return out


# -----------------------------
# Core builder
# -----------------------------

def build_privacy_pdf(
    title: str = "Privacy & Compliance Report",
    *,
    paths: ReportPaths = ReportPaths(),
    organization: str = "RWE Privacy & Compliance Playbook",
    author: str = "Carlos Victor Montefusco Pereira, PhD",
) -> Path:
    """
    Attempt PDF build (ReportLab). If unavailable or fails, write Markdown instead.
    Returns the output path.
    """
    _mkdirs(paths)

    privacy = _load_json(paths.privacy_json) or {}
    compliance = _load_json(paths.compliance_json) or {}
    roi = _load_json(paths.roi_json) or {}
    sens_rows = _load_json(paths.sensitivity_json) or []

    charts = _plots(privacy, compliance, roi, sens_rows, paths)

    # ---------- Try ReportLab (PDF) ----------
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

        doc = SimpleDocTemplate(str(paths.out_pdf), pagesize=A4, leftMargin=36, rightMargin=36, topMargin=40, bottomMargin=36)
        styles = getSampleStyleSheet()
        H1 = styles["Heading1"]; H2 = styles["Heading2"]; H3 = styles["Heading3"]; BODY = styles["BodyText"]
        BODY.spaceAfter = 6

        story = []
        story.append(Paragraph(title, H1))
        story.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), BODY))
        story.append(Paragraph(f"{organization} — {author}", BODY))
        story.append(Spacer(1, 10))

        # Overview
        story.append(Paragraph("Overview", H2))
        story.append(Paragraph(
            "This report summarizes the privacy and compliance posture of a real-world data asset. "
            "It includes identifier scans, de-identification metrics (k-anonymity, l-diversity), "
            "HIPAA/GDPR readiness, and ROI impact of safeguards.", BODY))

        # Privacy metrics table
        met = privacy.get("metrics", {})
        tbl = [["Metric", "Value", "Target/Note"]]
        tbl.append(["k-anonymity", _short(met.get("k_anonymity")), "≥ 5"])
        tbl.append(["l-diversity", _short(met.get("l_diversity")), "≥ 2"])
        tbl.append(["t-closeness", _short(met.get("t_closeness")), "Lower is better"])
        tbl.append(["Risk score", _short(met.get("risk_score")), "0 = good (lower is better)"])

        t = Table(tbl, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ]))
        story.append(Paragraph("Privacy Metrics", H2))
        story.append(t)
        story.append(Spacer(1, 8))

        # Identifier map image
        if charts["identifier_map"]:
            story.append(Paragraph("Identifier Map (Direct vs Quasi)", H3))
            story.append(Image(str(charts["identifier_map"]), width=460, height=260))
            story.append(Spacer(1, 8))

        # Compliance
        story.append(Paragraph("Compliance Readiness", H2))
        hipaa = (compliance.get("hipaa") or {})
        gdpr = (compliance.get("gdpr") or {})
        juris = (compliance.get("jurisdictions") or {})
        tblc = [["Framework", "Outcome", "Detail"]]
        tblc.append(["HIPAA Safe Harbor", "PASS" if hipaa.get("pass") else "REVIEW",
                     f"Found: {', '.join(hipaa.get('found_identifiers', [])) or 'None'}"])
        tblc.append(["GDPR Anonymization", "PASS" if gdpr.get("pass") else "REVIEW",
                     f"Risk index: {_short(gdpr.get('risk_index'))}"])
        dua = juris.get("DUA", {})
        tblc.append(["DUA Policy", f"Score: {_short(dua.get('score', 1.0))}",
                     f"Flags: {', '.join([k for k,v in (dua.get('flags') or {}).items() if v]) or 'None'}"])
        tblc.append(["Compliance Index", "", _short(juris.get("compliance_index"))])

        tc = Table(tblc, hAlign="LEFT")
        tc.setStyle(TableStyle([
            ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ]))
        story.append(tc)
        story.append(Spacer(1, 8))

        # Compliance bar image
        if charts["compliance_bar"]:
            story.append(Image(str(charts["compliance_bar"]), width=420, height=260))
            story.append(Spacer(1, 8))

        # ROI
        if roi:
            story.append(Paragraph("ROI of Safeguards (Illustrative)", H2))
            tblr = [["Component", "Value (USD)"]]
            for k in ["expected_loss_without","expected_loss_with","avoided_loss","time_benefit","safeguard_cost_npv","net_benefit","benefit_cost_ratio","payback_years"]:
                v = roi.get(k)
                tblr.append([k.replace("_"," ").title(), _short(v)])
            tr = Table(tblr, hAlign="LEFT")
            tr.setStyle(TableStyle([
                ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("ALIGN", (1,1), (-1,-1), "CENTER"),
                ("BOTTOMPADDING", (0,0), (-1,0), 6),
            ]))
            story.append(tr)
            story.append(Spacer(1, 8))
            if charts["roi_waterfall"]:
                story.append(Image(str(charts["roi_waterfall"]), width=460, height=260))
                story.append(Spacer(1, 8))

        # Sensitivity tornado (optional)
        if charts["tornado"]:
            story.append(Paragraph("Sensitivity (One-way Tornado)", H2))
            story.append(Image(str(charts["tornado"]), width=460, height=300))
            story.append(Spacer(1, 8))

        # Recommendations / Next actions / Usage
        story.append(Paragraph("Recommendations", H2))
        for r in _recommendations(privacy, compliance):
            story.append(Paragraph("• " + r, BODY))

        story.append(Paragraph("Next Actions", H2))
        for a in _next_actions(privacy, compliance):
            story.append(Paragraph("• " + a, BODY))

        story.append(Paragraph("How to Use These Results", H2))
        for u in _usage_guidance():
            story.append(Paragraph("• " + u, BODY))

        doc.build(story)
        print(f"[ok] PDF written → {paths.out_pdf}")
        return paths.out_pdf

    except Exception:
        # ---------- Markdown fallback ----------
        lines = []
        lines.append(f"# {title}\n\n")
        lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_  \n")
        lines.append(f"{organization} — {author}\n\n")

        # Metrics
        lines.append("## Privacy Metrics\n\n")
        met = (privacy.get("metrics") or {})
        lines.append("| Metric | Value | Target/Note |\n|---|---:|---|\n")
        lines.append(f"| k-anonymity | {_short(met.get('k_anonymity'))} | ≥ 5 |\n")
        lines.append(f"| l-diversity | {_short(met.get('l_diversity'))} | ≥ 2 |\n")
        lines.append(f"| t-closeness | {_short(met.get('t_closeness'))} | Lower is better |\n")
        lines.append(f"| Risk score | {_short(met.get('risk_score'))} | 0 = good (lower is better) |\n\n")

        # Compliance
        lines.append("## Compliance Readiness\n\n")
        hipaa = (compliance.get("hipaa") or {})
        gdpr = (compliance.get("gdpr") or {})
        juris = (compliance.get("jurisdictions") or {})
        lines.append("| Framework | Outcome | Detail |\n|---|---|---|\n")
        lines.append(f"| HIPAA Safe Harbor | {'PASS' if hipaa.get('pass') else 'REVIEW'} | Found: {', '.join(hipaa.get('found_identifiers', [])) or 'None'} |\n")
        lines.append(f"| GDPR Anonymization | {'PASS' if gdpr.get('pass') else 'REVIEW'} | Risk index: {_short(gdpr.get('risk_index'))} |\n")
        dua = juris.get("DUA", {})
        lines.append(f"| DUA Policy | Score: {_short(dua.get('score', 1.0))} | Flags: {', '.join([k for k,v in (dua.get('flags') or {}).items() if v]) or 'None'} |\n")
        lines.append(f"\n**Compliance Index:** {_short(juris.get('compliance_index'))}\n\n")

        # ROI
        if roi:
            lines.append("## ROI of Safeguards (Illustrative)\n\n")
            lines.append("| Component | Value (USD) |\n|---|---:|\n")
            for k in ["expected_loss_without","expected_loss_with","avoided_loss","time_benefit","safeguard_cost_npv","net_benefit","benefit_cost_ratio","payback_years"]:
                lines.append(f"| {k.replace('_',' ').title()} | {_short(roi.get(k))} |\n")
            lines.append("\n")

        # Chart references
        lines.append("## Figures\n\n")
        for label, p in charts.items():
            if p:
                lines.append(f"![{label}]({p.as_posix()})\n\n")

        # Rec/Next/Usage
        lines.append("## Recommendations\n")
        for r in _recommendations(privacy, compliance):
            lines.append(f"- {r}\n")
        lines.append("\n## Next Actions\n")
        for a in _next_actions(privacy, compliance):
            lines.append(f"- {a}\n")
        lines.append("\n## How to Use These Results\n")
        for u in _usage_guidance():
            lines.append(f"- {u}\n")

        paths.out_md.write_text("".join(lines))
        print(f"[ok] Markdown written → {paths.out_md}")
        return paths.out_md


# -----------------------------
# Content helpers
# -----------------------------

def _recommendations(privacy: Dict[str, Any], compliance: Dict[str, Any]):
    met = (privacy.get("metrics") or {})
    hipaa = (compliance.get("hipaa") or {})
    gdpr = (compliance.get("gdpr") or {})
    k = float(met.get("k_anonymity") or 0)
    l = float(met.get("l_diversity") or 0)
    risk = float(met.get("risk_score") or 0)
    found = hipaa.get("found_identifiers", [])
    risk_idx = gdpr.get("risk_index")

    recs = []
    if k < 5:
        recs.append("Increase k-anonymity: generalize dates to year/month, coarsen ZIP (ZIP3), bucket rare categories.")
    if l < 2:
        recs.append("Improve l-diversity: suppress or generalize highly homogeneous sensitive groups.")
    if risk > 0.6:
        recs.append("Reduce re-identification risk: remove direct identifiers, reduce quasi columns, add noise/aggregation.")
    if found:
        recs.append(f"Remove/replace direct identifiers flagged by HIPAA scan: {', '.join(found[:6])}...")
    if isinstance(risk_idx, (int, float)) and risk_idx >= 0.33:
        recs.append("Lower GDPR risk index: coarsen dates/geography and verify no direct identifiers remain.")
    if not recs:
        recs.append("Maintain safeguards and monitoring; verify DUA scope and retention settings.")
    return recs

def _next_actions(privacy: Dict[str, Any], compliance: Dict[str, Any]):
    return [
        "Run de-identification pipeline on high-risk columns; re-calc k/l and HIPAA/GDPR checks.",
        "Document DUA gates (purpose, retention, region, third parties) and attach this report to approvals.",
        "Calibrate ROI model with program-specific incident probabilities and fine ranges.",
        "Schedule quarterly re-assessment; add lineage & access controls to governance policy."
    ]

def _usage_guidance():
    return [
        "Regulatory/Legal: attach this report to DUA approvals and evidence packages.",
        "Data Stewardship: use the recommendations list as a remediation backlog.",
        "HEOR/MAx: link compliance readiness to payer confidence & evidence acceptance.",
        "Engineering: codify generalization/suppression rules into your ETL pipelines."
    ]


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build Privacy & Compliance report (PDF with charts, MD fallback)")
    ap.add_argument("--privacy", default="data/privacy_report.json", help="Path to privacy report JSON")
    ap.add_argument("--compliance", default="data/privacy_compliance_report.json", help="Path to compliance report JSON")
    ap.add_argument("--roi", default="data/privacy_roi_summary.json", help="Path to ROI summary JSON (optional)")
    ap.add_argument("--sensitivity", default="data/privacy_sensitivity.json", help="Path to sensitivity JSON (optional)")
    ap.add_argument("--title", default="Privacy & Compliance Report", help="Report title")
    args = ap.parse_args()

    paths = ReportPaths(
        privacy_json=Path(args.privacy),
        compliance_json=Path(args.compliance),
        roi_json=Path(args.roi),
        sensitivity_json=Path(args.sensitivity),
    )
    build_privacy_pdf(title=args.title, paths=paths)
