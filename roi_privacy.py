"""
roi_privacy.py — ROI model for privacy/compliance safeguards in RWD/RWE programs

What it does
------------
Translates privacy/compliance safeguards (de-identification, privacy-preserving analytics,
governance) into business terms:
- Avoided expected loss (regulatory fines, delays, reputational impact)
- Time-to-market benefit (months saved / risk of delay reduced)
- NPV of costs and benefits
- Clear ROI metrics for leadership

Exports
-------
- PrivacyScenario (dataclass)
- SafeguardBundle (dataclass)
- npv_cashflows(cashflows, annual_rate, freq)
- expected_loss_without(ts)
- expected_loss_with(ts, sg)
- time_to_market_benefit(ts, sg)
- roi_summary(ts, sg)
- sensitivity(ts, sg, param_grid, annual_rate=None)

Notes
-----
This is an executive-friendly model with conservative defaults. Calibrate to your portfolio
and policies (e.g., breach probability, fine ranges, delay costs).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple
from math import isfinite

try:
    import numpy as np
except Exception:  # numpy optional
    np = None


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------

@dataclass
class PrivacyScenario:
    """
    Program and risk baseline (annualized where applicable).
    """
    # Risk of a regulatory/privacy incident (per-year probability)
    prob_incident_per_year: float = 0.08

    # Monetary impacts if an incident occurs (one-off + delay)
    regulatory_fine_usd: float = 2_500_000.0
    reputation_cost_usd: float = 1_000_000.0
    expected_delay_months_if_incident: float = 4.0
    delay_cost_per_month_usd: float = 500_000.0  # burn + opportunity + holding costs

    # Economic value of being on-market (optional additional lens)
    program_monthly_value_usd: float = 3_000_000.0  # net benefit when commercialized

    # Finance
    discount_rate_annual: float = 0.10  # WACC / hurdle rate


@dataclass
class SafeguardBundle:
    """
    Privacy/compliance safeguards and their effects.
    """
    # One-time capital (implementation) costs
    cost_deid_capex_usd: float = 400_000.0
    cost_ppa_capex_usd: float = 250_000.0  # privacy-preserving analytics (e.g., synthetic data, federated)
    cost_governance_capex_usd: float = 150_000.0

    # Annual operating costs
    cost_annual_opex_usd: float = 300_000.0

    # Effectiveness
    risk_reduction_relative: float = 0.6   # reduces probability of incident (0..1)
    delay_reduction_relative: float = 0.5  # reduces expected delay months if incident (0..1)

    # Horizon for NPV of operating costs (years)
    horizon_years: int = 3


# -------------------------------------------------------------------
# Finance helpers
# -------------------------------------------------------------------

def npv_cashflows(cashflows: Iterable[float], annual_rate: float, freq: int = 12) -> float:
    """
    Net Present Value of period cashflows (first cashflow at t=1).
    annual_rate: APR/WACC; freq=12 for monthly series.
    """
    r = max(0.0, float(annual_rate))
    if r == 0.0:
        return float(sum(cashflows))
    per = r / float(freq)
    total = 0.0
    for t, cf in enumerate(cashflows, start=1):
        total += float(cf) / ((1.0 + per) ** t)
    return float(total)


# -------------------------------------------------------------------
# Core components
# -------------------------------------------------------------------

def expected_loss_without(ts: PrivacyScenario) -> float:
    """
    Expected yearly loss WITHOUT safeguards.
    E[Loss] = p(incident) * (fine + reputation + delay_costs)
    Delay costs = expected_delay_months_if_incident * delay_cost_per_month
    """
    p = max(0.0, min(1.0, ts.prob_incident_per_year))
    delay_costs = ts.expected_delay_months_if_incident * ts.delay_cost_per_month_usd
    return float(p * (ts.regulatory_fine_usd + ts.reputation_cost_usd + delay_costs))


def expected_loss_with(ts: PrivacyScenario, sg: SafeguardBundle) -> float:
    """
    Expected yearly loss WITH safeguards (after reduction).
    - Probability reduced by risk_reduction_relative
    - Delay months reduced by delay_reduction_relative
    """
    p0 = max(0.0, min(1.0, ts.prob_incident_per_year))
    p1 = p0 * max(0.0, (1.0 - sg.risk_reduction_relative))
    delay_months = ts.expected_delay_months_if_incident * max(0.0, (1.0 - sg.delay_reduction_relative))
    delay_costs = delay_months * ts.delay_cost_per_month_usd
    return float(p1 * (ts.regulatory_fine_usd + ts.reputation_cost_usd + delay_costs))


def time_to_market_benefit(ts: PrivacyScenario, sg: SafeguardBundle) -> float:
    """
    Benefit of earlier time-to-market from reduced incident probability and/or shorter delays.
    We proxy this as: expected months saved * monthly program value (discounted over 1 year).

    months_saved = p0*D0  -  p1*D1
        where p0 = baseline incident prob, D0 = baseline delay months,
              p1 = reduced prob,           D1 = reduced delay months.

    We value those months over the next year as a stream.
    """
    p0 = max(0.0, min(1.0, ts.prob_incident_per_year))
    p1 = p0 * max(0.0, (1.0 - sg.risk_reduction_relative))
    D0 = ts.expected_delay_months_if_incident
    D1 = ts.expected_delay_months_if_incident * max(0.0, (1.0 - sg.delay_reduction_relative))
    months_saved = max(0.0, p0 * D0 - p1 * D1)

    if months_saved <= 0.0 or ts.program_monthly_value_usd <= 0.0:
        return 0.0

    # Discount monthly benefits for 'months_saved' months
    cash = [ts.program_monthly_value_usd] * int(round(months_saved))
    return float(npv_cashflows(cash, annual_rate=ts.discount_rate_annual, freq=12))


def _safeguard_cost_npv(ts: PrivacyScenario, sg: SafeguardBundle) -> float:
    """
    NPV of safeguard costs = Capex (t0, undiscounted) + NPV of opex over horizon (monthly).
    """
    # Capex at t0
    capex = sg.cost_deid_capex_usd + sg.cost_ppa_capex_usd + sg.cost_governance_capex_usd

    # Opex over horizon, paid monthly
    months = max(1, int(12 * max(1, sg.horizon_years)))
    monthly_opex = float(sg.cost_annual_opex_usd) / 12.0
    opex_npv = npv_cashflows([monthly_opex] * months, annual_rate=ts.discount_rate_annual, freq=12)

    return float(capex + opex_npv)


# -------------------------------------------------------------------
# Summary & Sensitivity
# -------------------------------------------------------------------

def roi_summary(ts: PrivacyScenario, sg: SafeguardBundle) -> Dict[str, float]:
    """
    Returns a concise ROI breakdown (all values in USD, positive = benefit):
      - expected_loss_without
      - expected_loss_with
      - avoided_loss (delta)
      - time_benefit (NPV of months_saved * monthly program value)
      - safeguard_cost_npv (CapEx + discounted OpEx)
      - net_benefit = avoided_loss + time_benefit - safeguard_cost_npv
      - benefit_cost_ratio = (avoided_loss + time_benefit) / safeguard_cost_npv
      - payback_like (years) = safeguard_cost / (avoided_loss + time_benefit per year)  [rough]
    """
    loss_wo = expected_loss_without(ts)
    loss_w  = expected_loss_with(ts, sg)
    avoided = max(0.0, loss_wo - loss_w)
    ttm     = time_to_market_benefit(ts, sg)
    cost    = _safeguard_cost_npv(ts, sg)

    gross_benefit = avoided + ttm
    net = gross_benefit - cost
    bcr = (gross_benefit / cost) if cost > 0 else float("inf")

    # A rough payback proxy: years to recover cost if benefits repeat annually (avoid + a 12m slice of time benefit)
    annualized_benefit = avoided + min(ttm, ts.program_monthly_value_usd * 12.0)
    payback_years = (cost / annualized_benefit) if annualized_benefit > 0 else float("inf")

    return {
        "expected_loss_without": float(loss_wo),
        "expected_loss_with": float(loss_w),
        "avoided_loss": float(avoided),
        "time_benefit": float(ttm),
        "safeguard_cost_npv": float(cost),
        "net_benefit": float(net),
        "benefit_cost_ratio": float(bcr),
        "payback_years": float(payback_years),
    }


def sensitivity(
    ts: PrivacyScenario,
    sg: SafeguardBundle,
    param_grid: Dict[str, Iterable[float]],
    annual_rate: Optional[float] = None
) -> List[Dict[str, float]]:
    """
    One-way sensitivity analysis.

    param_grid: dict of {param_name: [values,...]} where param_name is an attribute of
                either PrivacyScenario or SafeguardBundle (dot notation supported: 'ts.prob_incident_per_year').
                If no dot supplied, we try ts first, then sg.
    returns: list of rows with 'param', 'value', and ROI summary fields.
    """
    rows: List[Dict[str, float]] = []

    for param, values in param_grid.items():
        for v in values:
            # Clone objects
            ts_ = PrivacyScenario(**asdict(ts))
            sg_ = SafeguardBundle(**asdict(sg))

            # Resolve target
            target = None
            attr = param
            if "." in param:
                head, attr = param.split(".", 1)
                if head.strip().lower() in ("ts","scenario","privacy","base"):
                    target = ts_
                elif head.strip().lower() in ("sg","safeguard","bundle"):
                    target = sg_
            else:
                # try ts then sg
                target = ts_ if hasattr(ts_, attr) else sg_ if hasattr(sg_, attr) else None

            if target is None or not hasattr(target, attr):
                # skip unknown parameter
                continue

            setattr(target, attr, v)
            res = roi_summary(ts_, sg_)
            res_row = {"param": param, "value": float(v) if isinstance(v, (int,float)) else v}
            res_row.update(res)
            rows.append(res_row)

    # Optional: sort by param then value
    try:
        rows.sort(key=lambda r: (str(r["param"]), float(r["value"]) if isinstance(r["value"], (int,float)) else 0.0))
    except Exception:
        pass
    return rows


# -------------------------------------------------------------------
# CLI preview (optional)
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test with defaults
    ts = PrivacyScenario()
    sg = SafeguardBundle()
    res = roi_summary(ts, sg)
    print("=== ROI SUMMARY (defaults) ===")
    for k, v in res.items():
        print(f"{k:22s} : {v:,.2f}")

    print("\n=== SENSITIVITY: prob_incident_per_year ===")
    grid = {"ts.prob_incident_per_year": [0.02, 0.05, 0.08, 0.12]}
    rows = sensitivity(ts, sg, grid)
    for r in rows:
        print(f"{r['param']}={r['value']:.2f}  ->  net_benefit={r['net_benefit']:,.0f}, BCR={r['benefit_cost_ratio']:.2f}")
