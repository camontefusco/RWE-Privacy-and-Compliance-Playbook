"""
compliance_matrix.py — Jurisdictional compliance checks for RWD/RWE

Exports:
- evaluate_hipaa_safe_harbor(columns: list[str]) -> dict
- evaluate_gdpr_anonymization(df: pd.DataFrame, *, quasi_cols=None, direct_cols=None) -> dict
- jurisdiction_scorecard(hipaa: dict, gdpr: dict, *, dua_flags=None) -> dict
- build_compliance_report(df: pd.DataFrame, *, quasi_cols=None, direct_cols=None, dua_flags=None) -> dict
- json_safe(obj) -> built-in types for JSON

Notes:
- HIPAA Safe Harbor: checks for presence of 18 direct identifiers by name heuristic
- GDPR anonymization: checklist around identifiability risk signals (uses df-level heuristics)
- DUA flags: configurable policy gates (e.g., purpose limitation, retention, geography)
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
import pandas as pd

# ----------------------------
# JSON-safe util
# ----------------------------

def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):   return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    if isinstance(obj, (np.integer,)):    return int(obj)
    if isinstance(obj, (np.floating,)):   return float(obj)
    return obj

# ----------------------------
# HIPAA Safe Harbor identifiers (name-based heuristics)
# ----------------------------

HIPAA_18 = {
    # Names
    "name","first_name","last_name","middle_name","full_name","patient_name","subject_name",
    # Geographic subdivisions smaller than a state (street address, city, county, precinct, ZIP <3 digits)
    "address","street","city","county","precinct","zip","zip_code","postal","postal_code",
    # All elements of dates (except year) for dates directly related to an individual (e.g., birth date)
    "date_of_birth","dob","birth_date","birth_dt","admission_date","discharge_date","death_date",
    # Telephone numbers
    "phone","telephone","mobile",
    # Fax numbers
    "fax",
    # Email addresses
    "email","email_address",
    # SSN
    "ssn","social_security_number",
    # Medical record numbers
    "medical_record_number","mrn",
    # Health plan beneficiary numbers
    "beneficiary_id","hpb_number",
    # Account numbers
    "account_number",
    # Certificate/license numbers
    "license","license_number",
    # Vehicle identifiers and serial numbers, including license plate numbers
    "vehicle_id","license_plate",
    # Device identifiers and serial numbers
    "device_id","device_serial",
    # Web URLs
    "url","uri",
    # IP addresses
    "ip","ip_address",
    # Biometric identifiers, including finger and voice prints
    "biometric","fingerprint","voiceprint",
    # Full-face photos and comparable images
    "face","face_image","photo",
    # Any other unique identifying number, characteristic, or code
    "unique_id","uid",
}

def evaluate_hipaa_safe_harbor(columns: Iterable[str]) -> Dict[str, Any]:
    """
    Check columns against HIPAA Safe Harbor direct identifiers by name.
    Returns dict with 'found', 'missing', and a naive 'pass' boolean (True if none found).
    """
    cols = [str(c).strip().lower() for c in columns]
    found = sorted({c for c in cols if any(k in c for k in HIPAA_18)})
    return {
        "policy": "HIPAA Safe Harbor (name-heuristic scan)",
        "found_identifiers": found,
        "pass": len(found) == 0
    }

# ----------------------------
# GDPR anonymization checklist
# ----------------------------

def _low_k_indicator(df: pd.DataFrame, quasi_cols: Optional[Iterable[str]]) -> float:
    """Return 1.0 if any equivalence class size < 5 (risky), else 0.0. If unknown, 0.5."""
    if not quasi_cols:
        return 0.5
    try:
        size = df.groupby(list(quasi_cols), dropna=False).size()
        return 1.0 if (len(size) and size.min() < 5) else 0.0
    except Exception:
        return 0.5

def _date_granularity_indicator(df: pd.DataFrame) -> float:
    """1.0 if sub-year date granularity detected in person-related fields; else 0.0; unknown 0.5."""
    cols = [c for c in df.columns if "date" in str(c).lower() or "dob" in str(c).lower() or "birth" in str(c).lower()]
    if not cols:
        return 0.5
    # if any date column seems full date (YYYY-MM-DD) -> risk indicator
    try:
        for c in cols:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                # if months/days vary, assume sub-year granularity present
                if s.dt.month.nunique(dropna=True) > 1 or s.dt.day.nunique(dropna=True) > 1:
                    return 1.0
        return 0.0
    except Exception:
        return 0.5

def _geo_precision_indicator(columns: Iterable[str]) -> float:
    """1.0 if fine-grain geo present (ZIP/postal smaller than state); else 0.0; unknown 0.5."""
    cols = [str(c).lower() for c in columns]
    risky = any(any(k in c for k in ["zip","postal","city","county","precinct","address","street"]) for c in cols)
    return 1.0 if risky else 0.5 if any("state" in c for c in cols) else 0.0

def _direct_present_indicator(direct_cols: Optional[Iterable[str]]) -> float:
    """1.0 if direct identifiers known/present; 0.0 otherwise; 0.5 unknown."""
    if direct_cols is None:
        return 0.5
    return 1.0 if len(list(direct_cols)) > 0 else 0.0

def evaluate_gdpr_anonymization(
    df: pd.DataFrame,
    *,
    quasi_cols: Optional[Iterable[str]] = None,
    direct_cols: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    """
    GDPR-style anonymization readiness (risk-based checklist).
    Indicators in [0,1]; lower is better (0 = good/low risk).
    - low_k: any equivalence class <5 persons
    - dates_subyear: sub-year date granularity present
    - geo_precision: ZIP/city/county present
    - direct_present: any direct identifiers present
    Aggregate 'risk_index' = mean of indicators, 'pass' if < 0.33
    """
    ind = {
        "low_k": _low_k_indicator(df, quasi_cols),
        "dates_subyear": _date_granularity_indicator(df),
        "geo_precision": _geo_precision_indicator(df.columns),
        "direct_present": _direct_present_indicator(direct_cols),
    }
    risk_vals = [v for v in ind.values() if isinstance(v, (int,float))]
    risk_index = float(np.mean(risk_vals)) if risk_vals else 0.5
    return {
        "policy": "GDPR anonymization (risk checklist)",
        "indicators": ind,
        "risk_index": float(risk_index),
        "pass": bool(risk_index < 0.33)
    }

# ----------------------------
# Jurisdictional scorecard
# ----------------------------

def jurisdiction_scorecard(
    hipaa: Dict[str, Any],
    gdpr: Dict[str, Any],
    *,
    dua_flags: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    """
    Combine HIPAA/GDPR outcomes and optional DUA gates into a jurisdictional scorecard.
    dua_flags: dict of project/policy gates (e.g., {'purpose_limited': True, 'retention_defined': True, 'in_region': True})
    """
    dua_flags = dua_flags or {}
    # Simple jurisdictional passes
    us_pass = bool(hipaa.get("pass", False))
    eu_pass = bool(gdpr.get("pass", False))

    # DUA score: fraction of True flags
    if dua_flags:
        dua_score = float(np.mean([1.0 if v else 0.0 for v in dua_flags.values()]))
    else:
        dua_score = 1.0  # assume ok if not provided

    # Aggregate compliance index (illustrative)
    # weights: HIPAA 0.4, GDPR 0.4, DUA 0.2
    comp_index = 0.4*(1.0 if us_pass else 0.0) + 0.4*(1.0 if eu_pass else 0.0) + 0.2*dua_score

    return {
        "US_HIPAA": {"pass": us_pass, "found_identifiers": hipaa.get("found_identifiers", [])},
        "EU_GDPR": {"pass": eu_pass, "risk_index": gdpr.get("risk_index")},
        "DUA": {"flags": dua_flags, "score": float(dua_score)},
        "compliance_index": float(comp_index)
    }

# ----------------------------
# One-shot report builder
# ----------------------------

def build_compliance_report(
    df: pd.DataFrame,
    *,
    quasi_cols: Optional[Iterable[str]] = None,
    direct_cols: Optional[Iterable[str]] = None,
    dua_flags: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    """
    Convenience wrapper to evaluate HIPAA + GDPR + DUA and return a single report object.
    """
    hipaa = evaluate_hipaa_safe_harbor(df.columns)
    gdpr = evaluate_gdpr_anonymization(df, quasi_cols=quasi_cols, direct_cols=direct_cols)
    score = jurisdiction_scorecard(hipaa, gdpr, dua_flags=dua_flags)
    return json_safe({
        "hipaa": hipaa,
        "gdpr": gdpr,
        "jurisdictions": score
    })
