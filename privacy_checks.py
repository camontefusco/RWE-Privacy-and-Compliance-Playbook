"""
privacy_checks.py — Identifier scanning and de-identification metrics for RWD/RWE

Functions:
- load_dataset(path) -> pd.DataFrame
- infer_column_roles(df) -> dict (id, dates, numeric, categorical)
- detect_direct_identifiers(df) -> set[str]
- detect_quasi_identifiers(df) -> set[str]
- k_anonymity(df, quasi_cols) -> int
- l_diversity(df, quasi_cols, sensitive_col, method="distinct") -> float
- t_closeness(df, quasi_cols, sensitive_col, method="emd") -> float
- risk_score(df, direct_ids, quasi_ids, k, l) -> float in [0,1]
- suggest_generalization(df, quasi_cols) -> dict[str, str]
- build_privacy_report(df, *, sensitive_col=None, quasi_override=None) -> dict
- json_safe(obj) -> built-in types for JSON

Notes:
- This module intentionally uses minimal dependencies (pandas, numpy).
- All functions are deterministic and side-effect free (pure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import math
import numpy as np
import pandas as pd

# ----------------------------
# Loading & type inference
# ----------------------------

def load_dataset(path: str | bytes | "Path") -> pd.DataFrame:
    """
    Load CSV/Parquet by extension. Falls back to CSV if unknown.
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    # default: CSV
    return pd.read_csv(p)

def infer_column_roles(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Heuristic role inference (non-destructive):
    - id_cols: patient_id, subject_id, mrn, ssn, email, name-like
    - date_cols: contains 'date', 'dob', 'birth'
    - numeric_cols: numeric dtype
    - categorical_cols: low-cardinality object/string
    """
    id_like_keywords = [
        "patient_id","subject_id","person_id","participant_id","mrn","ssn","national_id",
        "identifier","id","email","phone","name","first_name","last_name","address"
    ]
    date_like_keywords = ["date","dob","birth","visit","admission","discharge","dx_date"]
    cols = list(map(str, df.columns))
    lower = {c: c.lower() for c in cols}

    id_cols = [c for c in cols if any(k in lower[c] for k in id_like_keywords)]
    date_cols = [c for c in cols if any(k in lower[c] for k in date_like_keywords)]

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if (df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c]))]

    return {
        "id_cols": id_cols,
        "date_cols": date_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }

# ----------------------------
# Identifier detection
# ----------------------------

HIPAA_SAFE_HARBOR_DIRECT = set([
    # Classic direct identifiers (illustrative subset)
    "name","first_name","last_name","full_name","patient_name","subject_name",
    "address","street","email","phone","fax","ssn","social_security_number",
    "medical_record_number","mrn","account_number","license","vehicle_id","device_id",
    "ip_address","url","biometric","fingerprint","voiceprint","face_id",
])

ZIP_LIKE_NAMES = {"zip","zip_code","postal","postal_code"}
DOB_LIKE_NAMES = {"dob","date_of_birth","birth_date"}
AGE_NAME = {"age","age_years"}

def _is_zip_like(colname: str) -> bool:
    return any(k in colname.lower() for k in ZIP_LIKE_NAMES)

def _is_dob_like(colname: str) -> bool:
    return any(k in colname.lower() for k in DOB_LIKE_NAMES)

def detect_direct_identifiers(df: pd.DataFrame) -> Set[str]:
    """
    Column-level detection of likely direct identifiers using name heuristics.
    """
    directs = set()
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in HIPAA_SAFE_HARBOR_DIRECT or any(k in lc for k in HIPAA_SAFE_HARBOR_DIRECT):
            directs.add(c)
    return directs

def detect_quasi_identifiers(df: pd.DataFrame) -> Set[str]:
    """
    Quasi-identifiers are columns that don't directly identify but can re-ID in combination:
    - ZIP-like, geography, small-area codes
    - dates (visit, admission, birth, etc.)
    - age, year, month
    - low-cardinality categorical demographics
    """
    roles = infer_column_roles(df)
    quasi = set()

    # Dates, ages, years, months are quasi
    for c in roles["date_cols"]:
        quasi.add(c)
    for c in df.columns:
        lc = str(c).lower()
        if _is_zip_like(lc) or _is_dob_like(lc) or lc in AGE_NAME or lc.endswith("_year") or lc.endswith("_month"):
            quasi.add(c)

    # Geographies & demographics by name heuristics
    GEO_KEYS = ["state","county","city","region","country","location","geo","lat","lon"]
    DEMO_KEYS = ["sex","gender","race","ethnicity","language"]
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in GEO_KEYS + DEMO_KEYS):
            quasi.add(c)

    # Low-cardinality categoricals can be quasi (e.g., facility_id)
    for c in roles["categorical_cols"]:
        nun = df[c].nunique(dropna=True)
        if 2 <= nun <= min(50, max(5, int(len(df) * 0.05))):
            quasi.add(c)

    # Remove direct IDs from quasi set
    quasi -= detect_direct_identifiers(df)
    return quasi

# ----------------------------
# k-anonymity, l-diversity, t-closeness
# ----------------------------

def _group_counts(df: pd.DataFrame, quasi_cols: Iterable[str]) -> pd.Series:
    if not quasi_cols:
        return pd.Series([len(df)], index=[0])
    g = df.groupby(list(quasi_cols), dropna=False, as_index=False)
    return g.size().set_index(list(quasi_cols))["size"]

def k_anonymity(df: pd.DataFrame, quasi_cols: Iterable[str]) -> int:
    """
    Minimum group size across equivalence classes defined by quasi columns.
    """
    counts = _group_counts(df, quasi_cols)
    return int(counts.min()) if len(counts) else len(df)

def l_diversity(df: pd.DataFrame, quasi_cols: Iterable[str], sensitive_col: str, method: str = "distinct") -> float:
    """
    l-diversity (distinct) = min #distinct sensitive values per equivalence class.
    """
    if not quasi_cols or sensitive_col not in df.columns:
        return 0.0
    grp = df.groupby(list(quasi_cols), dropna=False)[sensitive_col].nunique(dropna=True)
    if grp.empty:
        return 0.0
    return float(grp.min())

def t_closeness(df: pd.DataFrame, quasi_cols: Iterable[str], sensitive_col: str, method: str = "emd") -> float:
    """
    t-closeness via Earth Mover's Distance (1D numeric approx).
    Lower is better (closer to global distribution). We return the worst (max) distance across groups.
    """
    if sensitive_col not in df.columns or not pd.api.types.is_numeric_dtype(df[sensitive_col]):
        return float("nan")
    if not quasi_cols:
        return 0.0

    s = df[sensitive_col].dropna().astype(float)
    if s.empty:
        return float("nan")
    global_sorted = np.sort(s.values)

    def emd_1d(a: np.ndarray, b: np.ndarray) -> float:
        # Simple discrete EMD approx via cumulative sums
        a = np.sort(a); b = np.sort(b)
        # pad to same length by interpolation
        n = max(len(a), len(b))
        qa = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(a)), a)
        qb = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(b)), b)
        return float(np.mean(np.abs(np.cumsum(qa - qb))))

    worst = 0.0
    for _, grp in df.groupby(list(quasi_cols), dropna=False):
        loc = grp[sensitive_col].dropna().astype(float).values
        if len(loc) == 0:
            continue
        worst = max(worst, emd_1d(loc, global_sorted))
    return float(worst)

# ----------------------------
# Risk scoring & suggestions
# ----------------------------

def risk_score(df: pd.DataFrame,
               direct_ids: Iterable[str],
               quasi_ids: Iterable[str],
               k: int,
               l: float) -> float:
    """
    Heuristic risk in [0,1]. Higher = riskier.
    - Direct identifiers add big risk
    - More quasi-identifiers add risk
    - Lower k and l add risk
    """
    n = max(1, len(df))
    d_count = len(list(direct_ids))
    q_count = len(list(quasi_ids))

    risk = 0.0
    # direct ids dominate
    risk += min(1.0, 0.4 + 0.1 * d_count)
    # quasi add moderate risk
    risk += min(0.3, 0.03 * q_count)
    # k-anon: target >= 5
    if k < 5:
        risk += 0.2 * (5 - k) / 5.0
    # l-div: target >= 2
    if l < 2:
        risk += 0.1 * (2 - l) / 2.0
    return float(max(0.0, min(1.0, risk)))

def suggest_generalization(df: pd.DataFrame, quasi_cols: Iterable[str]) -> Dict[str, str]:
    """
    Suggest simple transformations to increase k:
    - birth_date/date → year only
    - zip_code → first 3 digits (US HIPAA guideline)
    - high-cardinality categorical → bucket rare categories
    """
    suggestions: Dict[str, str] = {}
    for c in quasi_cols:
        lc = str(c).lower()
        if _is_dob_like(lc) or "date" in lc or lc.endswith("_date"):
            suggestions[c] = "Generalize to year or month (e.g., YYYY or YYYY-MM)"
        elif _is_zip_like(lc):
            suggestions[c] = "Generalize to first 3 digits (US ZIP3) or to region"
        elif pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == "object":
            nun = df[c].nunique(dropna=True)
            if nun > 20:
                suggestions[c] = "Bucket rare categories into 'Other' (e.g., top-10 + Other)"
        elif pd.api.types.is_numeric_dtype(df[c]):
            suggestions[c] = "Bin numeric into deciles/quantiles"
    return suggestions

# ----------------------------
# Report builder (JSON-ready)
# ----------------------------

def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):   return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    if isinstance(obj, (np.integer,)):    return int(obj)
    if isinstance(obj, (np.floating,)):   return float(obj)
    return obj

def build_privacy_report(
    df: pd.DataFrame,
    *,
    sensitive_col: Optional[str] = None,
    quasi_override: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    """
    Compute a compact privacy report suitable for saving as JSON.
    """
    roles = infer_column_roles(df)
    direct_ids = detect_direct_identifiers(df)

    if quasi_override is not None:
        quasi_ids = set(quasi_override) - set(direct_ids)
    else:
        quasi_ids = detect_quasi_identifiers(df)

    # metrics
    k = k_anonymity(df, quasi_ids)
    l = l_diversity(df, quasi_ids, sensitive_col, method="distinct") if sensitive_col else float("nan")
    t = t_closeness(df, quasi_ids, sensitive_col, method="emd") if sensitive_col else float("nan")
    r = risk_score(df, direct_ids, quasi_ids, k, l if not math.isnan(l) else 0.0)

    # suggestions
    sugg = suggest_generalization(df, quasi_ids)

    report = {
        "summary": {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "sensitive_col": sensitive_col,
        },
        "roles": roles,
        "direct_identifiers": sorted(map(str, direct_ids)),
        "quasi_identifiers": sorted(map(str, quasi_ids)),
        "metrics": {
            "k_anonymity": int(k),
            "l_diversity": float(l) if not math.isnan(l) else None,
            "t_closeness": float(t) if not (isinstance(t, float) and math.isnan(t)) else None,
            "risk_score": float(r),
        },
        "suggestions": sugg,
    }
    return json_safe(report)

# ----------------------------
# CLI preview (optional)
# ----------------------------

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Privacy checks for RWD/RWE datasets")
    ap.add_argument("path", help="CSV/Parquet file")
    ap.add_argument("--sensitive", default=None, help="Sensitive column name (optional)")
    ap.add_argument("--quasi", nargs="*", default=None, help="Quasi-identifier columns (override)")
    args = ap.parse_args()

    df_ = load_dataset(args.path)
    rep = build_privacy_report(df_, sensitive_col=args.sensitive, quasi_override=args.quasi)
    print(json.dumps(rep, indent=2))
