"""
TwinSim — Dataset Health Report  v6.0
=======================================
Validates both Layer 1 output tables against the v5.0 schema.

Session checks (20 checks):
  A1  uniqueness | A2 nullability | A3 dtypes | A4 ranges
  A5  timestamp  | A6 triangle constraint (excl geo_block)
  A7  event-type consistency + churn/reactivation mutex + reactivation-requires-prior-churn
  A8  session ordering | A9 segment distribution | A10 completion ordering
  A11 sparse satisfaction | A12 skip_intro guard (film/shortform/sport/documentary)
  A12b rewatch_flag=0 for live events            NEW v6.0
  A13 casting guard (smart_tv session-level)
  A14 bitrate-network correlation
  A15 drop flag semantics (completion_pct thresholds)
  A16 days_since NULL for session_number=1
  A17 session_depth content-type caps
  A18 network_stress_flag consistency
  A19 episode_position validity
  A_extra unavailability coherence + geo_block checks:
    geo_block_completeness (completion=0, depth=1)
    geo_block_event_type=session_start             NEW v6.0  (P1-C)
    geo_block_technical_signals all zero           NEW v6.0  (P1-B)
    geo_block_satisfaction=NULL                    NEW v6.0  (P1-A)

Profile checks (11 checks):
  B15 uniqueness | B16 nullability | B17 ranges (incl. new v5.0 fields)
  B18 price coherence | B19 trajectory validity (incl. fav_genre_confidence)
  B20 new profile fields (lifecycle, payment method, email, a/b test,
      campaign_response_rate, ppv_purchase_count, merchandise_purchase_flag)
      lifecycle_stage now includes "new" stage    NEW v6.0  (P2-A)
  B21 ticket null coherence | B22 ticket-payment correlation
  B23 ticket segment ordering | B24 binge_index segment correlation
  B25 casting_usage_pref=0 for smart_tv primary   NEW v6.0  (P2-F)

Cross-table (2 checks): C25, C26

Changes from v5.0:
  - VALID_LIFECYCLE_STAGES expanded: "new" stage added (P2-A)
  - A12: "documentary" added to skip_intro exclusion list (P3-C)
  - A12b: rewatch_flag=0 for is_live_event=1 (P3-D)
  - A_extra geo_block: three new sub-checks for event_type, technical signals,
    and content_satisfaction (P1-A, P1-B, P1-C)
  - B25: new check — casting_usage_pref=0 for primary_device=smart_tv (P2-F)

Exit codes: 0=all PASS/WARN, 1=any FAIL
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Module-level paths ────────────────────────────────────────────────────────

EVENTS_PATH    = "session_events.csv"
PROFILES_PATH  = "user_profiles.csv"
DICT_EVENTS    = "data_dictionary_session_events.json"
DICT_PROFILES  = "data_dictionary_user_profiles.json"
REPORT_PATH    = "dataset_health_report.csv"
SUMMARY_PATH   = "health_summary.txt"

# ── Expected distributions ─────────────────────────────────────────────────────

SEGMENT_WEIGHTS = {
    "binge_heavy": 0.25, "casual_dip": 0.30, "completion_obsessed": 0.20,
    "quick_churn": 0.15, "re_engager": 0.10,
}
COMPLETION_ORDER = [
    "quick_churn","casual_dip","re_engager","binge_heavy","completion_obsessed",
]
PLAN_BASE_PRICE = {"basic":5.99,"standard":10.99,"premium":15.99}
PLAN_DISCOUNT   = {"basic":0.20,"standard":0.15,"premium":0.10}
BUNDLE_ADD      = 3.00
TOLERANCE       = 1e-4

VALID_LIFECYCLE_STAGES  = {"active","at_risk","churned","reactivated","new"}
VALID_PAYMENT_METHODS   = {"card","mobile_money","carrier_billing"}
VALID_PLAN_TRAJECTORIES = {"none","upgraded","downgraded"}
VALID_COMM_CHANNELS     = {"email","push","sms"}
VALID_UNAVAIL_REASONS   = {"geo_block","content_pulled","search_no_result"}

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, list, list]:
    sess = pd.read_csv(EVENTS_PATH)
    prof = pd.read_csv(PROFILES_PATH)
    print(f"[INFO] session_events  : {len(sess):,} rows × {len(sess.columns)} cols")
    print(f"[INFO] user_profiles   : {len(prof):,} rows × {len(prof.columns)} cols")
    with open(DICT_EVENTS,   encoding="utf-8") as f: dd_ev = json.load(f)
    with open(DICT_PROFILES, encoding="utf-8") as f: dd_pr = json.load(f)
    return sess, prof, dd_ev, dd_pr

def r(check: str, column: str, status: str, detail: str) -> Dict[str, str]:
    severity = {"PASS":"OK","WARN":"WARN","FAIL":"ERROR","INFO":"INFO","SKIP":"SKIP"}.get(status,"OK")
    return {"check":check,"column":column,"status":status,"severity":severity,"detail":detail}

# ══════════════════════════════════════════════════════════════════════════════
# A. SESSION EVENTS (14 checks)
# ══════════════════════════════════════════════════════════════════════════════

def check_A1_uniqueness(sess: pd.DataFrame) -> List[Dict]:
    dup = int(sess["event_id"].duplicated().sum())
    return [r("A1_uniqueness","event_id","PASS" if dup==0 else "FAIL",
              f"All {len(sess):,} event_ids unique" if dup==0 else f"{dup:,} duplicate event_ids")]

def check_A2_nullability(sess: pd.DataFrame, dd: list) -> List[Dict]:
    results = []
    for entry in dd:
        col = entry["column"]
        if col not in sess.columns:
            results.append(r("A2_nullability",col,"FAIL","Column missing from session_events"))
            continue
        null_n = int(sess[col].isnull().sum())
        if not entry["nullable"] and null_n > 0:
            results.append(r("A2_nullability",col,"FAIL",f"{null_n:,} nulls in non-nullable column"))
        elif entry["nullable"] and null_n > 0:
            pct = null_n/len(sess)*100
            results.append(r("A2_nullability",col,"INFO",f"{null_n:,} nulls ({pct:.1f}%) — sparse field, expected"))
        else:
            results.append(r("A2_nullability",col,"PASS","No unexpected nulls"))
    return results

def check_A3_dtypes(sess: pd.DataFrame, dd: list) -> List[Dict]:
    results = []
    for entry in dd:
        col = entry["column"]
        if col not in sess.columns: continue
        expected = entry["dtype"]
        series   = sess[col].dropna()
        if len(series) == 0:
            results.append(r("A3_dtype",col,"INFO","All values null"))
            continue
        if expected == "float":
            try:
                pd.to_numeric(series, errors="raise")
                results.append(r("A3_dtype",col,"PASS","Numeric confirmed"))
            except Exception:
                results.append(r("A3_dtype",col,"FAIL","Non-numeric in float column"))
        elif expected == "integer":
            num = pd.to_numeric(series, errors="coerce")
            ni  = int((num != num.round()).sum())
            results.append(r("A3_dtype",col,"PASS" if ni==0 else "WARN",
                             "Integer confirmed" if ni==0 else f"{ni} non-integer values"))
        elif expected == "boolean":
            valid = set(series.astype(str).str.lower().unique())
            ok    = valid.issubset({"true","false","1","0"})
            results.append(r("A3_dtype",col,"PASS" if ok else "WARN",
                             "Boolean confirmed" if ok else f"Unexpected values: {valid}"))
        else:
            results.append(r("A3_dtype",col,"PASS",f"{series.nunique()} unique values"))
    return results

def check_A4_ranges(sess: pd.DataFrame) -> List[Dict]:
    RANGES = {
        "completion_pct":            (0.0, 1.0),
        "early_drop_rate":           (0.0, 1.0),
        "mid_session_drop_rate":     (0.0, 1.0),
        "early_drop_flag":           (0,   1),
        "mid_drop_flag":             (0,   1),
        "session_depth":             (1,   20),     # global cap; per-content-type caps checked in A17
        "session_duration_mins":     (1.0, 720.0),
        "days_since_last_session":   (1,   90),     # checked on non-null rows only (NULL valid for session 1)
        "window_day":                (1,   90),
        "avg_bitrate_mbps":          (1.0, 8.0),   # geo_block rows excluded (zeroed by design — P1-B)
        "avg_network_jitter":        (2.0, 80.0),  # geo_block rows excluded (zeroed by design — P1-B)
        "buffer_events":             (0,   20),
        "network_stress_flag":       (0,   1),
        "is_weekend":                (0,   1),
        "skip_intro_flag":           (0,   1),
        "rewatch_flag":              (0,   1),
        "casting_usage_flag":        (0,   1),
        "peak_hour_congestion_flag": (0,   1),
        "content_unavailable_flag":  (0,   1),
        "event_calendar_flag":       (0,   1),
        "is_live_event":             (0,   1),
        "is_exclusive":              (0,   1),
        "franchise_flag":            (0,   1),
        "content_satisfaction":      (1.0, 5.0),
    }
    # Columns where geo_block rows must be excluded — P1-B zeroes these fields
    # for geo_block sessions by design (no data transferred), so [0.0] values
    # are correct for those rows and must not be flagged as range violations.
    GEO_BLOCK_EXEMPT = {"avg_bitrate_mbps", "avg_network_jitter"}

    # Build geo_block mask once — reused for exempt columns
    if "unavailable_reason" in sess.columns:
        is_geo_block = sess["unavailable_reason"].fillna("") == "geo_block"
    else:
        is_geo_block = pd.Series([False] * len(sess), index=sess.index)

    results = []
    for col, (lo, hi) in RANGES.items():
        if col not in sess.columns:
            continue
        # Apply geo_block exclusion for exempt columns
        if col in GEO_BLOCK_EXEMPT:
            sub = sess[~is_geo_block]
            exempt_note = f" (geo_block sessions excluded — zeroed by design)"
        else:
            sub = sess
            exempt_note = ""
        # days_since_last_session is NULL for first sessions — check non-null rows only
        series = pd.to_numeric(sub[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        out_lo = int((series < lo - TOLERANCE).sum())
        out_hi = int((series > hi + TOLERANCE).sum())
        mn, mx = float(series.min()), float(series.max())
        if out_lo > 0 or out_hi > 0:
            results.append(r("A4_range", col, "FAIL",
                             f"Out of [{lo},{hi}]: {out_lo} below, {out_hi} above. "
                             f"Actual: [{mn:.4f},{mx:.4f}]{exempt_note}"))
        else:
            results.append(r("A4_range", col, "PASS",
                             f"All in [{lo},{hi}]. Actual: [{mn:.4f},{mx:.4f}]{exempt_note}"))
    return results

def check_A5_timestamp(sess: pd.DataFrame) -> List[Dict]:
    results = []
    ts  = pd.to_datetime(sess["timestamp"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
    bad = int(ts.isnull().sum())
    if bad == 0:
        results.append(r("A5_timestamp","timestamp","PASS",f"All valid. Range: {ts.min()} to {ts.max()}"))
    else:
        results.append(r("A5_timestamp","timestamp","FAIL",f"{bad:,} unparseable timestamps"))
    if "window_day" in sess.columns:
        wday = pd.to_numeric(sess["window_day"], errors="coerce").dropna()
        med  = float(wday.median())
        mid  = float(wday.max()) / 2
        results.append(r("A5_timestamp","window_day_recency","PASS" if med <= mid else "WARN",
                         f"Recency bias confirmed. Median={med:.1f} <= midpoint={mid:.1f}"
                         if med <= mid else f"Recency bias weak. Median={med:.1f} > midpoint={mid:.1f}"))
    return results

def check_A6_triangle_constraint(sess: pd.DataFrame) -> List[Dict]:
    """A6: triangle constraint — excludes geo_block sessions (different semantics)."""
    if not all(c in sess.columns for c in ["early_drop_rate","mid_session_drop_rate","completion_pct"]):
        return [r("A6_triangle","drop_rates","SKIP","Required columns absent")]
    # Exclude geo_block sessions — their session_duration is 2–5 min regardless of content
    not_geo = sess.get("unavailable_reason","").fillna("") != "geo_block" if "unavailable_reason" in sess.columns else pd.Series([True]*len(sess))
    sub   = sess[not_geo]
    early = pd.to_numeric(sub["early_drop_rate"],       errors="coerce").fillna(0)
    mid   = pd.to_numeric(sub["mid_session_drop_rate"], errors="coerce").fillna(0)
    comp  = pd.to_numeric(sub["completion_pct"],        errors="coerce").fillna(0)
    viol  = int(((early + mid) > (1.0 - comp + TOLERANCE)).sum())
    return [r("A6_triangle","early+mid_drop",
              "PASS" if viol==0 else "FAIL",
              f"Triangle constraint satisfied across {len(sub):,} non-geo-block sessions"
              if viol==0 else f"{viol:,} violations (geo_block sessions excluded)")]

def check_A7_event_type_consistency(sess: pd.DataFrame) -> List[Dict]:
    """A7: event-type consistency + churn/reactivation mutex + reactivation-requires-prior-churn."""
    results = []

    # Churn events: completion < 0.40, churn_flag=True
    churn_ev = sess[sess["event_type"]=="churn"]
    if len(churn_ev):
        high = int((pd.to_numeric(churn_ev["completion_pct"],errors="coerce") > 0.40).sum())
        results.append(r("A7_event_consistency","completion_pct@churn",
                         "PASS" if high==0 else "FAIL",
                         f"All {len(churn_ev):,} churn events completion<=0.40"
                         if high==0 else f"{high} churn events have completion>0.40"))
        flag_false = int(churn_ev["churn_flag"].astype(str).str.lower().eq("false").sum())
        results.append(r("A7_event_consistency","churn_flag@churn",
                         "PASS" if flag_false==0 else "FAIL",
                         "All churn events churn_flag=True" if flag_false==0
                         else f"{flag_false} churn events have churn_flag=False"))

    # Completion events: completion > 0.50
    comp_ev = sess[sess["event_type"]=="completion"]
    if len(comp_ev):
        low = int((pd.to_numeric(comp_ev["completion_pct"],errors="coerce") < 0.50).sum())
        results.append(r("A7_event_consistency","completion_pct@completion",
                         "PASS" if low==0 else "WARN",
                         f"All {len(comp_ev):,} completion events completion>=0.50"
                         if low==0 else f"{low} completion events have completion<0.50"))

    # Reactivation events: days_since >= 30 (exclude session_number=1 where days_since is NULL)
    react_ev = sess[sess["event_type"]=="reactivation"]
    if len(react_ev):
        # Only check rows where days_since is non-null (session_number > 1)
        react_with_gap = react_ev[react_ev["days_since_last_session"].notna()]
        if len(react_with_gap):
            short = int((pd.to_numeric(react_with_gap["days_since_last_session"],
                                       errors="coerce") < 30).sum())
            results.append(r("A7_event_consistency","days_since@reactivation",
                             "PASS" if short==0 else "FAIL",
                             f"All {len(react_with_gap):,} non-first reactivation events have days_since>=30"
                             if short==0 else f"{short} reactivation events have days_since<30"))

    # Churn + reactivation mutex — no session should have both True simultaneously
    if all(c in sess.columns for c in ["churn_flag","reactivation_flag"]):
        churned = sess["churn_flag"].astype(str).str.lower().isin(["true","1"])
        reacted = sess["reactivation_flag"].astype(str).str.lower().isin(["true","1"])
        both    = int((churned & reacted).sum())
        results.append(r("A7_event_consistency","churn+reactivation_mutex",
                         "PASS" if both==0 else "FAIL",
                         "churn_flag and reactivation_flag mutually exclusive in all sessions"
                         if both==0 else
                         f"{both} sessions have BOTH churn_flag=True AND reactivation_flag=True"))

    # Reactivation requires prior churn (Fix A4) — sample up to 500 users for speed
    if all(c in sess.columns for c in ["user_id","reactivation_flag","churn_flag"]):
        react_users = sess[sess["reactivation_flag"].astype(str).str.lower().isin(
            ["true","1"])]["user_id"].unique()
        sample = react_users if len(react_users) <= 500 else np.random.choice(
            react_users, 500, replace=False)
        violations = 0
        for uid in sample:
            u = sess[sess["user_id"]==uid]
            has_churn = u["churn_flag"].astype(str).str.lower().isin(["true","1"]).any()
            if not has_churn:
                violations += 1
        results.append(r("A7_event_consistency","reactivation_requires_prior_churn",
                         "PASS" if violations==0 else "FAIL",
                         f"All {len(sample):,} sampled users with reactivation_flag have prior churn"
                         if violations==0 else
                         f"{violations}/{len(sample)} reactivation users have no prior churn event"))
    return results

def check_A8_session_ordering(sess: pd.DataFrame) -> List[Dict]:
    """A8: session_number ordering within users."""
    if not all(c in sess.columns for c in ["user_id","session_number","timestamp"]):
        return [r("A8_session_order","session_number","SKIP","Required columns absent")]
    violations = 0
    users_checked = 0
    sample_users = sess["user_id"].unique()
    if len(sample_users) > 1000:
        sample_users = np.random.choice(sample_users, 1000, replace=False)
    for uid in sample_users:
        u = sess[sess["user_id"]==uid].copy()
        if len(u) < 2: continue
        u["ts"] = pd.to_datetime(u["timestamp"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
        sn = u.sort_values("ts")["session_number"].values
        if not all(sn[i] <= sn[i+1] for i in range(len(sn)-1)):
            violations += 1
        users_checked += 1
    rate = violations / max(users_checked, 1)
    return [r("A8_session_order","session_number",
              "PASS" if rate < 0.25 else "WARN",
              f"session_number ordering acceptable. {violations}/{users_checked} users "
              f"have minor timestamp overlap ({rate*100:.1f}% < 25% threshold)."
              if rate < 0.25 else
              f"{violations}/{users_checked} ({rate*100:.1f}%) have non-monotonic session_number.")]

def check_A9_segment_distribution(sess: pd.DataFrame) -> List[Dict]:
    sc      = sess["segment_id"].value_counts(normalize=True)
    max_dev = max(abs(sc.get(s,0) - w) for s,w in SEGMENT_WEIGHTS.items())
    actual  = {s: round(float(sc.get(s,0)),3) for s in SEGMENT_WEIGHTS}
    return [r("A9_segment_dist","segment_id","PASS" if max_dev<0.05 else "WARN",
              f"Segment distribution within ±5%. Max deviation={max_dev:.3f}. Actual: {actual}"
              if max_dev<0.05 else
              f"Max deviation={max_dev:.3f} (>0.05). Actual: {actual}")]

def check_A10_completion_ordering(sess: pd.DataFrame) -> List[Dict]:
    seg_means = {}
    for seg in COMPLETION_ORDER:
        sub = pd.to_numeric(sess[sess["segment_id"]==seg]["completion_pct"], errors="coerce").dropna()
        if len(sub): seg_means[seg] = float(sub.mean())
    present    = [s for s in COMPLETION_ORDER if s in seg_means]
    violations = [f"{present[i]}({seg_means[present[i]]:.3f})>{present[i+1]}({seg_means[present[i+1]]:.3f})"
                  for i in range(len(present)-1) if seg_means[present[i]] > seg_means[present[i+1]] + 0.05]
    summary    = ", ".join(f"{s}={seg_means[s]:.3f}" for s in present)
    return [r("A10_completion_order","completion_pct@segment","PASS" if not violations else "WARN",
              f"Ordering correct: {summary}" if not violations else f"Violations: {violations}. Full: {summary}")]

def check_A11_sparse_satisfaction(sess: pd.DataFrame) -> List[Dict]:
    results = []
    if "content_satisfaction" not in sess.columns:
        return [r("A11_sparse_satisfaction","content_satisfaction","SKIP","Column absent")]
    col    = pd.to_numeric(sess["content_satisfaction"], errors="coerce")
    fill   = col.notna().mean() * 100
    n_rated= int(col.notna().sum())
    if 5.0 <= fill <= 60.0:
        results.append(r("A11_sparse_satisfaction","satisfaction_fill","PASS",
                         f"Fill rate={fill:.1f}% (expected 5–60%). {n_rated:,}/{len(sess):,} sessions"))
    else:
        results.append(r("A11_sparse_satisfaction","satisfaction_fill","WARN",
                         f"Fill rate={fill:.1f}% outside 5–60%"))
    rated = col.dropna()
    if len(rated):
        out   = int(((rated < 1.0-TOLERANCE) | (rated > 5.0+TOLERANCE)).sum())
        zeros = int((rated == 0).sum())
        results.append(r("A11_sparse_satisfaction","satisfaction_range",
                         "PASS" if out==0 else "FAIL",
                         "All in [1.0,5.0]" if out==0 else f"{out} outside [1.0,5.0]"))
        results.append(r("A11_sparse_satisfaction","satisfaction_no_zero",
                         "PASS" if zeros==0 else "FAIL",
                         "No zero values" if zeros==0 else f"{zeros} zero values found"))
    if "completion_pct" in sess.columns:
        hi = pd.to_numeric(sess[pd.to_numeric(sess["completion_pct"],errors="coerce")>=0.80]["content_satisfaction"],errors="coerce").mean()
        lo = pd.to_numeric(sess[pd.to_numeric(sess["completion_pct"],errors="coerce")<0.30]["content_satisfaction"],errors="coerce").mean()
        results.append(r("A11_sparse_satisfaction","satisfaction_completion_correlation",
                         "PASS" if hi > lo + 0.3 else "WARN",
                         f"High completion mean={hi:.2f} > low completion mean={lo:.2f} (gap>{0.3:.1f})"
                         if hi > lo + 0.3 else f"Correlation weak. hi={hi:.2f}, lo={lo:.2f}"))
    return results

def check_A12_skip_intro_guard(sess: pd.DataFrame) -> List[Dict]:
    """A12: skip_intro must be 0 for film, shortform, sport, documentary (P3-C).
    A12b: rewatch_flag must be 0 for live events (P3-D)."""
    results = []
    if not all(c in sess.columns for c in ["skip_intro_flag","content_type"]):
        results.append(r("A12_skip_intro_guard","skip_intro_flag","SKIP","Required columns absent"))
    else:
        # P3-C: documentary added alongside film/shortform/sport
        bad = sess[(sess["content_type"].isin(["film","shortform","sport","documentary"])) &
                   (pd.to_numeric(sess["skip_intro_flag"], errors="coerce") == 1)]
        results.append(r("A12_skip_intro_guard","skip_intro_flag",
                  "PASS" if len(bad)==0 else "FAIL",
                  "skip_intro_flag=0 for all film/shortform/sport/documentary (no intro sequence)"
                  if len(bad)==0 else
                  f"{len(bad)} film/shortform/sport/documentary sessions have skip_intro_flag=1"))

    # P3-D: rewatch_flag must be 0 for live events — cannot rewatch a live broadcast
    if all(c in sess.columns for c in ["rewatch_flag","is_live_event"]):
        live    = pd.to_numeric(sess["is_live_event"], errors="coerce") == 1
        rewatch = pd.to_numeric(sess["rewatch_flag"],  errors="coerce") == 1
        bad_rw  = int((live & rewatch).sum())
        live_n  = int(live.sum())
        results.append(r("A12_skip_intro_guard","rewatch_flag@live_event",
                  "PASS" if bad_rw==0 else "FAIL",
                  f"rewatch_flag=0 for all {live_n:,} live event sessions (cannot rewatch live)"
                  if bad_rw==0 else
                  f"{bad_rw} live event sessions have rewatch_flag=1 (semantically invalid)"))
    return results

def check_A13_casting_guard(sess: pd.DataFrame) -> List[Dict]:
    """A13 NEW: casting_usage_flag must be 0 when device_type=smart_tv."""
    if not all(c in sess.columns for c in ["casting_usage_flag","device_type"]):
        return [r("A13_casting_guard","casting_usage_flag","SKIP","Required columns absent")]
    bad = sess[(sess["device_type"]=="smart_tv") &
               (pd.to_numeric(sess["casting_usage_flag"], errors="coerce") == 1)]
    return [r("A13_casting_guard","casting_usage_flag",
              "PASS" if len(bad)==0 else "FAIL",
              "casting_usage_flag=0 for all smart_tv sessions (correct — casting from smart_tv is impossible)"
              if len(bad)==0 else
              f"{len(bad)} smart_tv sessions have casting_usage_flag=1 (semantically invalid)")]

def check_A14_bitrate_network_correlation(sess: pd.DataFrame) -> List[Dict]:
    """A14: Fiber avg_bitrate > 4G avg_bitrate."""
    if not all(c in sess.columns for c in ["avg_bitrate_mbps","network_type"]):
        return [r("A14_bitrate_network","avg_bitrate_mbps","SKIP","Required columns absent")]
    fiber_mean = float(pd.to_numeric(sess[sess["network_type"]=="Fiber"]["avg_bitrate_mbps"], errors="coerce").mean())
    g4_mean    = float(pd.to_numeric(sess[sess["network_type"]=="4G"]["avg_bitrate_mbps"], errors="coerce").mean())
    if fiber_mean > g4_mean + 0.5:
        return [r("A14_bitrate_network","avg_bitrate_mbps","PASS",
                  f"Bitrate correlated with network_type. Fiber mean={fiber_mean:.2f} > 4G mean={g4_mean:.2f} Mbps")]
    return [r("A14_bitrate_network","avg_bitrate_mbps","WARN",
              f"Fiber({fiber_mean:.2f}) not meaningfully > 4G({g4_mean:.2f}). Expected gap >0.5 Mbps.")]


def check_A15_drop_flag_semantics(sess: pd.DataFrame) -> List[Dict]:
    """A15 NEW v5.0: early_drop_flag and mid_drop_flag are per-session facts from completion_pct.
    Validates: early=1 iff completion<0.25; mid=1 iff 0.25<=completion<0.75; mutually exclusive.
    Geo-block sessions (completion=0 by design) are excluded — they are bounce events, not drops."""
    results = []
    if not all(c in sess.columns for c in ["early_drop_flag","mid_drop_flag","completion_pct"]):
        return [r("A15_drop_flags","early_drop_flag","SKIP","Required columns absent")]

    # Exclude geo_block sessions
    not_geo = sess.get("unavailable_reason","").fillna("") != "geo_block" if "unavailable_reason" in sess.columns else pd.Series([True]*len(sess))
    sub = sess[not_geo].copy()
    comp = pd.to_numeric(sub["completion_pct"], errors="coerce")
    ef   = pd.to_numeric(sub["early_drop_flag"], errors="coerce")
    mf   = pd.to_numeric(sub["mid_drop_flag"],   errors="coerce")

    # early_drop_flag = 1 iff completion_pct < 0.25
    early_mismatch = int(((comp < 0.25) != (ef == 1)).sum())
    results.append(r("A15_drop_flags","early_drop_flag",
                     "PASS" if early_mismatch==0 else "FAIL",
                     f"early_drop_flag = (completion_pct < 0.25) for all {len(sub):,} non-geo-block sessions"
                     if early_mismatch==0 else
                     f"{early_mismatch} sessions: early_drop_flag inconsistent with completion_pct threshold"))

    # mid_drop_flag = 1 iff 0.25 <= completion_pct < 0.75
    mid_mismatch = int(((comp >= 0.25) & (comp < 0.75) != (mf == 1)).sum())
    results.append(r("A15_drop_flags","mid_drop_flag",
                     "PASS" if mid_mismatch==0 else "FAIL",
                     f"mid_drop_flag = (0.25 <= completion_pct < 0.75) for all non-geo-block sessions"
                     if mid_mismatch==0 else
                     f"{mid_mismatch} sessions: mid_drop_flag inconsistent with completion_pct threshold"))

    # Mutually exclusive — both cannot be 1 simultaneously
    both = int(((ef==1) & (mf==1)).sum())
    results.append(r("A15_drop_flags","drop_flags_mutex",
                     "PASS" if both==0 else "FAIL",
                     "early_drop_flag and mid_drop_flag mutually exclusive in all sessions"
                     if both==0 else f"{both} sessions have both flags=1"))
    return results


def check_A16_days_since_first_session(sess: pd.DataFrame) -> List[Dict]:
    """A16 NEW v5.0: days_since_last_session must be NULL for session_number=1.
    First sessions have no prior observed session — the gap is undefined."""
    if not all(c in sess.columns for c in ["session_number","days_since_last_session"]):
        return [r("A16_days_since_first","days_since_last_session","SKIP","Required columns absent")]
    s1 = sess[pd.to_numeric(sess["session_number"], errors="coerce") == 1]
    non_null_first = int(s1["days_since_last_session"].notna().sum())
    total_s1 = len(s1)
    if non_null_first == 0:
        return [r("A16_days_since_first","days_since_last_session","PASS",
                  f"All {total_s1:,} session_number=1 rows have days_since_last_session=NULL (correct)")]
    return [r("A16_days_since_first","days_since_last_session","FAIL",
              f"{non_null_first:,}/{total_s1:,} session_number=1 rows have non-NULL days_since_last_session "
              f"(semantically invalid — no prior session exists in the observation window)")]


def check_A17_session_depth_caps(sess: pd.DataFrame) -> List[Dict]:
    """A17 NEW v5.0: session_depth content-type caps.
    film/sport: max 3 | series/documentary: max 8 | shortform: max 20."""
    if not all(c in sess.columns for c in ["session_depth","content_type"]):
        return [r("A17_depth_caps","session_depth","SKIP","Required columns absent")]
    CAPS = {"film":3,"sport":3,"series":8,"documentary":8,"shortform":20}
    results = []
    for ct, cap in CAPS.items():
        sub  = sess[sess["content_type"]==ct]
        if len(sub) == 0:
            continue
        viol = int((pd.to_numeric(sub["session_depth"], errors="coerce") > cap).sum())
        mx   = int(pd.to_numeric(sub["session_depth"], errors="coerce").max())
        results.append(r("A17_depth_caps",f"session_depth@{ct}",
                         "PASS" if viol==0 else "FAIL",
                         f"{ct}: all session_depth <= {cap} (max observed={mx})"
                         if viol==0 else
                         f"{ct}: {viol} sessions exceed cap of {cap} (max={mx})"))
    return results


def check_A18_network_stress_flag(sess: pd.DataFrame) -> List[Dict]:
    """A18 NEW v5.0: network_stress_flag = 1 iff avg_network_jitter > 50ms."""
    if not all(c in sess.columns for c in ["network_stress_flag","avg_network_jitter"]):
        return [r("A18_network_stress","network_stress_flag","SKIP","Required columns absent")]
    jitter = pd.to_numeric(sess["avg_network_jitter"], errors="coerce")
    flag   = pd.to_numeric(sess["network_stress_flag"], errors="coerce")
    mismatch = int(((jitter > 50.0) != (flag == 1)).sum())
    stress_pct = float((flag==1).mean()*100)
    return [r("A18_network_stress","network_stress_flag",
              "PASS" if mismatch==0 else "FAIL",
              f"network_stress_flag = (jitter > 50ms) for all sessions. "
              f"{int((flag==1).sum()):,} stress sessions ({stress_pct:.1f}%)"
              if mismatch==0 else
              f"{mismatch} sessions: network_stress_flag inconsistent with avg_network_jitter threshold")]


def check_A19_episode_position(sess: pd.DataFrame) -> List[Dict]:
    """A19: episode_position validity.
    - NULL only for film/shortform/sport (non-episodic).
    - Non-null only for series/documentary.
    - Values restricted to known set.
    - premiere only on user's first watch of each title (P2-D: per-title counter,
      not platform session_number — a user's 5th session overall can be their
      first watch of a new title and correctly receive premiere).
    """
    if "episode_position" not in sess.columns:
        return [r("A19_episode_position","episode_position","SKIP","Column absent")]

    results = []
    ep  = sess["episode_position"].fillna("__null__")
    ct  = sess["content_type"].astype(str)

    # NULL only for film/shortform/sport
    non_episodic = ct.isin(["film","shortform","sport"])
    null_on_episodic   = int(((~non_episodic) & (ep == "__null__")).sum())
    non_null_on_non_ep = int((non_episodic & (ep != "__null__")).sum())

    if null_on_episodic == 0 and non_null_on_non_ep == 0:
        results.append(r("A19_episode_position","episode_position_nullability","PASS",
                         "NULL only for film/shortform/sport; populated for all series/documentary"))
    else:
        detail = []
        if null_on_episodic:   detail.append(f"{null_on_episodic} series/documentary rows have NULL episode_position")
        if non_null_on_non_ep: detail.append(f"{non_null_on_non_ep} film/shortform/sport rows have non-NULL episode_position")
        results.append(r("A19_episode_position","episode_position_nullability","FAIL","; ".join(detail)))

    # Valid values
    valid_vals = {"premiere","mid_season","penultimate_arc","finale","__null__"}
    actual_vals = set(ep.unique())
    invalid = actual_vals - valid_vals
    results.append(r("A19_episode_position","episode_position_values",
                     "PASS" if not invalid else "FAIL",
                     f"All values valid: {actual_vals - {'__null__'}}"
                     if not invalid else f"Invalid values: {invalid}"))

    # premiere only on user's first watch of each title (P2-D: per-title counter).
    # Check: for every (user_id, content_id) pair, premiere must only appear on
    # the row with the earliest timestamp in that pair.
    # Method: rank rows within each (user_id, content_id) group by timestamp,
    # then assert premiere rows are only rank=1.
    if all(c in sess.columns for c in ["user_id","content_id","timestamp"]):
        # Work on episodic content only (premiere is undefined for film/sport/shortform)
        episodic = sess[~non_episodic].copy()
        if len(episodic):
            episodic["_ts"] = pd.to_datetime(
                episodic["timestamp"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce"
            )
            episodic["_title_rank"] = (
                episodic.groupby(["user_id","content_id"])["_ts"]
                .rank(method="first", ascending=True)
            )
            ep_ep = episodic["episode_position"].fillna("__null__")
            # premiere on rank > 1 = violation
            premiere_wrong_rank = int(
                ((ep_ep == "premiere") & (episodic["_title_rank"] > 1)).sum()
            )
            # non-premiere on rank = 1 (for series/doc) is fine — user may have
            # started mid-arc on first platform visit (title_sn advances if
            # they've watched it before on another session — not possible here,
            # but we only flag premiere on wrong rank, not absence of premiere).
            results.append(r("A19_episode_position","premiere_on_first_title_watch",
                             "PASS" if premiere_wrong_rank==0 else "FAIL",
                             "All premiere sessions are the user's first watch of that title"
                             if premiere_wrong_rank==0 else
                             f"{premiere_wrong_rank} premiere sessions are not the user's "
                             f"first watch of that title (per-title rank > 1)"))
    else:
        results.append(r("A19_episode_position","premiere_on_first_title_watch",
                         "SKIP","user_id, content_id, or timestamp column absent"))

    # Distribution summary
    dist = sess["episode_position"].value_counts().to_dict()
    null_n = int(sess["episode_position"].isnull().sum())
    results.append(r("A19_episode_position","episode_position_distribution","INFO",
                     f"Distribution: {dist} | NULL (non-episodic): {null_n:,}"))
    return results

def check_A_unavailability(sess: pd.DataFrame) -> List[Dict]:
    """A_extra: unavailable_reason coherence and geo_block zero-completion."""
    results = []
    if "content_unavailable_flag" not in sess.columns:
        return [r("A_unavailability","content_unavailable_flag","SKIP","Column absent")]

    flag = pd.to_numeric(sess["content_unavailable_flag"], errors="coerce")
    # unavailable_reason is NULL when flag=0, non-null when flag=1
    if "unavailable_reason" in sess.columns:
        unavail_rows = sess[flag == 1]
        null_reason_when_unavail = int(unavail_rows["unavailable_reason"].isnull().sum())
        non_null_reason_when_avail = int(sess[flag == 0]["unavailable_reason"].notna().sum())
        if null_reason_when_unavail == 0 and non_null_reason_when_avail == 0:
            results.append(r("A_unavailability","unavailable_reason","PASS",
                             f"unavailable_reason coherent with content_unavailable_flag. "
                             f"{int(flag.sum()):,} unavailable sessions all have reason."))
        else:
            detail = []
            if null_reason_when_unavail: detail.append(f"{null_reason_when_unavail} unavailable sessions have NULL reason")
            if non_null_reason_when_avail: detail.append(f"{non_null_reason_when_avail} available sessions have non-NULL reason")
            results.append(r("A_unavailability","unavailable_reason","FAIL","; ".join(detail)))

        # Validate reason values
        reasons = sess["unavailable_reason"].dropna().unique()
        invalid = [x for x in reasons if x not in VALID_UNAVAIL_REASONS]
        results.append(r("A_unavailability","unavailable_reason_values",
                         "PASS" if not invalid else "FAIL",
                         f"All reasons valid: {set(reasons)}"
                         if not invalid else f"Invalid reasons: {invalid}"))

        # geo_block sessions must have completion=0, depth=1
        geo_bl = sess[sess["unavailable_reason"] == "geo_block"]
        if len(geo_bl):
            comp_nonzero = int((pd.to_numeric(geo_bl["completion_pct"],errors="coerce") > 0.001).sum())
            depth_non1   = int((pd.to_numeric(geo_bl["session_depth"],errors="coerce") != 1).sum())
            if comp_nonzero == 0 and depth_non1 == 0:
                results.append(r("A_unavailability","geo_block_completeness","PASS",
                                 f"All {len(geo_bl):,} geo_block sessions: completion=0, depth=1 (correct bounce events)"))
            else:
                detail = []
                if comp_nonzero: detail.append(f"{comp_nonzero} geo_block sessions have completion>0")
                if depth_non1:   detail.append(f"{depth_non1} geo_block sessions have depth!=1")
                results.append(r("A_unavailability","geo_block_completeness","FAIL","; ".join(detail)))

            # P1-C: geo_block must have event_type=session_start — not completion or progress
            if "event_type" in sess.columns:
                bad_et = int((geo_bl["event_type"] != "session_start").sum())
                results.append(r("A_unavailability","geo_block_event_type",
                                 "PASS" if bad_et==0 else "FAIL",
                                 f"All {len(geo_bl):,} geo_block sessions have event_type=session_start"
                                 if bad_et==0 else
                                 f"{bad_et} geo_block sessions have event_type != session_start (bounce events must not be completion/progress)"))

            # P1-B: geo_block must have all technical signals zeroed — no data transferred
            tech_checks = [
                ("buffer_events",             0, 0.001),
                ("avg_bitrate_mbps",          0, 0.001),
                ("avg_network_jitter",        0, 0.001),
                ("peak_hour_congestion_flag", 0, 0.001),
                ("network_stress_flag",       0, 0.001),
            ]
            tech_fails = []
            for col, expected, tol in tech_checks:
                if col not in sess.columns:
                    continue
                nonzero = int((pd.to_numeric(geo_bl[col], errors="coerce").fillna(0) > tol).sum())
                if nonzero:
                    tech_fails.append(f"{col}: {nonzero} non-zero")
            results.append(r("A_unavailability","geo_block_technical_signals",
                             "PASS" if not tech_fails else "FAIL",
                             f"All technical signals zero for {len(geo_bl):,} geo_block sessions (no data transferred)"
                             if not tech_fails else
                             f"Non-zero technical signals in geo_block sessions: {'; '.join(tech_fails)}"))

            # P1-A: geo_block must have content_satisfaction=NULL — user never watched
            if "content_satisfaction" in sess.columns:
                sat_nonnull = int(geo_bl["content_satisfaction"].notna().sum())
                results.append(r("A_unavailability","geo_block_satisfaction",
                                 "PASS" if sat_nonnull==0 else "FAIL",
                                 f"All {len(geo_bl):,} geo_block sessions have content_satisfaction=NULL (user never watched)"
                                 if sat_nonnull==0 else
                                 f"{sat_nonnull} geo_block sessions have non-NULL content_satisfaction (invalid — no watch event occurred)"))
    return results

# ══════════════════════════════════════════════════════════════════════════════
# B. USER PROFILES (10 checks)
# ══════════════════════════════════════════════════════════════════════════════

def check_B15_profile_uniqueness(prof: pd.DataFrame, sess: pd.DataFrame) -> List[Dict]:
    results = []
    dup = int(prof["user_id"].duplicated().sum())
    results.append(r("B15_uniqueness","user_id@profiles","PASS" if dup==0 else "FAIL",
                     f"All {len(prof):,} user_ids unique" if dup==0 else f"{dup} duplicate user_ids"))
    su, pu = sess["user_id"].nunique(), prof["user_id"].nunique()
    results.append(r("B15_uniqueness","user_count_match","PASS" if su==pu else "WARN",
                     f"Profile count ({pu:,}) matches session user count ({su:,})"
                     if su==pu else f"Profile ({pu:,}) != session ({su:,}) user count"))
    return results

def check_B16_profile_nullability(prof: pd.DataFrame, dd: list) -> List[Dict]:
    results = []
    for entry in dd:
        col = entry["column"]
        if col not in prof.columns:
            results.append(r("B16_nullability",col,"FAIL","Column missing from user_profiles"))
            continue
        null_n = int(prof[col].isnull().sum())
        if not entry["nullable"] and null_n > 0:
            results.append(r("B16_nullability",col,"FAIL",f"{null_n:,} nulls in non-nullable column"))
        elif entry["nullable"] and null_n > 0:
            results.append(r("B16_nullability",col,"INFO",f"{null_n:,} nulls ({null_n/len(prof)*100:.1f}%) — sparse, expected"))
        else:
            results.append(r("B16_nullability",col,"PASS","No unexpected nulls"))
    return results

def check_B17_profile_ranges(prof: pd.DataFrame) -> List[Dict]:
    RANGES = {
        "monthly_price":             (4.79, 19.00),
        "tenure_months":             (1,    60),
        "is_bundle":                 (0,    1),
        "discount_flag":             (0,    1),
        "casting_usage_pref":        (0,    1),
        "promo_code_usage":          (0,    1),
        "push_notification_opt_in":  (0,    1),
        "sports_dependency_score":   (0.0,  1.0),
        "binge_index":               (0.0,  1.0),
        "avg_watch_gap_days":        (0.0,  90.0),  # P3-A+B: sub-hour binge gaps can round to 0.0
        "last_active_days_ago":      (0,    90),
        "account_health_score":      (0.0,  1.0),
        "ltv_to_date":               (4.0,  1200.0),
        "fav_genre_confidence":      (0.0,  1.0),
        "campaign_response_rate":    (0.0,  1.0),
        "email_open_rate":           (0.0,  1.0),
        "email_click_rate":          (0.0,  1.0),
        "ppv_purchase_count":        (1,    8),
        "payment_failure_count":     (1,    6),
        "ticket_count":              (1,    4),
        "ticket_avg_resolution_hrs": (0.5,  48.0),
    }
    results = []
    for col, (lo, hi) in RANGES.items():
        if col not in prof.columns: continue
        series = pd.to_numeric(prof[col], errors="coerce").dropna()
        if len(series) == 0: continue
        out_lo = int((series < lo - TOLERANCE).sum())
        out_hi = int((series > hi + TOLERANCE).sum())
        mn, mx = float(series.min()), float(series.max())
        results.append(r("B17_range",col,"PASS" if out_lo==0 and out_hi==0 else "FAIL",
                         f"All in [{lo},{hi}]. Actual: [{mn:.4f},{mx:.4f}]"
                         if out_lo==0 and out_hi==0 else
                         f"Out of [{lo},{hi}]: {out_lo} below, {out_hi} above. Actual: [{mn:.4f},{mx:.4f}]"))
    return results

def check_B18_price_coherence(prof: pd.DataFrame) -> List[Dict]:
    if not all(c in prof.columns for c in ["monthly_price","plan_type","is_bundle","discount_flag"]):
        return [r("B18_price","monthly_price","SKIP","Required columns absent")]
    violations = 0
    n_checked  = 0
    for _, row in prof[["monthly_price","plan_type","is_bundle","discount_flag"]].dropna().iterrows():
        plan     = str(row["plan_type"]).lower()
        base     = PLAN_BASE_PRICE.get(plan, 10.99)
        bundle   = BUNDLE_ADD if str(row["is_bundle"]) in ("1","True","true") else 0.0
        disc     = PLAN_DISCOUNT.get(plan,0.0) if str(row["discount_flag"]) in ("1","True","true") else 0.0
        expected = round((base + bundle) * (1 - disc), 2)
        if abs(float(row["monthly_price"]) - expected) > 0.10:
            violations += 1
        n_checked += 1
    return [r("B18_price","monthly_price","PASS" if violations==0 else "FAIL",
              f"All {n_checked:,} monthly_price values consistent with plan+bundle+discount"
              if violations==0 else f"{violations:,}/{n_checked:,} inconsistent")]

def check_B19_trajectory_validity(prof: pd.DataFrame) -> List[Dict]:
    results = []
    for col, (lo, hi) in {
        "binge_index":(0.0,1.0), "avg_watch_gap_days":(0.0,90.0), "last_active_days_ago":(0,90),
        "account_health_score":(0.0,1.0), "ltv_to_date":(4.0,1200.0),
    }.items():
        if col not in prof.columns:
            results.append(r("B19_trajectory",col,"SKIP","Column absent"))
            continue
        null_n = int(prof[col].isnull().sum())
        if null_n > 0:
            results.append(r("B19_trajectory",col,"FAIL",f"{null_n:,} nulls in derived field"))
        else:
            series = pd.to_numeric(prof[col], errors="coerce")
            out    = int(((series < lo) | (series > hi)).sum())
            mn, mx = float(series.min()), float(series.max())
            results.append(r("B19_trajectory",col,"PASS" if out==0 else "WARN",
                             f"All in [{lo},{hi}]. Actual: [{mn:.4f},{mx:.4f}]"
                             if out==0 else f"{out} outside [{lo},{hi}]. Actual: [{mn:.4f},{mx:.4f}]"))
    for col in ["fav_genre"]:
        if col in prof.columns:
            null_n = int(prof[col].isnull().sum())
            results.append(r("B19_trajectory",col,"PASS" if null_n==0 else "FAIL",
                             f"Populated for all {len(prof):,} users"
                             if null_n==0 else f"{null_n:,} null values"))

    # fav_genre_confidence: non-null, [0,1]
    if "fav_genre_confidence" in prof.columns:
        null_n = int(prof["fav_genre_confidence"].isnull().sum())
        if null_n > 0:
            results.append(r("B19_trajectory","fav_genre_confidence","FAIL",
                             f"{null_n:,} null values in fav_genre_confidence"))
        else:
            conf = pd.to_numeric(prof["fav_genre_confidence"], errors="coerce")
            mn, mx = float(conf.min()), float(conf.max())
            mean   = float(conf.mean())
            results.append(r("B19_trajectory","fav_genre_confidence","PASS",
                             f"Range [{mn:.3f},{mx:.3f}], mean={mean:.3f}. "
                             f"{int((conf < 0.4).sum()):,} users with weak genre preference (confidence<0.4)"))
    return results

def check_B20_new_profile_fields(prof: pd.DataFrame) -> List[Dict]:
    """B20 NEW: validate new v4.0 profile fields — lifecycle_stage, payment_method, etc."""
    results = []

    # lifecycle_stage valid values
    if "lifecycle_stage" in prof.columns:
        vals = set(prof["lifecycle_stage"].dropna().unique())
        invalid = vals - VALID_LIFECYCLE_STAGES
        results.append(r("B20_new_fields","lifecycle_stage","PASS" if not invalid else "FAIL",
                         f"All lifecycle stages valid: {vals}"
                         if not invalid else f"Invalid stages: {invalid}"))

    # payment_method_type valid values
    if "payment_method_type" in prof.columns:
        vals = set(prof["payment_method_type"].dropna().unique())
        invalid = vals - VALID_PAYMENT_METHODS
        results.append(r("B20_new_fields","payment_method_type","PASS" if not invalid else "FAIL",
                         f"All payment methods valid: {vals}"
                         if not invalid else f"Invalid methods: {invalid}"))

    # upgrade_downgrade_history valid values
    if "upgrade_downgrade_history" in prof.columns:
        vals = set(prof["upgrade_downgrade_history"].dropna().unique())
        invalid = vals - VALID_PLAN_TRAJECTORIES
        results.append(r("B20_new_fields","upgrade_downgrade_history","PASS" if not invalid else "FAIL",
                         f"All trajectories valid: {vals}"
                         if not invalid else f"Invalid: {invalid}"))

    # preferred_communication_channel valid values
    if "preferred_communication_channel" in prof.columns:
        vals = set(prof["preferred_communication_channel"].dropna().unique())
        invalid = vals - VALID_COMM_CHANNELS
        results.append(r("B20_new_fields","preferred_communication_channel",
                         "PASS" if not invalid else "FAIL",
                         f"All channels valid: {vals}"
                         if not invalid else f"Invalid: {invalid}"))

    # email_click_rate <= email_open_rate
    if all(c in prof.columns for c in ["email_open_rate","email_click_rate"]):
        open_r  = pd.to_numeric(prof["email_open_rate"],  errors="coerce")
        click_r = pd.to_numeric(prof["email_click_rate"], errors="coerce")
        both    = open_r.notna() & click_r.notna()
        violations = int((click_r[both] > open_r[both] + TOLERANCE).sum())
        results.append(r("B20_new_fields","email_click_rate<=open_rate",
                         "PASS" if violations==0 else "FAIL",
                         "email_click_rate <= email_open_rate for all users with email data"
                         if violations==0 else f"{violations} users have click_rate > open_rate"))

    # email_open_rate and email_click_rate null together
    if all(c in prof.columns for c in ["email_open_rate","email_click_rate"]):
        open_null  = prof["email_open_rate"].isnull()
        click_null = prof["email_click_rate"].isnull()
        incoherent = int((open_null != click_null).sum())
        results.append(r("B20_new_fields","email_nullability_coherence",
                         "PASS" if incoherent==0 else "FAIL",
                         "email_open_rate and email_click_rate null together"
                         if incoherent==0 else f"{incoherent} users have mismatched email field nullability"))

    # ab_test_group_history valid values
    if "ab_test_group_history" in prof.columns:
        vals    = set(prof["ab_test_group_history"].dropna().unique())
        invalid = vals - {"treatment","control","not_tested"}
        null_n  = int(prof["ab_test_group_history"].isnull().sum())
        results.append(r("B20_new_fields","ab_test_group_history",
                         "PASS" if not invalid and null_n==0 else "FAIL",
                         f"All values valid: {vals}. No nulls."
                         if not invalid and null_n==0 else
                         f"Invalid values: {invalid}. Nulls: {null_n}"))

    # campaign_response_rate: sparse [0,1], ~40% null
    if "campaign_response_rate" in prof.columns:
        cr   = pd.to_numeric(prof["campaign_response_rate"], errors="coerce")
        fill = cr.notna().mean() * 100
        out  = int(((cr.dropna() < 0) | (cr.dropna() > 1)).sum())
        status = "PASS" if (30 <= fill <= 75) and out == 0 else "WARN" if out == 0 else "FAIL"
        results.append(r("B20_new_fields","campaign_response_rate",status,
                         f"Fill rate={fill:.1f}% (expected 30–75%). Values in [0,1]: {out==0}"))

    # ppv_purchase_count: sparse, non-null values in [1,8]
    if "ppv_purchase_count" in prof.columns:
        ppv  = pd.to_numeric(prof["ppv_purchase_count"], errors="coerce")
        fill = ppv.notna().mean() * 100
        out  = int(((ppv.dropna() < 1) | (ppv.dropna() > 8)).sum())
        results.append(r("B20_new_fields","ppv_purchase_count",
                         "PASS" if out==0 else "FAIL",
                         f"Fill rate={fill:.1f}% (expected <15%). All counts in [1,8]."
                         if out==0 else f"{out} values outside [1,8]"))

    # merchandise_purchase_flag: sparse binary (0 or 1 when non-null)
    if "merchandise_purchase_flag" in prof.columns:
        mf   = pd.to_numeric(prof["merchandise_purchase_flag"], errors="coerce")
        fill = mf.notna().mean() * 100
        out  = int(((mf.dropna() != 0) & (mf.dropna() != 1)).sum())
        results.append(r("B20_new_fields","merchandise_purchase_flag",
                         "PASS" if out==0 else "FAIL",
                         f"Fill rate={fill:.1f}% (expected <10%). All values binary when non-null."
                         if out==0 else f"{out} non-binary values in merchandise_purchase_flag"))
    return results

def check_B21_sparse_tickets(prof: pd.DataFrame) -> List[Dict]:
    results = []
    ticket_cols = ["ticket_count","ticket_issue_types","ticket_avg_resolution_hrs"]
    if not all(c in prof.columns for c in ticket_cols):
        return [r("B21_sparse_tickets","ticket_fields","SKIP","One or more ticket columns absent")]
    tc = prof["ticket_count"].isnull()
    ti = prof["ticket_issue_types"].isnull()
    tr = prof["ticket_avg_resolution_hrs"].isnull()
    viol = int(((tc != ti) | (tc != tr)).sum())
    results.append(r("B21_sparse_tickets","ticket_null_coherence","PASS" if viol==0 else "FAIL",
                     "All three ticket fields null together"
                     if viol==0 else f"{viol} users have inconsistent ticket field nullability"))
    n_usr = len(prof); n_tick = int(prof["ticket_count"].notna().sum()); fill = n_tick/n_usr*100
    results.append(r("B21_sparse_tickets","ticket_fill","PASS" if 5.0<=fill<=40.0 else "WARN",
                     f"Fill rate={fill:.1f}% (expected 5–40%). {n_tick:,}/{n_usr:,} users"))
    valid_issues = {"buffering","billing","login","content_unavailable","other"}
    invalid = sum(1 for val in prof["ticket_issue_types"].dropna()
                  if not set(str(val).split(",")).issubset(valid_issues))
    results.append(r("B21_sparse_tickets","ticket_issue_values","PASS" if invalid==0 else "FAIL",
                     "All issue types valid" if invalid==0 else f"{invalid} rows with invalid types"))
    return results

def check_B22_ticket_payment_correlation(prof: pd.DataFrame) -> List[Dict]:
    if not all(c in prof.columns for c in ["payment_failure_count","ticket_count"]):
        return [r("B22_ticket_payment","ticket+payment","SKIP","Required columns absent")]
    has_fail  = prof["payment_failure_count"].notna()
    tick_fail = float(prof[has_fail]["ticket_count"].notna().mean())
    tick_nofail = float(prof[~has_fail]["ticket_count"].notna().mean())
    return [r("B22_ticket_payment","ticket_rate@payment_failure",
              "PASS" if tick_fail > tick_nofail + 0.05 else "WARN",
              f"Users with payment failures have higher ticket rate. fail={tick_fail:.3f} > no_fail={tick_nofail:.3f}"
              if tick_fail > tick_nofail + 0.05 else
              f"Payment failure → ticket lift insufficient. fail={tick_fail:.3f}, no_fail={tick_nofail:.3f}")]

def check_B23_ticket_segment_ordering(prof: pd.DataFrame) -> List[Dict]:
    if not all(c in prof.columns for c in ["segment_id","ticket_count"]):
        return [r("B23_ticket_segment","ticket@segment","SKIP","Required columns absent")]
    rates = {seg: float(prof[prof["segment_id"]==seg]["ticket_count"].notna().mean())
             for seg in ["quick_churn","casual_dip","re_engager","binge_heavy","completion_obsessed"]
             if len(prof[prof["segment_id"]==seg]) > 0}
    qc = rates.get("quick_churn",0.0); bh = rates.get("binge_heavy",1.0)
    rate_str = ", ".join(f"{s}={v:.3f}" for s,v in rates.items())
    return [r("B23_ticket_segment","ticket_count@segment","PASS" if qc>bh+0.10 else "WARN",
              f"quick_churn({qc:.3f}) > binge_heavy({bh:.3f}). Full: {rate_str}"
              if qc>bh+0.10 else f"quick_churn({qc:.3f}) not > binge_heavy({bh:.3f}) + 0.10. Full: {rate_str}")]

def check_B24_trajectory_segment_correlation(prof: pd.DataFrame) -> List[Dict]:
    if not all(c in prof.columns for c in ["segment_id","binge_index"]):
        return [r("B24_trajectory_segment","binge_index@segment","SKIP","Required columns absent")]
    bh = float(pd.to_numeric(prof[prof.segment_id=="binge_heavy"]["binge_index"],errors="coerce").mean())
    qc = float(pd.to_numeric(prof[prof.segment_id=="quick_churn"]["binge_index"],errors="coerce").mean())
    return [r("B24_trajectory_segment","binge_index@segment",
              "PASS" if bh>qc+0.05 else "WARN",
              f"binge_heavy({bh:.4f}) > quick_churn({qc:.4f}). Multi-session trajectory working."
              if bh>qc+0.05 else
              f"binge_heavy({bh:.4f}) not > quick_churn({qc:.4f}) + 0.05. More sessions/user recommended.")]

def check_B25_casting_pref_guard(prof: pd.DataFrame) -> List[Dict]:
    """B25 NEW v6.0: casting_usage_pref must be 0 for primary_device=smart_tv (P2-F fix).
    Smart_tv is the cast destination — it cannot initiate a cast."""
    if not all(c in prof.columns for c in ["casting_usage_pref","primary_device"]):
        return [r("B25_casting_pref_guard","casting_usage_pref","SKIP","Required columns absent")]
    bad = prof[(prof["primary_device"]=="smart_tv") &
               (pd.to_numeric(prof["casting_usage_pref"], errors="coerce") == 1)]
    return [r("B25_casting_pref_guard","casting_usage_pref",
              "PASS" if len(bad)==0 else "FAIL",
              "casting_usage_pref=0 for all smart_tv primary users (smart_tv cannot initiate a cast)"
              if len(bad)==0 else
              f"{len(bad)} smart_tv primary users have casting_usage_pref=1 (semantically invalid)")]

# ══════════════════════════════════════════════════════════════════════════════
# C. CROSS-TABLE INTEGRITY (2 checks)
# ══════════════════════════════════════════════════════════════════════════════

def check_C25_sessions_have_profiles(sess: pd.DataFrame, prof: pd.DataFrame) -> List[Dict]:
    missing = set(sess["user_id"].unique()) - set(prof["user_id"].unique())
    return [r("C25_cross_table","user_id:sessions→profiles","PASS" if not missing else "FAIL",
              f"All {sess['user_id'].nunique():,} session user_ids have a profile row"
              if not missing else f"{len(missing):,} session user_ids have no profile row")]

def check_C26_profiles_have_sessions(prof: pd.DataFrame, sess: pd.DataFrame) -> List[Dict]:
    orphans = set(prof["user_id"].unique()) - set(sess["user_id"].unique())
    return [r("C26_cross_table","user_id:profiles→sessions","PASS" if not orphans else "FAIL",
              f"All {prof['user_id'].nunique():,} profile user_ids appear in session_events"
              if not orphans else f"{len(orphans):,} profile user_ids have no session events")]

# ══════════════════════════════════════════════════════════════════════════════
# COVERAGE STATS + WRITERS
# ══════════════════════════════════════════════════════════════════════════════

def coverage_stats(sess: pd.DataFrame, prof: pd.DataFrame) -> Dict[str, Any]:
    unavail = int((pd.to_numeric(sess.get("content_unavailable_flag",0),errors="coerce")==1).sum())
    geo_bl  = int((sess.get("unavailable_reason","").fillna("")=="geo_block").sum()) if "unavailable_reason" in sess.columns else "N/A"
    return {
        "total_sessions":             len(sess),
        "total_session_columns":      len(sess.columns),
        "unique_users_in_sessions":   sess["user_id"].nunique(),
        "unique_content_titles":      sess["content_id"].nunique(),
        "segments":                   sess["segment_id"].nunique(),
        "sessions_per_user_mean":     round(sess.groupby("user_id").size().mean(), 2),
        "avg_completion_pct":         round(pd.to_numeric(sess["completion_pct"],errors="coerce").mean(), 4),
        "churn_flag_rate":            round(sess["churn_flag"].astype(str).str.lower().eq("true").mean(), 4),
        "satisfaction_fill_pct":      round(sess["content_satisfaction"].notna().mean()*100 if "content_satisfaction" in sess.columns else 0, 1),
        "content_unavail_sessions":   unavail,
        "geo_block_sessions":         geo_bl,
        "network_stress_sessions":    int((pd.to_numeric(sess.get("network_stress_flag",0),errors="coerce")==1).sum()),
        "exclusive_sessions":         int((pd.to_numeric(sess.get("is_exclusive",0),errors="coerce")==1).sum()),
        "total_profiles":             len(prof),
        "total_profile_columns":      len(prof.columns),
        "payment_failure_fill_pct":   round(prof["payment_failure_count"].notna().mean()*100 if "payment_failure_count" in prof.columns else 0, 1),
        "ticket_fill_pct":            round(prof["ticket_count"].notna().mean()*100 if "ticket_count" in prof.columns else 0, 1),
        "email_data_fill_pct":        round(prof["email_open_rate"].notna().mean()*100 if "email_open_rate" in prof.columns else 0, 1),
        "avg_binge_index":            round(pd.to_numeric(prof["binge_index"],errors="coerce").mean(), 4) if "binge_index" in prof.columns else "N/A",
        "avg_watch_gap_days":         round(pd.to_numeric(prof["avg_watch_gap_days"],errors="coerce").mean(), 2) if "avg_watch_gap_days" in prof.columns else "N/A",
        "first_session_null_days_pct": round(
            sess[pd.to_numeric(sess.get("session_number",1),errors="coerce")==1]["days_since_last_session"].isnull().mean()*100
            if "session_number" in sess.columns and "days_since_last_session" in sess.columns else 0, 1),
        "avg_fav_genre_confidence":   round(pd.to_numeric(prof["fav_genre_confidence"],errors="coerce").mean(), 3) if "fav_genre_confidence" in prof.columns else "N/A",
        "ppv_purchasers_pct":         round(prof["ppv_purchase_count"].notna().mean()*100, 1) if "ppv_purchase_count" in prof.columns else "N/A",
        "campaign_data_fill_pct":     round(prof["campaign_response_rate"].notna().mean()*100, 1) if "campaign_response_rate" in prof.columns else "N/A",
        "avg_ltv_to_date":            round(pd.to_numeric(prof["ltv_to_date"],errors="coerce").mean(), 2) if "ltv_to_date" in prof.columns else "N/A",
        "lifecycle_stage_dist":       prof["lifecycle_stage"].value_counts().to_dict() if "lifecycle_stage" in prof.columns else "N/A",
        "payment_method_dist":        prof["payment_method_type"].value_counts().to_dict() if "payment_method_type" in prof.columns else "N/A",
    }

def write_summary(all_results: List[Dict], stats: Dict, path: str) -> None:
    total  = len(all_results)
    passes = sum(1 for x in all_results if x["status"]=="PASS")
    warns  = sum(1 for x in all_results if x["status"]=="WARN")
    fails  = sum(1 for x in all_results if x["status"]=="FAIL")
    infos  = sum(1 for x in all_results if x["status"]=="INFO")
    overall= "HEALTHY" if fails==0 else "NEEDS ATTENTION"

    lines = ["="*80,"TWINSIM — DATASET HEALTH REPORT  v6.0","="*80,
             f"Session events : {EVENTS_PATH}",f"User profiles  : {PROFILES_PATH}",""]
    lines += ["── COVERAGE STATISTICS ─────────────────────────────────────────────────────"]
    for k,v in stats.items():
        lines.append(f"  {k:<44} {v}")
    lines += ["","── CHECK SUMMARY ───────────────────────────────────────────────────────────",
              f"  Total checks : {total}",f"  PASS         : {passes}",
              f"  WARN         : {warns}", f"  INFO         : {infos}",
              f"  FAIL         : {fails}", f"\n  OVERALL      : {overall}",""]

    ICONS = {"PASS":"✓","FAIL":"✗","WARN":"⚠","INFO":"ℹ","SKIP":"·"}
    for group_name, prefix in [
        ("A — Session events",        "A"),
        ("B — User profiles",         "B"),
        ("C — Cross-table integrity", "C"),
    ]:
        group = [x for x in all_results if x["check"].startswith(prefix)]
        if not group: continue
        lines.append(f"── {group_name} {'─'*(75-len(group_name))}")
        for res in group:
            icon = ICONS.get(res["status"],"·")
            lines.append(f"  [{icon}] [{res['status']:<5}] {res['column']:<42} {res['detail']}")
        lines.append("")

    with open(path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[DONE] Health summary → {path}")

def write_report_csv(all_results: List[Dict], path: str) -> None:
    fields = ["check","column","status","severity","detail"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"[DONE] Health report  → {path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global EVENTS_PATH, PROFILES_PATH, DICT_EVENTS, DICT_PROFILES, REPORT_PATH, SUMMARY_PATH

    ap = argparse.ArgumentParser(description="TwinSim Dataset Health Report v6.0")
    ap.add_argument("--events",        default=None)
    ap.add_argument("--profiles",      default=None)
    ap.add_argument("--dict_events",   default=None)
    ap.add_argument("--dict_profiles", default=None)
    ap.add_argument("--out","-o",      default=None)
    ap.add_argument("--summary","-s",  default=None)
    args = ap.parse_args()

    if args.events:        EVENTS_PATH    = args.events
    if args.profiles:      PROFILES_PATH  = args.profiles
    if args.dict_events:   DICT_EVENTS    = args.dict_events
    if args.dict_profiles: DICT_PROFILES  = args.dict_profiles
    if args.out:           REPORT_PATH    = args.out
    if args.summary:       SUMMARY_PATH   = args.summary

    for path, label in [(EVENTS_PATH,"session_events CSV"),(PROFILES_PATH,"user_profiles CSV"),
                         (DICT_EVENTS,"session dict"),(DICT_PROFILES,"profiles dict")]:
        if not os.path.isfile(path):
            print(f"[ERROR] {label} not found: {path}")
            raise SystemExit(1)

    print("[INFO] Loading datasets...")
    sess, prof, dd_ev, dd_pr = load_all()
    print("[INFO] Running health checks...")

    all_results: List[Dict] = []

    # A — Session events
    all_results += check_A1_uniqueness(sess)
    all_results += check_A2_nullability(sess, dd_ev)
    all_results += check_A3_dtypes(sess, dd_ev)
    all_results += check_A4_ranges(sess)
    all_results += check_A5_timestamp(sess)
    all_results += check_A6_triangle_constraint(sess)
    all_results += check_A7_event_type_consistency(sess)
    all_results += check_A8_session_ordering(sess)
    all_results += check_A9_segment_distribution(sess)
    all_results += check_A10_completion_ordering(sess)
    all_results += check_A11_sparse_satisfaction(sess)
    all_results += check_A12_skip_intro_guard(sess)
    all_results += check_A13_casting_guard(sess)
    all_results += check_A14_bitrate_network_correlation(sess)
    all_results += check_A15_drop_flag_semantics(sess)
    all_results += check_A16_days_since_first_session(sess)
    all_results += check_A17_session_depth_caps(sess)
    all_results += check_A18_network_stress_flag(sess)
    all_results += check_A19_episode_position(sess)
    all_results += check_A_unavailability(sess)

    # B — User profiles
    all_results += check_B15_profile_uniqueness(prof, sess)
    all_results += check_B16_profile_nullability(prof, dd_pr)
    all_results += check_B17_profile_ranges(prof)
    all_results += check_B18_price_coherence(prof)
    all_results += check_B19_trajectory_validity(prof)
    all_results += check_B20_new_profile_fields(prof)
    all_results += check_B21_sparse_tickets(prof)
    all_results += check_B22_ticket_payment_correlation(prof)
    all_results += check_B23_ticket_segment_ordering(prof)
    all_results += check_B24_trajectory_segment_correlation(prof)
    all_results += check_B25_casting_pref_guard(prof)

    # C — Cross-table integrity
    all_results += check_C25_sessions_have_profiles(sess, prof)
    all_results += check_C26_profiles_have_sessions(prof, sess)

    stats = coverage_stats(sess, prof)
    write_summary(all_results, stats, SUMMARY_PATH)
    write_report_csv(all_results, REPORT_PATH)

    passes = sum(1 for x in all_results if x["status"]=="PASS")
    warns  = sum(1 for x in all_results if x["status"]=="WARN")
    fails  = sum(1 for x in all_results if x["status"]=="FAIL")
    infos  = sum(1 for x in all_results if x["status"]=="INFO")
    print(f"\n[SUMMARY] {len(all_results)} checks — "
          f"{passes} PASS / {warns} WARN / {infos} INFO / {fails} FAIL")
    if fails > 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()