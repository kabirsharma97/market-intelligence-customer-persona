"""
TwinSim — Feature Store Assessment  v4.4.0
============================================
Validates feature_store.csv before passing to clustering_engine.py.

41 features total across 5 feature groups.
12 checks covering completeness, nulls, ranges, variance, behavioral
ordering, trajectory quality, sparse field handling, and new v4.0 features.

Changes from v4.3 (v4.4.0):
  Check 13 note updated: FS-12 adaptive ceiling in feature_store.py v4.1.0
  means avg_watch_gap_norm var should exceed 0.05 on 100k+ datasets.
  ASSESSMENT_VERSION constant added.

Changes from v4.0 (v4.1.0):
  CLUSTERING_FEATURES reduced from 33 → 27: six high-sparsity features
  (>40% zeros) moved to SPARSE_CLUSTERING_AUXILIARY and merged into
  AUXILIARY_FEATURES. This mirrors clustering_engine.py v4.1.0 which
  excludes them from PCA to prevent the uniform-density problem that
  caused HDBSCAN to classify >89% of events as noise.
  Moved: binge_index_score, reactivation_signal, satisfaction_score,
         support_friction_score, drop_pattern_score, campaign_receptivity.
  ALL_FEATURES count unchanged — all 41 features still validated.
  Restoration trigger documented in SPARSE_CLUSTERING_AUXILIARY.

Usage:
    python feature_store_assessment.py --input feature_store.csv

Exit codes:
    0 — all PASS or WARN
    1 — any FAIL (DO NOT proceed to clustering_engine.py)
"""

import argparse
import sys
import pandas as pd
import numpy as np

# Windows cp1252 fix — force UTF-8 so arrow/special chars don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ASSESSMENT_VERSION: str = "4.4.0"

# ── FSA-05: Named constants for Check 8 thresholds ───────────────────────────
# tenure_weight encodes tenure_months / 24. A 12-month subscriber maps to 12/24 = 0.5.
# The high_value_churn_flag requires tenure >= 12 months, so the threshold is 12/24.
TENURE_12M_THRESHOLD: float = 12 / 24          # = 0.5 (12-month tenure floor)
# subscription_tier_score = 1.0 encodes the premium plan tier exclusively.
# Values below 1.0 indicate standard or basic — not premium.
PREMIUM_TIER_THRESHOLD: float = 1.0 - 1e-9     # = 0.9999… (exclusive premium gate)

# ── Feature sets ──────────────────────────────────────────────────────────────

CLUSTERING_FEATURES = [
    # FG-01: Session Engagement (3)
    # attention_quality_score removed: r=0.97 with completion_rate_smooth on fresh data.
    # completion / (1 + buffer_rate) ≈ completion when buffer_rate is low (typical).
    # buffer modulation is captured independently by friction_index. Moved to auxiliary.
    "session_depth_score",
    "session_intensity_score",
    "binge_signal",
    # FG-02: Completion & Retention (4)
    "completion_rate_smooth",
    "days_since_last_normalised",
    "attention_decay_curve",
    "avg_watch_gap_norm",
    # FG-03: Churn & Friction (3)
    "churn_risk_score",
    "churn_velocity",
    "friction_index",
    # FG-04: Content Affinity (5)
    "content_engagement_score",
    "viewing_context_score",
    "event_type_weight",
    "fav_genre_confidence",
    "episode_position_score",
    # FG-05: Contextual Modifiers (7)
    "recency_weight",
    "tenure_weight",
    "network_quality_score",
    "ltv_score",
    "account_health_score",
    "plan_trajectory_score",
    "lifecycle_stage_score",
]

# Sparse features (>40% zeros) excluded from clustering PCA but used for
# post-hoc cluster profile enrichment and persona characterisation.
# Restoration trigger for binge_index_score and reactivation_signal:
#   avg_watch_gap_norm var > 0.05 AND binge_index_score zeros < 30%
#   (confirms P3-A+B session_events are flowing through the pipeline).
# subscription_tier_score moved here (FS-06): collinear with tenure_weight through
#   ltv_score construction (ltv = price × tenure × discount; price = f(plan_type)).
#   tenure_weight is the primary subscription-value dimension in clustering.
SPARSE_CLUSTERING_AUXILIARY = [
    "binge_index_score",        # 64.6% zeros — restore post P3-A+B rerun
    "reactivation_signal",      # 92.8% zeros
    "satisfaction_score",       # 71.5% zeros
    "support_friction_score",   # 81.8% zeros
    "drop_pattern_score",       # 59.6% zeros
    "campaign_receptivity",     # 41.3% zeros
    "subscription_tier_score",  # FS-06: collinear with tenure_weight through ltv_score
    "completion_tier",          # FS-05: ordinal binning of completion_rate_smooth
    "completion_variance_signal",  # FS-05: deterministic transform of completion_pct
    "recency_adjusted_completion", # FS-04: exact product of completion_rate_smooth × recency_weight
    "attention_quality_score",  # empirical r=0.97 with completion_rate_smooth on fresh data;
                                # buffer modulation independently captured by friction_index
]

AUXILIARY_FEATURES = [
    "rewatch_engagement_flag",
    "completion_tier",
    "content_type_weight",
    "weekend_viewing_flag",
    "geo_tier",
    "churn_flag_encoded",
    "high_value_churn_flag",
    "satisfaction_trend",
    "superfan_score",
] + SPARSE_CLUSTERING_AUXILIARY

ALL_FEATURES = list(dict.fromkeys(CLUSTERING_FEATURES + AUXILIARY_FEATURES))

TOLERANCE = 1e-4

RANGE_RULES = {
    "session_depth_score":          (0.0, 1.0),
    "session_intensity_score":      (0.0, 1.0),
    "binge_signal":                 (0.0, 1.0),
    "attention_quality_score":      (0.0, 1.0),
    "binge_index_score":            (0.0, 1.0),
    "completion_rate_smooth":       (0.0, 1.0),
    "completion_tier":              (1,   3),
    "completion_variance_signal":   (0.0, 1.0),
    "recency_adjusted_completion":  (0.0, 1.0),
    "days_since_last_normalised":   (0.0, 1.0),
    "attention_decay_curve":        (-1.0, 1.0),
    "avg_watch_gap_norm":           (0.0, 1.0),
    "churn_risk_score":             (0.0, 1.0),
    "churn_velocity":               (0.0, 1.0),
    "reactivation_signal":          (0.0, 1.0),
    "friction_index":               (0.0, 1.0),
    "satisfaction_score":           (0.0, 1.0),
    "satisfaction_trend":           (-1.0, 1.0),
    "drop_pattern_score":           (0.0, 1.0),
    "content_engagement_score":     (0.0, 1.0),
    "content_type_weight":          (0.0, 1.0),
    "viewing_context_score":        (0.0, 1.0),
    "event_type_weight":            (0.0, 1.0),
    "fav_genre_confidence":         (0.0, 1.0),
    "episode_position_score":       (0.0, 1.0),
    "recency_weight":               (0.0, 1.0),
    "subscription_tier_score":      (0.0, 1.0),
    "tenure_weight":                (0.0, 1.0),
    "network_quality_score":        (0.0, 1.0),
    "geo_tier":                     (1,   3),
    "support_friction_score":       (0.0, 1.0),
    "ltv_score":                    (0.0, 1.0),
    "account_health_score":         (0.0, 1.0),
    "campaign_receptivity":         (0.0, 1.0),
    "superfan_score":               (0.0, 1.0),
    "plan_trajectory_score":        (0.0, 1.0),
    "lifecycle_stage_score":        (0.0, 1.0),
    "rewatch_engagement_flag":      (0,   1),
    "weekend_viewing_flag":         (0,   1),
    "churn_flag_encoded":           (0,   1),
    "high_value_churn_flag":        (0,   1),
}

# Sparse-safe: 0.0 encodes absent signal — expected to have many zeros, no nulls.
SPARSE_FEATURES = {
    "satisfaction_score",
    "support_friction_score",
    "campaign_receptivity",
    "superfan_score",
}

COMPLETION_ORDER = [
    "quick_churn", "casual_dip", "re_engager",
    "binge_heavy", "completion_obsessed",
]


def run_assessment(input_path: str) -> bool:
    df = pd.read_csv(input_path)
    n  = len(df)
    print(f"Loaded : {n:,} rows × {len(df.columns)} columns")
    print(f"File   : {input_path}")
    if "user_id" in df.columns:
        print(f"Users  : {df['user_id'].nunique():,}")

    results = {}

    # ── Check 1: Feature completeness ──────────────────────────────────────────
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    results["01_feature_completeness"] = (
        ("PASS", f"All {len(ALL_FEATURES)} features present")
        if not missing else
        ("FAIL", f"Missing {len(missing)} features: {missing}")
    )

    # ── Check 2: Zero nulls ────────────────────────────────────────────────────
    present  = [f for f in ALL_FEATURES if f in df.columns]
    null_map = {f: int(df[f].isnull().sum()) for f in present if df[f].isnull().sum() > 0}
    results["02_null_check"] = (
        ("PASS", f"Zero nulls across all {len(present)} features")
        if not null_map else
        ("FAIL", f"{sum(null_map.values())} nulls — {null_map}")
    )

    # ── Check 3: Range validation ──────────────────────────────────────────────
    range_fails = []
    for feat, (lo, hi) in RANGE_RULES.items():
        if feat not in df.columns:
            continue
        s = pd.to_numeric(df[feat], errors="coerce").dropna()
        if len(s) == 0:
            continue
        mn, mx = float(s.min()), float(s.max())
        if mn < lo - TOLERANCE or mx > hi + TOLERANCE:
            range_fails.append(f"{feat}: [{mn:.4f},{mx:.4f}] expected [{lo},{hi}]")
    results["03_range_check"] = (
        ("PASS", f"All {len(RANGE_RULES)} bounded features within declared range")
        if not range_fails else
        ("FAIL", "; ".join(range_fails))
    )

    # ── Check 4: No zero-variance clustering features ──────────────────────────
    zero_var = [
        f for f in CLUSTERING_FEATURES
        if f in df.columns and pd.to_numeric(df[f], errors="coerce").std() < 1e-6
    ]
    results["04_variance_check"] = (
        ("PASS", f"All {len(CLUSTERING_FEATURES)} clustering features have variance")
        if not zero_var else
        ("FAIL", f"Zero-variance (breaks HDBSCAN): {zero_var}")
    )

    # ── Check 13: Pairwise collinearity in clustering features ────────────────
    # FSA-03: No collinearity gate existed anywhere in the pipeline. FS-05 identified
    # 6-8 structurally collinear features with r up to 0.995. PCA compresses these into
    # a dominant first component; HDBSCAN then produces completion-quantile buckets, not
    # behavioral personas. This gate must block the pipeline before clustering runs.
    # Pearson is used for continuous features only. Binary/ordinal features excluded.
    COLLINEARITY_BINARY = {
        "completion_tier", "churn_flag_encoded",
        "rewatch_engagement_flag", "weekend_viewing_flag", "high_value_churn_flag",
    }
    continuous_clustering = [
        f for f in CLUSTERING_FEATURES
        if f in df.columns and f not in COLLINEARITY_BINARY
    ]
    if len(continuous_clustering) >= 2:
        col_sample = df[continuous_clustering].apply(pd.to_numeric, errors="coerce").dropna()
        if len(col_sample) > 10_000:
            col_sample = col_sample.sample(10_000, random_state=42)
        corr_matrix = col_sample.corr(method="pearson")
        hard_fails, warns, high_pairs = [], [], []
        for i in range(len(continuous_clustering)):
            for j in range(i + 1, len(continuous_clustering)):
                f1, f2 = continuous_clustering[i], continuous_clustering[j]
                r = float(corr_matrix.loc[f1, f2])  # cast to Python float for clean comparison
                if abs(r) > 0.95:
                    hard_fails.append(f"{f1} ↔ {f2}: r={r:.6f}")
                elif abs(r) > 0.80:
                    warns.append(f"{f1} ↔ {f2}: r={r:.6f}")
                elif abs(r) > 0.50:
                    high_pairs.append(f"{f1} ↔ {f2}: r={r:.4f}")
        if hard_fails:
            results["13_collinearity_check"] = (
                "FAIL",
                f"Algebraic redundancy — {len(hard_fails)} pairs |r|>0.95: "
                f"{hard_fails}. Remove redundant features before clustering.",
            )
        elif warns:
            results["13_collinearity_check"] = (
                "WARN",
                f"High collinearity — {len(warns)} pairs 0.80<|r|<=0.95: {warns}. "
                f"Review before Layer 3. Notable pairs (|r|>0.50): {high_pairs}",
            )
        else:
            results["13_collinearity_check"] = (
                "PASS",
                f"No pairs |r|>0.80 across {len(continuous_clustering)} continuous "
                f"clustering features (n={len(col_sample):,}). "
                f"Notable pairs (|r|>0.50): {high_pairs if high_pairs else 'none'}",
            )
    else:
        results["13_collinearity_check"] = ("SKIP", "Fewer than 2 continuous clustering features present")

    # ── Check 5: Churn risk vs churn_flag_encoded ──────────────────────────────
    if "churn_risk_score" in df.columns and "churn_flag_encoded" in df.columns:
        churned     = pd.to_numeric(df[df["churn_flag_encoded"]==1]["churn_risk_score"], errors="coerce")
        not_churned = pd.to_numeric(df[df["churn_flag_encoded"]==0]["churn_risk_score"], errors="coerce")
        if len(churned) > 0 and len(not_churned) > 0:
            gap    = float(churned.mean() - not_churned.mean())
            status = "PASS" if gap > 0.05 else "WARN"
            results["05_churn_risk_consistency"] = (
                status,
                f"churned={churned.mean():.4f}, not_churned={not_churned.mean():.4f}, gap={gap:.4f}"
                + (" ✓" if gap > 0.05 else " — need gap>0.05")
            )
        else:
            results["05_churn_risk_consistency"] = ("SKIP", "Insufficient data")
    else:
        results["05_churn_risk_consistency"] = ("SKIP", "Required columns absent")

    # ── Check 6: Segment × completion ordering ─────────────────────────────────
    # SYNTHETIC VALIDATION SCAFFOLDING ONLY.
    # This check validates that completion_rate_smooth is ordered correctly
    # across the known simulation segment groups (quick_churn < casual_dip <
    # re_engager < binge_heavy < completion_obsessed). It queries the feature
    # store by segment_id — a field generated by generate_signals.py as a
    # simulation control variable that does NOT exist in real client data.
    #
    # On real client data:
    #   - If segment_id is absent → this check emits SKIP (correct behaviour).
    #   - SKIP is not a failure. It means production mode is active.
    #
    # FSA-02: When segment_id IS present (synthetic run), ordering violation
    # is FAIL — a violation means the simulation data is behaving incorrectly
    # and must block clustering. The production-equivalent check is clustering
    # audit Check 5 (behavioral label ordering on discovered clusters).
    if "segment_id" in df.columns and "completion_rate_smooth" in df.columns:
        seg_means = {
            seg: float(df[df["segment_id"]==seg]["completion_rate_smooth"].astype(float).mean())
            for seg in COMPLETION_ORDER
            if len(df[df["segment_id"]==seg]) > 0
        }
        present_segs = [s for s in COMPLETION_ORDER if s in seg_means]
        violations = [
            f"{present_segs[i]}({seg_means[present_segs[i]]:.3f})"
            f">{present_segs[i+1]}({seg_means[present_segs[i+1]]:.3f})"
            for i in range(len(present_segs)-1)
            if seg_means[present_segs[i]] > seg_means[present_segs[i+1]] + TOLERANCE
        ]
        summary = ", ".join(f"{s}={seg_means[s]:.3f}" for s in present_segs)
        results["06_segment_completion_ordering"] = (
            ("PASS", f"[SYNTHETIC ONLY] Ordering correct: {summary}")
            if not violations else
            ("FAIL", f"[SYNTHETIC ONLY] Ordering violation — blocks clustering: {violations}. Full: {summary}")
        )
    else:
        results["06_segment_completion_ordering"] = (
            "SKIP",
            ("[SYNTHETIC ONLY] segment_id absent — production mode. "
             "This is correct for real client data. "
             "Production ordering is validated by clustering_audit.py Check 5 ")
        )

    # ── Check 7: Recency weight > 0 ───────────────────────────────────────────
    if "recency_weight" in df.columns:
        zeros = int((pd.to_numeric(df["recency_weight"], errors="coerce") <= 0).sum())
        results["07_recency_weight"] = (
            ("PASS", "All recency_weight > 0")
            if zeros == 0 else
            ("WARN", f"{zeros} sessions have recency_weight <= 0")
        )
    else:
        results["07_recency_weight"] = ("SKIP", "recency_weight absent")

    # ── Check 8: High-value churn flag conditions ──────────────────────────────
    if all(c in df.columns for c in
           ["high_value_churn_flag","churn_flag_encoded","tenure_weight","subscription_tier_score"]):
        hv = df[df["high_value_churn_flag"]==1]
        if len(hv) > 0:
            not_churned = int((hv["churn_flag_encoded"]==0).sum())
            low_tenure  = int((pd.to_numeric(hv["tenure_weight"],errors="coerce") < TENURE_12M_THRESHOLD).sum())
            not_premium = int((pd.to_numeric(hv["subscription_tier_score"],errors="coerce") < PREMIUM_TIER_THRESHOLD).sum())
            violations  = []
            if not_churned: violations.append(f"{not_churned} not churned")
            if low_tenure:  violations.append(f"{low_tenure} tenure_weight<0.5")
            if not_premium: violations.append(f"{not_premium} not premium tier")
            results["08_high_value_churn_flag"] = (
                ("FAIL", f"Violations: {', '.join(violations)}")
                if violations else
                ("PASS", f"All {len(hv):,} high_value_churn events satisfy all 3 conditions")
            )
        else:
            results["08_high_value_churn_flag"] = (
                "PASS", "No high_value_churn_flag=1 events (valid if no premium long-tenure churners)"
            )
    else:
        results["08_high_value_churn_flag"] = ("SKIP", "Required columns absent")

    # ── Check 9: Sparse feature handling ──────────────────────────────────────
    sparse_issues = []
    sparse_info   = []
    for feat in SPARSE_FEATURES:
        if feat not in df.columns:
            continue
        s = pd.to_numeric(df[feat], errors="coerce")
        null_n = int(s.isnull().sum())
        if null_n > 0:
            sparse_issues.append(f"{feat}: {null_n} nulls (should be 0.0, not NULL)")
        neg_n = int((s < 0).sum())
        if neg_n > 0:
            sparse_issues.append(f"{feat}: {neg_n} negative values")
        non_zero_pct = float((s > 0).mean() * 100)
        if non_zero_pct == 0.0:
            sparse_issues.append(f"{feat}: all zeros — sparse signal entirely absent")
        elif non_zero_pct > 85.0:
            sparse_issues.append(f"{feat}: {non_zero_pct:.1f}% non-zero — unusually high for sparse")
        else:
            sparse_info.append(f"{feat}: {non_zero_pct:.1f}% non-zero")
    results["09_sparse_feature_handling"] = (
        ("PASS", f"All sparse features NULL-safe. Fill rates: {'; '.join(sparse_info)}")
        if not sparse_issues else
        (
            "FAIL" if any("null" in i or "negative" in i for i in sparse_issues) else "WARN",
            "; ".join(sparse_issues)
        )
    )

    # ── Check 10: episode_position_score valid values ──────────────────────────
    # Valid encoded values: 0.50 (mid_season/NULL), 0.75 (penultimate_arc), 1.00 (premiere/finale)
    if "episode_position_score" in df.columns:
        s = pd.to_numeric(df["episode_position_score"], errors="coerce")
        null_n  = int(s.isnull().sum())
        valid_vals = {0.50, 0.75, 1.00}
        actual_vals = set(round(float(v), 2) for v in s.dropna().unique())
        invalid = actual_vals - valid_vals
        dist = {v: int((s.round(2) == v).sum()) for v in sorted(valid_vals)}
        if invalid or null_n > 0:
            results["10_episode_position_score"] = (
                "FAIL",
                f"Invalid values: {invalid}. Nulls: {null_n}. Distribution: {dist}"
            )
        else:
            results["10_episode_position_score"] = (
                "PASS",
                f"All values in {{0.50, 0.75, 1.00}}. Distribution: {dist}"
            )
    else:
        results["10_episode_position_score"] = ("SKIP", "episode_position_score absent")

    # ── Check 11: lifecycle and plan trajectory value validity ─────────────────
    issues = []
    if "lifecycle_stage_score" in df.columns:
        ls = pd.to_numeric(df["lifecycle_stage_score"], errors="coerce")
        null_n = int(ls.isnull().sum())
        if null_n > 0:
            issues.append(f"lifecycle_stage_score: {null_n} nulls")
        # P2-A: "new" stage added (tenure_months ≤ 2) → lifecycle_stage_score = 0.70
        valid_ls = {0.0, 0.4, 0.7, 0.8, 1.0}
        actual_ls = set(round(float(v), 1) for v in ls.dropna().unique())
        invalid_ls = actual_ls - valid_ls
        if invalid_ls:
            issues.append(f"lifecycle_stage_score invalid values: {invalid_ls}")
        else:
            dist_ls = {v: int((ls.round(1) == v).sum()) for v in sorted(valid_ls)}
            issues.append(None)
            issues = [i for i in issues if i]
            if not issues:
                # Good — report dist as info
                pass

    if "plan_trajectory_score" in df.columns:
        pt = pd.to_numeric(df["plan_trajectory_score"], errors="coerce")
        valid_pt = {0.0, 0.5, 1.0}
        actual_pt = set(round(float(v), 1) for v in pt.dropna().unique())
        invalid_pt = actual_pt - valid_pt
        if invalid_pt:
            issues.append(f"plan_trajectory_score invalid values: {invalid_pt}")

    if not issues:
        ls_dist = {}
        pt_dist = {}
        if "lifecycle_stage_score" in df.columns:
            ls = pd.to_numeric(df["lifecycle_stage_score"], errors="coerce")
            ls_dist = {round(v,1): int((ls.round(1)==round(v,1)).sum()) for v in [0.0,0.4,0.7,0.8,1.0]}
        if "plan_trajectory_score" in df.columns:
            pt = pd.to_numeric(df["plan_trajectory_score"], errors="coerce")
            pt_dist = {round(v,1): int((pt.round(1)==round(v,1)).sum()) for v in [0.0,0.5,1.0]}
        results["11_trajectory_score_validity"] = (
            "PASS",
            f"lifecycle_stage_score: {ls_dist}. plan_trajectory_score: {pt_dist}"
        )
    else:
        results["11_trajectory_score_validity"] = ("FAIL", "; ".join(issues))

    # ── Check 12: superfan_score vs ltv_score correlation ─────────────────────
    if "superfan_score" in df.columns and "ltv_score" in df.columns:
        sf = pd.to_numeric(df["superfan_score"], errors="coerce")
        lv = pd.to_numeric(df["ltv_score"],      errors="coerce")
        n_superfan = int((sf > 0).sum())
        pct_sf     = n_superfan / max(n, 1) * 100
        if n_superfan == 0:
            results["12_superfan_ltv_correlation"] = (
                "PASS", f"No superfan sessions ({pct_sf:.1f}%) — sparse field correct"
            )
        else:
            sf_ltv     = float(lv[sf > 0].mean())
            nonsf_ltv  = float(lv[sf == 0].mean())
            status     = "PASS" if sf_ltv > nonsf_ltv else "WARN"
            results["12_superfan_ltv_correlation"] = (
                status,
                f"superfan ltv={sf_ltv:.4f} {'>' if sf_ltv > nonsf_ltv else '<='} "
                f"non-superfan={nonsf_ltv:.4f}. "
                f"{n_superfan:,} superfan sessions ({pct_sf:.1f}%)"
            )
    else:
        results["12_superfan_ltv_correlation"] = ("SKIP", "Required columns absent")

    # ── Check 13: HDBSCAN restoration trigger evaluation (FSA-04) ─────────────
    # CAL-001 closed: HDBSCAN is not the current default. This check evaluates
    # the restoration criteria at runtime and prompts the operator if both
    # conditions are met, suggesting a deliberate HDBSCAN attempt.
    # Criteria: avg_watch_gap_norm var > 0.05 AND binge_index_score zeros < 30%.
    # FS-12: feature_store.py v4.1.0 adaptive p75 ceiling should bring
    # avg_watch_gap_norm var above 0.05 on 100k+ datasets. If this check
    # still fails after FS-12, re-run feature_store.py v4.1.0 first.
    trigger_parts = []
    trigger_met   = True
    if "avg_watch_gap_norm" in df.columns:
        gap_var = float(pd.to_numeric(df["avg_watch_gap_norm"], errors="coerce").var())
        if gap_var > 0.05:
            trigger_parts.append(f"avg_watch_gap_norm var={gap_var:.4f} > 0.05 ✓")
        else:
            trigger_parts.append(f"avg_watch_gap_norm var={gap_var:.4f} ≤ 0.05 ✗")
            trigger_met = False
    else:
        trigger_parts.append("avg_watch_gap_norm absent ✗")
        trigger_met = False
    if "binge_index_score" in df.columns:
        binge_zeros = float((pd.to_numeric(df["binge_index_score"], errors="coerce") == 0).mean() * 100)
        if binge_zeros < 30.0:
            trigger_parts.append(f"binge_index_score zeros={binge_zeros:.1f}% < 30% ✓")
        else:
            trigger_parts.append(f"binge_index_score zeros={binge_zeros:.1f}% ≥ 30% ✗")
            trigger_met = False
    else:
        trigger_parts.append("binge_index_score absent ✗")
        trigger_met = False
    trigger_detail = "; ".join(trigger_parts)
    if trigger_met:
        results["13_hdbscan_restoration_trigger"] = (
            "WARN",
            (f"FSA-04: HDBSCAN restoration criteria MET — {trigger_detail}. "
             f"Consider running: python clustering_engine.py --algorithm hdbscan "
             f"and checking logged silhouette. Accept if sil ≥ 0.05 and noise < 30%.")
        )
    else:
        results["13_hdbscan_restoration_trigger"] = (
            "PASS",
            f"FSA-04: HDBSCAN restoration criteria not met — {trigger_detail}. K-Means default correct."
        )

    # ── Print results ──────────────────────────────────────────────────────────
    print()
    print("=" * 76)
    print("TWINSIM — FEATURE STORE ASSESSMENT  v4.4.0")
    print("=" * 76)
    print(f"  Features total     : {len(ALL_FEATURES)}")
    print(f"  Clustering features: {len(CLUSTERING_FEATURES)} "
          f"(collinear + sparse features moved to auxiliary)")
    print(f"  Auxiliary (sparse) : {len(SPARSE_CLUSTERING_AUXILIARY)} excluded from PCA")
    print(f"     Sparse: binge_index_score, reactivation_signal, satisfaction_score,")
    print(f"             support_friction_score, drop_pattern_score, campaign_receptivity")
    print(f"     Collinear (FS-04/05/06 + empirical): recency_adjusted_completion,")
    print(f"             completion_tier, completion_variance_signal,")
    print(f"             subscription_tier_score, attention_quality_score (r=0.97 fresh data)")
    print()

    all_pass = True
    for check_name, result in results.items():
        status, msg = result
        icon = {"PASS":"[PASS]","FAIL":"[FAIL]","WARN":"[WARN]","SKIP":"[SKIP]"}[status]
        print(f"  {icon} {check_name}")
        print(f"         {msg}")
        if status == "FAIL":
            all_pass = False

    print()
    print("=" * 76)
    overall = (
        "ALL CHECKS PASSED — safe to proceed to clustering_engine.py"
        if all_pass else
        "ISSUES FOUND — DO NOT proceed to clustering_engine.py"
    )
    print(f"  OVERALL: {overall}")
    print("=" * 76)
    return all_pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TwinSim Feature Store Assessment v4.3.0")
    ap.add_argument("--input", default="feature_store.csv")
    args = ap.parse_args()
    ok = run_assessment(args.input)
    raise SystemExit(0 if ok else 1)