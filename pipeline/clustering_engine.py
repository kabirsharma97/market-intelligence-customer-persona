"""
TwinSim — Behavioral Clustering Engine
=========================================
Layer 3: Behavioral Persona Clustering

Reads feature_store.csv (Layer 2 output) and produces:
  - cluster_assignments.csv  : per-event cluster label + 2D embed coordinates
  - cluster_metadata.json    : full cluster profiles with behavioral priors
  - clustering_report.txt    : pipeline log + cluster summary table

Algorithm design:
  Primary  — K-Means with composite k-selection (silhouette + elbow + CH criterion).
             K-Means is the confirmed permanent default (CAL-001 closed 2026-04-07).
             HDBSCAN was attempted on 123k events; fell back to kmeans_fallback on
             both 20k/90d and 123k runs. HDBSCAN available via --algorithm hdbscan
             for future re-evaluation if feature space changes significantly.
  Noise    — Conservative 3σ distance-based outlier pass (~1.5% noise on 123k data).

Labelling design (v4.0.0 — CAL-002):
  TRIGGER_RULES declarative scoring engine replaces LABEL_RULES lambda list.
  Each label has: gates (hard AND prerequisites), score_fn (continuous ranking),
  multi_assign flag (quick_churn may claim multiple clusters).
  rank_personas() implements a two-pass assignment:
    Pass 1 — single-assign labels in priority order (completion_obsessed, binge_heavy,
             re_engager, casual_dip). Highest-scoring eligible unassigned cluster wins.
    Pass 2 — multi-assign labels (quick_churn). All remaining eligible clusters claimed.
    Pass 3 — fallback: mixed_behavior.

Architecture notes:
  - Input: feature_store.csv only. No parallel signals.csv required.
    All behavioral context is carried in the feature matrix.
  - Cluster metadata includes behavioral priors per cluster:
    base_completion, base_churn_30d, base_reactivation — these feed
    directly into Layer 4 (Digital Twin Engine) as segment priors.
  - Focused mode (--content_refs / --segments) matches generate_signals.py
    and feature_store.py.
  - --mode flag: full (default) or compare (side-by-side cluster profiles
    for two content_refs — maps to the old single/compare pipeline mode).

Usage:
    # Full clustering
    python clustering_engine.py --input feature_store.csv --out cluster_assignments.csv

    # Focused: single content title
    python clustering_engine.py --input feature_store.csv --content_refs title_001

    # Focused: two titles in compare mode (side-by-side profiles)
    python clustering_engine.py --input feature_store.csv \\
        --content_refs title_001 title_002 --mode compare

    # Force K-Means (default — same as omitting --algorithm)
    python clustering_engine.py --input feature_store.csv --algorithm kmeans

    # Attempt HDBSCAN (will fall back to K-Means if sil < 0.05 or noise > 30%)
    python clustering_engine.py --input feature_store.csv --algorithm hdbscan

    # Run audit after
    python clustering_audit.py --assignments cluster_assignments.csv \\
        --metadata cluster_metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Thread control — must be set before numpy import
os.environ["LOKY_MAX_CPU_COUNT"]  = "1"
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"]= "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_SEED    = 42
ENGINE_VERSION = "4.1.0"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Clustering feature set ────────────────────────────────────────────────────
# Single authoritative list — must match CLUSTERING_FEATURES in
# feature_store_assessment.py exactly. (FSA-01/XL-01)
# Rationale for removals vs v3.0:
#   FS-04: recency_adjusted_completion removed — product of two features already present
#   FS-05: completion_tier, completion_variance_signal removed — deterministic transforms
#   FS-06: subscription_tier_score moved to ENGINE_AUXILIARY_FEATURES — collinear with
#          tenure_weight through ltv_score construction
#   empirical: attention_quality_score removed — r=0.97 with completion_rate_smooth on
#          fresh 5k-user data; buffer signal independently captured by friction_index
#   Added vs v3.0: attention_decay_curve, avg_watch_gap_norm, fav_genre_confidence,
#          episode_position_score, ltv_score, account_health_score, plan_trajectory_score,
#          lifecycle_stage_score

CLUSTERING_FEATURES: List[str] = [
    # FG-01: Session Engagement (3)
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

# Auxiliary features: excluded from PCA clustering but included in cluster
# profile enrichment and persona characterisation (FS-04/05/06).
ENGINE_AUXILIARY_FEATURES: List[str] = [
    "binge_index_score",
    "reactivation_signal",      # used in TRIGGER_RULES (re_engager gate) via priors dict — read from profile, not clustered
    "satisfaction_score",
    "support_friction_score",
    "drop_pattern_score",
    "campaign_receptivity",
    "subscription_tier_score",  # FS-06: collinear with tenure_weight through ltv_score
    "completion_tier",          # FS-05: ordinal binning of completion_rate_smooth
    "completion_variance_signal",  # FS-05: deterministic transform
    "recency_adjusted_completion", # FS-04: product of two features already in clustering set
    "attention_quality_score",  # empirical r=0.97 with completion_rate_smooth; buffer signal
                                # captured by friction_index which remains in clustering set
]

# Behavioral prior fields extracted from feature store for cluster profiles
# These become base_* priors in Layer 4 Digital Twin models
PRIOR_FIELDS = {
    "base_completion":      "completion_rate_smooth",
    "base_churn_30d":       "churn_flag_encoded",
    "base_reactivation":    "reactivation_signal",
    "base_session_depth":   "session_depth_score",
    "base_churn_risk":      "churn_risk_score",
}

# CAL-002 — Declarative TRIGGER_RULES scoring engine
#
# Replaces the first-match-wins LABEL_RULES lambda list. The old design had two
# structural failures on K-Means output:
#   1. binge_heavy rule depended on binge_signal (auxiliary, not clustered) — near-zero
#      profile means caused the rule to silently never fire.
#   2. Dedup assigned each label to exactly one cluster — with k=6 and 5 labels,
#      the two legitimate quick_churn populations (high-churn, low-completion) were
#      forced into mixed_behavior.
#
# New design:
#   TRIGGER_RULES: dict of label → {gates, score_fn, multi_assign}
#     gates      — hard prerequisite conditions (AND). Cluster ineligible if any fail.
#     score_fn   — continuous strength score (0–1). Used to rank eligible clusters.
#     multi_assign — if True, label can be claimed by multiple clusters (quick_churn).
#
# Thresholds calibrated from the validated 6-cluster K-Means output on 123k events (v5.0 data):
#   Cluster 0: compl=0.836, churn=0.284, react=0.003  → completion_obsessed
#   Cluster 1: compl=0.271, churn=0.424, react=0.271  → re_engager
#   Cluster 2: compl=0.834, churn=0.271, react=0.001  → binge_heavy
#   Cluster 3: compl=0.219, churn=0.667, react=0.008  → quick_churn
#   Cluster 4: compl=0.235, churn=0.699, react=0.009  → casual_dip / quick_churn border
#   Cluster 5: compl=0.033, churn=0.380, react=0.002  → quick_churn (near-zero completion)
#
# Recalibrated for generate_signals.py v5.2.0 (P5-A/B segment parameter changes):
#   quick_churn churn_30d_prob raised 0.55→0.70: centroid churn expected 0.70–0.85
#   casual_dip completion Beta(4,5) mean=0.444: centroid completion expected 0.42–0.50
#   casual_dip churn contaminated by quick_churn: centroid churn observed 0.70–0.80
#   casual_dip churn ceiling raised 0.75→0.85 to accommodate elevated centroid.
#   binge_heavy depth lognorm(3.0): centroid depth_score expected higher
#   completion_obsessed depth lognorm(1.4): centroid depth_score expected lower
#   quick_churn completion gate unchanged at < 0.35 (centroid compl 0.04–0.08 safely below)
#
# From 10k v5.2.0 K-Means k=5 run:
#   Cluster 0: compl=0.433, churn=0.760  → casual_dip (fails gate at 0.75 ceiling — fix applied)
#   Cluster 1: compl=0.825, churn=0.350  → binge_heavy
#   Cluster 2: compl=0.041, churn=0.547  → quick_churn
#   Cluster 3: compl=0.847, churn=0.333  → completion_obsessed
#   Cluster 4: compl=0.295, churn=0.337  → re_engager

TRIGGER_RULES: Dict[str, Dict] = {
    "completion_obsessed": {
        # High completion, low churn. Distinguished from binge_heavy by lower churn.
        # Gate uses completion_rate_smooth (in clustering set — reliable profile mean).
        "gates": [
            lambda p, pr: p.get("completion_rate_smooth", 0) > 0.80,
            lambda p, pr: p.get("churn_risk_score", 1)      < 0.30,
        ],
        # Score: reward high completion, penalise churn. Tiebreak between C0 and C2:
        # C0 (churn=0.284) scores lower than C2 (churn=0.271) → C2 wins completion_obsessed
        # if both qualify (correct: C2 is the purer high-completion, low-churn cluster).
        "score_fn": lambda p, pr: (
            p.get("completion_rate_smooth", 0) * 0.6
            - p.get("churn_risk_score", 0)     * 0.4
        ),
        "multi_assign": False,
    },
    "binge_heavy": {
        # High completion, higher churn tolerance than completion_obsessed.
        # Does NOT gate on binge_signal (auxiliary, unreliable in profile means).
        # Distinguisher: completion > 0.72 AND does NOT already own completion_obsessed.
        "gates": [
            lambda p, pr: p.get("completion_rate_smooth", 0) > 0.72,
        ],
        "score_fn": lambda p, pr: (
            p.get("completion_rate_smooth", 0) * 0.5
            + p.get("session_intensity_score", 0) * 0.3
            + p.get("session_depth_score", 0)     * 0.2
        ),
        "multi_assign": False,
    },
    "re_engager": {
        # Defined by reactivation signal. base_reactivation passed via priors dict
        # (reactivation_signal is auxiliary — not in clustering feature set).
        # Threshold 0.12 is well above the noise floor (C1=0.271, all others ≤ 0.009).
        "gates": [
            lambda p, pr: pr.get("base_reactivation", 0) > 0.12,
        ],
        "score_fn": lambda p, pr: (
            pr.get("base_reactivation", 0) * 0.7
            + p.get("days_since_last_normalised", 0) * 0.3
        ),
        "multi_assign": False,
    },
    "quick_churn": {
        # High churn, low completion. multi_assign=True: with k=6 and 5 labels,
        # two behaviorally distinct quick_churn populations exist (C3 high-churn,
        # C5 near-zero-completion). Both should be labeled quick_churn, not mixed_behavior.
        # Completion gate < 0.35 excludes casual_dip boundary clusters.
        "gates": [
            lambda p, pr: pr.get("base_churn_30d", 0)        > 0.35,
            lambda p, pr: p.get("completion_rate_smooth", 1)  < 0.35,
        ],
        "score_fn": lambda p, pr: (
            pr.get("base_churn_30d", 0)         * 0.6
            + (1 - p.get("completion_rate_smooth", 1)) * 0.4
        ),
        "multi_assign": True,
    },
    "casual_dip": {
        # Moderate completion, moderate churn, low session depth.
        # Churn ceiling raised 0.75→0.85 for generate_signals.py v5.2.0 compatibility:
        #   quick_churn churn floor raised to 0.70 pushes its centroid to 0.70–0.85,
        #   but contamination from quick_churn users elevates the casual_dip centroid
        #   to 0.70–0.80. A ceiling of 0.75 caused the casual_dip cluster to fail the
        #   gate and be labelled mixed_behavior (observed: churn=0.760, misses by 0.010).
        # Ceiling 0.85 remains safely below the quick_churn centroid floor (>0.85 at
        #   churn_30d_prob=0.70), ensuring quick_churn clusters still fail this gate.
        # Completion floor 0.15 excludes near-zero-completion quick_churn clusters.
        # Completion ceiling 0.55 excludes binge_heavy/completion_obsessed clusters.
        "gates": [
            lambda p, pr: p.get("completion_rate_smooth", 0)  < 0.55,
            lambda p, pr: p.get("completion_rate_smooth", 0)  > 0.15,
            lambda p, pr: pr.get("base_churn_30d", 0)         < 0.85,
        ],
        "score_fn": lambda p, pr: (
            p.get("session_depth_score", 0)      * 0.5
            + p.get("completion_rate_smooth", 0) * 0.3
            + (1 - pr.get("base_churn_30d", 0)) * 0.2
        ),
        "multi_assign": False,
    },
}

# Label priority order for single-assign labels (processed first, in order).
# multi_assign labels (quick_churn) are processed after all single-assign labels
# so they absorb remaining eligible clusters without competing.
_SINGLE_ASSIGN_ORDER = [
    "completion_obsessed",
    "binge_heavy",
    "re_engager",
    "casual_dip",
]
_MULTI_ASSIGN_ORDER = ["quick_churn"]


def _score_triggers(
    profile: Dict[str, float],
    priors: Dict[str, float],
    label: str,
) -> Optional[float]:
    """
    Evaluate TRIGGER_RULES for a single label against a cluster's profile + priors.
    Returns the score (float) if all gates pass, or None if any gate fails.
    profile: feature_profile means (clustering features only).
    priors:  behavioral_priors dict (base_completion, base_churn_30d, base_reactivation, …).
    """
    rule = TRIGGER_RULES.get(label)
    if rule is None:
        return None
    for gate in rule["gates"]:
        try:
            if not gate(profile, priors):
                return None
        except (KeyError, TypeError):
            return None
    try:
        return float(rule["score_fn"](profile, priors))
    except (KeyError, TypeError):
        return None


def rank_personas(clusters: List[Dict]) -> Dict[int, str]:
    """
    CAL-002: Assign behavioral labels to clusters using the TRIGGER_RULES scoring engine.

    Pass 1 — single-assign labels (in _SINGLE_ASSIGN_ORDER):
      For each label, find all eligible clusters (gates pass). Assign the label to
      the highest-scoring eligible unassigned cluster. Each label claimed once.

    Pass 2 — multi-assign labels (quick_churn):
      For each remaining unassigned cluster, check eligibility. If gates pass,
      assign the label regardless of how many clusters already carry it.

    Pass 3 — fallback:
      Any still-unassigned cluster → mixed_behavior.

    Returns dict of cluster_index → behavioral_label.
    """
    assigned: Dict[int, str] = {}

    # Pass 1: single-assign labels
    for label in _SINGLE_ASSIGN_ORDER:
        candidates = []
        for cl in clusters:
            if cl["cluster_index"] in assigned:
                continue
            score = _score_triggers(cl["feature_profile"], cl["behavioral_priors"], label)
            if score is not None:
                candidates.append((score, cl["cluster_index"]))
        if candidates:
            candidates.sort(reverse=True)
            best_idx = candidates[0][1]
            assigned[best_idx] = label

    # Pass 2: multi-assign labels
    for label in _MULTI_ASSIGN_ORDER:
        for cl in clusters:
            if cl["cluster_index"] in assigned:
                continue
            score = _score_triggers(cl["feature_profile"], cl["behavioral_priors"], label)
            if score is not None:
                assigned[cl["cluster_index"]] = label

    # Pass 3: fallback
    for cl in clusters:
        if cl["cluster_index"] not in assigned:
            assigned[cl["cluster_index"]] = "mixed_behavior"

    return assigned


def _auto_label(profile: Dict[str, float], priors: Optional[Dict[str, float]] = None) -> str:
    """
    Single-cluster label assignment used during initial cluster profile construction
    (before the full rank_personas dedup pass). Kept for compatibility with the
    build_cluster_metadata per-cluster loop; the rank_personas pass overwrites these
    labels with the globally consistent assignment.
    priors dict exposes auxiliary fields (e.g. base_reactivation) to the gates.
    """
    p = dict(profile)
    pr = dict(priors) if priors else {}
    # Try single-assign labels first, then multi-assign
    for label in _SINGLE_ASSIGN_ORDER + _MULTI_ASSIGN_ORDER:
        score = _score_triggers(p, pr, label)
        if score is not None:
            return label
    return "mixed_behavior"


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(fdf: pd.DataFrame, lines: List[str]) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """
    Extract clustering features, StandardScale.
    The feature store guarantees no NULLs for non-sparse features and maps
    sparse NULLs to 0.0. Any NULL reaching this point is an upstream bug.
    CE-01: fillna(0.0) instead of median — median imputation masked upstream failures.
    A WARNING is emitted for any NULL detected so the failure is visible.
    """
    available = [f for f in CLUSTERING_FEATURES if f in fdf.columns]
    missing   = [f for f in CLUSTERING_FEATURES if f not in fdf.columns]
    if missing:
        lines.append(f"[WARN] Missing features (skipped): {missing}")
    if len(available) < 5:
        raise ValueError(
            f"Too few clustering features available ({len(available)}). "
            f"Run feature_store.py first."
        )

    X_raw = fdf[available].copy()
    for col in available:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
        null_count = int(X_raw[col].isnull().sum())
        if null_count > 0:
            lines.append(
                f"[WARN] CE-01: {null_count} NULLs in clustering feature '{col}' — "
                f"upstream feature_store.py bug. Filling with 0.0. "
                f"Run feature_store_assessment.py to diagnose."
            )
            X_raw[col] = X_raw[col].fillna(0.0)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.values)
    lines.append(f"[PREP] {len(available)}/{len(CLUSTERING_FEATURES)} features  |  X shape {X_scaled.shape}")
    return X_scaled, available, scaler


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_pca(
    X: np.ndarray, var_threshold: float, lines: List[str]
) -> Tuple[np.ndarray, PCA]:
    """PCA to retain var_threshold of variance. Min 5 components."""
    n_max    = min(X.shape[1], X.shape[0] - 1, 50)
    pca_full = PCA(n_components=n_max, random_state=RANDOM_SEED).fit(X)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    n_keep   = max(int(np.searchsorted(cumvar, var_threshold) + 1), 5)
    n_keep   = min(n_keep, n_max)
    pca      = PCA(n_components=n_keep, random_state=RANDOM_SEED)
    X_pca    = pca.fit_transform(X)
    var_explained = float(np.sum(pca.explained_variance_ratio_))
    lines.append(f"[PCA] {n_keep} components → {var_explained:.3f} variance explained")
    return X_pca, pca


def embed_2d(X_pca: np.ndarray, lines: List[str]) -> np.ndarray:
    """2D projection for visualisation. PCA-2 used here; swap to UMAP in production."""
    X_2d = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X_pca)
    lines.append("[DIM] 2D embed via PCA-2 (replace with UMAP for production visualisation)")
    return X_2d


# ── HDBSCAN clustering (primary) ──────────────────────────────────────────────

def run_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    lines: List[str],
) -> Tuple[np.ndarray, bool]:
    """
    HDBSCAN on PCA-reduced behavioral feature space.

    Preferred over K-Means because:
    - Behavioral segment boundaries are non-convex
    - Segment sizes differ significantly (binge_heavy ~25% vs re_engager ~10%)
    - Noise events (genuinely anomalous behavior) should be labeled -1, not
      forced into the nearest centroid

    Returns (labels, is_valid) where is_valid=False triggers K-Means fallback.
    """
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",  # excess of mass — better for varied densities

    )
    labels   = hdb.fit_predict(X)
    n_clusters = len(set(labels[labels >= 0]))
    n_noise    = int((labels == -1).sum())
    noise_pct  = n_noise / len(labels) * 100

    lines.append(
        f"[HDBSCAN] clusters={n_clusters}  noise={n_noise} ({noise_pct:.1f}%)"
    )

    # Validity gate: too few clusters or too much noise → use K-Means fallback
    is_valid = n_clusters >= 3 and noise_pct <= 30.0
    if not is_valid:
        lines.append(
            f"[HDBSCAN] Invalid result (clusters={n_clusters}, noise={noise_pct:.1f}%). "
            f"Falling back to K-Means."
        )
    return labels, is_valid


# ── K-Means fallback ──────────────────────────────────────────────────────────

def select_k(
    X: np.ndarray, k_min: int, k_max: int, lines: List[str]
) -> Tuple[int, Dict[int, float], Dict[int, float]]:
    """
    Composite k-selection retained from v2.0:
      score(k) = sil_norm(k) × 0.45
               + elbow_norm(k) × 0.30   (second derivative of inertia)
               + ch_norm(k)   × 0.25    (Calinski-Harabász)

    k range rebased for behavioral segments: default k_min=5, k_max=7.
    5 known segments ±2. Prevents collapse to k_min on near-homogeneous synthetic data.
    Previous k_min=4 allowed the selector to find k=4 even with 5 distinct segments.
    """
    n       = len(X)
    k_max   = min(k_max, n // 50)
    k_range = list(range(max(k_min, 2), k_max + 1))

    sil_scores: Dict[int, float] = {}
    inertias:   Dict[int, float] = {}
    ch_scores:  Dict[int, float] = {}

    # Segment-stratified sample for evaluation
    sample_size = min(5000, n)
    idx_sample  = np.random.choice(n, sample_size, replace=False)
    X_sample    = X[idx_sample]

    for k in k_range:
        km = MiniBatchKMeans(
            n_clusters=k, random_state=RANDOM_SEED,
            n_init=5, max_iter=300, batch_size=min(10000, n)
        )
        lab = km.fit_predict(X)
        if len(set(lab)) < 2:
            continue
        sil = silhouette_score(X_sample, lab[idx_sample])
        ch  = calinski_harabasz_score(X_sample, lab[idx_sample])
        sil_scores[k] = float(sil)
        inertias[k]   = float(km.inertia_)
        ch_scores[k]  = float(ch)
        lines.append(f"[K-SEL] k={k:>2}  sil={sil:.4f}  CH={ch:.1f}  inertia={km.inertia_:.1f}")

    if not sil_scores:
        return k_min, sil_scores, inertias

    ks = sorted(sil_scores.keys())

    def _norm(d: Dict) -> Dict:
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        return {k: (v - mn) / (mx - mn + 1e-9) for k, v in d.items()}

    sil_n  = _norm(sil_scores)
    ch_n   = _norm(ch_scores)

    iner_vals = [inertias[k] for k in ks]
    if len(iner_vals) >= 3:
        d1 = [iner_vals[i] - iner_vals[i + 1] for i in range(len(iner_vals) - 1)]
        d2 = [d1[i] - d1[i + 1] for i in range(len(d1) - 1)]
        elbow_raw = {ks[i + 1]: max(d2[i], 0) for i in range(len(d2))}
        for k in ks:
            elbow_raw.setdefault(k, 0.0)
    else:
        elbow_raw = {k: 0.0 for k in ks}
    elbow_n = _norm(elbow_raw)

    composite = {
        k: sil_n[k] * 0.45 + elbow_n.get(k, 0) * 0.30 + ch_n[k] * 0.25
        for k in ks
    }
    best_k = max(composite, key=composite.get)
    lines.append(f"[K-SEL] Composite: { {k: round(v, 4) for k, v in composite.items()} }")
    lines.append(
        f"[K-SEL] >> Best k={best_k}  "
        f"(composite={composite[best_k]:.4f}  sil={sil_scores[best_k]:.4f})"
    )
    return best_k, sil_scores, inertias


def run_kmeans(X: np.ndarray, k: int, lines: List[str]) -> Tuple[np.ndarray, Any]:
    """MiniBatchKMeans for n > 200k, full KMeans otherwise. Retained from v2.0."""
    n = len(X)
    if n > 200_000:
        km = MiniBatchKMeans(
            n_clusters=k, random_state=RANDOM_SEED,
            n_init=10, max_iter=500, batch_size=min(20000, n)
        )
    else:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20, max_iter=500)
    labels = km.fit_predict(X)
    lines.append(f"[KMEANS] k={k}  inertia={km.inertia_:.2f}")
    return labels, km


def kmeans_outlier_pass(
    X: np.ndarray, labels: np.ndarray, km_model: Any, lines: List[str],
    sigma_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft outlier detection for K-Means path.
    Replaces the DBSCAN noise pass which was eliminating 56% of events on
    weakly-structured synthetic data (eps=1.5 was calibrated for a different
    feature space and is too aggressive here).

    Method: for each point, compute distance to its assigned centroid.
    Points beyond sigma_threshold standard deviations from their centroid's
    distance distribution are marked as noise (label=-1).

    This is conservative by design — on well-formed behavioral data expect
    <5% noise. On flat synthetic data expect near-zero noise, which is correct:
    K-Means should assign all points; only genuine geometric outliers are removed.
    """
    centroids  = km_model.cluster_centers_
    final      = labels.copy()
    noise_mask = np.zeros(len(X), dtype=bool)
    total_noise = 0

    for c in np.unique(labels):
        mask  = labels == c
        pts   = X[mask]
        dists = np.linalg.norm(pts - centroids[c], axis=1)
        threshold = dists.mean() + sigma_threshold * dists.std()
        outliers  = dists > threshold
        idx = np.where(mask)[0][outliers]
        final[idx]      = -1
        noise_mask[idx] = True
        total_noise    += outliers.sum()

    pct = total_noise / len(X) * 100
    lines.append(
        f"[OUTLIER] K-Means outlier pass ({sigma_threshold}σ): "
        f"{total_noise} ({pct:.1f}%) events marked noise"
    )
    if pct > 10.0:
        lines.append(
            f"[WARN] Outlier rate {pct:.1f}% > 10% — feature space may be too flat "
            f"for meaningful clustering. Consider running with --algorithm hdbscan "
            f"or increasing --sessions_per_user in generate_signals.py."
        )
    return final, noise_mask


# ── Stability check ───────────────────────────────────────────────────────────

def stability_check(
    X: np.ndarray, labels: np.ndarray, n_boot: int, lines: List[str]
) -> Dict:
    """
    Bootstrap silhouette stability on PCA-reduced feature space.

    CE-07 DESIGN NOTE: stability is measured in X_pca (the compressed projection),
    not the original 22-feature behavioral space. The STABLE/MARGINAL/UNSTABLE label
    therefore reflects cluster separation in the PCA projection, not full feature
    space separation. These can differ if PCA has compressed discriminating dimensions.
    Downstream consumers (persona_engine.py, clustering_audit.py) must be aware that
    the stability label is a PCA-space measure. It is documented as such in
    cluster_metadata.json under stability.measurement_space = "pca".
    """
    valid  = labels >= 0
    Xv, Lv = X[valid], labels[valid]
    n      = len(Xv)
    scores = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=min(3000, n), replace=True)
        if len(set(Lv[idx])) < 2:
            continue
        try:
            scores.append(float(silhouette_score(Xv[idx], Lv[idx])))
        except Exception:
            pass
    mean_ = float(np.mean(scores)) if scores else 0.0
    std_  = float(np.std(scores))  if scores else 0.0
    ci_lo = mean_ - 1.96 * std_
    ci_hi = mean_ + 1.96 * std_
    label = (
        "STABLE"   if mean_ > 0.35 else
        "MARGINAL" if mean_ > 0.20 else
        "UNSTABLE"
    )
    lines.append(
        f"[STAB] Bootstrap sil={mean_:.4f} ±{std_:.4f}  "
        f"CI=[{ci_lo:.4f},{ci_hi:.4f}]  → {label}"
    )
    return {
        "mean_silhouette":   mean_,
        "std_silhouette":    std_,
        "ci_lower":          ci_lo,
        "ci_upper":          ci_hi,
        "stability_label":   label,
        "n_bootstrap":       n_boot,
        "measurement_space": "pca",  # CE-07: silhouette measured in PCA space, not original feature space
    }


# ── Cluster metadata ──────────────────────────────────────────────────────────

def build_cluster_metadata(
    fdf:            pd.DataFrame,
    labels:         np.ndarray,
    X_pca:          np.ndarray,
    X_2d:           np.ndarray,
    stability:      Dict,
    super_clusters: Dict,
    feature_cols:   List[str],
    algorithm_used: str,
    synthetic_validation_mode: bool = False,
) -> Dict:
    """
    Build full behavioral cluster profiles.

    Each cluster profile includes:
    - feature_profile: mean of all clustering features
    - behavioral_priors: base_completion, base_churn_30d, base_reactivation
      (consumed directly by Layer 4 Digital Twin Engine)
    - behavioral_label: auto-assigned from feature profile (e.g. binge_heavy)
    - segment_composition: distribution of Layer 1 segment_ids in cluster
      (used by persona_engine.py to map clusters back to known segments)
    - content_distribution: top content titles and types in cluster
    - top_discriminating_features: features most different from global mean
    - drift_baseline: global feature means for drift detection in production
    """
    cluster_ids  = sorted(set(labels[labels >= 0]))
    global_means = {
        col: float(pd.to_numeric(fdf[col], errors="coerce").mean())
        for col in feature_cols if col in fdf.columns
    }

    clusters = []
    for c in cluster_ids:
        mask  = labels == c
        c_fdf = fdf[mask]
        c_2d  = X_2d[mask]
        n     = int(mask.sum())

        # Feature profile means
        feat_profile: Dict[str, float] = {}
        for col in feature_cols:
            if col in c_fdf.columns:
                v = pd.to_numeric(c_fdf[col], errors="coerce").dropna()
                feat_profile[col] = round(float(v.mean()), 4) if len(v) else 0.0

        # Top discriminating features vs global mean
        deviations   = {
            col: abs(feat_profile.get(col, 0.0) - global_means.get(col, 0.0))
            for col in feature_cols
        }
        top_features = sorted(deviations, key=deviations.get, reverse=True)[:7]

        # Behavioral priors — direct inputs to Layer 4 twin models
        # CE-03: base_churn_30d must be user-level churn rate, not session mean.
        # churn_flag_encoded is per-session binary. Session mean systematically
        # underestimates churn for multi-session users (a 10-session user with one
        # churn event contributes 0.1; a single-session churner contributes 1.0).
        priors: Dict[str, float] = {}
        for prior_name, src_col in PRIOR_FIELDS.items():
            if src_col not in c_fdf.columns:
                continue
            if prior_name == "base_churn_30d" and "user_id" in c_fdf.columns:
                # Compute fraction of USERS who have churned (any session flagged)
                user_churn = (
                    c_fdf.groupby("user_id")["churn_flag_encoded"]
                    .apply(lambda x: int(pd.to_numeric(x, errors="coerce").max() == 1))
                )
                priors[prior_name] = round(float(user_churn.mean()), 4) if len(user_churn) else 0.0
            else:
                v = pd.to_numeric(c_fdf[src_col], errors="coerce").dropna()
                priors[prior_name] = round(float(v.mean()), 4) if len(v) else 0.0

        # Auto behavioral label — pass priors so re_engager rule can access base_reactivation
        behavioral_label = _auto_label(feat_profile, priors)

        # Segment composition — maps cluster back to known Layer 1 segments.
        # SYNTHETIC VALIDATION SCAFFOLDING ONLY — guarded behind
        # --synthetic_validation_mode flag. segment_id is a simulation control
        # variable generated by generate_signals.py and does not exist in real
        # client data. The flag ensures this block never executes on real inputs.
        segment_composition: Dict[str, Any] = {}
        if synthetic_validation_mode and "segment_id" in c_fdf.columns:
            sc = c_fdf["segment_id"].value_counts(normalize=True)
            segment_composition = {
                "dominant_segment":   str(sc.index[0]) if len(sc) else "unknown",
                "segment_distribution": {
                    str(k): round(float(v), 4) for k, v in sc.items()
                },
                "_synthetic_validation_only": True,
            }

        # Content distribution
        content_dist: Dict[str, Any] = {}
        if "content_id" in c_fdf.columns:
            top_content = c_fdf["content_id"].value_counts().head(5).to_dict()
            content_dist["top_content_ids"] = {str(k): int(v) for k, v in top_content.items()}
        if "content_type" in c_fdf.columns:
            # content_type is not in the feature store directly — carried as raw column
            pass
        if "event_type_raw" in c_fdf.columns:
            et = c_fdf["event_type_raw"].value_counts(normalize=True)
            content_dist["event_type_mix"] = {
                str(k): round(float(v), 4) for k, v in et.items()
            }

        # Churn and reactivation rates
        churn_rate = float(
            pd.to_numeric(c_fdf.get("churn_flag_encoded", pd.Series([])),
                          errors="coerce").mean()
        ) if "churn_flag_encoded" in c_fdf.columns else None

        reactivation_rate = float(
            pd.to_numeric(c_fdf.get("reactivation_signal", pd.Series([])),
                          errors="coerce").mean()
        ) if "reactivation_signal" in c_fdf.columns else None

        clusters.append({
            "cluster_id":                  str(uuid.uuid4()),
            "cluster_index":               int(c),
            "super_cluster":               super_clusters.get(c, 1),
            "behavioral_label":            behavioral_label,
            "size":                        n,
            "size_pct":                    round(n / len(labels) * 100, 2),
            "centroid_2d":                 [
                round(float(c_2d[:, 0].mean()), 4),
                round(float(c_2d[:, 1].mean()), 4),
            ],
            "feature_profile":             feat_profile,
            "top_discriminating_features": top_features,
            "behavioral_priors":           priors,
            "segment_composition":         segment_composition,
            "content_distribution":        content_dist,
            "churn_rate":                  round(churn_rate, 4) if churn_rate is not None else None,
            "reactivation_rate":           round(reactivation_rate, 4) if reactivation_rate is not None else None,
            "computed_at":                 datetime.utcnow().isoformat(),
            "engine_version":              ENGINE_VERSION,
        })

    # ── CAL-002: rank_personas label assignment pass ──────────────────────────
    # Replaces the old priority-based dedup. rank_personas() uses TRIGGER_RULES
    # (declarative gate + score_fn per label) with multi_assign support for
    # quick_churn, which legitimately covers multiple clusters when k > n_labels.
    assigned = rank_personas(clusters)

    # Apply final labels and log any changes
    for cl in clusters:
        old_label = cl["behavioral_label"]
        new_label = assigned.get(cl["cluster_index"], "mixed_behavior")
        cl["behavioral_label"] = new_label
        if old_label != new_label:
            print(
                f"[LABEL] Cluster #{cl['cluster_index']} "
                f"{old_label} → {new_label}"
            )

    # ── CAL-002: Behavioral coverage check ───────────────────────────────────
    # After labelling, verify all expected behavioral labels are represented.
    # A missing label indicates k was too small to separate all behavioral
    # segments — the unlabelled segment has been absorbed into another cluster.
    # At 100k+ users, k=5 is often insufficient; use --k_min 6 to force more
    # clusters and give the selector room to recover the missing segment.
    EXPECTED_LABELS = {"completion_obsessed", "binge_heavy", "re_engager",
                       "casual_dip", "quick_churn"}
    produced_labels = {cl["behavioral_label"] for cl in clusters} - {"mixed_behavior"}
    missing_labels  = EXPECTED_LABELS - produced_labels
    duplicate_labels = {
        lbl for lbl in produced_labels
        if sum(1 for cl in clusters if cl["behavioral_label"] == lbl) > 1
    }
    if missing_labels:
        print(
            f"[WARN] CAL-002: Missing behavioral labels: {sorted(missing_labels)}. "
            f"k={len(cluster_ids)} clusters is likely insufficient to separate all "
            f"5 behavioral segments at this dataset scale. "
            f"Re-run with --k_min {len(cluster_ids) + 1} to allow the selector to "
            f"produce an additional cluster. Duplicate labels: {sorted(duplicate_labels)}."
        )
    if duplicate_labels:
        print(
            f"[WARN] CAL-002: Duplicate labels: {sorted(duplicate_labels)}. "
            f"Multi-assign (quick_churn) is expected; other duplicates indicate "
            f"threshold overlap. Missing: {sorted(missing_labels)}."
        )

    return {
        "version":       ENGINE_VERSION,
        "computed_at":   datetime.utcnow().isoformat(),
        "algorithm":     algorithm_used,
        "n_events":      len(labels),
        "n_clusters":    len(cluster_ids),
        "noise_events":  int((labels == -1).sum()),
        "noise_pct":     round((labels == -1).mean() * 100, 2),
        "stability":     stability,
        "clusters":      clusters,
        "drift_baseline": {
            "timestamp":   datetime.utcnow().isoformat(),
            "feature_means": {
                col: round(float(pd.to_numeric(fdf[col], errors="coerce").mean()), 6)
                for col in feature_cols if col in fdf.columns
            },
            "cluster_size_distribution": {
                str(c): int((labels == c).sum()) for c in cluster_ids
            },
        },
    }


# ── Compare mode ──────────────────────────────────────────────────────────────

def compare_mode_report(
    meta: Dict, fdf: pd.DataFrame, content_refs: List[str], lines: List[str]
) -> None:
    """
    Side-by-side cluster profile comparison for two content titles.
    Maps to the old pipeline's compare mode.
    Adds comparison block to lines — included in clustering_report.txt.
    """
    lines += ["", "=" * 70, "COMPARE MODE — CLUSTER PROFILE DELTA", "=" * 70]
    if len(content_refs) < 2:
        lines.append("[WARN] Compare mode requires exactly 2 content_refs. Skipped.")
        return

    ref_a, ref_b = content_refs[0], content_refs[1]
    lines.append(f"  Title A: {ref_a}    Title B: {ref_b}")
    lines.append("")

    # Per-title cluster composition
    for ref in [ref_a, ref_b]:
        lines.append(f"  [{ref}] behavioral label distribution:")
        # CE-05: removed dead loop (cl_events_in_ref computed but immediately overridden)
        if "content_id" in fdf.columns:
            sub = fdf[fdf["content_id"] == ref]
            if "segment_id" in sub.columns:
                dist = sub["segment_id"].value_counts(normalize=True)
                for seg, pct in dist.items():
                    lines.append(f"    {seg:<28} {pct*100:.1f}%")
        lines.append("")

    # Key behavioral metric delta
    DELTA_FEATURES = [
        "completion_rate_smooth",
        "churn_risk_score",
        "binge_signal",
        "friction_index",
        "session_intensity_score",
    ]
    lines.append(f"  {'Feature':<32} {'Title A':>10} {'Title B':>10} {'Delta':>10}")
    lines.append("  " + "-" * 64)
    for feat in DELTA_FEATURES:
        if feat not in fdf.columns:
            continue
        if "content_id" in fdf.columns:
            va = pd.to_numeric(fdf[fdf["content_id"] == ref_a][feat], errors="coerce").mean()
            vb = pd.to_numeric(fdf[fdf["content_id"] == ref_b][feat], errors="coerce").mean()
            delta = vb - va
            lines.append(f"  {feat:<32} {va:>10.4f} {vb:>10.4f} {delta:>+10.4f}")


# ── Output writers ────────────────────────────────────────────────────────────

def write_report(meta: Dict, lines: List[str], path: str) -> None:
    cluster_ids = sorted(c["cluster_index"] for c in meta["clusters"])

    lines += [
        "",
        "=" * 70,
        "CLUSTER SUMMARY",
        "=" * 70,
        f"{'Index':>6} {'Label':<24} {'Size':>7} {'%':>6}  "
        f"{'Compl':>6} {'Churn':>6} {'Reactiv':>8}  Dominant segment",
        "-" * 70,
    ]
    for cl in sorted(meta["clusters"], key=lambda c: c["cluster_index"]):
        idx   = cl["cluster_index"]
        label = cl["behavioral_label"]
        size  = cl["size"]
        pct   = cl["size_pct"]
        comp  = cl["behavioral_priors"].get("base_completion", 0.0)
        churn = cl["behavioral_priors"].get("base_churn_30d", 0.0)
        react = cl["behavioral_priors"].get("base_reactivation", 0.0)
        seg_comp = cl.get("segment_composition", {})
        dom = seg_comp.get("dominant_segment", "—")
        if dom != "—":
            dom = f"{dom} [SYNTH]"
        lines.append(
            f"{idx:>6} {label:<24} {size:>7} {pct:>5.1f}%  "
            f"{comp:>6.4f} {churn:>6.4f} {react:>8.4f}  {dom}"
        )

    lines.append(
        f"\nNoise: {meta['noise_events']} ({meta['noise_pct']:.1f}%)  "
        f"| Algorithm: {meta['algorithm']}  "
        f"| Stability: {meta['stability']['stability_label']} "
        f"({meta['stability']['mean_silhouette']:.4f})"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[DONE] Clustering report → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="TwinSim Behavioral Clustering Engine — Layer 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",    default="feature_store.csv",
                    help="Feature matrix CSV from feature_store.py (Layer 2 output).")
    ap.add_argument("--out",      default="cluster_assignments.csv",
                    help="Output cluster assignment CSV.")
    ap.add_argument("--metadata", default="cluster_metadata.json",
                    help="Output cluster metadata JSON.")
    ap.add_argument("--report",   default="clustering_report.txt",
                    help="Output pipeline log + summary TXT.")
    ap.add_argument("--content_refs", nargs="+", default=None,
                    help="Focused mode: one or more content IDs.")
    ap.add_argument("--segments",     nargs="+", default=None,
                    help="Focused mode: one or more segment IDs.")
    ap.add_argument("--mode",    choices=["full", "compare"], default="full",
                    help="full = standard clustering. compare = side-by-side delta (requires 2 content_refs).")
    ap.add_argument("--algorithm", choices=["auto", "hdbscan", "kmeans"], default="auto",
                    help="auto = HDBSCAN with K-Means fallback. hdbscan/kmeans = force.")
    ap.add_argument("--min_cluster_size", type=int, default=30,
                    help="HDBSCAN min_cluster_size. Default 30.")
    ap.add_argument("--min_samples",      type=int, default=5,
                    help="HDBSCAN min_samples. Default 5.")
    ap.add_argument("--k_min",   type=int, default=5,
                    help="K-Means fallback: min k. Default 5 (5 known behavioral segments).")
    ap.add_argument("--k_max",   type=int, default=7,
                    help="K-Means fallback: max k. Default 7 (5 segments ±2).")
    ap.add_argument("--n_boot",  type=int, default=30,
                    help="Bootstrap stability iterations. Default 30.")
    ap.add_argument("--pca_var", type=float, default=0.90,
                    help="PCA variance threshold. Default 0.90.")
    ap.add_argument("--synthetic_validation_mode", action="store_true",
                    help=(
                        "Enable synthetic data validation scaffolding. "
                        "When set, segment_id-dependent checks (segment_composition block, "
                        "dominant_segment reporting) are computed and written to cluster metadata. "
                        "MUST NOT be used on real client data — segment_id is a simulation "
                        "control variable that does not exist in production inputs. "
                        "Omit this flag for all production runs."
                    ))
    args = ap.parse_args()

    lines = [
        "=" * 70,
        f"TWINSIM BEHAVIORAL CLUSTERING ENGINE v{ENGINE_VERSION} — PIPELINE LOG",
        f"Run at: {datetime.utcnow().isoformat()}",
        "=" * 70,
    ]

    # Load feature store
    print(f"[INFO] Loading {args.input}...")
    fdf = pd.read_csv(args.input)
    print(f"[INFO] Feature matrix: {fdf.shape[0]:,} events × {fdf.shape[1]} columns")
    lines.append(f"[INFO] Feature matrix: {fdf.shape}")

    if args.synthetic_validation_mode:
        print("[SYNTH] --synthetic_validation_mode ACTIVE — segment_id scaffolding enabled.")
        print("[SYNTH] WARNING: Do not use this flag on real client data.")
        lines.append("[SYNTH] synthetic_validation_mode active — segment_composition block enabled.")
    else:
        lines.append("[INFO] Production mode — segment_composition block disabled (no segment_id dependency).")

    # Focused mode filtering
    # CE-04 DESIGN NOTE: when --content_refs or --segments filters are applied,
    # StandardScaler and PCA are fit on the FILTERED subset only. Cluster boundaries
    # and PCA components are relative to the subset, not the global population.
    # Persona profiles from a focused run are NOT comparable to a full-population run.
    # cluster_metadata.json will include a focused_mode warning when active.
    focused_mode = bool(args.content_refs or args.segments)
    if args.content_refs:
        if "content_id" in fdf.columns:
            before = len(fdf)
            fdf = fdf[fdf["content_id"].isin(args.content_refs)].reset_index(drop=True)
            lines.append(f"[FOCUS] content_refs filter: {before:,} → {len(fdf):,} events ({args.content_refs})")
        else:
            lines.append("[WARN] content_refs specified but content_id not in feature store. Ignored.")

    if args.segments:
        if "segment_id" in fdf.columns:
            before = len(fdf)
            fdf = fdf[fdf["segment_id"].isin(args.segments)].reset_index(drop=True)
            lines.append(f"[FOCUS] segments filter: {before:,} → {len(fdf):,} events ({args.segments})")
        else:
            lines.append("[WARN] segments specified but segment_id not in feature store. Ignored.")

    if len(fdf) < 50:
        print(f"[ERROR] Only {len(fdf)} events after filtering — too few to cluster.")
        raise SystemExit(1)

    # Preprocess + PCA
    X_scaled, feat_cols, _scaler = preprocess(fdf, lines)
    X_pca, _pca                  = reduce_pca(X_scaled, args.pca_var, lines)
    X_2d                         = embed_2d(X_pca, lines)

    # Clustering
    algorithm_used = "hdbscan"
    noise_mask     = np.zeros(len(X_pca), dtype=bool)

    use_hdbscan = args.algorithm in ("auto", "hdbscan")
    use_kmeans  = args.algorithm == "kmeans"

    # ── HDBSCAN restoration trigger check (CAL-001) ───────────────────────────
    # In "auto" mode, evaluate whether the feature data meets the restoration
    # criteria before attempting HDBSCAN. If the criteria are not met, HDBSCAN
    # will produce degenerate results (near-flat density → negative silhouette).
    # Trigger: avg_watch_gap_norm var > 0.05 AND binge_index_score zeros < 30%.
    # These confirm P3-A+B session data is flowing (segment-differentiated gaps
    # and genuine binge density), which is the prerequisite for HDBSCAN to find
    # meaningful density gradients.
    if use_hdbscan and args.algorithm == "auto":
        trigger_met = True
        trigger_log = []

        if "avg_watch_gap_norm" in fdf.columns:
            gap_var = float(pd.to_numeric(fdf["avg_watch_gap_norm"], errors="coerce").var())
            if gap_var > 0.05:
                trigger_log.append(f"avg_watch_gap_norm var={gap_var:.4f} > 0.05 ✓")
            else:
                trigger_log.append(f"avg_watch_gap_norm var={gap_var:.4f} <= 0.05 ✗")
                trigger_met = False
        else:
            trigger_log.append("avg_watch_gap_norm absent ✗")
            trigger_met = False

        if "binge_index_score" in fdf.columns:
            binge_zero_pct = float((pd.to_numeric(fdf["binge_index_score"], errors="coerce") == 0).mean() * 100)
            if binge_zero_pct < 30.0:
                trigger_log.append(f"binge_index_score zeros={binge_zero_pct:.1f}% < 30% ✓")
            else:
                trigger_log.append(f"binge_index_score zeros={binge_zero_pct:.1f}% >= 30% ✗")
                trigger_met = False
        else:
            trigger_log.append("binge_index_score absent ✗")
            trigger_met = False

        if trigger_met:
            lines.append(f"[TRIGGER] HDBSCAN restoration criteria MET: {'; '.join(trigger_log)}")
        else:
            lines.append(
                f"[TRIGGER] HDBSCAN restoration criteria NOT MET: {'; '.join(trigger_log)}. "
                f"Skipping HDBSCAN — using K-Means directly. "
                f"(Use --algorithm hdbscan to force HDBSCAN regardless.)"
            )
            use_hdbscan    = False
            use_kmeans     = True
            algorithm_used = "kmeans"

    if use_hdbscan:
        labels, hdb_valid = run_hdbscan(X_pca, args.min_cluster_size, args.min_samples, lines)
        if not hdb_valid:
            use_kmeans     = True
            algorithm_used = "kmeans_fallback"
        else:
            # Additional validity gate: negative or near-zero silhouette means
            # HDBSCAN found no meaningful density structure — fall back to K-Means.
            noise_mask = labels == -1
            valid_mask = labels >= 0
            if valid_mask.sum() > 1 and len(set(labels[valid_mask])) > 1:
                sample_n = min(5000, int(valid_mask.sum()))
                idx = np.random.choice(np.where(valid_mask)[0], sample_n, replace=False)
                hdb_sil = float(silhouette_score(X_pca[idx], labels[idx]))
                lines.append(f"[HDBSCAN] Post-validity silhouette={hdb_sil:.4f}")
                if hdb_sil < 0.05:
                    lines.append(
                        f"[HDBSCAN] Silhouette {hdb_sil:.4f} < 0.05 — degenerate structure. "
                        f"Falling back to K-Means."
                    )
                    use_kmeans     = True
                    algorithm_used = "kmeans_fallback"
                    noise_mask     = np.zeros(len(X_pca), dtype=bool)
            else:
                use_kmeans     = True
                algorithm_used = "kmeans_fallback"
                noise_mask     = np.zeros(len(X_pca), dtype=bool)
    else:
        hdb_valid = False

    if use_kmeans:
        if algorithm_used != "kmeans_fallback":
            algorithm_used = "kmeans"
        best_k, _, _ = select_k(X_pca, args.k_min, args.k_max, lines)
        labels, km_model = run_kmeans(X_pca, best_k, lines)
        labels, noise_mask = kmeans_outlier_pass(X_pca, labels, km_model, lines)

    # Stability check
    stability = stability_check(X_pca, labels, args.n_boot, lines)

    # Hierarchical super-clusters (group similar clusters for persona engine)
    # CE-06 DESIGN NOTE: Ward linkage on PCA centroid means. Given that the first
    # PCA component is dominated by completion-related features (even after FS-04/05/06
    # removals), Ward linkage systematically groups clusters by completion similarity
    # rather than broader behavioral taxonomy. Super-cluster groupings passed to Layer 4
    # reflect completion proximity, not holistic behavioral similarity.
    # This will improve once HDBSCAN restoration criteria are met and the PCA space
    # is more evenly distributed across behavioral dimensions.
    cluster_ids   = sorted(set(labels[labels >= 0]))
    super_clusters: Dict[int, int] = {}
    if len(cluster_ids) >= 3:
        try:
            centroids = np.array([X_pca[labels == c].mean(axis=0) for c in cluster_ids])
            Z         = linkage(centroids, method="ward")
            n_super   = min(3, len(cluster_ids))
            cuts      = fcluster(Z, t=n_super, criterion="maxclust")
            super_clusters = {c: int(cuts[i]) for i, c in enumerate(cluster_ids)}
            lines.append(f"[HIER] Super-clusters: {super_clusters}")
        except Exception as e:
            lines.append(f"[HIER] Failed: {e}")
            super_clusters = {c: 1 for c in cluster_ids}
    else:
        super_clusters = {c: 1 for c in cluster_ids}

    # Validation metrics
    valid_mask = labels >= 0
    if valid_mask.sum() > 1 and len(set(labels[valid_mask])) > 1:
        sil = silhouette_score(
            X_pca[valid_mask], labels[valid_mask],
            sample_size=min(5000, int(valid_mask.sum()))
        )
        dbi = davies_bouldin_score(X_pca[valid_mask], labels[valid_mask])
        lines.append(f"[VALID] Final silhouette={sil:.4f}  DBI={dbi:.4f}")

    # Build metadata
    meta = build_cluster_metadata(
        fdf, labels, X_pca, X_2d,
        stability, super_clusters, feat_cols, algorithm_used,
        synthetic_validation_mode=args.synthetic_validation_mode,
    )

    # CE-04: document focused mode — scaler and PCA fit on subset, not global population.
    # Profiles from a focused run are not comparable to a full-population run.
    meta["focused_mode"] = focused_mode
    if focused_mode:
        meta["focused_mode_warning"] = (
            "CE-04: StandardScaler and PCA fit on filtered subset only. "
            "Cluster boundaries are relative to the subset, not the global population. "
            "Simulation priors from this run are subset-specific and must not be "
            "compared directly to full-population persona profiles."
        )
        lines.append("[WARN] CE-04: Focused mode active — scaler/PCA fit on subset. "
                     "Priors are subset-specific.")

    # CA-04/XL-05: write feature_store_n to metadata so clustering_audit.py and
    # persona_engine.py can perform event dropout checks.
    meta["feature_store_n"] = len(fdf) + int((labels == -1).sum())

    # Compare mode delta report
    if args.mode == "compare" and args.content_refs:
        compare_mode_report(meta, fdf, args.content_refs, lines)

    # Assignment CSV
    assign_df = pd.DataFrame({
        "event_id":        fdf["event_id"].values    if "event_id"    in fdf.columns else range(len(fdf)),
        "user_id":         fdf["user_id"].values     if "user_id"     in fdf.columns else "",
        "content_id":      fdf["content_id"].values  if "content_id"  in fdf.columns else "",
        "segment_id":      fdf["segment_id"].values  if "segment_id"  in fdf.columns else "",
        "cluster_index":   labels,
        "behavioral_label": [
            next((c["behavioral_label"] for c in meta["clusters"] if c["cluster_index"] == l), "noise")
            for l in labels
        ],
        "is_noise":        noise_mask.astype(int),
        "embed_x":         X_2d[:, 0].round(4),
        "embed_y":         X_2d[:, 1].round(4),
        "cluster_version": ENGINE_VERSION,
        "clustered_at":    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    assign_df.to_csv(args.out, index=False)
    print(f"[DONE] Cluster assignments → {args.out}")

    with open(args.metadata, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[DONE] Cluster metadata → {args.metadata}")

    write_report(meta, lines, args.report)

    n_clusters = meta["n_clusters"]
    noise_pct  = meta["noise_pct"]
    sil_mean   = stability["mean_silhouette"]
    stab_label = stability["stability_label"]
    print(
        f"\n[SUMMARY] {n_clusters} clusters | noise={noise_pct:.1f}% | "
        f"sil={sil_mean:.3f} ({stab_label}) | algorithm={algorithm_used}"
    )
    print(f"[NEXT] Run audit: python clustering_audit.py "
          f"--assignments {args.out} --metadata {args.metadata}")


if __name__ == "__main__":
    main()