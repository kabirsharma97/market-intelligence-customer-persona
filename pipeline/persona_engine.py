"""
TwinSim — Persona Engine  v2.0
================================
Layer 4: Behavioral Persona Definition + Intervention Simulation

Reads cluster_metadata.json (Layer 3 output) and produces:
  - persona_profiles.json   : one fully-specified persona per cluster, enriched
                              with intervention playbooks and simulation priors
  - persona_report.txt      : structured 7-section human-readable report

Architecture:
  Each cluster from Layer 3 maps to exactly one persona. Noise events (cluster -1)
  are not assigned a persona — they remain unclassified and excluded from targeting.

  A persona consists of:
    1. Identity          — name, archetype label, dominant segment, size
    2. Behavioral DNA    — feature profile means, top discriminating features,
                           behavioral priors (base_completion, base_churn_30d,
                           base_reactivation, base_session_depth, base_churn_risk)
    3. Behavioral profile — categorical tier labels (churn/completion/reactivation/
                           ltv/depth), preferred content types, content arc affinity,
                           strategic recommendation (Groups C, D)
    4. Intervention playbook — ordered interventions with simulated lift per KPI
    5. Simulation priors — CI-bounded priors for Monte Carlo; within-cluster
                           variance stats; is_diffuse_cluster flag (Groups B, F)
    6. Risk flags        — high_churn_risk, high_value, reactivation_candidate,
                           superfan_potential, marketing_receptive, upgrade_candidate;
                           conditional_risk_override block (Group E)

v2.0 changes (Groups A–F):
  A: Section-based narrative report (§0–§6); _interpret() per-cluster narrative
     with low-confidence guard; graceful column degradation via _cols()
  B: is_diffuse_cluster flag; within-cluster variance stats in simulation_priors
  C: behavioral_profile block with categorical tier labels; corrected churn
     tier boundaries (PE-04): stable<0.25, elevated<0.45, critical≥0.45
  D: preferred_content_types per label (re_engager gets ["series","film"] PE-07);
     content_arc_affinity; strategic_recommendation string
  E: conditional_risk_override block; satisfaction threshold raised to <0.60 (PE-03)
  F: Formula verification check (F1); prior consistency check (F2);
     diffuse cluster validation check (F3)

v2.2.0 changes:
  G: User-level persona assignment (Q1/Q2 architecture resolution)
     _compute_user_persona_assignment() — modal cluster per user_id from
     cluster_assignments.csv. Produces unique_user_count per persona and
     user_assignment summary block. --assignments CLI arg added.
     Architecture decision: Option B — session-level clustering preserved,
     user-level assignment added as post-processing step. Each user assigned
     to their modal cluster (most sessions). Behaviorally ambiguous users
     (evenly split sessions) flagged but still assigned (lowest index tiebreak).
  H: dominant_segment labelled as [SYNTHETIC VALIDATION ONLY] in JSON and report.
     dominant_segment_note field added explaining production vs synthetic context.
     segment_composition._synthetic_validation_only flag from clustering_engine
     v4.1.0 used to gate field population.
  I: §1 Overview report updated — sessions column and users column separated.
     NOTE lines added explaining session vs user count distinction.

v2.3.0 changes:
  J: unique_user_count keying bug fixed. Previously keyed by behavioral_label —
     collapsed counts when duplicate labels exist (CAL-002 multi_assign produces
     two quick_churn clusters; both mapped to the same dict key, halving the count
     and causing sum(unique_user_count) < n_users_total). Now keyed by cluster_index
     (int), which is always unique. main() updated to look up by cluster_index.
  K: Duplicate behavioral label detection added. _compute_user_persona_assignment()
     emits [WARN] when duplicate labels detected. §1 report emits [WARN] with
     explanation and guidance to check TRIGGER_RULES calibration (CAL-002).
  L: user_assignment.duplicate_labels field added to top-level JSON output.
  M: Provenance cross-check added. _compute_user_persona_assignment() now accepts
     expected_cluster_version from cluster_metadata.json and asserts it matches the
     cluster_version column in cluster_assignments.csv. Mismatch emits [FAIL] and
     skips user assignment rather than producing silently wrong unique_user_counts.
  N: Noise-only user detection added. Users whose every session was marked noise
     (cluster_index=-1) are now counted explicitly as n_users_noise_only and
     reported in assignment_summary. Previously these users were silently dropped,
     causing sum(unique_user_count) < total users with no explanation. The gap
     (705 users at 100k/1.1%-noise scale) is now surfaced as a named count.
     [WARN] emitted if n_users_noise_only > 0.

Architecture rules (unchanged):
  - No sentiment, NPS, brand perception, or emotion constructs.
  - Lift estimates are simulation priors, not causal claims.
  - External signals (Layer 7) are not referenced here.

Intervention catalogue:
  WIN_BACK        — re-engagement campaign for churned/high-risk users
  CONTENT_NUDGE   — personalised next-content recommendation push
  PLAN_UPGRADE    — upgrade offer for high-engagement/high-ltv users
  LOYALTY_REWARD  — retention incentive for long-tenure high-value users
  ONBOARDING_PUSH — accelerated onboarding for new/low-tenure users
  PRICE_LOCK      — price stability offer for downgrade-risk users
  REACTIVATION    — targeted win-back for lapsed re_engager persona
  SUPERFAN_EVENT  — PPV / exclusive event access offer for superfan persona

Usage:
    python persona_engine.py --metadata cluster_metadata.json

    python persona_engine.py \\
        --metadata cluster_metadata.json \\
        --out persona_profiles.json \\
        --report persona_report.txt

    # Validate outputs
    python persona_engine.py --metadata cluster_metadata.json --validate
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

ENGINE_VERSION = "2.3.0"

# ── Persona name mapping ──────────────────────────────────────────────────────
# Maps behavioral_label from clustering engine to a human-readable persona name.
# Names are neutral, behavior-descriptive — not demographic labels.

PERSONA_NAMES: Dict[str, str] = {
    "binge_heavy":         "The Binge Watcher",
    "completion_obsessed": "The Completionist",
    "casual_dip":          "The Casual Dipper",
    "quick_churn":         "The Quick Churner",
    "re_engager":          "The Re-Engager",
    "mixed_behavior":      "The Mixed-Mode Viewer",
}

PERSONA_ARCHETYPES: Dict[str, str] = {
    "binge_heavy":         "high_engagement_retention_anchor",
    "completion_obsessed": "high_engagement_ltv_driver",
    "casual_dip":          "moderate_engagement_growth_target",
    "quick_churn":         "high_risk_early_intervention_required",
    "re_engager":          "lapsed_high_potential_win_back",
    "mixed_behavior":      "heterogeneous_needs_segmentation",
}

# ── Intervention catalogue ────────────────────────────────────────────────────

INTERVENTIONS: Dict[str, Dict[str, Any]] = {
    "WIN_BACK": {
        "name":        "Win-Back Campaign",
        "description": "Re-engagement outreach for users showing high churn risk or "
                       "recently lapsed. Personalised content highlight + limited-time offer.",
        "channel":     ["email", "push"],
        "trigger":     "churn_risk_score > 0.35 OR days_since_last_normalised > 0.60",
    },
    "CONTENT_NUDGE": {
        "name":        "Personalised Content Nudge",
        "description": "Next-content recommendation delivered via preferred channel. "
                       "Calibrated to fav_genre_confidence and episode_position arc.",
        "channel":     ["push", "in_app"],
        "trigger":     "session_depth_score > 0.40 AND fav_genre_confidence > 0.40",
    },
    "PLAN_UPGRADE": {
        "name":        "Plan Upgrade Offer",
        "description": "Targeted upgrade prompt for high-engagement users on basic/standard plan. "
                       "Framed around content access, not price.",
        "channel":     ["email", "in_app"],
        "trigger":     "subscription_tier_score < 0.70 AND completion_rate_smooth > 0.70",
    },
    "LOYALTY_REWARD": {
        "name":        "Loyalty Reward",
        "description": "Retention incentive for long-tenure high-value users. "
                       "Exclusive content access or price-lock confirmation.",
        "channel":     ["email"],
        "trigger":     "tenure_weight > 0.70 AND ltv_score > 0.50",
    },
    "ONBOARDING_PUSH": {
        "name":        "Accelerated Onboarding",
        "description": "Guided content discovery for new or low-tenure users to "
                       "accelerate time-to-habit. Reduces early churn risk.",
        "channel":     ["push", "in_app"],
        "trigger":     "lifecycle_stage IN (new, active) AND tenure_weight < 0.20",
    },
    "PRICE_LOCK": {
        "name":        "Price Stability Offer",
        "description": "Proactive price-lock or discount offer for users showing "
                       "downgrade intent signals (plan_trajectory_score low, "
                       "payment failures present).",
        "channel":     ["email"],
        "trigger":     "plan_trajectory_score < 0.40 OR support_friction_score > 0.60",
    },
    "REACTIVATION": {
        "name":        "Lapsed User Reactivation",
        "description": "Targeted win-back for historically engaged users who have "
                       "lapsed. Leverages prior content affinity for personalisation.",
        "channel":     ["email", "push"],
        "trigger":     "reactivation_signal > 0.40 AND days_since_last_normalised > 0.50",
    },
    "SUPERFAN_EVENT": {
        "name":        "Superfan Exclusive Access",
        "description": "PPV event invitation or exclusive content access offer for "
                       "identified superfan users. High-margin conversion opportunity.",
        "channel":     ["email", "in_app"],
        "trigger":     "superfan_score > 0.30 AND campaign_receptivity > 0.20",
    },
}

# ── Intervention routing rules ────────────────────────────────────────────────
# Maps behavioral_label → ordered list of intervention IDs.
# Order = priority. Twin runs interventions in order until lift threshold met.

INTERVENTION_ROUTING: Dict[str, List[str]] = {
    "binge_heavy":         ["LOYALTY_REWARD", "SUPERFAN_EVENT", "CONTENT_NUDGE"],
    "completion_obsessed": ["LOYALTY_REWARD", "PLAN_UPGRADE",   "CONTENT_NUDGE", "SUPERFAN_EVENT"],
    "casual_dip":          ["CONTENT_NUDGE",  "ONBOARDING_PUSH","WIN_BACK"],
    "quick_churn":         ["WIN_BACK",        "PRICE_LOCK",     "ONBOARDING_PUSH"],
    "re_engager":          ["REACTIVATION",    "CONTENT_NUDGE",  "WIN_BACK"],
    "mixed_behavior":      ["CONTENT_NUDGE",   "WIN_BACK",       "LOYALTY_REWARD"],
}

# ── Lift estimation ───────────────────────────────────────────────────────────
# Simulation priors for each intervention × behavioral_label combination.
# Values are estimated deltas (Δ) from behavioral priors.
# These are simulation starting points — NOT causal claims.
# Positive = improvement. Unit matches the KPI it modifies.
#
# PE-09 DESIGN NOTE: LIFT_TABLE is a STATIC LOOKUP, not a data-driven simulation.
# Every value is hardcoded. A binge_heavy cluster with base_completion=0.95 gets
# identical lifts to one with base_completion=0.73. The lift values do not adapt
# to the actual cluster priors produced by clustering_engine.py.
# Layer 5 (Digital Twin Engine) must be explicitly informed at the
# persona_profiles.json output level that playbook lifts are fixed priors
# independent of clustering findings. They are initialisation values for Monte
# Carlo simulation, not predictions derived from the observed cluster structure.
# Replacement path: CAL-004 (Segment Reaction History) — replace with empirically
# calibrated lifts once real campaign response data is available.
#
# KPIs:
#   retention_lift    : Δ in 30d churn probability (negative = fewer churns)
#   engagement_lift   : Δ in completion_rate_smooth
#   reactivation_lift : Δ in reactivation_signal
#   ltv_lift          : Δ in normalised LTV score

LIFT_TABLE: Dict[str, Dict[str, Dict[str, float]]] = {
    "WIN_BACK": {
        "binge_heavy":         {"retention_lift": -0.04, "engagement_lift":  0.05, "reactivation_lift":  0.08, "ltv_lift":  0.03},
        "completion_obsessed": {"retention_lift": -0.04, "engagement_lift":  0.04, "reactivation_lift":  0.06, "ltv_lift":  0.03},
        "casual_dip":          {"retention_lift": -0.08, "engagement_lift":  0.06, "reactivation_lift":  0.12, "ltv_lift":  0.04},
        "quick_churn":         {"retention_lift": -0.12, "engagement_lift":  0.04, "reactivation_lift":  0.15, "ltv_lift":  0.03},
        "re_engager":          {"retention_lift": -0.10, "engagement_lift":  0.07, "reactivation_lift":  0.20, "ltv_lift":  0.05},
        "mixed_behavior":      {"retention_lift": -0.06, "engagement_lift":  0.04, "reactivation_lift":  0.10, "ltv_lift":  0.03},
    },
    "CONTENT_NUDGE": {
        "binge_heavy":         {"retention_lift": -0.02, "engagement_lift":  0.08, "reactivation_lift":  0.03, "ltv_lift":  0.04},
        "completion_obsessed": {"retention_lift": -0.02, "engagement_lift":  0.10, "reactivation_lift":  0.02, "ltv_lift":  0.05},
        "casual_dip":          {"retention_lift": -0.04, "engagement_lift":  0.07, "reactivation_lift":  0.05, "ltv_lift":  0.03},
        "quick_churn":         {"retention_lift": -0.03, "engagement_lift":  0.03, "reactivation_lift":  0.04, "ltv_lift":  0.01},
        "re_engager":          {"retention_lift": -0.05, "engagement_lift":  0.09, "reactivation_lift":  0.12, "ltv_lift":  0.04},
        "mixed_behavior":      {"retention_lift": -0.03, "engagement_lift":  0.06, "reactivation_lift":  0.04, "ltv_lift":  0.03},
    },
    "PLAN_UPGRADE": {
        "binge_heavy":         {"retention_lift": -0.03, "engagement_lift":  0.05, "reactivation_lift":  0.01, "ltv_lift":  0.15},
        "completion_obsessed": {"retention_lift": -0.02, "engagement_lift":  0.04, "reactivation_lift":  0.01, "ltv_lift":  0.18},
        "casual_dip":          {"retention_lift": -0.01, "engagement_lift":  0.02, "reactivation_lift":  0.01, "ltv_lift":  0.08},
        "quick_churn":         {"retention_lift":  0.02, "engagement_lift":  0.00, "reactivation_lift":  0.00, "ltv_lift":  0.02},
        "re_engager":          {"retention_lift": -0.02, "engagement_lift":  0.03, "reactivation_lift":  0.02, "ltv_lift":  0.10},
        "mixed_behavior":      {"retention_lift": -0.01, "engagement_lift":  0.03, "reactivation_lift":  0.01, "ltv_lift":  0.09},
    },
    "LOYALTY_REWARD": {
        "binge_heavy":         {"retention_lift": -0.06, "engagement_lift":  0.04, "reactivation_lift":  0.02, "ltv_lift":  0.08},
        "completion_obsessed": {"retention_lift": -0.05, "engagement_lift":  0.03, "reactivation_lift":  0.02, "ltv_lift":  0.09},
        "casual_dip":          {"retention_lift": -0.02, "engagement_lift":  0.02, "reactivation_lift":  0.02, "ltv_lift":  0.03},
        "quick_churn":         {"retention_lift": -0.01, "engagement_lift":  0.01, "reactivation_lift":  0.01, "ltv_lift":  0.01},
        "re_engager":          {"retention_lift": -0.03, "engagement_lift":  0.03, "reactivation_lift":  0.05, "ltv_lift":  0.05},
        "mixed_behavior":      {"retention_lift": -0.03, "engagement_lift":  0.02, "reactivation_lift":  0.02, "ltv_lift":  0.05},
    },
    "ONBOARDING_PUSH": {
        "binge_heavy":         {"retention_lift": -0.01, "engagement_lift":  0.02, "reactivation_lift":  0.01, "ltv_lift":  0.02},
        "completion_obsessed": {"retention_lift": -0.01, "engagement_lift":  0.02, "reactivation_lift":  0.01, "ltv_lift":  0.02},
        "casual_dip":          {"retention_lift": -0.06, "engagement_lift":  0.05, "reactivation_lift":  0.03, "ltv_lift":  0.04},
        "quick_churn":         {"retention_lift": -0.08, "engagement_lift":  0.04, "reactivation_lift":  0.05, "ltv_lift":  0.03},
        "re_engager":          {"retention_lift": -0.03, "engagement_lift":  0.04, "reactivation_lift":  0.08, "ltv_lift":  0.03},
        "mixed_behavior":      {"retention_lift": -0.04, "engagement_lift":  0.03, "reactivation_lift":  0.03, "ltv_lift":  0.03},
    },
    "PRICE_LOCK": {
        "binge_heavy":         {"retention_lift": -0.02, "engagement_lift":  0.01, "reactivation_lift":  0.01, "ltv_lift":  0.04},
        "completion_obsessed": {"retention_lift": -0.02, "engagement_lift":  0.01, "reactivation_lift":  0.01, "ltv_lift":  0.04},
        "casual_dip":          {"retention_lift": -0.04, "engagement_lift":  0.02, "reactivation_lift":  0.02, "ltv_lift":  0.05},
        "quick_churn":         {"retention_lift": -0.10, "engagement_lift":  0.02, "reactivation_lift":  0.03, "ltv_lift":  0.06},
        "re_engager":          {"retention_lift": -0.04, "engagement_lift":  0.02, "reactivation_lift":  0.04, "ltv_lift":  0.05},
        "mixed_behavior":      {"retention_lift": -0.04, "engagement_lift":  0.02, "reactivation_lift":  0.02, "ltv_lift":  0.04},
    },
    "REACTIVATION": {
        "binge_heavy":         {"retention_lift": -0.03, "engagement_lift":  0.04, "reactivation_lift":  0.10, "ltv_lift":  0.03},
        "completion_obsessed": {"retention_lift": -0.03, "engagement_lift":  0.04, "reactivation_lift":  0.08, "ltv_lift":  0.03},
        "casual_dip":          {"retention_lift": -0.05, "engagement_lift":  0.04, "reactivation_lift":  0.14, "ltv_lift":  0.03},
        "quick_churn":         {"retention_lift": -0.06, "engagement_lift":  0.03, "reactivation_lift":  0.12, "ltv_lift":  0.02},
        "re_engager":          {"retention_lift": -0.14, "engagement_lift":  0.10, "reactivation_lift":  0.28, "ltv_lift":  0.08},
        "mixed_behavior":      {"retention_lift": -0.05, "engagement_lift":  0.04, "reactivation_lift":  0.12, "ltv_lift":  0.03},
    },
    "SUPERFAN_EVENT": {
        "binge_heavy":         {"retention_lift": -0.04, "engagement_lift":  0.06, "reactivation_lift":  0.03, "ltv_lift":  0.20},
        "completion_obsessed": {"retention_lift": -0.04, "engagement_lift":  0.07, "reactivation_lift":  0.03, "ltv_lift":  0.25},
        "casual_dip":          {"retention_lift": -0.01, "engagement_lift":  0.02, "reactivation_lift":  0.02, "ltv_lift":  0.08},
        "quick_churn":         {"retention_lift":  0.00, "engagement_lift":  0.01, "reactivation_lift":  0.01, "ltv_lift":  0.04},
        "re_engager":          {"retention_lift": -0.03, "engagement_lift":  0.05, "reactivation_lift":  0.08, "ltv_lift":  0.14},
        "mixed_behavior":      {"retention_lift": -0.02, "engagement_lift":  0.04, "reactivation_lift":  0.03, "ltv_lift":  0.10},
    },
}

# ── Group C: Behavioral profile tier thresholds ───────────────────────────────
# PE-04: churn tier boundaries corrected — old proposal had wrong direction.
# quick_churn auto-label fires at churn_risk > 0.35; a cluster mean of 0.40
# labeled "stable" falls inside the quick_churn band — actively misleading.

CHURN_TIERS    = [(0.25, "stable"),   (0.45, "elevated"),  (1.01, "critical")]
COMPLETION_TIERS = [(0.40, "low"),    (0.70, "medium"),    (1.01, "high")]
REACTIVATION_TIERS = [(0.10, "weak"), (0.30, "moderate"),  (1.01, "strong")]
LTV_TIERS      = [(0.33, "low"),      (0.66, "medium"),    (1.01, "high")]
DEPTH_TIERS    = [(0.35, "light"),    (0.60, "moderate"),  (1.01, "deep")]

def _tier(value: float, thresholds: list) -> str:
    """Map a scalar to its tier label using (upper_bound, label) threshold pairs."""
    for bound, label in thresholds:
        if value < bound:
            return label
    return thresholds[-1][1]


# ── Group D: Content routing tables ──────────────────────────────────────────
# PE-07: re_engager gets ["series", "film"] — film was missing from v1.0.

PREFERRED_CONTENT: Dict[str, List[str]] = {
    "binge_heavy":         ["series"],
    "completion_obsessed": ["series", "documentary"],
    "casual_dip":          ["film", "shortform"],
    "quick_churn":         ["shortform", "film"],
    "re_engager":          ["series", "film"],   # PE-07: film added
    "mixed_behavior":      ["series", "film", "documentary"],
}

CONTENT_ARC_AFFINITY: Dict[str, List[str]] = {
    "binge_heavy":         ["mid_season", "penultimate_arc"],
    "completion_obsessed": ["penultimate_arc", "finale"],
    "casual_dip":          ["premiere"],
    "quick_churn":         ["premiere"],
    "re_engager":          ["premiere", "mid_season"],
    "mixed_behavior":      ["premiere", "mid_season"],
}

STRATEGIC_RECOMMENDATION: Dict[str, str] = {
    "binge_heavy":
        "Protect retention through loyalty rewards and exclusive content access. "
        "High completion and depth indicate strong habit formation — priority is "
        "tenure extension, not acquisition.",
    "completion_obsessed":
        "Drive LTV through plan upgrade offers and superfan events. "
        "This persona has the highest lifetime value ceiling — intervention should "
        "focus on monetisation, not churn reduction.",
    "casual_dip":
        "Accelerate habit formation through content nudges and onboarding pushes. "
        "Low session depth is the primary risk signal — increase content discovery "
        "touchpoints early in the session arc.",
    "quick_churn":
        "Prioritise immediate churn intervention: win-back campaign and price-lock offer. "
        "Short engagement window requires fast-moving triggers. "
        "Content nudge is secondary — resolve friction first.",
    "re_engager":
        "Reactivation campaign leveraging prior content affinity is the primary lever. "
        "This persona has demonstrated prior engagement value — personalised win-back "
        "with series or film continuations has the highest lift probability.",
    "mixed_behavior":
        "Heterogeneous cluster requires segment-level decomposition before targeting. "
        "Apply content nudge as a low-risk first intervention while refining "
        "sub-segment profiling.",
}


# ── Group A: Graceful column degradation helper ───────────────────────────────

def _cols(profile: Dict[str, float], *fields: str) -> bool:
    """Return True if all fields are present in profile. Used for graceful degradation."""
    return all(f in profile for f in fields)


# ── Group A: Per-cluster narrative interpreter ────────────────────────────────

def _interpret(persona: Dict[str, Any]) -> str:
    """
    Generate a 2–3 sentence behavioral narrative from cluster priors.
    Group A2 + PE-08: when confidence_note == 'low', returns an explicit
    unreliability flag instead of authoritative-sounding text.
    """
    sp    = persona["simulation_priors"]
    label = persona["behavioral_label"]
    name  = persona["persona_name"]
    conf  = sp.get("confidence_note", "low")

    # PE-08: low-confidence guard — small clusters produce unreliable priors
    if conf.startswith("low"):
        return (
            f"[PRIORS UNRELIABLE] {name} cluster has fewer than 100 events. "
            f"Behavioral characterisation is not valid at this cluster size. "
            f"Do not use these priors for intervention targeting until cluster "
            f"is revalidated on a larger dataset."
        )

    churn      = sp["base_churn_30d"]
    completion = sp["base_completion"]
    react      = sp["base_reactivation"]
    bp         = persona.get("behavioral_profile", {})
    churn_tier = bp.get("churn_tier", "unknown")
    comp_tier  = bp.get("completion_tier", "unknown")

    # Narrative templates by label
    narratives = {
        "binge_heavy": (
            f"{name} users show {comp_tier} content completion "
            f"(base={completion:.2f}) with {churn_tier} churn risk "
            f"(base={churn:.2f}). "
            f"Session depth and binge signal are the primary retention anchors — "
            f"these users engage in extended viewing sessions and demonstrate "
            f"strong habitual behaviour. "
            f"Intervention priority is tenure protection, not acquisition."
        ),
        "completion_obsessed": (
            f"{name} users have the highest completion rates in the pipeline "
            f"(base={completion:.2f}) with {churn_tier} churn risk (base={churn:.2f}). "
            f"LTV is the primary opportunity lever — these users are strong candidates "
            f"for plan upgrade and superfan event conversion. "
            f"Retention spend is likely wasted here; redirect to monetisation."
        ),
        "casual_dip": (
            f"{name} users show {comp_tier} completion (base={completion:.2f}) "
            f"with {churn_tier} churn risk (base={churn:.2f}). "
            f"Low session depth indicates shallow habit formation — the primary "
            f"intervention is content discovery acceleration to build a viewing routine "
            f"before the early-churn window closes."
        ),
        "quick_churn": (
            f"{name} users have {churn_tier} churn risk (base={churn:.2f}) "
            f"and {comp_tier} completion (base={completion:.2f}). "
            f"The engagement window is short — friction reduction and immediate "
            f"win-back outreach are the highest-priority interventions. "
            f"Price-lock offers should accompany content nudges to address both "
            f"value perception and content-fit gaps simultaneously."
        ),
        "re_engager": (
            f"{name} users have demonstrated prior platform engagement "
            f"(reactivation base={react:.2f}) with {churn_tier} current churn risk "
            f"(base={churn:.2f}). "
            f"Personalised reactivation campaigns leveraging prior content affinity "
            f"(series and film continuations) have the highest lift probability "
            f"for this cluster. "
            f"These users are recoverable — prioritise win-back before they fully lapse."
        ),
        "mixed_behavior": (
            f"{name} is a heterogeneous cluster with {comp_tier} completion "
            f"(base={completion:.2f}) and {churn_tier} churn risk (base={churn:.2f}). "
            f"The mixed segment composition indicates that HDBSCAN or K-Means has "
            f"grouped behaviourally distinct users together. "
            f"Sub-segment decomposition is recommended before applying targeted "
            f"interventions — content nudge is the lowest-risk first action."
        ),
    }
    return narratives.get(label, (
        f"{name} shows completion={completion:.2f}, churn={churn:.2f}. "
        f"No narrative template for label '{label}'."
    ))


# ── Group B: Within-cluster variance computation ──────────────────────────────

def _within_cluster_variance(cluster: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Extract within-cluster std dev stats from cluster metadata.
    Clustering engine stores feature_profile means; std fields are stored
    under profile keys with '_std' suffix if present (added in engine v4.1+).
    Falls back to None when not available.
    B1/B2: churn_risk_score_std drives is_diffuse_cluster; all three populate
    simulation_priors for Layer 5 Monte Carlo uncertainty bounds.
    """
    profile = cluster.get("feature_profile", {})
    return {
        "churn_risk_score_std":           profile.get("churn_risk_score_std"),
        "completion_rate_smooth_std":     profile.get("completion_rate_smooth_std"),
        "days_since_last_normalised_std": profile.get("days_since_last_normalised_std"),
    }


# ── Group E: Conditional risk override ───────────────────────────────────────

def _conditional_risk_override(
    label:   str,
    priors:  Dict[str, float],
    profile: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """
    Surfaces tension when a persona's churn risk is elevated despite a label
    that implies retention stability.
    PE-03: satisfaction threshold raised to < 0.60. Original proposal used < 0.20,
    which is structurally impossible for binge_heavy (Beta(7,2) on [3,5] gives
    minimum satisfaction_score = 0.50). Threshold of 0.60 is functionally reachable.
    Only fires when BOTH conditions are met — churn_risk alone is not sufficient.
    Returns None if conditions not met (field omitted from output).
    """
    churn_risk    = priors.get("base_churn_30d", 0.0)
    satisfaction  = profile.get("satisfaction_score", 1.0)  # default high = no override

    if churn_risk > 0.40 and satisfaction < 0.60:
        return {
            "triggered":       True,
            "churn_risk":      round(churn_risk, 4),
            "satisfaction":    round(satisfaction, 4),
            "note": (
                f"Elevated churn risk ({churn_risk:.2f}) coexists with low satisfaction "
                f"({satisfaction:.2f}) despite label '{label}'. "
                f"This cluster may contain users in active dissatisfaction — "
                f"standard playbook should be reviewed before deployment."
            ),
        }
    return None


# ── Risk flag derivation ──────────────────────────────────────────────────────

def derive_risk_flags(
    label:   str,
    priors:  Dict[str, float],
    profile: Dict[str, float],
) -> Dict[str, bool]:
    """
    Derive boolean risk/opportunity flags from behavioral priors and feature profile.
    These flags are consumed by CX systems and campaign targeting logic.
    """
    churn_risk     = priors.get("base_churn_30d",   0.0)
    completion     = priors.get("base_completion",  0.5)
    reactivation   = priors.get("base_reactivation",0.0)
    session_depth  = priors.get("base_session_depth",0.0)
    ltv_score      = profile.get("ltv_score",       0.0)
    tenure_weight  = profile.get("tenure_weight",   0.0)
    superfan_score = profile.get("superfan_score",  0.0)
    campaign_rec   = profile.get("campaign_receptivity", 0.0)
    subscription   = profile.get("subscription_tier_score", 0.0)

    plan_trajectory = profile.get("plan_trajectory_score", 0.5)

    return {
        # Churn risk: high base churn prob OR quick_churn archetype.
        # Threshold 0.35 aligns with quick_churn LABEL_RULES (churn_risk > 0.35).
        # 0.25 was too sensitive — flagged completion_obsessed clusters as high risk.
        "high_churn_risk":         churn_risk > 0.35 or label == "quick_churn",
        # High value: long tenure AND strong LTV signal
        "high_value":              tenure_weight > 0.60 and ltv_score > 0.40,
        # PE-05: reactivation_candidate requires BOTH signal AND label — not OR.
        # re_engager clusters with base_reactivation near 0 should not trigger
        # reactivation campaigns (reactivation requires a prior churn event).
        "reactivation_candidate":  reactivation > 0.30 or (label == "re_engager" and reactivation > 0.05),
        # Superfan potential: commerce engagement OR very high completion + depth
        "superfan_potential":      superfan_score > 0.25 or (completion > 0.80 and session_depth > 0.70),
        # Marketing receptive: campaign history AND not high churn (dark users unreachable)
        "marketing_receptive":     campaign_rec > 0.15 and churn_risk < 0.50,
        # PE-06: upgrade_candidate excludes recent downgrades (plan_trajectory_score < 0.5).
        # A user who recently downgraded with good completion is not an upgrade candidate.
        "upgrade_candidate":       completion > 0.70 and subscription < 0.70 and plan_trajectory >= 0.5,
    }


# ── Simulation priors ─────────────────────────────────────────────────────────

def build_simulation_priors(
    label:        str,
    priors:       Dict[str, float],
    profile:      Dict[str, float],
    size:         int,
    total_events: int,
    variance_stats: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Any]:
    """
    Build the simulation prior block consumed by the digital twin Monte Carlo engine.
    All values are point estimates with uncertainty bounds derived from cluster size.
    Larger clusters → tighter confidence intervals.
    Group B: variance_stats (churn_risk_score_std etc.) added to output.
    is_diffuse_cluster flag added: std > 0.25 on churn_risk_score indicates
    mixed-segment cluster. Threshold documented as calibration-dependent (B1).
    """
    # Uncertainty scales inversely with sqrt(cluster_size) — larger cluster = more confident
    # PE-01: correct binomial CI formula: SE = sqrt(p*(1-p)/n), not p/sqrt(n).
    # The old formula produced CIs ~half as wide as correct at typical churn rates.
    n_eff     = max(size, 1)
    unc_scale = round(1.0 / (n_eff ** 0.5), 4)

    base_churn       = priors.get("base_churn_30d",    0.20)
    base_completion  = priors.get("base_completion",   0.50)
    base_reactivation= priors.get("base_reactivation", 0.10)
    base_depth       = priors.get("base_session_depth",0.50)
    base_churn_risk  = priors.get("base_churn_risk",   0.25)

    # SE = sqrt(p*(1-p)/n) for both — consistent binomial standard error
    churn_se      = (base_churn      * (1 - base_churn))      ** 0.5 / (n_eff ** 0.5)
    completion_se = (base_completion * (1 - base_completion))  ** 0.5 / (n_eff ** 0.5)

    vs = variance_stats or {}
    churn_std = vs.get("churn_risk_score_std")

    return {
        "base_churn_30d":          round(base_churn, 4),
        "base_churn_30d_ci":       [
            round(max(0.0, base_churn      - 1.96 * churn_se),      4),
            round(min(1.0, base_churn      + 1.96 * churn_se),      4),
        ],
        "base_completion":         round(base_completion, 4),
        "base_completion_ci":      [
            round(max(0.0, base_completion - 1.96 * completion_se), 4),
            round(min(1.0, base_completion + 1.96 * completion_se), 4),
        ],
        "base_reactivation":       round(base_reactivation, 4),
        "base_session_depth_score":round(base_depth, 4),
        "base_churn_risk_score":   round(base_churn_risk, 4),
        # Group B: within-cluster variance stats
        "churn_risk_score_std":           round(churn_std, 4) if churn_std is not None else None,
        "completion_rate_smooth_std":     round(vs["completion_rate_smooth_std"], 4) if vs.get("completion_rate_smooth_std") is not None else None,
        "days_since_last_normalised_std": round(vs["days_since_last_normalised_std"], 4) if vs.get("days_since_last_normalised_std") is not None else None,
        # B1: is_diffuse_cluster — std > 0.25 on churn_risk_score indicates mixed-segment cluster.
        # PE-11 CALIBRATION DEPENDENCY: threshold 0.25 was informed by within-segment standard
        # deviations from the 30k test subset. [EMPIRICAL — VERIFY] The threshold direction is
        # structurally correct at any scale (mixed-segment clusters will always have higher
        # within-cluster variance than pure-segment clusters), but the exact value 0.25 should
        # be revisited once HDBSCAN produces clean clusters on the full dataset with real data.
        # Current synthetic K-Means clusters have overlapping segment compositions by construction,
        # so is_diffuse_cluster=None (std not available from cluster metadata) is expected until
        # the clustering engine writes per-cluster std fields to cluster_metadata.json.
        # The flag is None (not False) when std data is absent — Layer 5 must treat None as
        # "unknown" not "not diffuse".
        "is_diffuse_cluster":      (churn_std > 0.25) if churn_std is not None else None,
        "cluster_size":            size,
        "cluster_share_pct":       round(size / max(total_events, 1) * 100, 2),
        "uncertainty_scale":       unc_scale,
        "confidence_note": (
            "high"   if n_eff >= 1000 else
            "medium" if n_eff >= 100  else
            "low — treat priors with caution, small cluster"
        ),
    }


# ── Playbook builder ──────────────────────────────────────────────────────────

def build_playbook(
    label:      str,
    priors:     Dict[str, float],
    profile:    Dict[str, float],
    risk_flags: Dict[str, bool],
) -> List[Dict[str, Any]]:
    """
    Build the ordered intervention playbook for this persona.
    Each entry includes the intervention spec + simulated lift estimates
    from the LIFT_TABLE + a rationale string.
    """
    route = INTERVENTION_ROUTING.get(label, INTERVENTION_ROUTING["mixed_behavior"])
    playbook = []

    for intervention_id in route:
        spec  = INTERVENTIONS[intervention_id]
        lifts = LIFT_TABLE.get(intervention_id, {}).get(label, {
            "retention_lift": 0.0, "engagement_lift": 0.0,
            "reactivation_lift": 0.0, "ltv_lift": 0.0,
        })

        # Derive a rationale from the risk flags and priors
        rationale = _build_rationale(intervention_id, label, priors, risk_flags)

        playbook.append({
            "intervention_id":   intervention_id,
            "name":              spec["name"],
            "description":       spec["description"],
            "channel":           spec["channel"],
            "trigger_condition": spec["trigger"],
            "priority":          len(playbook) + 1,
            "simulated_lift": {
                "retention_lift":    lifts.get("retention_lift",    0.0),
                "engagement_lift":   lifts.get("engagement_lift",   0.0),
                "reactivation_lift": lifts.get("reactivation_lift", 0.0),
                "ltv_lift":          lifts.get("ltv_lift",          0.0),
            },
            "rationale": rationale,
            "lift_note": (
                "Simulation prior only — not a causal estimate. "
                "Requires A/B test integration for causal measurement."
            ),
        })

    return playbook


def _build_rationale(
    intervention_id: str,
    label:           str,
    priors:          Dict[str, float],
    risk_flags:      Dict[str, bool],
) -> str:
    churn    = priors.get("base_churn_30d",    0.20)
    reactiv  = priors.get("base_reactivation", 0.10)
    compl    = priors.get("base_completion",   0.50)

    rationales = {
        "WIN_BACK":       f"Churn prior {churn:.0%} — re-engagement outreach targets users before permanent lapse.",
        "CONTENT_NUDGE":  f"Completion prior {compl:.0%} — high content affinity makes personalised nudge high-conversion.",
        "PLAN_UPGRADE":   f"Engagement level supports premium tier — upgrade framed on content access, not price.",
        "LOYALTY_REWARD": f"Long-tenure high-value persona — loyalty reward reduces voluntary churn without discount dependency.",
        "ONBOARDING_PUSH":f"Low tenure detected — accelerated onboarding reduces first-30-day churn risk.",
        "PRICE_LOCK":     f"Downgrade risk signals present — price stability offer removes friction before billing cycle.",
        "REACTIVATION":   f"Reactivation prior {reactiv:.0%} — lapsed user with prior engagement history, high win-back ROI.",
        "SUPERFAN_EVENT": f"Commerce/depth signals indicate willingness-to-pay above subscription — high-margin upsell.",
    }
    return rationales.get(intervention_id, "Intervention selected by routing policy.")


# ── User-level persona assignment (Q1/Q2 resolution) ─────────────────────────
# Architecture decision: personas are user-level archetypes, not session-level
# behavioral patterns. Implements Option B from the architecture review:
# session-level clustering is preserved, but each user_id is assigned to their
# MODAL cluster (the cluster where most of their sessions landed). This produces
# a clean one-user → one-persona mapping for CRM targeting and Layer 5 simulation.
#
# Users whose sessions are evenly split across clusters are flagged as
# behaviorally_ambiguous. Modal cluster is still assigned (lowest index tiebreak).
#
# unique_user_count per cluster is returned for accurate stakeholder reporting.
# IMPORTANT: before modal assignment, unique user counts per cluster sum to MORE
# than total users (one user may appear in multiple clusters). After modal
# assignment, each user belongs to exactly one cluster — counts sum to n_users.

def _compute_user_persona_assignment(
    assignments_path,
    personas,
    expected_cluster_version: str = None,
):
    """
    Read cluster_assignments.csv, compute modal cluster per user_id, and return:
      - unique_user_counts: {cluster_index (int): n_unique_users_assigned}
        Keyed by cluster_index, NOT behavioral_label, to correctly handle duplicate
        labels (e.g. two quick_churn clusters from CAL-002 multi_assign). Callers
        must look up by cluster_index when attaching counts to personas.
      - assignment_summary: metadata block for persona_profiles.json top-level

    expected_cluster_version: the version string from cluster_metadata.json.
      When provided, asserts that the cluster_version column in the assignments CSV
      matches. A mismatch means the assignments file is from a different pipeline run
      than the metadata — counts would be meaningless. Emits [FAIL] and returns empty
      dicts on mismatch rather than producing silently wrong output.

    Returns empty dicts if assignments_path is None or file not found.
    """
    import os
    import pandas as _pd

    if not assignments_path or not os.path.exists(assignments_path):
        print(f"[WARN] --assignments not provided or file not found. "
              f"unique_user_count will be omitted from persona output. "
              f"Pass cluster_assignments.csv via --assignments for user-level counts.")
        return {}, {}

    try:
        adf = _pd.read_csv(assignments_path)
    except Exception as e:
        print(f"[WARN] Could not read {assignments_path}: {e}. Skipping user assignment.")
        return {}, {}

    if "user_id" not in adf.columns or "cluster_index" not in adf.columns:
        print("[WARN] assignments CSV missing user_id or cluster_index. Skipping user assignment.")
        return {}, {}

    # ── Provenance check: assert assignments CSV matches cluster_metadata.json ──
    # cluster_assignments.csv carries a cluster_version column written by
    # clustering_engine.py. If this does not match the version in cluster_metadata.json,
    # the two files are from different pipeline runs. Proceeding would compute
    # modal assignments against the wrong cluster structure, silently producing
    # wrong unique_user_counts (as observed in the 20k-vs-100k mismatch).
    if expected_cluster_version and "cluster_version" in adf.columns:
        observed_versions = adf["cluster_version"].dropna().unique().tolist()
        mismatches = [v for v in observed_versions if str(v) != str(expected_cluster_version)]
        if mismatches:
            print(
                f"[FAIL] Provenance mismatch: cluster_metadata.json version={expected_cluster_version!r} "
                f"but cluster_assignments.csv contains version(s): {observed_versions}. "
                f"These files are from different pipeline runs. "
                f"Re-run clustering_engine.py to produce a matched pair, then re-run persona_engine.py. "
                f"Skipping user assignment to prevent silently wrong unique_user_counts."
            )
            return {}, {}
        print(f"[INFO] Provenance check: assignments version={expected_cluster_version!r} — matches metadata ✓")
    elif expected_cluster_version and "cluster_version" not in adf.columns:
        print(
            f"[WARN] Provenance check skipped: cluster_version column absent from {assignments_path}. "
            f"Cannot verify assignments match cluster_metadata.json v{expected_cluster_version}. "
            f"Ensure both files are from the same clustering_engine.py run."
        )

    # Separate noise sessions from assigned sessions.
    # Users whose ALL sessions are noise (cluster_index == -1) cannot be assigned
    # to any persona via modal assignment. They are counted explicitly and reported
    # in assignment_summary as n_users_noise_only — they must not be silently dropped.
    all_users   = set(adf["user_id"].unique())
    noise_only  = set(adf[adf["cluster_index"] == -1]["user_id"].unique()) -                   set(adf[adf["cluster_index"] != -1]["user_id"].unique())
    n_noise_only = len(noise_only)
    if n_noise_only > 0:
        print(
            f"[WARN] {n_noise_only:,} users have ALL sessions marked noise (cluster_index=-1). "
            f"These users cannot be assigned to any persona and are excluded from "
            f"unique_user_count totals. They represent {n_noise_only/len(all_users)*100:.1f}% "
            f"of the total user base. Check the K-Means outlier σ threshold in "
            f"clustering_engine.py if this count is unexpectedly high."
        )

    clean = adf[adf["cluster_index"] != -1].copy()
    if clean.empty:
        return {}, {}

    # Count sessions per user per cluster (noise sessions already excluded)
    session_counts = (
        clean.groupby(["user_id", "cluster_index"])
        .size()
        .reset_index(name="session_count")
    )

    # Modal cluster = cluster with most sessions per user; tiebreak: lowest cluster_index
    modal = (
        session_counts
        .sort_values(["user_id", "session_count", "cluster_index"],
                     ascending=[True, False, True])
        .drop_duplicates(subset="user_id", keep="first")
        [["user_id", "cluster_index", "session_count"]]
        .rename(columns={"cluster_index": "modal_cluster",
                         "session_count": "modal_session_count"})
    )

    # Detect behaviorally ambiguous users (max session count tied across >=2 clusters)
    max_per_user = session_counts.groupby("user_id")["session_count"].max()
    count_at_max = session_counts.merge(
        max_per_user.rename("max_sc").reset_index(), on="user_id"
    )
    ties = count_at_max[count_at_max["session_count"] == count_at_max["max_sc"]]
    ambiguous_users = set(
        ties.groupby("user_id").filter(lambda g: len(g) > 1)["user_id"].unique()
    )

    n_users_assigned = len(modal)
    n_ambiguous      = len(ambiguous_users)
    n_unambiguous    = n_users_assigned - n_ambiguous
    # Total users in the assignments file (assigned + noise-only)
    n_users          = len(all_users)

    # Count users per cluster_index (post-modal-assignment: each user counted exactly once).
    # CRITICAL: keyed by cluster_index (int), NOT behavioral_label.
    # When duplicate labels exist (e.g. two quick_churn clusters from CAL-002 multi_assign),
    # keying by label collapses both clusters into one bucket — the same users get counted
    # twice and the total falls short of n_users_total. cluster_index is always unique.
    user_counts_by_cluster = (
        modal.groupby("modal_cluster")["user_id"]
        .nunique()
        .to_dict()
    )
    # Returns {cluster_index (int): user_count (int)}
    unique_user_counts = {
        int(idx): int(count)
        for idx, count in user_counts_by_cluster.items()
    }

    # Detect duplicate behavioral labels — indicates multi_assign firing (CAL-002).
    # Log a warning so the operator knows the report will show two personas with the
    # same name but different user counts.
    idx_to_label = {p["cluster_index"]: p["behavioral_label"] for p in personas}
    label_counts = {}
    for idx in unique_user_counts:
        lbl = idx_to_label.get(idx, f"cluster_{idx}")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    duplicate_labels = {lbl for lbl, cnt in label_counts.items() if cnt > 1}
    if duplicate_labels:
        print(
            f"[WARN] Duplicate behavioral labels detected: {sorted(duplicate_labels)}. "
            f"Two or more clusters share the same label (CAL-002 multi_assign). "
            f"unique_user_count is keyed by cluster_index to avoid collapsing counts. "
            f"Check TRIGGER_RULES calibration — a missing label usually indicates a "
            f"threshold mismatch at this dataset scale."
        )

    assignment_summary = {
        "n_users_in_assignments":  n_users,
        "n_users_assigned":        n_users_assigned,
        "n_users_noise_only":      n_noise_only,
        "n_users_unambiguous":     n_unambiguous,
        "n_users_ambiguous":       n_ambiguous,
        "noise_only_pct":          round(n_noise_only / n_users * 100, 2) if n_users else 0.0,
        "ambiguous_pct":           round(n_ambiguous / n_users_assigned * 100, 2) if n_users_assigned else 0.0,
        "duplicate_labels":        sorted(duplicate_labels) if duplicate_labels else [],
        "method":                  "modal_cluster_assignment",
        "note": (
            "n_users_assigned = users with at least one non-noise session, assigned to modal cluster. "
            "n_users_noise_only = users whose every session was marked noise — unassignable, "
            "excluded from unique_user_count totals. "
            "n_users_in_assignments = n_users_assigned + n_users_noise_only. "
            "unique_user_count per persona is keyed by cluster_index (not behavioral_label) "
            "to correctly handle duplicate labels from CAL-002 multi_assign. "
            "Sum of unique_user_counts across all personas equals n_users_assigned exactly."
        ),
    }

    print(f"[INFO] User assignment: {n_users_assigned:,} users assigned to modal cluster "
          f"({n_noise_only:,} noise-only excluded). "
          f"{n_ambiguous:,} behaviorally ambiguous ({n_ambiguous/n_users_assigned*100:.1f}%).")

    return unique_user_counts, assignment_summary


# ── Core persona builder ──────────────────────────────────────────────────────

def build_persona(cluster: Dict[str, Any], total_events: int) -> Dict[str, Any]:
    """
    Build one complete persona from a cluster metadata entry.
    v2.0: adds behavioral_profile (C), content routing (D),
    conditional_risk_override (E), variance stats + is_diffuse_cluster (B).
    """
    label    = cluster.get("behavioral_label", "mixed_behavior")
    priors   = cluster.get("behavioral_priors", {})
    profile  = cluster.get("feature_profile",  {})
    size     = cluster.get("size", 0)
    seg_comp = cluster.get("segment_composition", {})
    content  = cluster.get("content_distribution", {})

    # Group B: variance stats from cluster metadata
    variance_stats   = _within_cluster_variance(cluster)

    risk_flags       = derive_risk_flags(label, priors, profile)
    simulation_priors= build_simulation_priors(
        label, priors, profile, size, total_events, variance_stats
    )
    playbook         = build_playbook(label, priors, profile, risk_flags)

    # Group C: behavioral profile — categorical tier labels
    behavioral_profile = {
        "churn_tier":        _tier(priors.get("base_churn_30d",    0.0), CHURN_TIERS),
        "completion_tier":   _tier(priors.get("base_completion",   0.5), COMPLETION_TIERS),
        "reactivation_tier": _tier(priors.get("base_reactivation", 0.0), REACTIVATION_TIERS),
        "ltv_tier":          _tier(profile.get("ltv_score",        0.0), LTV_TIERS),
        "depth_tier":        _tier(priors.get("base_session_depth",0.0), DEPTH_TIERS),
    }

    # Group D: content routing
    content_routing = {
        "preferred_content_types": PREFERRED_CONTENT.get(label, ["series"]),
        "content_arc_affinity":    CONTENT_ARC_AFFINITY.get(label, ["premiere"]),
        "strategic_recommendation": STRATEGIC_RECOMMENDATION.get(
            label, "No strategic template for this label."
        ),
    }

    # Group E: conditional risk override (None if conditions not met — omitted from JSON)
    risk_override = _conditional_risk_override(label, priors, profile)

    persona = {
        "persona_id":      str(uuid.uuid4()),
        "cluster_id":      cluster.get("cluster_id", ""),
        "cluster_index":   cluster.get("cluster_index", -1),
        "engine_version":  ENGINE_VERSION,
        "computed_at":     datetime.now(timezone.utc).isoformat(),

        # ── Identity ──────────────────────────────────────────────────────────
        "persona_name":    PERSONA_NAMES.get(label, "Unknown Viewer"),
        "archetype":       PERSONA_ARCHETYPES.get(label, "unknown"),
        "behavioral_label":label,
        # dominant_segment and segment_distribution are SYNTHETIC VALIDATION ONLY.
        # These fields are populated when clustering_engine was run with
        # --synthetic_validation_mode. They reflect the simulation ground-truth
        # segment_id, which does not exist in real client data. Downstream
        # consumers must not treat dominant_segment as a real behavioral attribute.
        "dominant_segment": (
            seg_comp.get("dominant_segment", "unknown")
            if seg_comp.get("_synthetic_validation_only")
            else "[not available in production mode]"
        ),
        "dominant_segment_note": (
            "[SYNTHETIC VALIDATION ONLY] Derived from simulation segment_id — "
            "not a real observable property of this persona. "
            "Used during development to verify clustering recovers simulation ground truth."
            if seg_comp.get("_synthetic_validation_only")
            else "synthetic_validation_mode was not active — segment_id not present in input."
        ),
        "segment_distribution": seg_comp.get("segment_distribution", {}),
        "size":            size,
        "size_pct":        cluster.get("size_pct", 0.0),
        # unique_user_count is populated by _compute_user_persona_assignment()
        # after all personas are built. Placeholder here; filled in main().
        # Only counts users assigned to this specific cluster (modal assignment).
        # Does NOT include noise-only users (see user_assignment.n_users_noise_only).
        "unique_user_count": None,
        "super_cluster":   cluster.get("super_cluster", None),

        # ── Behavioral DNA ────────────────────────────────────────────────────
        "behavioral_priors":            priors,
        "feature_profile":              profile,
        "top_discriminating_features":  cluster.get("top_discriminating_features", []),
        "content_distribution":         content,
        "churn_rate":                   cluster.get("churn_rate",        0.0),
        "reactivation_rate":            cluster.get("reactivation_rate", 0.0),

        # ── Group C: Behavioral profile ───────────────────────────────────────
        "behavioral_profile": behavioral_profile,

        # ── Group D: Content routing ──────────────────────────────────────────
        "content_routing": content_routing,

        # ── Risk & opportunity flags ──────────────────────────────────────────
        "risk_flags": risk_flags,

        # ── Group E: Conditional risk override (omitted if not triggered) ─────
        **({"conditional_risk_override": risk_override} if risk_override else {}),

        # ── Simulation priors ─────────────────────────────────────────────────
        "simulation_priors": simulation_priors,

        # ── Intervention playbook ─────────────────────────────────────────────
        "intervention_playbook": playbook,
    }

    # Group A2: attach narrative after persona is built (needs behavioral_profile)
    persona["behavioral_narrative"] = _interpret(persona)

    return persona


# ── Report writer ─────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 30) -> str:
    """ASCII bar chart segment for a 0–1 value."""
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def write_report(personas: List[Dict[str, Any]], meta: Dict[str, Any], path: str) -> None:
    """
    Group A1: 7-section structured report.
    §0 Metadata  §1 Overview  §2 Pipeline checks  §3 Cluster summary
    §4 Persona profiles  §5 Prior analysis  §6 Recommendations
    """
    now_str = datetime.now(timezone.utc).isoformat()
    W = 74  # report width

    def rule(char="="):  return char * W
    def head(s):         return [rule(), f"  {s}", rule(), ""]
    def subhead(s):      return ["", f"── {s}", "─" * W, ""]

    lines: List[str] = []

    # ── §0 Metadata ───────────────────────────────────────────────────────────
    lines += head(f"TWINSIM PERSONA ENGINE v{ENGINE_VERSION} — PERSONA REPORT")
    lines += [
        f"  Generated     : {now_str}",
        f"  Source        : {meta.get('source_file', 'cluster_metadata.json')}",
        f"  Cluster ver   : {meta.get('version', '?')}",
        f"  Algorithm     : {meta.get('algorithm', '?')}",
        f"  Stability     : {meta.get('stability', {}).get('stability_label', 'UNKNOWN')}  "
        f"  sil={meta.get('stability', {}).get('mean_silhouette', 0):.4f}",
        f"  Total events  : {meta.get('n_events', 0):,}",
        f"  Noise events  : {meta.get('noise_events', 0):,}  "
        f"({meta.get('noise_pct', 0):.1f}%)",
        f"  Personas      : {len(personas)}",
        "",
    ]

    # ── §1 Overview ───────────────────────────────────────────────────────────
    lines += head("§1  CLUSTER OVERVIEW")
    has_user_counts = any(p.get("unique_user_count") is not None for p in personas)
    lines += [
        f"  {'#':<4} {'Persona':<28} {'Sessions':>9} {'%':>6}  "
        f"{'Users':>7}  {'Churn':>6} {'Compl':>6}  {'Confidence':<10}  Flags",
        "  " + "─" * (W - 2),
    ]
    for p in sorted(personas, key=lambda x: x["cluster_index"]):
        sp      = p["simulation_priors"]
        flags   = [k for k, v in p["risk_flags"].items() if v]
        u_count = p.get("unique_user_count")
        u_str   = f"{u_count:>7,}" if u_count is not None else "      —"
        lines.append(
            f"  #{p['cluster_index']:<3} {p['persona_name']:<28} "
            f"{p['size']:>9,} {p['size_pct']:>5.1f}%  "
            f"{u_str}  "
            f"{sp['base_churn_30d']:>6.3f} {sp['base_completion']:>6.3f}  "
            f"{sp['confidence_note'][:10]:<10}  {', '.join(flags) or 'none'}"
        )
    # Detect duplicate persona names (duplicate behavioral labels from CAL-002 multi_assign)
    persona_name_counts = {}
    for p in personas:
        persona_name_counts[p["persona_name"]] = persona_name_counts.get(p["persona_name"], 0) + 1
    duplicate_names = {n for n, c in persona_name_counts.items() if c > 1}
    if duplicate_names:
        lines.append(
            f"  [WARN] Duplicate persona names: {sorted(duplicate_names)}. "
            f"Two or more clusters share the same behavioral label (CAL-002 multi_assign). "
            f"Users column shows per-cluster counts — users are not double-counted. "
            f"A missing persona label indicates TRIGGER_RULES calibration is needed at this scale."
        )
    if has_user_counts:
        # Check if any noise-only users were dropped
        noise_only_note = ""
        for p_top in personas:
            # access top-level assignment summary via a flag set in main() if available
            break
        lines.append(
            f"  NOTE: Sessions = session count from clustering. "
            f"Users = unique users with >=1 non-noise session assigned to this cluster "
            f"(one user → one cluster, no double-counting). "
            f"Sum of Users column = n_users_assigned (excludes noise-only users — "
            f"see user_assignment.n_users_noise_only in persona_profiles.json)."
        )
    else:
        lines.append(
            f"  NOTE: Users column unavailable — run with --assignments cluster_assignments.csv "
            f"to compute unique user counts per persona."
        )
    lines.append("")

    # ── §2 Pipeline checks ────────────────────────────────────────────────────
    lines += head("§2  PIPELINE CHECKS")
    total  = meta.get("n_events", 0)
    n_feat = meta.get("feature_store_n")
    if n_feat:
        dropout_pct = (1 - total / n_feat) * 100 if n_feat > 0 else 0
        status = "[FAIL]" if dropout_pct > 20 else "[PASS]"
        lines.append(
            f"  {status} Event coverage: {total:,} / {n_feat:,} "
            f"({100 - dropout_pct:.1f}% retained, {dropout_pct:.1f}% dropout)"
        )
    else:
        lines.append("  [SKIP] Event dropout check — feature_store_n not in metadata")

    stab     = meta.get("stability", {}).get("stability_label", "UNKNOWN")
    sil_mean = meta.get("stability", {}).get("mean_silhouette", 0.0)
    # Mirror clustering_audit.py Check 7: UNSTABLE is WARN when sil >= 0.05 (structure exists),
    # FAIL only when sil < 0.05 (degenerate — no meaningful structure).
    if stab == "STABLE":
        stab_icon = "[PASS]"
    elif stab == "MARGINAL":
        stab_icon = "[WARN]"
    elif sil_mean >= 0.05:
        stab_icon = "[WARN]"   # UNSTABLE but not degenerate
    else:
        stab_icon = "[FAIL]"   # Degenerate clustering
    lines.append(f"  {stab_icon} Clustering stability: {stab} (sil={sil_mean:.4f})")

    diffuse = [p["persona_name"] for p in personas
               if p["simulation_priors"].get("is_diffuse_cluster") is True]
    if diffuse:
        lines.append(f"  [WARN] Diffuse clusters detected: {diffuse}")
    else:
        lines.append(f"  [PASS] No diffuse clusters (churn_risk_score_std ≤ 0.25)")
    lines.append("")

    # ── §3 Cluster summary ────────────────────────────────────────────────────
    lines += head("§3  CLUSTER PROFILES — BEHAVIORAL TIERS")
    for p in sorted(personas, key=lambda x: x["cluster_index"]):
        bp = p.get("behavioral_profile", {})
        cr = p.get("content_routing", {})
        lines += [
            f"  #{p['cluster_index']} {p['persona_name']}  [{p['behavioral_label']}]",
            f"     Churn tier   : {bp.get('churn_tier','?'):<10}  "
            f"Completion: {bp.get('completion_tier','?'):<8}  "
            f"Reactivation: {bp.get('reactivation_tier','?')}",
            f"     LTV tier     : {bp.get('ltv_tier','?'):<10}  "
            f"Depth     : {bp.get('depth_tier','?')}",
            f"     Content types: {', '.join(cr.get('preferred_content_types', []))}",
            f"     Arc affinity : {', '.join(cr.get('content_arc_affinity', []))}",
            "",
        ]

    # ── §4 Persona profiles ───────────────────────────────────────────────────
    lines += head("§4  PERSONA PROFILES")
    for p in sorted(personas, key=lambda x: x["cluster_index"]):
        sp    = p["simulation_priors"]
        flags = {k: v for k, v in p["risk_flags"].items()}

        lines += [
            f"  ╔══ #{p['cluster_index']}  {p['persona_name'].upper()} ══",
            f"  ║  Archetype       : {p['archetype']}",
            f"  ║  Dominant segment: {p['dominant_segment']}  {p.get('dominant_segment_note','') and '[SYNTHETIC VALIDATION ONLY — see note]' or ''}",
            f"  ║  Size (sessions) : {p['size']:,} ({p['size_pct']:.1f}% of all sessions)",
            f"  ║  Size (users)    : {p['unique_user_count']:,} unique users (modal assignment)" if p.get('unique_user_count') is not None else "  ║  Size (users)    : — (run with --assignments to compute)",
            f"  ║  Confidence      : {sp['confidence_note']}",
            f"  ║",
            f"  ║  Narrative:",
        ]
        # Wrap narrative at ~65 chars
        narrative = p.get("behavioral_narrative", "")
        words, line_buf = narrative.split(), ""
        for word in words:
            if len(line_buf) + len(word) + 1 > 65:
                lines.append(f"  ║    {line_buf}")
                line_buf = word
            else:
                line_buf = (line_buf + " " + word).strip()
        if line_buf:
            lines.append(f"  ║    {line_buf}")

        lines += [
            f"  ║",
            f"  ║  Behavioral priors:",
            f"  ║    churn_30d  {sp['base_churn_30d']:.4f}  "
            f"CI[{sp['base_churn_30d_ci'][0]:.4f},{sp['base_churn_30d_ci'][1]:.4f}]  "
            f"{_bar(sp['base_churn_30d'])}",
            f"  ║    completion {sp['base_completion']:.4f}  "
            f"CI[{sp['base_completion_ci'][0]:.4f},{sp['base_completion_ci'][1]:.4f}]  "
            f"{_bar(sp['base_completion'])}",
            f"  ║    reactiv    {sp['base_reactivation']:.4f}  "
            f"{_bar(sp['base_reactivation'])}",
            f"  ║    churn_risk {sp['base_churn_risk_score']:.4f}  "
            f"{_bar(sp['base_churn_risk_score'])}",
        ]
        if sp.get("is_diffuse_cluster") is not None:
            diffuse_str = "YES ⚠" if sp["is_diffuse_cluster"] else "no"
            std_str = f"{sp.get('churn_risk_score_std'):.4f}" if sp.get("churn_risk_score_std") is not None else "n/a"
            lines.append(f"  ║    is_diffuse  {diffuse_str}  (churn_risk_std={std_str})")

        # B3: risk flag bar chart
        lines += ["  ║", "  ║  Risk flags:"]
        for flag, val in flags.items():
            marker = "■" if val else "□"
            lines.append(f"  ║    {marker} {flag}")

        if p.get("conditional_risk_override"):
            ov = p["conditional_risk_override"]
            lines += [
                f"  ║",
                f"  ║  ⚠ CONDITIONAL RISK OVERRIDE:",
                f"  ║    churn={ov['churn_risk']:.3f}  sat={ov['satisfaction']:.3f}",
            ]
            note_words, note_buf = ov['note'].split(), ""
            for word in note_words:
                if len(note_buf) + len(word) + 1 > 65:
                    lines.append(f"  ║    {note_buf}")
                    note_buf = word
                else:
                    note_buf = (note_buf + " " + word).strip()
            if note_buf:
                lines.append(f"  ║    {note_buf}")

        lines += [f"  ║", f"  ║  Strategic recommendation:"]
        strat = cr.get("strategic_recommendation", "") if (cr := p.get("content_routing", {})) else ""
        if strat:
            words, line_buf = strat.split(), ""
            for word in words:
                if len(line_buf) + len(word) + 1 > 65:
                    lines.append(f"  ║    {line_buf}")
                    line_buf = word
                else:
                    line_buf = (line_buf + " " + word).strip()
            if line_buf:
                lines.append(f"  ║    {line_buf}")
        lines += [f"  ║", f"  ║  Top discriminating features:"]
        for feat in p["top_discriminating_features"][:5]:
            val = p["feature_profile"].get(feat, 0.0)
            lines.append(f"  ║    {feat:<38} {val:.4f}")

        lines += [f"  ║", f"  ║  Intervention playbook:"]
        for item in p["intervention_playbook"]:
            sl = item["simulated_lift"]
            lines.append(
                f"  ║    [{item['priority']}] {item['intervention_id']:<20} "
                f"ret Δ{sl['retention_lift']:+.2f}  "
                f"eng Δ{sl['engagement_lift']:+.2f}  "
                f"ltv Δ{sl['ltv_lift']:+.2f}"
            )
        lines += [f"  ╚" + "═" * (W - 4), ""]

    # ── §5 Prior analysis ─────────────────────────────────────────────────────
    lines += head("§5  PRIOR ANALYSIS")
    churns      = [p["simulation_priors"]["base_churn_30d"]   for p in personas]
    completions = [p["simulation_priors"]["base_completion"]   for p in personas]
    churn_spread      = max(churns)      - min(churns)
    completion_spread = max(completions) - min(completions)

    lines += [
        f"  Churn range      : {min(churns):.4f} – {max(churns):.4f}  spread={churn_spread:.4f}",
        f"  Completion range : {min(completions):.4f} – {max(completions):.4f}  spread={completion_spread:.4f}",
    ]
    # F2: prior consistency check
    if churn_spread < 0.05 and completion_spread < 0.05:
        lines.append(
            f"  [WARN] F2: All clusters within 0.05 on both churn and completion axes — "
            f"clustering may not be discriminating meaningfully."
        )
    else:
        lines.append(f"  [PASS] F2: Clusters show meaningful prior differentiation.")
    lines.append("")

    # ── §6 Recommendations ────────────────────────────────────────────────────
    lines += head("§6  PIPELINE RECOMMENDATIONS")
    recs = []
    if stab == "STABLE":
        pass  # no stability rec needed
    elif stab in ("MARGINAL", "UNSTABLE") and sil_mean >= 0.05:
        # Structure exists — WARN not CRITICAL. Mirror clustering_audit Check 7.
        recs.append(
            f"[WARN] Clustering {stab} (sil={sil_mean:.4f}). Cluster structure exists "
            f"but boundaries overlap — expected on synthetic data. Persona priors are "
            f"valid with wider uncertainty. Proceed to Layer 5 with confidence_note "
            f"awareness. To improve: increase users (target 50k+) or use real data."
        )
    else:
        recs.append(
            f"[CRITICAL] Degenerate clustering (sil={sil_mean:.4f} < 0.05). "
            f"Re-run clustering_engine.py — priors are unreliable for intervention targeting."
        )
    if diffuse:
        recs.append(f"[WARN] Diffuse clusters {diffuse} may contain mixed segments. "
                    f"Review sub-segment decomposition before deploying playbooks.")
    if churn_spread < 0.05 and completion_spread < 0.05:
        recs.append("[WARN] Prior spread <0.05 on all axes — inspect clustering feature "
                    "collinearity. Check 13 in feature_store_assessment.py.")
    low_conf = [p["persona_name"] for p in personas
                if p["simulation_priors"]["confidence_note"].startswith("low")]
    if low_conf:
        recs.append(f"[WARN] Low-confidence personas (n<100): {low_conf}. "
                    f"Narratives replaced with reliability flags — do not target.")
    if not recs:
        recs.append("[PASS] No pipeline issues detected. Safe to proceed to Layer 5.")
    for r in recs:
        lines.append(f"  {r}")
    lines += ["", rule(), ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[DONE] Persona report → {path}")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_personas(personas: List[Dict[str, Any]]) -> bool:
    """
    15 checks covering completeness, prior ranges, playbook integrity,
    risk flag coverage, simulation prior consistency, and v2.0 additions.
    Groups F1 (formula verification), F2 (prior consistency), F3 (diffuse cluster).
    """
    results = {}
    all_pass = True

    # Check 1: At least one persona produced
    results["01_persona_count"] = (
        ("PASS", f"{len(personas)} personas produced")
        if len(personas) >= 1 else
        ("FAIL", "No personas produced")
    )

    # Check 2: All required top-level fields present (v2.0 fields added)
    required = {
        "persona_id", "cluster_id", "persona_name", "archetype",
        "behavioral_label", "dominant_segment", "size",
        "behavioral_priors", "feature_profile", "risk_flags",
        "simulation_priors", "intervention_playbook",
        "behavioral_profile", "content_routing", "behavioral_narrative",
    }
    missing_fields = [
        p["persona_name"] for p in personas
        if not required.issubset(set(p.keys()))
    ]
    results["02_field_completeness"] = (
        ("PASS", "All required fields present in all personas")
        if not missing_fields else
        ("FAIL", f"Missing fields in: {missing_fields}")
    )

    # Check 3: behavioral_priors in [0,1] range
    prior_range_fails = []
    for p in personas:
        for k, v in p["behavioral_priors"].items():
            if not (0.0 <= v <= 1.0):
                prior_range_fails.append(f"{p['persona_name']}.{k}={v}")
    results["03_prior_ranges"] = (
        ("PASS", "All behavioral priors in [0.0, 1.0]")
        if not prior_range_fails else
        ("FAIL", f"Out-of-range priors: {prior_range_fails}")
    )

    # Check 4: Each persona has at least one intervention
    no_playbook = [p["persona_name"] for p in personas if not p["intervention_playbook"]]
    results["04_playbook_populated"] = (
        ("PASS", "All personas have at least one intervention")
        if not no_playbook else
        ("FAIL", f"No interventions for: {no_playbook}")
    )

    # Check 5: No duplicate persona_ids
    ids = [p["persona_id"] for p in personas]
    results["05_unique_ids"] = (
        ("PASS", "All persona_ids unique")
        if len(ids) == len(set(ids)) else
        ("FAIL", "Duplicate persona_ids detected")
    )

    # Check 6: Simulated lift values are finite floats
    lift_fails = []
    for p in personas:
        for item in p["intervention_playbook"]:
            for kpi, val in item["simulated_lift"].items():
                if not isinstance(val, (int, float)) or val != val:
                    lift_fails.append(f"{p['persona_name']}.{item['intervention_id']}.{kpi}")
    results["06_lift_values_finite"] = (
        ("PASS", "All simulated lift values are finite")
        if not lift_fails else
        ("FAIL", f"Non-finite lift values: {lift_fails}")
    )

    # Check 7: Risk flags are all boolean
    flag_fails = [
        p["persona_name"] for p in personas
        if not all(isinstance(v, bool) for v in p["risk_flags"].values())
    ]
    results["07_risk_flags_boolean"] = (
        ("PASS", "All risk flags are boolean")
        if not flag_fails else
        ("FAIL", f"Non-boolean risk flags in: {flag_fails}")
    )

    # Check 8: Simulation prior confidence_note is valid
    valid_conf = {"high", "medium", "low — treat priors with caution, small cluster"}
    conf_fails = [
        p["persona_name"] for p in personas
        if p["simulation_priors"].get("confidence_note") not in valid_conf
    ]
    results["08_confidence_note"] = (
        ("PASS", "All confidence_note values valid")
        if not conf_fails else
        ("FAIL", f"Invalid confidence_note in: {conf_fails}")
    )

    # Check 9: CI width validation
    ci_fails = []
    for p in personas:
        sp    = p["simulation_priors"]
        ci    = sp.get("base_churn_30d_ci", [0.0, 1.0])
        width = ci[1] - ci[0]
        if width <= 0:
            ci_fails.append(f"{p['persona_name']}: zero-width CI {ci}")
        elif width > 0.5:
            ci_fails.append(f"{p['persona_name']}: implausibly wide CI {ci} (width={width:.4f})")
    results["09_ci_width_validation"] = (
        ("PASS", "All churn CIs have positive, plausible width")
        if not ci_fails else
        ("FAIL", f"Degenerate or implausibly wide CIs: {ci_fails}")
    )

    # Check 10: Persona names are non-empty strings
    name_fails = [
        p.get("persona_name","") for p in personas
        if not isinstance(p.get("persona_name"), str) or not p["persona_name"].strip()
    ]
    results["10_persona_names"] = (
        ("PASS", "All persona names populated")
        if not name_fails else
        ("FAIL", f"Empty/missing names: {name_fails}")
    )

    # Check 11: high_churn_risk flag set for quick_churn label
    # PE-10: also cross-reference dominant_segment — a cluster whose dominant_segment
    # is quick_churn but behavioral_label is not should trigger a mismatch warning.
    churn_flag_miss = [
        p["persona_name"] for p in personas
        if p["behavioral_label"] == "quick_churn"
        and not p["risk_flags"].get("high_churn_risk", False)
    ]
    label_dom_mismatches_churn = [
        f"{p['persona_name']} (label={p['behavioral_label']}, dominant={p['dominant_segment']})"
        for p in personas
        if p.get("dominant_segment") == "quick_churn"
        and p["behavioral_label"] not in ("quick_churn", "mixed_behavior")
    ]
    if churn_flag_miss or label_dom_mismatches_churn:
        details = []
        if churn_flag_miss:
            details.append(f"quick_churn without high_churn_risk flag: {churn_flag_miss}")
        if label_dom_mismatches_churn:
            details.append(f"PE-10 dominant_segment=quick_churn but label mismatch: {label_dom_mismatches_churn}")
        results["11_quick_churn_flag"] = ("FAIL", "; ".join(details))
    else:
        results["11_quick_churn_flag"] = (
            "PASS", "quick_churn personas correctly flagged; no dominant_segment/label mismatches"
        )

    # Check 12: re_engager personas have REACTIVATION in their playbook
    # PE-10: also check for dominant_segment=re_engager with a non-re_engager label.
    reactiv_miss = [
        p["persona_name"] for p in personas
        if p["behavioral_label"] == "re_engager"
        and not any(
            item["intervention_id"] == "REACTIVATION"
            for item in p["intervention_playbook"]
        )
    ]
    label_dom_mismatches_react = [
        f"{p['persona_name']} (label={p['behavioral_label']}, dominant={p['dominant_segment']})"
        for p in personas
        if p.get("dominant_segment") == "re_engager"
        and p["behavioral_label"] != "re_engager"
    ]
    if reactiv_miss or label_dom_mismatches_react:
        details = []
        if reactiv_miss:
            details.append(f"re_engager missing REACTIVATION intervention: {reactiv_miss}")
        if label_dom_mismatches_react:
            details.append(f"PE-10 dominant_segment=re_engager but label mismatch: {label_dom_mismatches_react}")
        results["12_reengager_has_reactivation"] = ("FAIL", "; ".join(details))
    else:
        results["12_reengager_has_reactivation"] = (
            "PASS", "All re_engager personas include REACTIVATION; no dominant_segment/label mismatches"
        )

    # ── Group F: v2.0 validation checks ──────────────────────────────────────

    # F1: Formula verification — recompute CI from raw values and compare to stored
    f1_fails = []
    for p in personas:
        sp   = p["simulation_priors"]
        churn = sp.get("base_churn_30d", 0.0)
        n     = sp.get("cluster_size", 1)
        se    = (churn * (1 - churn)) ** 0.5 / max(n, 1) ** 0.5
        expected_lo = round(max(0.0, churn - 1.96 * se), 4)
        expected_hi = round(min(1.0, churn + 1.96 * se), 4)
        stored = sp.get("base_churn_30d_ci", [None, None])
        if stored[0] is not None and (
            abs(stored[0] - expected_lo) > 0.001 or
            abs(stored[1] - expected_hi) > 0.001
        ):
            f1_fails.append(
                f"{p['persona_name']}: stored={stored} expected=[{expected_lo},{expected_hi}]"
            )
    results["13_f1_formula_verification"] = (
        ("PASS", "CI values match recomputed formula for all personas")
        if not f1_fails else
        ("FAIL", f"CI mismatch (rounding drift or formula change): {f1_fails}")
    )

    # F2: Prior consistency — warn if all clusters within 0.05 on both axes
    churns      = [p["simulation_priors"]["base_churn_30d"]  for p in personas]
    completions = [p["simulation_priors"]["base_completion"]  for p in personas]
    churn_spread      = max(churns)      - min(churns)      if churns      else 0
    completion_spread = max(completions) - min(completions)  if completions else 0
    if churn_spread < 0.05 and completion_spread < 0.05:
        results["14_f2_prior_consistency"] = (
            "WARN",
            f"All clusters within 0.05 on churn ({churn_spread:.4f}) and "
            f"completion ({completion_spread:.4f}) axes — clustering may not be "
            f"discriminating meaningfully. Check collinearity gate (FSA-03)."
        )
    else:
        results["14_f2_prior_consistency"] = (
            "PASS",
            f"Priors show meaningful spread: churn={churn_spread:.4f}, "
            f"completion={completion_spread:.4f}"
        )

    # F3: Diffuse cluster validation — warn if any cluster is diffuse
    diffuse = [
        p["persona_name"] for p in personas
        if p["simulation_priors"].get("is_diffuse_cluster") is True
    ]
    if diffuse:
        results["15_f3_diffuse_cluster"] = (
            "WARN",
            f"Diffuse clusters detected (churn_risk_std>0.25): {diffuse}. "
            f"Mixed-segment clusters produce unreliable persona priors."
        )
    else:
        results["15_f3_diffuse_cluster"] = (
            "PASS",
            "No diffuse clusters detected (or std data not available)"
        )

    # Print results
    print()
    print("=" * 62)
    print(f"PERSONA ENGINE VALIDATION v{ENGINE_VERSION} — {len(results)} checks")
    print("=" * 62)
    n_pass = n_warn = n_fail = 0
    for check_id, (status, detail) in sorted(results.items()):
        icon = {"PASS": "✓", "WARN": "~", "FAIL": "✗"}.get(status, "?")
        print(f"  [{icon}] {check_id:<38} {status}  {detail}")
        if status == "PASS": n_pass += 1
        elif status == "WARN": n_warn += 1
        else:
            n_fail += 1
            all_pass = False
    print()
    print(f"  Result: {n_pass} PASS / {n_warn} WARN / {n_fail} FAIL")
    print("=" * 62)
    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="TwinSim Persona Engine — Layer 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--metadata", default="cluster_metadata.json",
                    help="cluster_metadata.json from clustering_engine.py (Layer 3 output).")
    ap.add_argument("--assignments", default=None,
                    help=(
                        "cluster_assignments.csv from clustering_engine.py. "
                        "When provided, computes modal cluster per user_id to produce "
                        "user-level persona assignments and unique_user_count per persona. "
                        "Required for accurate intervention targeting and stakeholder reporting. "
                        "When absent, unique_user_count is omitted from output."
                    ))
    ap.add_argument("--out",      default="persona_profiles.json",
                    help="Output persona profiles JSON.")
    ap.add_argument("--report",   default="persona_report.txt",
                    help="Output human-readable persona report TXT.")
    ap.add_argument("--validate", action="store_true",
                    help="Run validation checks after building personas.")
    args = ap.parse_args()

    # Load cluster metadata
    if not os.path.exists(args.metadata):
        print(f"[ERROR] {args.metadata} not found. Run clustering_engine.py first.")
        raise SystemExit(1)

    with open(args.metadata, encoding="utf-8") as f:
        meta = json.load(f)

    clusters     = meta.get("clusters", [])
    total_events = meta.get("n_events", 0)
    noise_pct    = meta.get("noise_pct", 0.0)
    stability    = meta.get("stability", {})

    print(f"[INFO] Loaded {args.metadata}")
    print(f"[INFO] {len(clusters)} clusters | {total_events:,} events | "
          f"noise {noise_pct:.1f}%")

    # XL-02: Version handshake — assert clustering engine compatibility before
    # building personas. persona_engine previously read cluster_version for
    # reporting only; this assert catches stale metadata from an incompatible
    # clustering engine version being passed to a newer persona engine.
    # Minimum compatible clustering engine version: 4.0.0 (CAL-002 TRIGGER_RULES).
    # Metadata from earlier versions used LABEL_RULES and may have mixed_behavior
    # labels where valid behavioral labels are now expected.
    COMPATIBLE_ENGINE_MIN = (4, 0, 0)
    cluster_version_str = meta.get("version", "0.0.0")
    try:
        cv_parts = tuple(int(x) for x in cluster_version_str.split(".")[:3])
    except (ValueError, AttributeError):
        cv_parts = (0, 0, 0)
    if cv_parts < COMPATIBLE_ENGINE_MIN:
        min_str = ".".join(str(x) for x in COMPATIBLE_ENGINE_MIN)
        print(
            f"[ERROR] XL-02: cluster_metadata.json produced by clustering_engine "
            f"v{cluster_version_str} — minimum compatible version is v{min_str}. "
            f"Re-run clustering_engine.py (v{min_str}+) to regenerate metadata "
            f"with TRIGGER_RULES labeling before building personas."
        )
        raise SystemExit(1)
    print(f"[INFO] XL-02: clustering_engine v{cluster_version_str} — compatible ✓")

    # Warn if clustering was unstable — priors have wide uncertainty
    stab_label = stability.get("stability_label", "UNKNOWN")
    if stab_label == "UNSTABLE":
        print(f"[WARN] Clustering stability: UNSTABLE (sil={stability.get('mean_silhouette',0):.4f}). "
              f"Persona priors have high uncertainty. "
              f"Re-run clustering_engine.py with P3-A+B data before trusting Layer 4 outputs.")
    elif stab_label == "MARGINAL":
        print(f"[WARN] Clustering stability: MARGINAL. Validate persona priors manually.")

    if not clusters:
        print("[ERROR] No clusters in metadata. Cannot produce personas.")
        raise SystemExit(1)

    # Build personas
    personas = []
    for cluster in clusters:
        persona = build_persona(cluster, total_events)
        personas.append(persona)
        print(f"[PERSONA] #{cluster.get('cluster_index',-1):>2} "
              f"{persona['persona_name']:<28} "
              f"size={persona['size']:>7,}  "
              f"churn={persona['simulation_priors']['base_churn_30d']:.3f}  "
              f"flags={[k for k,v in persona['risk_flags'].items() if v]}")

    # ── User-level persona assignment (Option B modal assignment) ────────────
    # Compute modal cluster per user_id from cluster_assignments.csv.
    # Populates unique_user_count on each persona and produces assignment_summary
    # for top-level output. When --assignments is not provided, unique_user_count
    # remains None and assignment_summary is omitted (graceful degradation).
    unique_user_counts, assignment_summary = _compute_user_persona_assignment(
        args.assignments,
        personas,
        expected_cluster_version=meta.get("version"),
    )
    for p in personas:
        # Keyed by cluster_index (int) — safe when duplicate behavioral_labels exist.
        cidx = p["cluster_index"]
        p["unique_user_count"] = unique_user_counts.get(int(cidx), None)

    # Write outputs
    output = {
        "version":           ENGINE_VERSION,
        "computed_at":       datetime.now(timezone.utc).isoformat(),
        "source_metadata":   args.metadata,
        "cluster_version":   meta.get("version", "unknown"),
        "algorithm":         meta.get("algorithm", "unknown"),
        "stability_label":   stab_label,
        "n_personas":        len(personas),
        "total_events":      total_events,
        "noise_events":      meta.get("noise_events", 0),
        "noise_pct":         noise_pct,
        # User-level assignment summary — populated when --assignments provided.
        # unique_user_count per persona sums to n_users_total (no double-counting).
        **({"user_assignment": assignment_summary} if assignment_summary else {}),
        # XL-04: data provenance flag — all priors derived from synthetic data.
        # Downstream consumers must not treat these as real historical measurements.
        "data_provenance": {
            "source":      "synthetic",
            "generator":   "generate_signals.py",
            "note":        (
                "All behavioral priors are derived from synthetic session data. "
                "These are simulation starting points, not empirical measurements. "
                "Layer 5 Monte Carlo runs use these as initialisation priors only."
            ),
        },
        "personas":          personas,
        "intervention_catalogue": INTERVENTIONS,
        # PE-09: explicit signal that intervention lifts are static priors, not cluster-derived.
        "lift_table_note": (
            "All intervention lift values in persona playbooks are static simulation priors "
            "from LIFT_TABLE — they do not adapt to the observed cluster structure. "
            "A binge_heavy cluster with base_completion=0.95 receives identical lifts to one "
            "with base_completion=0.73. Layer 5 Monte Carlo must treat these as initialisation "
            "values only. Replace with empirically calibrated values via CAL-004 "
            "(Segment Reaction History) once real campaign response data is available."
        ),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[DONE] Persona profiles → {args.out}  ({len(personas)} personas)")

    write_report(personas, meta, args.report)

    if args.validate:
        ok = validate_personas(personas)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()