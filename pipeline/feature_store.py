"""
TwinSim — Behavioral Feature Store  v4.0
==========================================
Layer 2: Feature Engineering from Two-Table Internal Data

Reads two Layer 1 output tables and produces:
  feature_store.csv        — engineered behavioral feature matrix
  feature_registry.json    — metadata for every feature
  feature_store_report.txt — summary stats, trajectory features, sparse fill rates
  baseline.json            — 90-day segment × content baseline (--generate_baseline)

Input tables:
  session_events.csv  — multi-session behavioral events (one row per session)
  user_profiles.csv   — static + trajectory user attributes (one row per user)

The two tables are joined on user_id. Profile fields are sourced from
user_profiles. All session-level features are sourced from session_events.

NOTE on network_type: moved from user_profiles to session_events in v4.0.
It is available directly on the joined df from the session side — not pulled
from the profile join.

Feature Groups
--------------
FG-01  Session Engagement     — depth and intensity of viewing behaviour
FG-02  Completion & Retention — how far users get; trajectory over sessions
FG-03  Churn & Friction       — leading indicators + support signals
FG-04  Content Affinity       — segment × content interaction patterns
FG-05  Contextual Modifiers   — recency, subscription, network, geo

Trajectory features (require multi-session data):
  binge_index_score       FG01 — session cadence intensity from profile
  attention_decay_curve   FG02 — slope of completion across user sessions
  avg_watch_gap_norm      FG02 — normalised inter-session gap from profile
  satisfaction_score      FG03 — sparse in-app rating, NULL-safe
  satisfaction_trend      FG03 — slope of ratings across user sessions
  drop_pattern_score      FG03 — per-session drop flags (early/mid completion thresholds)
  support_friction_score  FG05 — sparse support ticket signal

New in v4.1:
  avg_watch_gap_norm      FS-12: adaptive p75 ceiling replaces fixed /30.
                          Fixes variance collapse at 100k+ scale caused by
                          reactivation gap inflation in avg_watch_gap_days.

New in v4.0:
  ltv_score               FG05 — clip(ltv_to_date/500, 0, 1)
  account_health_score    FG05 — CRM composite [0,1] from profile
  fav_genre_confidence    FG04 — content affinity signal quality modifier
  superfan_score          FG05 — PPV buyer × merchandise signal
  campaign_receptivity    FG05 — historical campaign response rate (sparse-safe)
  plan_trajectory_score   FG05 — upgrade/downgrade history encoded
  lifecycle_stage_score   FG05 — customer lifecycle stage encoded
  episode_position_score  FG04 — engagement arc position (premiere→finale)

Changes from v3.0:
  - pause_count REMOVED (field dropped from session_events in v4.0)
  - attention_quality_score: completion / (1 + buffer_rate), not pause_rate
  - friction_index: buffer_rate*0.5 + network_stress_flag*0.5
  - drop_pattern_score: uses early_drop_flag/mid_drop_flag (per-session facts),
    not early_drop_rate/mid_session_drop_rate (content-level aggregates)
  - network_type: sourced from session_events, not user_profiles
  - days_since_last_session: NULL for session_number=1 → _safe() maps to 0.0

Sparse field philosophy:
  NULL = absent signal, not zero. The feature store maps:
    NULL satisfaction → satisfaction_score = 0.0 (no rating given)
    NULL tickets      → support_friction_score = 0.0 (no tickets raised)
    NULL campaign     → campaign_receptivity = 0.0 (no campaign history)
    NULL ppv/merch    → superfan_score = 0.0 (no commerce data)
  These are NOT imputed from medians — absence has its own meaning.

Architecture notes:
  - No sentiment, NPS, emotion, or brand perception constructs.
  - Baseline mode produces the normalization reference for Layer 7.
  - Focused mode (--content_refs / --segments) matches generate_signals.py.

Usage:
  python feature_store.py --events session_events.csv --profiles user_profiles.csv

  python feature_store.py --events session_events.csv --profiles user_profiles.csv \\
      --content_refs title_001

  python feature_store.py --events session_events.csv --profiles user_profiles.csv \\
      --content_refs title_001 title_002 --segments binge_heavy casual_dip

  python feature_store.py --events baseline_events.csv --profiles baseline_profiles.csv \\
      --generate_baseline --baseline_out baseline.json

  python feature_store_assessment.py --input feature_store.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Windows cp1252 fix — force UTF-8 so arrow/special chars don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_STORE_VERSION = "4.1.0"

# ── Feature Registry ──────────────────────────────────────────────────────────

FEATURE_REGISTRY: List[Dict[str, Any]] = [

    # ── FG-01: Session Engagement ─────────────────────────────────────────────

    {
        "feature_id": "FG01_001", "group": "session_engagement",
        "name": "session_depth_score", "dtype": "float",
        "description": (
            "log1p(session_depth) / log1p(20). Range [0,1]. "
            "Key discriminator: binge_heavy vs casual_dip."
        ),
        "source_fields": ["session_depth"], "aggregation": "log_normalize",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG01_002", "group": "session_engagement",
        "name": "session_intensity_score", "dtype": "float",
        "description": (
            "Composite: depth_norm*0.4 + duration_norm*0.4 + skip_intro*0.2. "
            "depth_norm=clip(depth/10,0,1). duration_norm=clip(dur/120,0,1)."
        ),
        "source_fields": ["session_depth", "session_duration_mins", "skip_intro_flag"],
        "aggregation": "weighted_composite",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG01_003", "group": "session_engagement",
        "name": "binge_signal", "dtype": "float",
        "description": (
            "1.0 if session_depth>=3 AND completion>=0.7, weighted by skip_intro. "
            "Strong predictor of binge_heavy membership."
        ),
        "source_fields": ["session_depth", "completion_pct", "skip_intro_flag"],
        "aggregation": "conditional_composite",
        "consumed_by": ["Layer 3"], "nullable": False,
    },
    {
        "feature_id": "FG01_004", "group": "session_engagement",
        "name": "attention_quality_score", "dtype": "float",
        "description": (
            "completion_pct / (1 + buffer_rate). buffer_rate=clip(buffer_events/5,0,1). "
            "High = attentive session. Low = buffering friction or distracted viewing. "
            "v4.0: pause_count removed (field dropped); buffer_rate is now the sole "
            "friction denominator — a more reliable technical signal."
        ),
        "source_fields": ["completion_pct", "buffer_events", "session_depth"],
        "aggregation": "ratio",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG01_005", "group": "session_engagement",
        "name": "rewatch_engagement_flag", "dtype": "int",
        "description": "1 if rewatch_flag=1 AND completion>=0.5. High content affinity signal.",
        "source_fields": ["rewatch_flag", "completion_pct"],
        "aggregation": "binary_conditional",
        "consumed_by": ["Layer 3"], "nullable": False,
    },
    {
        "feature_id": "FG01_006", "group": "session_engagement",
        "name": "binge_index_score", "dtype": "float",
        "description": (
            "USER-LEVEL TRAJECTORY. binge_index from user_profiles (fraction of "
            "consecutive session pairs with gap < 24h). Broadcast to all sessions "
            "for this user. Requires multi-session data."
        ),
        "source_fields": ["user_profiles.binge_index"],
        "aggregation": "passthrough_from_profile",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },

    # ── FG-02: Completion & Retention ─────────────────────────────────────────

    {
        "feature_id": "FG02_001", "group": "completion_retention",
        "name": "completion_rate_smooth", "dtype": "float",
        "description": "clip(completion_pct, 0, 1). Primary input to base_completion_probability.",
        "source_fields": ["completion_pct"], "aggregation": "passthrough_clip",
        "consumed_by": ["Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG02_002", "group": "completion_retention",
        "name": "completion_tier", "dtype": "int",
        "description": "3=high(>=0.80), 2=mid(0.40-0.79), 1=low(<0.40). Categorical for clustering.",
        "source_fields": ["completion_pct"], "aggregation": "ordinal_encode",
        "consumed_by": ["Layer 3"], "nullable": False,
    },
    {
        "feature_id": "FG02_003", "group": "completion_retention",
        "name": "completion_variance_signal", "dtype": "float",
        "description": "abs(completion_pct - 0.5) * 2. 0=uncertain, 1=decisive. Distinguishes obsessed from casual.",
        "source_fields": ["completion_pct"], "aggregation": "deviation_from_midpoint",
        "consumed_by": ["Layer 3"], "nullable": False,
    },
    {
        "feature_id": "FG02_004", "group": "completion_retention",
        "name": "recency_adjusted_completion", "dtype": "float",
        "description": "completion_pct × exp(-window_day/30). Recent completions weighted higher.",
        "source_fields": ["completion_pct", "window_day"], "aggregation": "recency_weighted",
        "consumed_by": ["Layer 4", "Layer 7"], "nullable": False,
    },
    {
        "feature_id": "FG02_005", "group": "completion_retention",
        "name": "days_since_last_normalised", "dtype": "float",
        "description": (
            "clip(days_since_last_session/90, 0, 1). 0=active today, 1=90+ day gap. "
            "days_since_last_session is NULL for session_number=1 (no prior session in window). "
            "_safe() maps NULL→0.0, encoding first sessions as 'active today' — "
            "semantically correct since recency is undefined, not a long gap."
        ),
        "source_fields": ["days_since_last_session"], "aggregation": "normalize",
        "consumed_by": ["Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG02_006", "group": "completion_retention",
        "name": "attention_decay_curve", "dtype": "float",
        "description": (
            "USER-LEVEL TRAJECTORY. Linear regression slope of completion_pct over "
            "session_number for each user (OLS on sessions ordered by session_number). "
            "Positive = engagement rising across sessions. Negative = declining. "
            "Clipped to [-0.5, 0.5] and normalised to [-1, 1]. Zero for single-session users."
        ),
        "source_fields": ["session_events.completion_pct", "session_events.session_number"],
        "aggregation": "ols_slope_per_user",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG02_007", "group": "completion_retention",
        "name": "avg_watch_gap_norm", "dtype": "float",
        "description": (
            "USER-LEVEL TRAJECTORY. clip(avg_watch_gap_days / p75_ceil, 0, 1) from user_profiles. "
            "Ceiling = p75 of non-zero avg_watch_gap_days values (adaptive, not fixed). "
            "FS-12: replaces fixed /30 ceiling which collapsed variance at 100k+ scale due to "
            "reactivation gap inflation in avg_watch_gap_days. p75 ceiling preserves relative "
            "ordering while maximising within-dataset variance for PCA separation. "
            "0 = gap <= p75. 1 = gap at/above p75 of dataset. Broadcast to all sessions."
        ),
        "source_fields": ["user_profiles.avg_watch_gap_days"],
        "aggregation": "normalize_from_profile",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },

    # ── FG-03: Churn & Friction ───────────────────────────────────────────────

    {
        "feature_id": "FG03_001", "group": "churn_friction",
        "name": "churn_risk_score", "dtype": "float",
        "description": (
            "Composite: days_norm*0.40 + (1-completion)*0.35 + buffer_rate*0.15 + pay_friction*0.10. "
            "pay_friction from plan_type (basic=0.30, standard=0.15, premium=0.05)."
        ),
        "source_fields": ["days_since_last_session", "completion_pct", "buffer_events", "plan_type"],
        "aggregation": "weighted_composite",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG03_002", "group": "churn_friction",
        "name": "churn_velocity", "dtype": "float",
        "description": "days_since_norm × (1 - completion_pct). How fast is this user moving toward churn?",
        "source_fields": ["days_since_last_session", "completion_pct"], "aggregation": "product",
        "consumed_by": ["Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG03_003", "group": "churn_friction",
        "name": "churn_flag_encoded", "dtype": "int",
        "description": "1 if churn_flag=True. Ground-truth churn label for twin model training.",
        "source_fields": ["churn_flag"], "aggregation": "binary_encode",
        "consumed_by": ["Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG03_004", "group": "churn_friction",
        "name": "reactivation_signal", "dtype": "float",
        "description": "reactivation_flag × clip(days_since/60, 0, 1). Higher = returned after longer gap.",
        "source_fields": ["reactivation_flag", "days_since_last_session"],
        "aggregation": "conditional_weighted",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG03_005", "group": "churn_friction",
        "name": "friction_index", "dtype": "float",
        "description": (
            "Technical streaming friction: buffer_rate*0.5 + network_stress*0.5. "
            "buffer_rate=clip(buffer_events/5,0,1). network_stress=network_stress_flag(0/1). "
            "v4.0: pause_count removed (field dropped); network_stress_flag added — "
            "jitter>50ms represents service-failure-threshold conditions that directly cause churn."
        ),
        "source_fields": ["buffer_events", "network_stress_flag"], "aggregation": "weighted_composite",
        "consumed_by": ["Layer 3", "Layer 7"], "nullable": False,
    },
    {
        "feature_id": "FG03_006", "group": "churn_friction",
        "name": "satisfaction_score", "dtype": "float",
        "description": (
            "SPARSE-SAFE. (content_satisfaction - 1) / 4 normalised to [0,1]. "
            "NULL rating → 0.0 (absent signal, not negative). "
            "Non-zero only for sessions where user left a rating (~28%)."
        ),
        "source_fields": ["content_satisfaction"],
        "aggregation": "sparse_normalize",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG03_007", "group": "churn_friction",
        "name": "satisfaction_trend", "dtype": "float",
        "description": (
            "USER-LEVEL TRAJECTORY. OLS slope of content_satisfaction over session_number, "
            "computed only on sessions where rating is present. Positive = improving satisfaction. "
            "Negative = declining. 0.0 for users with <2 rated sessions. Range [-1, 1]."
        ),
        "source_fields": ["content_satisfaction", "session_number"],
        "aggregation": "ols_slope_per_user_sparse",
        "consumed_by": ["Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG03_008", "group": "churn_friction",
        "name": "drop_pattern_score", "dtype": "float",
        "description": (
            "Per-session drop composite: early_drop_flag*0.4 + mid_drop_flag*0.6. "
            "v4.0 REDESIGNED: uses per-session binary flags (derived from completion_pct "
            "thresholds) rather than early_drop_rate/mid_session_drop_rate (which are "
            "content-level population aggregates repeated per row — not per-session observations). "
            "early_drop_flag=1 when completion_pct<0.25; mid_drop_flag=1 when 0.25<=completion<0.75."
        ),
        "source_fields": ["early_drop_flag", "mid_drop_flag"],
        "aggregation": "weighted_composite",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },

    # ── FG-04: Content Affinity ───────────────────────────────────────────────

    {
        "feature_id": "FG04_001", "group": "content_affinity",
        "name": "content_engagement_score", "dtype": "float",
        "description": "completion*0.5 + depth_norm*0.3 + rewatch*0.2. Range [0,1].",
        "source_fields": ["completion_pct", "session_depth", "rewatch_flag"],
        "aggregation": "weighted_composite",
        "consumed_by": ["Layer 4", "Layer 5"], "nullable": False,
    },
    {
        "feature_id": "FG04_002", "group": "content_affinity",
        "name": "content_type_weight", "dtype": "float",
        "description": "series=1.0, documentary=0.95, sport=0.90, film=0.85, shortform=0.75.",
        "source_fields": ["content_type"], "aggregation": "lookup_encode",
        "consumed_by": ["Layer 3"], "nullable": False,
    },
    {
        "feature_id": "FG04_003", "group": "content_affinity",
        "name": "viewing_context_score", "dtype": "float",
        "description": "device_weight × time_weight. smart_tv+evening=1.0 (max).",
        "source_fields": ["device_type", "time_of_day"], "aggregation": "lookup_product",
        "consumed_by": ["Layer 3"], "nullable": False,
    },
    {
        "feature_id": "FG04_004", "group": "content_affinity",
        "name": "weekend_viewing_flag", "dtype": "int",
        "description": "1 if is_weekend=1 AND session_depth>=2. Leisure-mode signal.",
        "source_fields": ["is_weekend", "session_depth"], "aggregation": "binary_conditional",
        "consumed_by": ["Layer 3"], "nullable": False,
    },
    {
        "feature_id": "FG04_005", "group": "content_affinity",
        "name": "event_type_weight", "dtype": "float",
        "description": "completion=1.0, reactivation=0.90, progress=0.75, session_start=0.60, churn=0.50.",
        "source_fields": ["event_type"], "aggregation": "lookup_encode",
        "consumed_by": ["Layer 4"], "nullable": False,
    },

    # ── FG-05: Contextual Modifiers ───────────────────────────────────────────

    {
        "feature_id": "FG05_001", "group": "contextual_modifiers",
        "name": "recency_weight", "dtype": "float",
        "description": "exp(-window_day/30). Recent events weight more in twin training.",
        "source_fields": ["window_day"], "aggregation": "exp_decay",
        "consumed_by": ["Layer 4", "Layer 7"], "nullable": False,
    },
    {
        "feature_id": "FG05_002", "group": "contextual_modifiers",
        "name": "subscription_tier_score", "dtype": "float",
        "description": "premium=1.0, standard=0.65, basic=0.30. From user_profiles.plan_type.",
        "source_fields": ["user_profiles.plan_type"], "aggregation": "lookup_encode",
        "consumed_by": ["Layer 4", "Layer 7"], "nullable": False,
    },
    {
        "feature_id": "FG05_003", "group": "contextual_modifiers",
        "name": "tenure_weight", "dtype": "float",
        "description": "clip(tenure_months/24, 0, 1). From user_profiles. Long tenure + churn = high-value alert.",
        "source_fields": ["user_profiles.tenure_months"], "aggregation": "normalize",
        "consumed_by": ["Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG05_004", "group": "contextual_modifiers",
        "name": "network_quality_score", "dtype": "float",
        "description": (
            "Fiber=1.0, 5G=0.85, Broadband=0.70, 4G=0.55. "
            "v4.0: sourced from session_events.network_type (session-level, variable per session). "
            "Previously from user_profiles.network_type (static). "
            "Session-level is more accurate — users switch between networks."
        ),
        "source_fields": ["session_events.network_type"], "aggregation": "lookup_encode",
        "consumed_by": ["Layer 3"], "nullable": False,
    },
    {
        "feature_id": "FG05_005", "group": "contextual_modifiers",
        "name": "geo_tier", "dtype": "int",
        "description": "3=mature(US/UK/AU/CA), 2=growth(ZA/KE), 1=emerging(IN/NG/GH/BR).",
        "source_fields": ["user_profiles.geo_country"], "aggregation": "lookup_encode",
        "consumed_by": ["Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG05_006", "group": "contextual_modifiers",
        "name": "high_value_churn_flag", "dtype": "int",
        "description": "1 if churn_flag AND tenure>=12mo AND plan=premium. Highest-priority intervention signal.",
        "source_fields": ["churn_flag", "user_profiles.tenure_months", "user_profiles.plan_type"],
        "aggregation": "binary_conditional",
        "consumed_by": ["Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG05_007", "group": "contextual_modifiers",
        "name": "support_friction_score", "dtype": "float",
        "description": (
            "SPARSE-SAFE USER-LEVEL. Composite from user_profiles ticket fields. "
            "0.0 when no tickets (NULL). When tickets present: "
            "ticket_count_norm*0.6 + resolution_norm*0.4. "
            "ticket_count_norm=clip(ticket_count/4,0,1). resolution_norm=clip(avg_res_hrs/48,0,1). "
            "High = frequent unresolved support issues → churn precursor."
        ),
        "source_fields": ["user_profiles.ticket_count", "user_profiles.ticket_avg_resolution_hrs"],
        "aggregation": "sparse_composite_from_profile",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },

    # ── FG-05: New v4.0 Contextual Modifiers ──────────────────────────────────

    {
        "feature_id": "FG05_008", "group": "contextual_modifiers",
        "name": "ltv_score", "dtype": "float",
        "description": (
            "clip(ltv_to_date / p95(ltv_to_date), 0, 1). ltv_to_date = monthly_price × tenure_months × "
            "payment_reliability_discount. Enables high-value persona prioritization. "
            "FS-11: ceiling computed dynamically as p95 of non-null ltv_to_date values — "
            "replaces hardcoded $500 which saturated long-tenure premium users (observed "
            "max=$1,139 on 123k run). Fallback ceiling=$500 when <10 non-null LTV values. "
            "Broadcast from user_profiles to all sessions."
        ),
        "source_fields": ["user_profiles.ltv_to_date"], "aggregation": "normalize_from_profile",
        "consumed_by": ["Layer 4", "Layer 5"], "nullable": False,
    },
    {
        "feature_id": "FG05_009", "group": "contextual_modifiers",
        "name": "account_health_score", "dtype": "float",
        "description": (
            "Passthrough [0,1] from user_profiles. Derived composite: "
            "recency_score*0.35 + completion_score*0.30 + payment_score*0.20 - ticket_penalty. "
            "Low = at-risk. High = healthy engaged user. "
            "Direct twin KPI target for CX intervention simulations."
        ),
        "source_fields": ["user_profiles.account_health_score"], "aggregation": "passthrough_from_profile",
        "consumed_by": ["Layer 4", "Layer 5"], "nullable": False,
    },
    {
        "feature_id": "FG05_010", "group": "contextual_modifiers",
        "name": "campaign_receptivity", "dtype": "float",
        "description": (
            "SPARSE-SAFE. campaign_response_rate passthrough from user_profiles. "
            "NULL → 0.0 (no campaign history). Range [0,1]. "
            "Defines 'marketing-receptive' vs 'dark' persona dimension. "
            "Twin uses to calibrate campaign lift assumptions in intervention simulations."
        ),
        "source_fields": ["user_profiles.campaign_response_rate"],
        "aggregation": "sparse_passthrough_from_profile",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG05_011", "group": "contextual_modifiers",
        "name": "superfan_score", "dtype": "float",
        "description": (
            "SPARSE-SAFE. ppv_buyer_flag*0.6 + merchandise_flag*0.4. "
            "ppv_buyer_flag=1 if ppv_purchase_count is non-null (purchased at least once). "
            "merchandise_flag=1 if merchandise_purchase_flag=1. "
            "0.0 for non-purchasers (NULL commerce data). "
            "Superfan persona dimension — defines high-willingness-to-pay archetype "
            "that viewing behaviour alone cannot replicate."
        ),
        "source_fields": ["user_profiles.ppv_purchase_count", "user_profiles.merchandise_purchase_flag"],
        "aggregation": "sparse_composite_from_profile",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG05_012", "group": "contextual_modifiers",
        "name": "plan_trajectory_score", "dtype": "float",
        "description": (
            "upgrade_downgrade_history encoded: upgraded=1.0, none=0.5, downgraded=0.0. "
            "Personas that previously downgraded are elevated churn risk. "
            "Upgraded = high satisfaction / value perception. "
            "Broadcast from user_profiles to all sessions."
        ),
        "source_fields": ["user_profiles.upgrade_downgrade_history"],
        "aggregation": "lookup_encode_from_profile",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG05_013", "group": "contextual_modifiers",
        "name": "lifecycle_stage_score", "dtype": "float",
        "description": (
            "lifecycle_stage encoded: active=1.0, new=0.70, reactivated=0.80, at_risk=0.40, churned=0.0. "
            "new = tenure_months ≤ 2 (engaged but not yet proven retained). "
            "Derived from behavioral fields at profile generation. "
            "Twin simulates stage transition probabilities under interventions. "
            "Broadcast from user_profiles to all sessions."
        ),
        "source_fields": ["user_profiles.lifecycle_stage"],
        "aggregation": "lookup_encode_from_profile",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },

    # ── FG-04: New v4.0 Content Affinity ──────────────────────────────────────

    {
        "feature_id": "FG04_006", "group": "content_affinity",
        "name": "fav_genre_confidence", "dtype": "float",
        "description": (
            "Passthrough [0,1] from user_profiles. Computed as sessions_on_winning_genre / total_sessions. "
            "FS-13 DESIGN NOTE: this feature measures SESSION CONCENTRATION on a single genre, "
            "not preference signal reliability. With multiple genres and one comprising a significant "
            "catalogue share, a user with no genuine preference can accumulate high concentration "
            "by catalogue skew alone. The description 'preference signal quality modifier' is "
            "inaccurate — it is a concentration measure. The correct fix is computing confidence "
            "relative to catalogue base rates (sessions_on_genre / expected_by_catalogue_share), "
            "but this requires catalogue exposure data not currently tracked per-session. "
            "Current use: gate at > 0.4 before treating genre affinity as reliable. "
            "Consumers (Layer 3, Layer 4) must be aware that high confidence can reflect "
            "catalogue skew rather than genuine preference. Broadcast to all sessions."
        ),
        "source_fields": ["user_profiles.fav_genre_confidence"],
        "aggregation": "passthrough_from_profile",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
    {
        "feature_id": "FG04_007", "group": "content_affinity",
        "name": "episode_position_score", "dtype": "float",
        "description": (
            "Engagement arc position encoded: premiere=1.0, penultimate_arc=0.75, "
            "mid_season=0.50, finale=1.0, NULL=0.50 (non-episodic content). "
            "premiere and finale share 1.0 — both are high-engagement events "
            "(discovery intent at premiere; completion/resolution at finale). "
            "mid_season=0.50 is the baseline. penultimate_arc=0.75 signals deepening commitment. "
            "NULL (film/shortform/sport) maps to 0.50 — neutral arc position."
        ),
        "source_fields": ["episode_position"], "aggregation": "lookup_encode",
        "consumed_by": ["Layer 3", "Layer 4"], "nullable": False,
    },
]

FEATURE_NAMES = [f["name"] for f in FEATURE_REGISTRY]

# ── Lookup tables ─────────────────────────────────────────────────────────────

CONTENT_TYPE_WEIGHT = {"series":1.00,"documentary":0.95,"sport":0.90,"film":0.85,"shortform":0.75}
DEVICE_WEIGHT       = {"smart_tv":1.00,"laptop":0.85,"desktop":0.80,"tablet":0.75,"mobile":0.65}
TIME_WEIGHT         = {"evening":1.00,"night":0.90,"afternoon":0.80,"morning":0.70}
EVENT_TYPE_WEIGHT   = {"completion":1.00,"reactivation":0.90,"progress":0.75,"session_start":0.60,"churn":0.50}
PLAN_TIER_SCORE     = {"premium":1.00,"standard":0.65,"basic":0.30}
NETWORK_QUALITY     = {"Fiber":1.00,"5G":0.85,"Broadband":0.70,"4G":0.55}
GEO_TIER            = {"US":3,"UK":3,"AU":3,"CA":3,"DE":3,
                        "ZA":2,"KE":2,"SG":2,"AE":2,"FR":2,
                        "IN":1,"NG":1,"GH":1,"BR":1}
PLAN_FRICTION       = {"basic":0.30,"standard":0.15,"premium":0.05}
EPISODE_POSITION_SCORE = {
    # FS-12 DESIGN NOTE: premiere and finale both encode to 1.00.
    # These represent meaningfully different behavioral states:
    #   premiere  → discovery/acquisition risk (user may not return after first episode)
    #   finale    → renewal opportunity (user has completed the arc; churn/reactivation decision point)
    # Encoding them identically makes differentiation impossible from this feature alone.
    # Layer 4 cannot distinguish a premiere cluster from a finale cluster using
    # episode_position_score. The encoding is intentional (both are high-engagement
    # arc positions vs mid_season baseline), but the limitation must be documented
    # before Layer 5 calibrates intervention response by episode arc position.
    # Fix path: split into two separate binary features (is_premiere, is_finale) if
    # Layer 5 requires arc-position-specific intervention routing.
    "premiere":        1.00,
    "penultimate_arc": 0.75,
    "mid_season":      0.50,
    "finale":          1.00,
}
PLAN_TRAJECTORY_SCORE = {"upgraded": 1.0, "none": 0.5, "downgraded": 0.0}
LIFECYCLE_STAGE_SCORE = {
    "active":      1.00,
    "new":         0.70,  # P2-A: tenure_months ≤ 2 — engaged but not yet proven retained
    "reactivated": 0.80,
    "at_risk":     0.40,
    "churned":     0.00,
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default

def _bool_col(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().isin(["true","1","yes"]).values

def _round(arr: np.ndarray) -> list:
    return [round(float(v), 6) for v in arr]

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ── Trajectory feature computation ───────────────────────────────────────────

def compute_user_slopes(
    df: pd.DataFrame,
    value_col: str,
    order_col: str = "session_number",
    sparse: bool = False,
) -> pd.Series:
    """
    Compute per-user OLS slope of value_col over order_col.

    sparse=True: only use rows where value_col is non-null (for satisfaction_trend).
    sparse=False: use all rows (for attention_decay_curve).

    Returns a Series indexed by user_id with the slope clipped to [-0.5, 0.5]
    and normalised to [-1, 1]. Users with <2 usable observations get slope=0.0.
    """
    slopes: Dict[str, float] = {}
    for uid, group in df.groupby("user_id"):
        g = group[[order_col, value_col]].copy()
        g[order_col] = pd.to_numeric(g[order_col], errors="coerce")
        g[value_col] = pd.to_numeric(g[value_col], errors="coerce")
        if sparse:
            g = g.dropna(subset=[value_col])
        else:
            g = g.dropna()
        if len(g) < 2:
            slopes[uid] = 0.0
            continue
        x = g[order_col].values.astype(float)
        y = g[value_col].values.astype(float)
        # OLS slope: cov(x,y)/var(x)
        xm, ym = x.mean(), y.mean()
        denom  = float(np.sum((x - xm)**2))
        if denom < 1e-9:
            slopes[uid] = 0.0
        else:
            slope = float(np.sum((x - xm) * (y - ym)) / denom)
            # Clip to [-0.5, 0.5] then normalise to [-1, 1]
            slopes[uid] = float(np.clip(slope, -0.5, 0.5) / 0.5)
    return pd.Series(slopes, name="slope")


# ── Join helper ───────────────────────────────────────────────────────────────

def join_tables(
    sess: pd.DataFrame,
    prof: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join session_events with user_profiles on user_id.
    Profile fields are broadcast to every session row for that user.
    Sessions with no matching profile get NaN profile fields (handled in compute).

    NOTE: network_type is NOT pulled from user_profiles — it was moved to
    session_events in v4.0 (session-level, variable per session). It is already
    present in sess and available on the joined df from the session side.
    """
    profile_cols = [
        "user_id",
        # Subscription
        "plan_type", "tenure_months",
        # Geo (network_type removed — now session-level)
        "geo_country",
        # Behavioral trajectory
        "binge_index", "avg_watch_gap_days",
        "fav_genre_confidence",
        # Sparse support
        "ticket_count", "ticket_avg_resolution_hrs",
        # New v4.0 profile fields
        "ltv_to_date", "account_health_score",
        "campaign_response_rate",
        "ppv_purchase_count", "merchandise_purchase_flag",
        "upgrade_downgrade_history", "lifecycle_stage",
    ]
    available = [c for c in profile_cols if c in prof.columns]
    missing   = [c for c in profile_cols if c not in prof.columns and c != "user_id"]
    if missing:
        print(f"[WARN] Profile columns not found (will be NaN): {missing}")
    df = sess.merge(prof[available], on="user_id", how="left", suffixes=("", "_profile"))
    print(f"[INFO] Joined: {len(df):,} rows × {len(df.columns)} cols "
          f"({df['user_id'].nunique():,} users)")
    return df


# ── Core feature computation ──────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 41 behavioral features from the joined two-table dataframe.
    Session-level features operate row-wise (vectorised numpy).
    Trajectory features are computed per-user then broadcast back.
    Sparse fields are handled NULL-safe — absence maps to 0.0, not median.
    """
    n = len(df)
    print(f"[INFO] Computing features: {n:,} sessions × {df['user_id'].nunique():,} users...")

    # ── Raw field extraction ──────────────────────────────────────────────────
    # Session fields
    completion   = df["completion_pct"].apply(_safe).values
    sess_depth   = df["session_depth"].apply(_safe).values
    sess_dur     = df["session_duration_mins"].apply(_safe).values
    days_since   = df["days_since_last_session"].apply(_safe).values   # NULL→0.0 for session 1
    buffer_ev    = df["buffer_events"].apply(_safe).values
    skip_intro   = df["skip_intro_flag"].apply(_safe).values
    rewatch      = df["rewatch_flag"].apply(_safe).values
    is_weekend   = df["is_weekend"].apply(_safe).values
    window_day   = df["window_day"].apply(_safe).values
    early_drop_f = df["early_drop_flag"].apply(_safe).values           # per-session binary
    mid_drop_f   = df["mid_drop_flag"].apply(_safe).values             # per-session binary
    net_stress   = df["network_stress_flag"].apply(_safe).values       # jitter>50ms binary
    churn_flag   = _bool_col(df["churn_flag"])
    react_flag   = _bool_col(df["reactivation_flag"])

    # Categorical session fields
    content_type     = df["content_type"].astype(str).str.lower().values
    device_type      = df["device_type"].astype(str).str.lower().values
    time_of_day      = df["time_of_day"].astype(str).str.lower().values
    event_type       = df["event_type"].astype(str).str.lower().values
    network_type     = df["network_type"].astype(str).fillna("4G").values  # session-level
    episode_position = df["episode_position"].astype(str).values           # may be "nan"

    # Profile fields — sourced from join (broadcast per user)
    plan_type    = df["plan_type"].astype(str).str.lower().fillna("standard").values
    tenure_mo    = df["tenure_months"].apply(_safe).values
    geo_country  = df["geo_country"].astype(str).fillna("US").values

    # Profile trajectory fields (scalar per user, broadcast)
    binge_idx_raw  = df["binge_index"].apply(_safe).values
    watch_gap_raw  = df["avg_watch_gap_days"].apply(_safe, args=(2.0,)).values

    # Sparse profile fields — NaN preserved for NULL-safe handling
    ticket_count_raw = pd.to_numeric(df["ticket_count"], errors="coerce").values
    ticket_res_raw   = pd.to_numeric(df["ticket_avg_resolution_hrs"], errors="coerce").values

    # New v4.0 profile fields — NaN preserved where sparse
    ltv_raw           = pd.to_numeric(df.get("ltv_to_date",          pd.Series([np.nan]*n)), errors="coerce").values
    acct_health_raw   = pd.to_numeric(df.get("account_health_score", pd.Series([np.nan]*n)), errors="coerce").values
    fav_conf_raw      = pd.to_numeric(df.get("fav_genre_confidence",  pd.Series([np.nan]*n)), errors="coerce").values
    campaign_raw      = pd.to_numeric(df.get("campaign_response_rate",pd.Series([np.nan]*n)), errors="coerce").values
    ppv_raw           = pd.to_numeric(df.get("ppv_purchase_count",    pd.Series([np.nan]*n)), errors="coerce").values
    merch_raw         = pd.to_numeric(df.get("merchandise_purchase_flag", pd.Series([np.nan]*n)), errors="coerce").values
    plan_traj_raw     = df.get("upgrade_downgrade_history", pd.Series(["none"]*n)).astype(str).values
    lifecycle_raw     = df.get("lifecycle_stage", pd.Series(["active"]*n)).astype(str).values

    # Sparse session field — NaN preserved
    satisfaction_raw = pd.to_numeric(df.get("content_satisfaction", pd.Series([np.nan]*n)),
                                     errors="coerce").values

    # ── FG-01: Session Engagement ─────────────────────────────────────────────

    session_depth_score = np.clip(np.log1p(sess_depth) / math.log1p(20), 0.0, 1.0)

    dur_norm   = np.clip(sess_dur / 120.0, 0.0, 1.0)
    depth_norm = np.clip(sess_depth / 10.0, 0.0, 1.0)
    session_intensity_score = np.clip(depth_norm*0.4 + dur_norm*0.4 + skip_intro*0.2, 0.0, 1.0)

    binge_base   = ((sess_depth >= 3) & (completion >= 0.7)).astype(float)
    binge_signal = np.clip(binge_base * (0.5 + skip_intro * 0.5), 0.0, 1.0)

    # v4.0: pause_rate removed — buffer_rate is the friction denominator
    buffer_rate             = np.clip(buffer_ev / 5.0, 0.0, 1.0)
    attention_quality_score = np.clip(completion / (1.0 + buffer_rate), 0.0, 1.0)

    rewatch_engagement_flag = ((rewatch == 1) & (completion >= 0.5)).astype(int)

    binge_index_score = np.clip(binge_idx_raw, 0.0, 1.0)

    # ── FG-02: Completion & Retention ─────────────────────────────────────────

    completion_rate_smooth = np.clip(completion, 0.0, 1.0)

    completion_tier = np.where(
        completion >= 0.80, 3, np.where(completion >= 0.40, 2, 1)
    ).astype(int)

    completion_variance_signal = np.abs(completion - 0.5) * 2.0

    # Single recency weight computation — reused for recency_adjusted_completion (FG02)
    # and recency_weight feature (FG05). Defined once to prevent silent formula drift.
    recency_w = np.exp(-window_day / 30.0)
    recency_adjusted_completion = np.clip(completion * recency_w, 0.0, 1.0)

    # days_since NULL for session_number=1 → _safe() maps to 0.0 (treated as active today)
    days_since_last_normalised = np.clip(days_since / 90.0, 0.0, 1.0)

    print("[INFO] Computing attention_decay_curve (OLS per user)...")
    slope_series = compute_user_slopes(df, "completion_pct", "session_number", sparse=False)
    slope_map    = slope_series.to_dict()
    attention_decay_curve = np.array([slope_map.get(uid, 0.0) for uid in df["user_id"]])

    # FS-12: avg_watch_gap_norm adaptive ceiling.
    # Fixed /30 ceiling was calibrated on 20k/90d and compresses variance at 100k scale.
    # Root cause: avg_watch_gap_days includes reactivation gaps (30–90d) which inflate
    # per-user means and collapse inter-segment separation in normalised space.
    # Fix: compute p75 of non-zero gaps as the normalisation ceiling. p75 is robust to
    # the re_engager/reactivation tail while preserving binge_heavy vs casual_dip spread.
    # Fallback to 30.0 when fewer than 10 non-zero values (e.g. very small datasets).
    _gap_nonzero = watch_gap_raw[watch_gap_raw > 0.1]
    if len(_gap_nonzero) >= 10:
        _gap_ceil = float(np.percentile(_gap_nonzero, 75))
        _gap_ceil = max(_gap_ceil, 1.0)   # floor: never divide by less than 1
    else:
        _gap_ceil = 30.0                  # fallback for tiny datasets
    avg_watch_gap_norm = np.clip(watch_gap_raw / _gap_ceil, 0.0, 1.0)
    print(f"[INFO] avg_watch_gap_norm: p75_ceiling={_gap_ceil:.2f}d  "
          f"mean_norm={avg_watch_gap_norm.mean():.4f}  var={avg_watch_gap_norm.var():.4f}")

    # ── FG-03: Churn & Friction ───────────────────────────────────────────────

    days_norm    = np.clip(days_since / 90.0, 0.0, 1.0)
    pay_friction = np.array([PLAN_FRICTION.get(p, 0.15) for p in plan_type])
    churn_risk_score = np.clip(
        days_norm*0.40 + (1.0-completion)*0.35 + buffer_rate*0.15 + pay_friction*0.10,
        0.0, 1.0
    )

    # FS-07: churn_velocity collapses to 0 for ALL session_number=1 rows.
    # days_since_last_session is NULL for session_number=1 → _safe() → 0.0,
    # so churn_velocity = 0.0 * (1 - completion) = 0.0 regardless of completion.
    # Layer 4/5 must not treat churn_velocity as informative for first-session rows.
    # This is structural and dataset-size independent — 100% of session_number=1
    # rows will always have churn_velocity=0.0.
    churn_velocity = np.clip(days_since_last_normalised * (1.0 - completion), 0.0, 1.0)

    churn_flag_encoded = churn_flag.astype(int)

    gap_norm = np.clip(days_since / 60.0, 0.0, 1.0)
    reactivation_signal = np.clip(react_flag.astype(float) * gap_norm, 0.0, 1.0)

    # v4.0: buffer_rate*0.5 + network_stress*0.5 (pause_count removed)
    friction_index = np.clip(buffer_rate*0.5 + net_stress*0.5, 0.0, 1.0)

    satisfaction_score = np.where(
        np.isnan(satisfaction_raw),
        0.0,
        np.clip((satisfaction_raw - 1.0) / 4.0, 0.0, 1.0)
    )

    print("[INFO] Computing satisfaction_trend (OLS per user, sparse)...")
    if "content_satisfaction" in df.columns and "session_number" in df.columns:
        sat_slope_series = compute_user_slopes(
            df, "content_satisfaction", "session_number", sparse=True
        )
        sat_slope_map    = sat_slope_series.to_dict()
        satisfaction_trend = np.array([sat_slope_map.get(uid, 0.0) for uid in df["user_id"]])
    else:
        satisfaction_trend = np.zeros(n)

    # v4.0: uses per-session binary flags, not content-level aggregate rates.
    # FS-08: drop_pattern_score is a ninth transformation of completion_pct.
    # early_drop_flag = (completion < 0.25), mid_drop_flag = (0.25 ≤ completion < 0.75).
    # Both flags are threshold-derived from completion_pct, making drop_pattern_score
    # a deterministic function of completion. It is correctly excluded from
    # CLUSTERING_FEATURES (in SPARSE_CLUSTERING_AUXILIARY). Do NOT restore it to
    # the clustering set without a collinearity review — it will re-introduce a
    # completion derivative that Check 13 will catch and block.
    drop_pattern_score = np.clip(early_drop_f*0.4 + mid_drop_f*0.6, 0.0, 1.0)

    # ── FG-04: Content Affinity ───────────────────────────────────────────────

    depth_norm_aff = np.clip(sess_depth / 10.0, 0.0, 1.0)
    content_engagement_score = np.clip(
        completion*0.5 + depth_norm_aff*0.3 + rewatch*0.2, 0.0, 1.0
    )

    content_type_weight = np.array([CONTENT_TYPE_WEIGHT.get(ct, 0.80) for ct in content_type])

    dev_w  = np.array([DEVICE_WEIGHT.get(d, 0.70) for d in device_type])
    time_w = np.array([TIME_WEIGHT.get(t, 0.75) for t in time_of_day])
    viewing_context_score = np.clip(dev_w * time_w, 0.0, 1.0)

    weekend_viewing_flag = ((is_weekend == 1) & (sess_depth >= 2)).astype(int)

    event_type_weight = np.array([EVENT_TYPE_WEIGHT.get(et, 0.60) for et in event_type])

    # FG04_006 — fav_genre_confidence from profile (passthrough, broadcast)
    # NULL → 0.0 (null-as-signal: profile-unmatched users fail the confidence gate > 0.4,
    # preventing genre affinity from being treated as reliable for unknown users)
    fav_genre_confidence = np.clip(
        np.where(np.isnan(fav_conf_raw), 0.0, fav_conf_raw), 0.0, 1.0
    )

    # FG04_007 — episode_position_score
    episode_position_score = np.array([
        EPISODE_POSITION_SCORE.get(ep, 0.50)      # NULL/"nan" → 0.50 (neutral)
        for ep in episode_position
    ])

    # ── FG-05: Contextual Modifiers ───────────────────────────────────────────

    recency_weight          = recency_w  # unified with FG02 recency_w — see FS-09
    subscription_tier_score = np.array([PLAN_TIER_SCORE.get(p, 0.30) for p in plan_type])
    tenure_weight           = np.clip(tenure_mo / 24.0, 0.0, 1.0)
    # network_type now from session_events (variable per session)
    network_quality_score   = np.array([NETWORK_QUALITY.get(nt, 0.55) for nt in network_type])
    geo_tier                = np.array([GEO_TIER.get(c, 1) for c in geo_country])

    high_value_churn_flag = (
        churn_flag
        & (tenure_mo >= 12)
        & (np.array([p == "premium" for p in plan_type]))
    ).astype(int)

    ticket_norm  = np.where(np.isnan(ticket_count_raw), 0.0,
                            np.clip(ticket_count_raw / 4.0, 0.0, 1.0))
    res_norm     = np.where(np.isnan(ticket_res_raw), 0.0,
                            np.clip(ticket_res_raw / 48.0, 0.0, 1.0))
    has_tickets  = (~np.isnan(ticket_count_raw)).astype(float)
    support_friction_score = np.clip(
        has_tickets * (ticket_norm*0.6 + res_norm*0.4), 0.0, 1.0
    )

    # FG05_008 — ltv_score (FS-11: dynamic p95 ceiling replaces hardcoded $500)
    # The fixed $500 ceiling saturated long-tenure premium users: observed max=1139.40
    # on the 123k run, avg=333.11. p95 computed on non-null values only; fallback to
    # $500 if fewer than 10 non-null LTV values exist (cold-start / focused mode).
    ltv_non_null = ltv_raw[~np.isnan(ltv_raw)]
    if len(ltv_non_null) >= 10:
        ltv_ceiling = float(np.percentile(ltv_non_null, 95))
        if ltv_ceiling < 1.0:            # degenerate — all users have near-zero LTV
            ltv_ceiling = 500.0
    else:
        ltv_ceiling = 500.0              # fallback for small/focused runs
    ltv_score = np.clip(
        np.where(np.isnan(ltv_raw), 0.0, ltv_raw / ltv_ceiling), 0.0, 1.0
    )

    # FG05_009 — account_health_score passthrough
    # NULL → 0.0 (null-as-signal: profile-unmatched users are unknown health, not medium health)
    account_health_score = np.clip(
        np.where(np.isnan(acct_health_raw), 0.0, acct_health_raw), 0.0, 1.0
    )

    # FG05_010 — campaign_receptivity (NULL → 0.0)
    campaign_receptivity = np.clip(
        np.where(np.isnan(campaign_raw), 0.0, campaign_raw), 0.0, 1.0
    )

    # FG05_011 — superfan_score: ppv_buyer_flag*0.6 + merchandise*0.4
    ppv_buyer_flag  = (~np.isnan(ppv_raw)).astype(float)           # 1 if purchased PPV at least once
    merch_flag      = np.where(np.isnan(merch_raw), 0.0, merch_raw)
    superfan_score  = np.clip(ppv_buyer_flag*0.6 + merch_flag*0.4, 0.0, 1.0)

    # FG05_012 — plan_trajectory_score
    plan_trajectory_score = np.array([
        PLAN_TRAJECTORY_SCORE.get(pt.strip().lower(), 0.5) for pt in plan_traj_raw
    ])

    # FG05_013 — lifecycle_stage_score
    lifecycle_stage_score = np.array([
        LIFECYCLE_STAGE_SCORE.get(ls.strip().lower(), 0.5) for ls in lifecycle_raw
    ])

    # ── Assemble ──────────────────────────────────────────────────────────────
    feature_df = pd.DataFrame({
        # Identity keys
        "event_id":            df["event_id"].values,
        "user_id":             df["user_id"].values,
        "content_id":          df["content_id"].values,
        "segment_id":          df["segment_id"].values,
        "session_number":      df["session_number"].values if "session_number" in df.columns else 0,
        "event_type_raw":      df["event_type"].values,
        "timestamp":           df["timestamp"].values,
        "feature_version":     FEATURE_STORE_VERSION,
        "feature_computed_at": _now(),

        # FG-01: Session Engagement
        "session_depth_score":       _round(session_depth_score),
        "session_intensity_score":   _round(session_intensity_score),
        "binge_signal":              _round(binge_signal),
        "attention_quality_score":   _round(attention_quality_score),
        "rewatch_engagement_flag":   rewatch_engagement_flag.tolist(),
        "binge_index_score":         _round(binge_index_score),

        # FG-02: Completion & Retention
        "completion_rate_smooth":       _round(completion_rate_smooth),
        "completion_tier":              completion_tier.tolist(),
        "completion_variance_signal":   _round(completion_variance_signal),
        "recency_adjusted_completion":  _round(recency_adjusted_completion),
        "days_since_last_normalised":   _round(days_since_last_normalised),
        "attention_decay_curve":        _round(attention_decay_curve),
        "avg_watch_gap_norm":           _round(avg_watch_gap_norm),

        # FG-03: Churn & Friction
        "churn_risk_score":       _round(churn_risk_score),
        "churn_velocity":         _round(churn_velocity),
        "churn_flag_encoded":     churn_flag_encoded.tolist(),
        "reactivation_signal":    _round(reactivation_signal),
        "friction_index":         _round(friction_index),
        "satisfaction_score":     _round(satisfaction_score),
        "satisfaction_trend":     _round(satisfaction_trend),
        "drop_pattern_score":     _round(drop_pattern_score),

        # FG-04: Content Affinity
        "content_engagement_score":  _round(content_engagement_score),
        "content_type_weight":       _round(content_type_weight),
        "viewing_context_score":     _round(viewing_context_score),
        "weekend_viewing_flag":      weekend_viewing_flag.tolist(),
        "event_type_weight":         _round(event_type_weight),
        "fav_genre_confidence":      _round(fav_genre_confidence),
        "episode_position_score":    _round(episode_position_score),

        # FG-05: Contextual Modifiers
        "recency_weight":           _round(recency_weight),
        "subscription_tier_score":  _round(subscription_tier_score),
        "tenure_weight":            _round(tenure_weight),
        "network_quality_score":    _round(network_quality_score),
        "geo_tier":                 geo_tier.tolist(),
        "high_value_churn_flag":    high_value_churn_flag.tolist(),
        "support_friction_score":   _round(support_friction_score),
        "ltv_score":                _round(ltv_score),
        "account_health_score":     _round(account_health_score),
        "campaign_receptivity":     _round(campaign_receptivity),
        "superfan_score":           _round(superfan_score),
        "plan_trajectory_score":    _round(plan_trajectory_score),
        "lifecycle_stage_score":    _round(lifecycle_stage_score),
    })

    print(f"[INFO] Feature matrix shape: {feature_df.shape}")
    return feature_df


# ── Baseline computation ──────────────────────────────────────────────────────

def compute_baseline(feature_df: pd.DataFrame) -> dict:
    """
    90-day per-segment per-content baseline statistics.
    Consumed by Layer 7 normalization and Layer 4 twin prior calibration.

    Minimum sample guard: cells with n < MIN_BASELINE_N are skipped.
    p90 and max from n=1 cells are degenerate — they equal the single value
    and corrupt Layer 7 normalization if written. Threshold: 10 rows.
    """
    print("[INFO] Computing baseline statistics...")

    MIN_BASELINE_N = 10  # FS-10: minimum rows before writing any baseline cell

    BASELINE_FEATURES = [
        "completion_rate_smooth", "churn_risk_score", "churn_velocity",
        "friction_index", "recency_adjusted_completion",
        "session_depth_score", "session_intensity_score",
        "content_engagement_score", "attention_decay_curve",
        "satisfaction_score", "drop_pattern_score", "support_friction_score",
    ]

    baseline: dict = {}
    skipped_cells = 0
    for seg in feature_df["segment_id"].unique():
        baseline[seg] = {}
        for content in feature_df["content_id"].unique():
            mask   = (feature_df["segment_id"]==seg) & (feature_df["content_id"]==content)
            subset = feature_df[mask]
            if len(subset) < MIN_BASELINE_N:
                skipped_cells += 1
                continue
            stats: dict = {}
            for feat in BASELINE_FEATURES:
                if feat not in subset.columns:
                    continue
                s = pd.to_numeric(subset[feat], errors="coerce").dropna()
                if len(s) < MIN_BASELINE_N:
                    continue
                stats[feat] = {
                    "mean":   round(float(s.mean()),   4),
                    "median": round(float(s.median()), 4),
                    "p90":    round(float(s.quantile(0.90)), 4),
                    "max":    round(float(s.max()),    4),
                    "n":      len(s),
                }
            if stats:
                baseline[seg][content] = stats

    n_cells = sum(len(v) for v in baseline.values())
    print(f"[INFO] Baseline: {len(baseline)} segments × {n_cells} segment-content cells "
          f"({skipped_cells} skipped — n < {MIN_BASELINE_N})")
    return baseline


# ── Output writers ────────────────────────────────────────────────────────────

def write_registry(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "version":        FEATURE_STORE_VERSION,
            "generated_at":   _now(),
            "total_features": len(FEATURE_REGISTRY),
            "feature_groups": sorted(set(f["group"] for f in FEATURE_REGISTRY)),
            "new_in_v3": [
                "binge_index_score", "attention_decay_curve", "avg_watch_gap_norm",
                "satisfaction_score", "satisfaction_trend", "drop_pattern_score",
                "support_friction_score",
            ],
            "new_in_v4": [
                "ltv_score", "account_health_score", "fav_genre_confidence",
                "superfan_score", "campaign_receptivity", "plan_trajectory_score",
                "lifecycle_stage_score", "episode_position_score",
            ],
            "changed_in_v4": [
                "attention_quality_score (pause_count→buffer_rate)",
                "friction_index (pause_count removed, network_stress_flag added)",
                "drop_pattern_score (early/mid_drop_flag not rate aggregates)",
                "network_quality_score (session-level not profile-level)",
            ],
            "features": FEATURE_REGISTRY,
        }, f, indent=2)
    print(f"[DONE] Feature registry → {path}")


def write_report(fdf: pd.DataFrame, path: str) -> None:
    numeric_cols = [c for c in FEATURE_NAMES if c in fdf.columns]
    lines = [
        "=" * 80,
        f"TWINSIM — BEHAVIORAL FEATURE STORE REPORT  v{FEATURE_STORE_VERSION}",
        "=" * 80,
        f"Version         : {FEATURE_STORE_VERSION}",
        f"Records         : {len(fdf):,}",
        f"Users           : {fdf['user_id'].nunique():,}",
        f"Features        : {len(numeric_cols)}",
        f"Computed at     : {_now()}",
        f"Segments        : {fdf['segment_id'].nunique()} "
        f"({', '.join(sorted(fdf['segment_id'].unique()))})",
        f"Content titles  : {fdf['content_id'].nunique()}",
        "",
        f"{'Feature':<34} {'Min':>7} {'Mean':>7} {'Max':>7} {'Null%':>6}  Source",
        "-" * 80,
    ]

    TRAJECTORY = {"binge_index_score","attention_decay_curve","avg_watch_gap_norm",
                  "satisfaction_trend","fav_genre_confidence","account_health_score",
                  "ltv_score","plan_trajectory_score","lifecycle_stage_score"}
    SPARSE     = {"satisfaction_score","support_friction_score",
                  "campaign_receptivity","superfan_score"}

    for col in numeric_cols:
        s = pd.to_numeric(fdf[col], errors="coerce")
        null_pct = s.isna().mean() * 100
        tag = " [T]" if col in TRAJECTORY else (" [S]" if col in SPARSE else "    ")
        lines.append(
            f"{col:<34} {s.min():>7.4f} {s.mean():>7.4f} {s.max():>7.4f} "
            f"{null_pct:>5.1f}%{tag}"
        )

    lines += ["", "[T] = trajectory feature (user-level, multi-session)  "
              "[S] = sparse-safe (NULL→0.0)"]

    # Segment summary table
    lines += ["", "── Segment behavioral summary " + "─" * 50,
              f"  {'Segment':<26} {'Completion':>10} {'ChurnRisk':>10} "
              f"{'DecayCurve':>11} {'SatScore':>9}"]
    lines.append("  " + "-" * 70)
    for seg in sorted(fdf["segment_id"].unique()):
        m    = fdf["segment_id"] == seg
        comp = pd.to_numeric(fdf.loc[m,"completion_rate_smooth"], errors="coerce").mean()
        cr   = pd.to_numeric(fdf.loc[m,"churn_risk_score"],       errors="coerce").mean()
        dc   = pd.to_numeric(fdf.loc[m,"attention_decay_curve"],  errors="coerce").mean()
        ss   = pd.to_numeric(fdf.loc[m,"satisfaction_score"],     errors="coerce").mean()
        lines.append(
            f"  {seg:<26} {comp:>10.4f} {cr:>10.4f} {dc:>11.4f} {ss:>9.4f}"
        )

    # Sparse fill rates
    lines += ["", "── Sparse field fill rates " + "─" * 53]
    for col, label in [("satisfaction_score","content_satisfaction rated"),
                       ("support_friction_score","users with tickets")]:
        if col not in fdf.columns:
            continue
        s = pd.to_numeric(fdf[col], errors="coerce")
        non_zero = float((s > 0).mean() * 100)
        lines.append(f"  {col:<34} {non_zero:.1f}% non-zero")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[DONE] Feature store report → {path}")


def write_baseline(baseline: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "version":      FEATURE_STORE_VERSION,
            "generated_at": _now(),
            "description":  (
                "90-day per-segment per-content baseline. "
                "Consumed by Layer 7: clip((raw - baseline_90d) / (max_observed - baseline_90d), 0, 1)."
            ),
            "baseline": baseline,
        }, f, indent=2)
    print(f"[DONE] Baseline → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="TwinSim Behavioral Feature Store v4.0 — Layer 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--events",   default="session_events.csv",
                    help="session_events.csv from generate_signals.py")
    ap.add_argument("--profiles", default="user_profiles.csv",
                    help="user_profiles.csv from generate_signals.py")
    ap.add_argument("--out",      default="feature_store.csv")
    ap.add_argument("--registry", default="feature_registry.json")
    ap.add_argument("--report",   default="feature_store_report.txt")
    ap.add_argument("--baseline_out", default="baseline.json")
    ap.add_argument("--content_refs", nargs="+", default=None,
                    help="Focused mode: one or more content IDs.")
    ap.add_argument("--segments",     nargs="+", default=None,
                    help="Focused mode: one or more segment IDs.")
    ap.add_argument("--generate_baseline", action="store_true",
                    help="Compute 90-day baseline for Layer 7 normalization.")
    args = ap.parse_args()

    # Load
    print(f"[INFO] Loading {args.events}...")
    sess = pd.read_csv(args.events)
    print(f"[INFO] session_events: {len(sess):,} rows × {len(sess.columns)} cols")

    print(f"[INFO] Loading {args.profiles}...")
    prof = pd.read_csv(args.profiles)
    print(f"[INFO] user_profiles : {len(prof):,} rows × {len(prof.columns)} cols")

    # Join
    df = join_tables(sess, prof)

    # Focused mode filtering (applied to joined table)
    if args.content_refs:
        before = len(df)
        df = df[df["content_id"].isin(args.content_refs)].reset_index(drop=True)
        print(f"[INFO] content_refs filter: {before:,} → {len(df):,} ({args.content_refs})")
    if args.segments:
        before = len(df)
        df = df[df["segment_id"].isin(args.segments)].reset_index(drop=True)
        print(f"[INFO] segments filter: {before:,} → {len(df):,} ({args.segments})")

    if len(df) == 0:
        print("[ERROR] No rows after filtering. Check --content_refs / --segments.")
        raise SystemExit(1)

    # Compute
    fdf = compute_features(df)
    fdf.to_csv(args.out, index=False)
    print(f"[DONE] Feature store → {args.out}")

    write_registry(args.registry)
    write_report(fdf, args.report)

    if args.generate_baseline:
        # XL-03: baseline mode is intended for 90-day data. Running against <60 days
        # produces degenerate p90/max statistics that corrupt Layer 7 normalization.
        if "window_day" in fdf.columns:
            max_day = int(pd.to_numeric(fdf["window_day"], errors="coerce").max())
            if max_day < 60:
                print(
                    f"[ERROR] --generate_baseline requires at least 60 days of data. "
                    f"Max window_day in this dataset: {max_day}. "
                    f"Run generate_signals.py with --window_days 90 before computing baseline."
                )
                raise SystemExit(1)
        baseline = compute_baseline(fdf)
        write_baseline(baseline, args.baseline_out)

    print(f"\n[NEXT] python feature_store_assessment.py --input {args.out}")


if __name__ == "__main__":
    main()