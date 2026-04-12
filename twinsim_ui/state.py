"""EXL TwinSim Pipeline — Global Application State."""

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List

import reflex as rx
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
DATA_DIR     = PIPELINE_DIR / "data"
UPLOAD_DIR   = PROJECT_ROOT / "uploaded_files"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

# ── Health check descriptions (plain English for business stakeholders) ───────
CHECK_DESCRIPTIONS = {
    "A1":               "No duplicate events in session data",
    "A2":               "Required session fields have no missing values",
    "A3":               "All session fields have the correct data type",
    "A4":               "All session numeric values are within allowed ranges",
    "A5":               "Timestamps are valid and sessions are recent",
    "A6":               "Drop rates and completion % add up correctly",
    "A7":               "Churn, completion and reactivation events are logically consistent",
    "A8":               "Sessions are numbered in the correct time order per user",
    "A9":               "Customer segments are balanced within expected proportions",
    "A10":              "Binge watchers complete more content than churners",
    "A11":              "Satisfaction ratings are present and within the 1–5 scale",
    "A12":              "Skip-intro and rewatch flags are logically valid per content type",
    "A13":              "Cast-to-TV flag is only set for devices that support casting",
    "A14":              "Higher broadband speed correlates with less buffering",
    "A15":              "Early and mid-drop flags match the completion percentage",
    "A16":              "Days since last session is blank for a user's very first session",
    "A17":              "Number of sessions stays within per content-type limits",
    "A18":              "Network stress flag aligns with jitter and buffer readings",
    "A19":              "Episode position never exceeds the total episodes in a series",
    "A_unavailability": "Geo-blocked sessions have zero watch activity and no satisfaction score",
    "B15":              "No duplicate user IDs in profile data",
    "B16":              "Required profile fields have no missing values",
    "B17":              "All profile numeric values are within valid ranges",
    "B18":              "Subscription price matches the plan type, bundle and discount",
    "B19":              "Engagement scores and lifetime value are complete and in range",
    "B20":              "New profile fields contain only valid, expected values",
    "B21":              "Support ticket fields are consistent and within expected fill rate",
    "B22":              "Users with failed payments raise more support tickets",
    "B23":              "High-churn users raise more tickets than loyal binge watchers",
    "B24":              "Binge watchers have a higher binge score than quick churners",
    "B25":              "Smart TV users do not have casting preference switched on",
    "C25":              "Every session record has a matching user profile",
    "C26":              "Every user profile has at least one session record",
}

# ── Feature assessment descriptions (plain English for business stakeholders) ──
ASSESS_DESCRIPTIONS = {
    "Feature Completeness":          "All expected ML features are present in the feature store",
    "Null Check":                    "No missing values exist across any of the engineered features",
    "Range Check":                   "All numeric features fall within their declared valid ranges",
    "Variance Check":                "Every clustering feature has enough variation to be useful",
    "Churn Risk Consistency":        "Users who churned score higher on churn risk than active users",
    "Segment Completion Ordering":   "Binge watchers complete more content than quick churners",
    "Recency Weight":                "Every session has been assigned a positive recency score",
    "High Value Churn Flag":         "High-value churn events correctly meet all three required conditions",
    "Sparse Feature Handling":       "Optional features such as satisfaction and campaign scores handle blanks safely",
    "Episode Position Score":        "Episode position scores contain only the three allowed values",
    "Trajectory Score Validity":     "Lifecycle stage and plan trajectory scores are complete and valid",
    "Superfan Ltv Correlation":      "Superfan users generate higher lifetime value than non-superfans",
    "Collinearity Check":            "No two features are so similar that one becomes redundant for clustering",
    "Hdbscan Restoration Trigger":   "Checks whether the data needs a different clustering algorithm",
}

def _check_sort_key(grp: str) -> tuple:
    """Sort check groups: A1 < A2 < A19 < A_unavailability < B15 < C25."""
    import re
    prefix_order = {"A": 0, "B": 1, "C": 2}
    letter = grp[0].upper()
    rest   = grp[1:].lstrip("_")
    m      = re.match(r"^(\d+)", rest)
    num    = int(m.group(1)) if m else 999
    return (prefix_order.get(letter, 9), num)

# ── Persona label → display constants ─────────────────────────────────────────
PERSONA_NAMES = {
    "binge_heavy":         "The Binge Watcher",
    "completion_obsessed": "The Completionist",
    "casual_dip":          "The Casual Dipper",
    "quick_churn":         "The Quick Churner",
    "re_engager":          "The Re-Engager",
    "mixed_behavior":      "The Mixed-Mode Viewer",
}

PERSONA_AVATARS = {
    "binge_heavy":         "📺",
    "completion_obsessed": "🏆",
    "casual_dip":          "🌊",
    "quick_churn":         "🏃",
    "re_engager":          "🔁",
    "mixed_behavior":      "🎲",
}

PERSONA_DESCRIPTIONS = {
    "quick_churn":
        "A long-tenured subscriber whose engagement has quietly collapsed. "
        "They open the app, browse briefly, and leave before finishing anything. "
        "Despite high LTV, their sessions grow shallower each week — one missed "
        "re-engagement moment from cancelling for good.",
    "completion_obsessed":
        "A dedicated viewer who never leaves a series unfinished. They work "
        "methodically from premiere to finale in long focused sessions. The "
        "platform's most loyal users — they upgrade plans without prompting and "
        "are natural brand advocates.",
    "casual_dip":
        "A mood-driven viewer who drops in for a film or live match but rarely "
        "commits to a series. Their viewing is unpredictable and they haven't "
        "built a content habit, making them highly susceptible to churn when the "
        "next billing cycle arrives.",
    "re_engager":
        "A lapsed subscriber who returned after inactivity. They're cautiously "
        "rebuilding their viewing habit with growing session depth. Small in "
        "number but disproportionately valuable — the right content nudge converts "
        "them into long-term loyal users.",
    "binge_heavy":
        "An intense viewer who consumes content in long, uninterrupted sessions "
        "with strong completion signals. They invest deeply in the catalogue and "
        "represent the highest engagement tier — prime candidates for superfan "
        "and upgrade offers.",
    "mixed_behavior":
        "An unpredictable viewer who switches between devices, genres, and viewing "
        "times with no consistent routine. Their sessions are erratic, completion "
        "is low, and they carry the highest churn risk — needing personalised "
        "intervention before their next billing cycle.",
}

# ── Feature category definitions ───────────────────────────────────────────────
FEATURE_CATEGORIES = {
    "Identity Keys": [
        "event_id", "user_id", "session_id", "content_id", "segment_id",
    ],
    "Core Viewing Behaviour": [
        "completion_pct", "session_depth", "session_duration_mins",
        "early_drop_rate", "mid_session_drop_rate", "early_drop_flag",
        "mid_drop_flag", "skip_intro_flag", "rewatch_flag",
        "content_satisfaction", "days_since_last_session",
        "churn_flag", "reactivation_flag",
    ],
    "Content Metadata": [
        "content_type", "genre", "content_duration_minutes",
        "is_live_event", "is_exclusive", "franchise_flag", "episode_position",
    ],
    "Device & Network": [
        "device_type", "network_type", "buffer_events", "avg_bitrate_mbps",
        "avg_network_jitter", "peak_hour_congestion_flag",
        "network_stress_flag", "casting_usage_flag",
    ],
    "Subscription & Lifecycle": [
        "plan_type", "monthly_price", "tenure_months", "ltv_to_date",
        "lifecycle_stage", "discount_flag", "is_bundle",
        "upgrade_downgrade_history",
    ],
    "Engagement & Campaign": [
        "account_health_score", "campaign_response_rate",
        "push_notification_opt_in", "email_open_rate", "email_click_rate",
        "content_discovery_source", "ab_test_group_history", "promo_code_usage",
    ],
    "User Attributes": [
        "geo_country", "primary_device", "secondary_device", "fav_genre",
        "fav_genre_confidence", "binge_index", "sports_dependency_score",
        "avg_watch_gap_days", "payment_failure_count", "ticket_count",
    ],
}


# ── Sync helpers ───────────────────────────────────────────────────────────────
def _run_cmd(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
class PipelineState(rx.State):

    # ── Navigation ─────────────────────────────────────────────────────────────
    active_page: str = "upload"

    # ── Step statuses: 6 steps now ─────────────────────────────────────────────
    step_status: List[str] = ["pending","pending","pending","pending","pending","pending"]

    # ── Upload ─────────────────────────────────────────────────────────────────
    sessions_filename:  str  = ""
    profiles_filename:  str  = ""
    catalogue_filename: str  = ""
    sessions_path:      str  = ""
    profiles_path:      str  = ""
    catalogue_path:     str  = ""
    upload_error:       str  = ""
    all_uploaded:       bool = False

    # ── Step 2: Ingestion ───────────────────────────────────────────────────────
    s2_profiles_rows:     int        = 0
    s2_profiles_rows_fmt: str        = ""
    s2_profiles_cols:     int        = 0
    s2_sessions_rows:     int        = 0
    s2_sessions_rows_fmt: str        = ""
    s2_sessions_cols:     int        = 0
    s2_catalogue_rows:    int        = 0
    s2_catalogue_rows_fmt: str       = ""
    s2_catalogue_cols:    int        = 0
    s2_category_rows:     List[dict] = []
    s2_expanded_categories: List[str] = []   # which category rows are expanded
    s2_missing_features:  List[str]  = []
    s2_coverage_pct:      float      = 0.0
    s2_feature_checked:   bool       = False
    s2_gate_passed:       bool       = False
    s2_running:           bool       = False
    s2_error:             str        = ""
    s2_merge_running:     bool       = False
    s2_merge_done:        bool       = False
    s2_enriched_rows:     int        = 0
    s2_enriched_rows_fmt: str        = ""
    s2_enriched_cols:     int        = 0
    s2_merge_error:       str        = ""
    s2_enrich_path:       str        = ""

    # ── Step 3: Health ──────────────────────────────────────────────────────────
    s3_total_checks:    int        = 0
    s3_passed:          int        = 0
    s3_warned:          int        = 0
    s3_failed:          int        = 0
    s3_check_rows:      List[dict] = []
    s3_expanded_checks: List[str]  = []
    s3_gate_passed:     bool       = False
    s3_running:         bool       = False
    s3_error:           str        = ""
    s3_report_path:     str        = ""

    # ── Step 4: Feature Engineering ─────────────────────────────────────────────
    s4_total_features:     int        = 0
    s4_users:              int        = 0
    s4_users_fmt:          str        = ""
    s4_assessment_rows:    List[dict] = []
    s4_all_passed:         bool       = False
    s4_gate_passed:        bool       = False
    s4_running:            bool       = False
    s4_error:              str        = ""
    s4_feature_store_path: str        = ""
    s4_expanded_assess:    List[str]  = []

    # ── Step 5: Clustering ──────────────────────────────────────────────────────
    s5_running:           bool       = False
    s5_error:             str        = ""
    s5_done:              bool       = False
    s5_n_clusters:        int        = 0
    s5_n_events:          int        = 0
    s5_n_events_fmt:      str        = ""
    s5_noise_events:      int        = 0
    s5_noise_events_fmt:  str        = ""
    s5_noise_pct:        float      = 0.0
    s5_stability_label:  str        = ""
    s5_silhouette:       float      = 0.0
    s5_algorithm:        str        = ""
    s5_cluster_rows:     List[dict] = []
    s5_audit_rows:       List[dict] = []
    s5_audit_passed:     bool       = False
    s5_gate_passed:      bool       = False
    s5_assignments_path: str        = ""
    s5_metadata_path:    str        = ""

    # ── Step 6: Persona Intelligence ────────────────────────────────────────────
    s6_running:              bool       = False
    s6_error:                str        = ""
    s6_done:                 bool       = False
    s6_personas:             List[dict] = []   # overview card data
    s6_active_persona_idx:   int        = -1   # -1 = overview, >=0 = detail
    # Active persona detail vars (populated by set_active_persona)
    s6_active_name:          str        = ""
    s6_active_label:         str        = ""
    s6_active_avatar:        str        = ""
    s6_active_archetype:     str        = ""
    s6_active_size:          int        = 0
    s6_active_size_pct:      float      = 0.0
    s6_active_churn:         float      = 0.0
    s6_active_churn_level:   str        = "low"   # "high" | "medium" | "low"
    s6_active_churn_ci_lo:   float      = 0.0
    s6_active_churn_ci_hi:   float      = 0.0
    s6_active_completion:    float      = 0.0
    s6_active_reactivation:  float      = 0.0
    s6_active_ltv_tier:      str        = ""
    s6_active_depth_tier:    str        = ""
    s6_active_churn_tier:    str        = ""
    s6_active_confidence:    str        = ""
    s6_active_narrative:     str        = ""
    s6_active_strategic_rec: str        = ""
    s6_active_content:       str        = ""
    s6_active_arc:           str        = ""
    s6_active_description:   str        = ""
    # Risk flags
    s6_active_high_churn:    bool       = False
    s6_active_high_value:    bool       = False
    s6_active_reactivation_candidate: bool = False
    s6_active_superfan:      bool       = False
    s6_active_marketing:     bool       = False
    s6_active_upgrade:       bool       = False
    # Intervention playbook
    s6_active_interventions: List[dict] = []
    # Feature fingerprint
    s6_active_features:      List[dict] = []
    # Segment distribution
    s6_active_segments:      List[dict] = []

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def go_to(self, page: str):
        self.active_page = page

    def toggle_category(self, category: str):
        """Expand or collapse a feature category row."""
        if category in self.s2_expanded_categories:
            self.s2_expanded_categories = [
                c for c in self.s2_expanded_categories if c != category
            ]
        else:
            self.s2_expanded_categories = self.s2_expanded_categories + [category]

    def toggle_check(self, check_group: str):
        """Expand or collapse a health check reason row."""
        self.s3_check_rows = [
            {**row, "expanded": "false" if row["expanded"] == "true" else "true"}
            if row["check_group"] == check_group
            else row
            for row in self.s3_check_rows
        ]

    def toggle_assess(self, check: str):
        """Expand or collapse a feature assessment reason row."""
        self.s4_assessment_rows = [
            {**row, "expanded": "false" if row["expanded"] == "true" else "true"}
            if row["check"] == check
            else row
            for row in self.s4_assessment_rows
        ]

    def _refresh_upload_flag(self):
        self.all_uploaded = bool(
            self.sessions_path and
            self.profiles_path and
            self.catalogue_path
        )

    # ─────────────────────────────────────────────────────────────────────────
    # UPLOAD
    # ─────────────────────────────────────────────────────────────────────────
    async def _save_upload(self, f: rx.UploadFile, dest) -> None:
        """Write an uploaded file to disk in 1 MB chunks to handle large files."""
        CHUNK = 1024 * 1024  # 1 MB
        with open(dest, "wb") as out:
            while True:
                chunk = await f.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)

    async def handle_sessions_upload(self, files: list[rx.UploadFile]):
        self.upload_error = ""
        if not files:
            return
        f = files[0]
        dest = UPLOAD_DIR / f.filename
        await self._save_upload(f, dest)
        self.sessions_filename = f.filename
        self.sessions_path = str(dest)
        self._refresh_upload_flag()

    async def handle_profiles_upload(self, files: list[rx.UploadFile]):
        self.upload_error = ""
        if not files:
            return
        f = files[0]
        dest = UPLOAD_DIR / f.filename
        await self._save_upload(f, dest)
        self.profiles_filename = f.filename
        self.profiles_path = str(dest)
        self._refresh_upload_flag()

    async def handle_catalogue_upload(self, files: list[rx.UploadFile]):
        self.upload_error = ""
        if not files:
            return
        f = files[0]
        dest = UPLOAD_DIR / f.filename
        await self._save_upload(f, dest)
        self.catalogue_filename = f.filename
        self.catalogue_path = str(dest)
        self._refresh_upload_flag()

    def ingest_data(self):
        if not self.all_uploaded:
            self.upload_error = "Please upload all three files before proceeding."
            return
        self.active_page = "ingest"
        return PipelineState.run_ingestion

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2a — INGESTION
    # ─────────────────────────────────────────────────────────────────────────
    async def run_ingestion(self):
        sessions_path  = self.sessions_path
        profiles_path  = self.profiles_path
        catalogue_path = self.catalogue_path

        self.s2_running         = True
        self.s2_feature_checked = False
        self.s2_error           = ""
        self.step_status = ["complete","running","pending","pending","pending","pending"]
        yield

        try:
            loop = asyncio.get_running_loop()
            sessions, profiles, catalogue = await loop.run_in_executor(
                None,
                lambda: (
                    _load_csv(sessions_path),
                    _load_csv(profiles_path),
                    _load_csv(catalogue_path),
                ),
            )
            self.s2_profiles_rows      = int(profiles.shape[0])
            self.s2_profiles_rows_fmt  = f"{int(profiles.shape[0]):,}"
            self.s2_profiles_cols      = int(profiles.shape[1])
            self.s2_sessions_rows      = int(sessions.shape[0])
            self.s2_sessions_rows_fmt  = f"{int(sessions.shape[0]):,}"
            self.s2_sessions_cols      = int(sessions.shape[1])
            self.s2_catalogue_rows     = int(catalogue.shape[0])
            self.s2_catalogue_rows_fmt = f"{int(catalogue.shape[0]):,}"
            self.s2_catalogue_cols     = int(catalogue.shape[1])
            yield

            all_cols = set(sessions.columns) | set(profiles.columns) | set(catalogue.columns)
            cat_rows:    list[dict] = []
            missing_all: list[str]  = []
            total_exp = total_pre   = 0

            for cat, feats in FEATURE_CATEGORIES.items():
                check_feats = [f for f in feats if f != "episode_position"]
                n_pre  = sum(1 for f in check_feats if f in all_cols)
                missed = [f for f in check_feats if f not in all_cols]
                total_exp += len(check_feats)
                total_pre += n_pre
                missing_all.extend(missed)
                pct    = round(n_pre / len(check_feats) * 100) if check_feats else 100
                status = "green" if pct == 100 else ("amber" if pct >= 80 else "red")
                cat_rows.append({
                    "category":     cat,
                    "expected":     len(check_feats),
                    "present":      n_pre,
                    "missing":      len(check_feats) - n_pre,
                    "pct":          pct,
                    "status":       status,
                    "features_str": "  ·  ".join(check_feats),  # pre-joined for display
                })

            coverage = round(total_pre / total_exp * 100, 1) if total_exp else 0.0
            gate     = len(missing_all) == 0

            self.s2_category_rows    = cat_rows
            self.s2_missing_features = missing_all
            self.s2_coverage_pct     = coverage
            self.s2_gate_passed      = gate
            self.s2_feature_checked  = True
            self.s2_running          = False
            self.step_status = ["complete","running","pending","pending","pending","pending"]
            yield

        except Exception as exc:
            self.s2_error   = str(exc)
            self.s2_running = False
            self.step_status = ["complete","error","pending","pending","pending","pending"]
            yield

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2b — MERGE
    # ─────────────────────────────────────────────────────────────────────────
    async def run_merge(self):
        sessions_path  = self.sessions_path
        catalogue_path = self.catalogue_path
        enrich_out     = str(UPLOAD_DIR / "enriched_sessions.csv")

        self.s2_merge_running = True
        self.s2_merge_error   = ""
        yield

        try:
            loop = asyncio.get_running_loop()
            enrich_cmd = [
                PYTHON, str(PIPELINE_DIR / "enrich_sessions.py"),
                "--sessions", sessions_path,
                "--metadata", catalogue_path,
                "--out",      enrich_out,
            ]
            res = await loop.run_in_executor(None, lambda: _run_cmd(enrich_cmd, timeout=120))
            if res.returncode != 0:
                raise RuntimeError(
                    f"Session enrichment failed (exit {res.returncode}):\n"
                    f"{res.stderr or res.stdout}"
                )
            enriched = await loop.run_in_executor(None, lambda: _load_csv(enrich_out))
            self.s2_enriched_rows     = int(enriched.shape[0])
            self.s2_enriched_rows_fmt = f"{int(enriched.shape[0]):,}"
            self.s2_enriched_cols     = int(enriched.shape[1])
            self.s2_enrich_path   = enrich_out
            self.s2_merge_running = False
            self.s2_merge_done    = True
            self.step_status = ["complete","complete","pending","pending","pending","pending"]
            yield

        except Exception as exc:
            self.s2_merge_error   = str(exc)
            self.s2_merge_running = False
            self.step_status = ["complete","error","pending","pending","pending","pending"]
            yield

    def proceed_to_health(self):
        self.active_page = "health"
        return PipelineState.run_health_check

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — HEALTH CHECK
    # ─────────────────────────────────────────────────────────────────────────
    async def run_health_check(self):
        enrich_path   = self.s2_enrich_path
        profiles_path = self.profiles_path
        report_out    = str(UPLOAD_DIR / "health_report_run.csv")

        self.s3_running = True
        self.s3_error   = ""
        self.step_status = ["complete","complete","running","pending","pending","pending"]
        yield

        try:
            loop = asyncio.get_running_loop()
            cmd = [
                PYTHON, str(PIPELINE_DIR / "health_report.py"),
                "--events",        enrich_path,
                "--profiles",      profiles_path,
                "--dict_events",   str(DATA_DIR / "data_dictionary_session_events.json"),
                "--dict_profiles", str(DATA_DIR / "data_dictionary_user_profiles.json"),
                "--out",           report_out,
            ]
            res = await loop.run_in_executor(None, lambda: _run_cmd(cmd, timeout=300))
            if res.returncode != 0:
                raise RuntimeError(
                    f"health_report.py failed (exit {res.returncode}):\n"
                    f"{res.stderr or res.stdout}"
                )
            self.s3_report_path = report_out
            yield

            rdf = await loop.run_in_executor(None, lambda: _load_csv(report_out))
            total  = len(rdf)
            passed = int((rdf["status"] == "PASS").sum())
            warned = int((rdf["status"] == "WARN").sum())
            failed = int((rdf["status"] == "FAIL").sum())

            rdf["group"] = rdf["check"].str.extract(r"^([A-Za-z_]+\d*)")
            summary: list[dict] = []
            for grp, sub in rdf.groupby("group"):
                grp_str  = str(grp)
                n_pass   = int((sub["status"] == "PASS").sum())
                n_warn   = int((sub["status"] == "WARN").sum())
                n_fail   = int((sub["status"] == "FAIL").sum())
                # Overall status for the group
                grp_status = "FAIL" if n_fail > 0 else ("WARN" if n_warn > 0 else "PASS")
                # Collect reason text from non-passing rows (for expandable detail)
                bad_rows   = sub[sub["status"].isin(["FAIL", "WARN"])]
                reason_str = " · ".join(bad_rows["detail"].tolist()) if len(bad_rows) else ""
                summary.append({
                    "check_group": grp_str,
                    "description": CHECK_DESCRIPTIONS.get(grp_str, grp_str),
                    "passed":      n_pass,
                    "warned":      n_warn,
                    "failed":      n_fail,
                    "status":      grp_status,
                    "reason":      reason_str,
                    "expanded":    "false",
                })
            # Sort: A1 < A2 < ... < A19 < A_unavailability < B15 < C25
            summary.sort(key=lambda x: _check_sort_key(x["check_group"]))

            gate = failed == 0
            self.s3_total_checks = total
            self.s3_passed       = passed
            self.s3_warned       = warned
            self.s3_failed       = failed
            self.s3_check_rows   = summary
            self.s3_gate_passed  = gate
            self.s3_running      = False
            self.step_status = (
                ["complete","complete","complete","pending","pending","pending"] if gate
                else ["complete","complete","error","pending","pending","pending"]
            )
            yield

        except Exception as exc:
            self.s3_error   = str(exc)
            self.s3_running = False
            self.step_status = ["complete","complete","error","pending","pending","pending"]
            yield

    def proceed_to_features(self):
        self.active_page = "features"
        return PipelineState.run_feature_engineering

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────────────────────
    async def run_feature_engineering(self):
        enrich_path   = self.s2_enrich_path
        profiles_path = self.profiles_path
        fs_out        = str(UPLOAD_DIR / "feature_store_run.csv")

        self.s4_running = True
        self.s4_error   = ""
        self.step_status = ["complete","complete","complete","running","pending","pending"]
        yield

        try:
            loop = asyncio.get_running_loop()
            cmd_fs = [
                PYTHON, str(PIPELINE_DIR / "feature_store.py"),
                "--events",   enrich_path,
                "--profiles", profiles_path,
                "--out",      fs_out,
            ]
            res = await loop.run_in_executor(None, lambda: _run_cmd(cmd_fs, timeout=600))
            if res.returncode != 0:
                raise RuntimeError(
                    f"feature_store.py failed (exit {res.returncode}):\n"
                    f"{res.stderr or res.stdout}"
                )
            self.s4_feature_store_path = fs_out
            yield

            fsdf = await loop.run_in_executor(None, lambda: _load_csv(fs_out))
            self.s4_users          = int(fsdf.shape[0])
            self.s4_users_fmt      = f"{int(fsdf.shape[0]):,}"
            self.s4_total_features = int(fsdf.shape[1])
            yield

            cmd_assess = [
                PYTHON, str(PIPELINE_DIR / "feature_store_assessment.py"),
                "--input", fs_out,
            ]
            res2 = await loop.run_in_executor(None, lambda: _run_cmd(cmd_assess, timeout=120))

            assess_rows: list[dict] = []
            tag_map = {"[PASS]":"PASS","[FAIL]":"FAIL","[WARN]":"WARN","[SKIP]":"SKIP"}
            lines = res2.stdout.splitlines()
            for i, line in enumerate(lines):
                for tag, status in tag_map.items():
                    if tag in line:
                        raw_name = line.replace(tag, "").strip()
                        # Strip numeric prefix e.g. "01_feature_completeness" → "Feature Completeness"
                        m = re.match(r"^(\d+)_(.+)$", raw_name)
                        if m:
                            sort_num   = int(m.group(1))
                            clean_name = m.group(2).replace("_", " ").title()
                        else:
                            sort_num   = 999
                            clean_name = raw_name.replace("_", " ").title()
                        detail = ""
                        if i + 1 < len(lines):
                            nxt = lines[i + 1].strip()
                            if nxt and not any(
                                t in nxt for t in
                                ["[PASS]","[FAIL]","[WARN]","[SKIP]","===","OVERALL"]
                            ):
                                detail = nxt
                        assess_rows.append({
                            "check":       clean_name,
                            "description": ASSESS_DESCRIPTIONS.get(clean_name, clean_name),
                            "sort_num":    sort_num,
                            "status":      status,
                            "detail":      detail,
                            "expanded":    "false",
                        })
                        break
            # Sort ascending by original numeric prefix
            assess_rows.sort(key=lambda x: x["sort_num"])

            if not assess_rows:
                assess_rows = [{
                    "check":  "Assessment result",
                    "status": "PASS" if res2.returncode == 0 else "FAIL",
                    "detail": (res2.stdout or res2.stderr)[:400],
                }]

            all_pass = (res2.returncode == 0) and all(
                r["status"] in ("PASS","WARN","SKIP") for r in assess_rows
            )
            self.s4_assessment_rows = assess_rows
            self.s4_all_passed      = all_pass
            self.s4_gate_passed     = all_pass
            self.s4_running         = False
            self.step_status = (
                ["complete","complete","complete","complete","pending","pending"] if all_pass
                else ["complete","complete","complete","error","pending","pending"]
            )
            yield

        except Exception as exc:
            self.s4_error   = str(exc)
            self.s4_running = False
            self.step_status = ["complete","complete","complete","error","pending","pending"]
            yield

    def proceed_to_clustering(self):
        self.active_page = "clustering"
        return PipelineState.run_clustering

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — CLUSTERING
    # ─────────────────────────────────────────────────────────────────────────
    async def run_clustering(self):
        fs_path        = self.s4_feature_store_path
        assignments_out = str(UPLOAD_DIR / "cluster_assignments.csv")
        metadata_out    = str(UPLOAD_DIR / "cluster_metadata.json")
        report_out      = str(UPLOAD_DIR / "clustering_report.txt")

        self.s5_running = True
        self.s5_error   = ""
        self.s5_done    = False
        self.step_status = ["complete","complete","complete","complete","running","pending"]
        yield

        try:
            loop = asyncio.get_running_loop()

            # Run clustering engine
            cmd_cluster = [
                PYTHON, str(PIPELINE_DIR / "clustering_engine.py"),
                "--input",    fs_path,
                "--out",      assignments_out,
                "--metadata", metadata_out,
                "--report",   report_out,
            ]
            res = await loop.run_in_executor(None, lambda: _run_cmd(cmd_cluster, timeout=600))
            if res.returncode != 0:
                raise RuntimeError(
                    f"clustering_engine.py failed (exit {res.returncode}):\n"
                    f"{res.stderr or res.stdout}"
                )

            self.s5_assignments_path = assignments_out
            self.s5_metadata_path    = metadata_out
            yield

            # Parse cluster_metadata.json
            def _parse_metadata():
                with open(metadata_out, encoding="utf-8") as f:
                    return json.load(f)

            meta = await loop.run_in_executor(None, _parse_metadata)

            clusters   = meta.get("clusters", [])
            stability  = meta.get("stability", {})
            n_events   = int(meta.get("n_events", 0))
            noise_ev   = int(meta.get("noise_events", 0))
            noise_pct  = float(meta.get("noise_pct", 0.0))
            algorithm  = str(meta.get("algorithm", "kmeans"))
            sil        = float(stability.get("mean_silhouette", 0.0))
            stab_label = str(stability.get("stability_label", "UNKNOWN"))

            cluster_rows: list[dict] = []
            for cl in clusters:
                priors = cl.get("behavioral_priors", {})
                label  = cl.get("behavioral_label", "mixed_behavior")
                churn  = float(priors.get("base_churn_30d", 0.0))
                comp   = float(priors.get("base_completion", 0.0))
                react  = float(priors.get("base_reactivation", 0.0))
                size   = int(cl.get("size", 0))
                pct    = float(cl.get("cluster_share_pct", 0.0))
                churn_pct = round(churn * 100, 1)
                churn_level = "high" if churn_pct > 60 else ("medium" if churn_pct > 35 else "low")
                cluster_rows.append({
                    "label":       label,
                    "name":        PERSONA_NAMES.get(label, label.replace("_"," ").title()),
                    "avatar":      PERSONA_AVATARS.get(label, "👤"),
                    "size":        size,
                    "size_pct":    round(pct, 1),
                    "churn":       churn_pct,
                    "churn_level": churn_level,
                    "completion":  round(comp  * 100, 1),
                    "reactivation": round(react * 100, 1),
                })

            self.s5_n_clusters        = len(clusters)
            self.s5_n_events          = n_events
            self.s5_n_events_fmt      = f"{n_events:,}"
            self.s5_noise_events      = noise_ev
            self.s5_noise_events_fmt  = f"{noise_ev:,}"
            self.s5_noise_pct       = round(noise_pct, 1)
            self.s5_stability_label = stab_label
            self.s5_silhouette      = round(sil, 3)
            self.s5_algorithm       = algorithm.upper()
            self.s5_cluster_rows    = cluster_rows
            yield

            # Run clustering audit
            cmd_audit = [
                PYTHON, str(PIPELINE_DIR / "clustering_audit.py"),
                "--assignments", assignments_out,
                "--metadata",   metadata_out,
            ]
            res2 = await loop.run_in_executor(None, lambda: _run_cmd(cmd_audit, timeout=120))

            audit_rows: list[dict] = []
            tag_map = {"[PASS]":"PASS","[FAIL]":"FAIL","[WARN]":"WARN","[SKIP]":"SKIP"}
            for i, line in enumerate(res2.stdout.splitlines()):
                for tag, status in tag_map.items():
                    if tag in line:
                        check_name = line.replace(tag,"").strip()
                        detail = ""
                        lines2 = res2.stdout.splitlines()
                        if i + 1 < len(lines2):
                            nxt = lines2[i + 1].strip()
                            if nxt and not any(
                                t in nxt for t in
                                ["[PASS]","[FAIL]","[WARN]","[SKIP]","===","OVERALL","──"]
                            ):
                                detail = nxt
                        audit_rows.append({"check": check_name, "status": status, "detail": detail})
                        break

            if not audit_rows:
                audit_rows = [{
                    "check":  "Audit result",
                    "status": "PASS" if res2.returncode == 0 else "FAIL",
                    "detail": (res2.stdout or res2.stderr)[:300],
                }]

            audit_passed = res2.returncode == 0 and all(
                r["status"] in ("PASS","WARN","SKIP") for r in audit_rows
            )
            self.s5_audit_rows    = audit_rows
            self.s5_audit_passed  = audit_passed
            self.s5_gate_passed   = audit_passed
            self.s5_running       = False
            self.s5_done          = True
            self.step_status = (
                ["complete","complete","complete","complete","complete","pending"] if audit_passed
                else ["complete","complete","complete","complete","error","pending"]
            )
            yield

        except Exception as exc:
            self.s5_error   = str(exc)
            self.s5_running = False
            self.step_status = ["complete","complete","complete","complete","error","pending"]
            yield

    def proceed_to_persona(self):
        self.active_page = "persona"
        return PipelineState.run_persona_engine

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — PERSONA ENGINE
    # ─────────────────────────────────────────────────────────────────────────
    async def run_persona_engine(self):
        metadata_path   = self.s5_metadata_path
        personas_out    = str(UPLOAD_DIR / "personas.json")
        report_out      = str(UPLOAD_DIR / "persona_report.txt")

        self.s6_running = True
        self.s6_error   = ""
        self.s6_done    = False
        self.s6_active_persona_idx = -1
        self.step_status = ["complete","complete","complete","complete","complete","running"]
        yield

        try:
            loop = asyncio.get_running_loop()

            cmd = [
                PYTHON, str(PIPELINE_DIR / "persona_engine.py"),
                "--metadata", metadata_path,
                "--out",      personas_out,
                "--report",   report_out,
                "--validate",
            ]
            res = await loop.run_in_executor(None, lambda: _run_cmd(cmd, timeout=300))
            if res.returncode != 0:
                raise RuntimeError(
                    f"persona_engine.py failed (exit {res.returncode}):\n"
                    f"{res.stderr or res.stdout}"
                )
            yield

            # Parse personas.json
            def _parse_personas():
                with open(personas_out, encoding="utf-8") as f:
                    return json.load(f)

            data = await loop.run_in_executor(None, _parse_personas)
            raw_personas = data.get("personas", [])

            cards: list[dict] = []
            for p in raw_personas:
                label   = p.get("behavioral_label", "mixed_behavior")
                priors  = p.get("behavioral_priors", {})
                sim     = p.get("simulation_priors", {})
                flags   = p.get("risk_flags", {})
                profile = p.get("behavioral_profile", {})
                routing = p.get("content_routing", {})
                feat_p  = p.get("feature_profile", {})
                seg_d   = p.get("segment_distribution", {})
                interv  = p.get("intervention_playbook", [])

                churn  = float(priors.get("base_churn_30d", 0.0))
                comp   = float(priors.get("base_completion", 0.0))
                react  = float(priors.get("base_reactivation", 0.0))
                ci     = sim.get("base_churn_30d_ci", [churn, churn])

                # Flatten interventions
                flat_interventions: list[dict] = []
                for iv in interv:
                    lift = iv.get("simulated_lift", {})
                    flat_interventions.append({
                        "name":       str(iv.get("name", "")),
                        "channel":    " + ".join(iv.get("channel", [])),
                        "trigger":    str(iv.get("trigger_condition", iv.get("trigger", ""))),
                        "priority":   int(iv.get("priority", 1)),
                        "rationale":  str(iv.get("rationale", "")),
                        "ret_lift":   float(lift.get("retention_lift", 0.0)),
                        "eng_lift":   float(lift.get("engagement_lift", 0.0)),
                        "react_lift": float(lift.get("reactivation_lift", 0.0)),
                        "ltv_lift":   float(lift.get("ltv_lift", 0.0)),
                    })

                # Feature fingerprint (top 10 by value, sorted desc)
                feat_items = sorted(
                    [{"name": k, "val": round(float(v), 3), "pct": round(float(v) * 100, 1)}
                     for k, v in feat_p.items() if isinstance(v, (int, float))],
                    key=lambda x: x["val"], reverse=True
                )[:10]

                # Segment distribution
                seg_items = [
                    {"label": k, "pct": round(float(v) * 100, 1)}
                    for k, v in seg_d.items()
                    if isinstance(v, (int, float))
                ]

                churn_pct_p     = round(churn * 100, 1)
                churn_level_p   = "high" if churn_pct_p > 60 else ("medium" if churn_pct_p > 35 else "low")
                cards.append({
                    # Identity
                    "name":        p.get("persona_name", PERSONA_NAMES.get(label, label)),
                    "label":       label,
                    "avatar":      PERSONA_AVATARS.get(label, "👤"),
                    "archetype":   p.get("archetype", "").replace("_", " ").title(),
                    "description": PERSONA_DESCRIPTIONS.get(label, ""),
                    "size":        int(p.get("size", 0)),
                    "size_pct":    round(float(p.get("size_pct", 0.0)), 1),
                    # Priors
                    "churn":       churn_pct_p,
                    "churn_level": churn_level_p,
                    "completion":  round(comp  * 100, 1),
                    "reactivation": round(react * 100, 1),
                    "churn_ci_lo": round(float(ci[0]) * 100, 1) if ci else round(churn*100,1),
                    "churn_ci_hi": round(float(ci[1]) * 100, 1) if ci else round(churn*100,1),
                    "confidence":  str(sim.get("confidence_note", "high")),
                    # Tiers
                    "ltv_tier":    str(profile.get("ltv_tier", "")),
                    "depth_tier":  str(profile.get("depth_tier", "")),
                    "churn_tier":  str(profile.get("churn_tier", "")),
                    # Risk flags
                    "high_churn":  bool(flags.get("high_churn_risk", False)),
                    "high_value":  bool(flags.get("high_value", False)),
                    "reactivation_candidate": bool(flags.get("reactivation_candidate", False)),
                    "superfan":    bool(flags.get("superfan_potential", False)),
                    "marketing":   bool(flags.get("marketing_receptive", False)),
                    "upgrade":     bool(flags.get("upgrade_candidate", False)),
                    # Content
                    "preferred_content": ", ".join(routing.get("preferred_content_types", [])).title(),
                    "arc_affinity":      ", ".join(routing.get("content_arc_affinity", [])).title(),
                    "strategic_rec":     str(routing.get("strategic_recommendation", "")),
                    # Narrative
                    "narrative":   str(p.get("behavioral_narrative", "")),
                    # Interventions & fingerprint (for detail view)
                    "interventions": flat_interventions,
                    "features":      feat_items,
                    "segments":      seg_items,
                })

            self.s6_personas = cards
            self.s6_running  = False
            self.s6_done     = True
            self.step_status = ["complete","complete","complete","complete","complete","complete"]
            yield

        except Exception as exc:
            self.s6_error   = str(exc)
            self.s6_running = False
            self.step_status = ["complete","complete","complete","complete","complete","error"]
            yield

    def set_active_persona(self, idx: int):
        """Load persona detail vars from the selected persona card."""
        self.s6_active_persona_idx = idx
        if idx < 0 or idx >= len(self.s6_personas):
            return
        p = self.s6_personas[idx]
        self.s6_active_name          = p.get("name", "")
        self.s6_active_label         = p.get("label", "")
        self.s6_active_avatar        = p.get("avatar", "👤")
        self.s6_active_archetype     = p.get("archetype", "")
        self.s6_active_size          = p.get("size", 0)
        self.s6_active_size_pct      = p.get("size_pct", 0.0)
        self.s6_active_churn         = p.get("churn", 0.0)
        self.s6_active_churn_level   = p.get("churn_level", "low")
        self.s6_active_churn_ci_lo   = p.get("churn_ci_lo", 0.0)
        self.s6_active_churn_ci_hi   = p.get("churn_ci_hi", 0.0)
        self.s6_active_completion    = p.get("completion", 0.0)
        self.s6_active_reactivation  = p.get("reactivation", 0.0)
        self.s6_active_ltv_tier      = p.get("ltv_tier", "")
        self.s6_active_depth_tier    = p.get("depth_tier", "")
        self.s6_active_churn_tier    = p.get("churn_tier", "")
        self.s6_active_confidence    = p.get("confidence", "high")
        self.s6_active_narrative     = p.get("narrative", "")
        self.s6_active_strategic_rec = p.get("strategic_rec", "")
        self.s6_active_content       = p.get("preferred_content", "")
        self.s6_active_arc           = p.get("arc_affinity", "")
        self.s6_active_description   = p.get("description", "")
        self.s6_active_high_churn    = p.get("high_churn", False)
        self.s6_active_high_value    = p.get("high_value", False)
        self.s6_active_reactivation_candidate = p.get("reactivation_candidate", False)
        self.s6_active_superfan      = p.get("superfan", False)
        self.s6_active_marketing     = p.get("marketing", False)
        self.s6_active_upgrade       = p.get("upgrade", False)
        self.s6_active_interventions = p.get("interventions", [])
        self.s6_active_features      = p.get("features", [])
        self.s6_active_segments      = p.get("segments", [])

    def back_to_overview(self):
        self.s6_active_persona_idx = -1

    def download_persona_pdf(self, idx: int):
        """Generate and download a PDF report for the given persona."""
        import base64
        from fpdf import FPDF

        def _safe(text: str) -> str:
            """Replace Unicode characters that Helvetica cannot encode."""
            return (
                str(text)
                .replace("\u2014", "-")   # em dash  —
                .replace("\u2013", "-")   # en dash  –
                .replace("\u2019", "'")   # right single quote  '
                .replace("\u2018", "'")   # left single quote   '
                .replace("\u201c", '"')   # left double quote   "
                .replace("\u201d", '"')   # right double quote  "
                .replace("\u2022", "*")   # bullet  •
                .replace("\u2192", "->")  # arrow   →
                .replace("\u00e9", "e")   # é
                .replace("\u00e0", "a")   # à
                .replace("\u00fc", "u")   # ü
                .replace("\u00f6", "o")   # ö
                .encode("latin-1", errors="replace")
                .decode("latin-1")
            )

        if idx < 0 or idx >= len(self.s6_personas):
            return
        p = self.s6_personas[idx]

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ── Header bar ─────────────────────────────────────────────────────
        pdf.set_fill_color(232, 70, 30)
        pdf.rect(0, 0, 210, 18, "F")
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(10, 4)
        pdf.cell(0, 10, "EXL  |  Market Intelligence - Customer Persona Report")

        pdf.set_text_color(30, 30, 30)
        pdf.set_xy(10, 24)

        # ── Persona name & label ────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 10, _safe(p.get("name", "")), ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6,
                 _safe(
                     f"Segment: {p.get('label', '')}   |   "
                     f"Size: {p.get('size_pct', 0)}% of customer base   |   "
                     f"Churn level: {str(p.get('churn_level', '')).title()}"
                 ),
                 ln=True)
        pdf.ln(4)

        # ── Divider ─────────────────────────────────────────────────────────
        pdf.set_draw_color(220, 220, 220)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # ── Description ─────────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 7, "Persona Overview", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 5, _safe(p.get("description", "")))
        pdf.ln(5)

        # ── Key metrics ─────────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 7, "Key Metrics", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(60, 6, f"Churn Risk:       {p.get('churn', 0)}%")
        pdf.cell(60, 6, f"Completion Rate:  {p.get('completion', 0)}%")
        pdf.cell(60, 6, f"Re-engagement:    {p.get('reactivation', 0)}%", ln=True)
        pdf.ln(4)

        # ── Content preferences ─────────────────────────────────────────────
        if p.get("preferred_content") or p.get("arc_affinity"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 7, "Content Preferences", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            if p.get("preferred_content"):
                pdf.cell(0, 6, _safe(f"Preferred content:  {p.get('preferred_content')}"), ln=True)
            if p.get("arc_affinity"):
                pdf.cell(0, 6, _safe(f"Arc affinity:       {p.get('arc_affinity')}"), ln=True)
            pdf.ln(3)

        # ── Strategic recommendation ────────────────────────────────────────
        if p.get("strategic_rec"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 7, "Strategic Recommendation", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 5, _safe(p.get("strategic_rec", "")))
            pdf.ln(3)

        # ── Narrative ───────────────────────────────────────────────────────
        if p.get("narrative"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 7, "Behavioural Narrative", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 5, _safe(p.get("narrative", "")))
            pdf.ln(3)

        # ── Interventions ───────────────────────────────────────────────────
        interventions = p.get("interventions", [])
        if interventions:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 7, "Intervention Playbook", ln=True)
            for i, iv in enumerate(interventions, 1):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(232, 70, 30)
                pdf.cell(0, 6, _safe(f"{i}. {iv.get('name', '')}"), ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(60, 60, 60)
                if iv.get("action"):
                    pdf.multi_cell(0, 5, _safe(f"   Action: {iv.get('action', '')}"))
                if iv.get("why"):
                    pdf.multi_cell(0, 5, _safe(f"   Why: {iv.get('why', '')}"))
                pdf.ln(2)

        # ── Footer ──────────────────────────────────────────────────────────
        pdf.set_y(-20)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(160, 160, 160)
        pdf.cell(0, 5,
                 "Generated by EXL Market Intelligence Platform  |  Confidential",
                 align="C")

        # ── Encode & trigger download ────────────────────────────────────────
        pdf_bytes = pdf.output()
        pdf_b64   = base64.b64encode(pdf_bytes).decode()
        filename  = f"persona_{p.get('label', 'report')}.pdf"

        return rx.call_script(
            f"const a=document.createElement('a');"
            f"a.href='data:application/pdf;base64,{pdf_b64}';"
            f"a.download='{filename}';"
            f"document.body.appendChild(a);"
            f"a.click();"
            f"document.body.removeChild(a);"
        )
