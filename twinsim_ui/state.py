"""
TwinSim Pipeline — Global Application State.

All paths are derived relative to this file so the project is portable
and deployable without modification.

Pipeline layout (relative to project root):
  pipeline/
    enrich_sessions.py
    health_report.py
    feature_store.py
    feature_store_assessment.py
    data/
      data_dictionary_session_events.json
      data_dictionary_user_profiles.json
  uploaded_files/          ← runtime uploads (git-ignored)
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import List

import reflex as rx
import pandas as pd

# ── Paths (all relative — no hardcoded user paths) ────────────────────────────
# This file lives at:  <project_root>/twinsim_ui/state.py
# Project root is one level up.
PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
DATA_DIR     = PIPELINE_DIR / "data"
UPLOAD_DIR   = PROJECT_ROOT / "uploaded_files"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Use the same Python interpreter that is running this app
PYTHON = sys.executable

# ── Feature category definitions ──────────────────────────────────────────────
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


# ── Sync helpers (called from thread pool) ────────────────────────────────────
def _run_cmd(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
class PipelineState(rx.State):

    # ── Navigation ─────────────────────────────────────────────────────────────
    active_page: str = "upload"

    # ── Step statuses: pending | running | complete | error ─────────────────────
    step_status: List[str] = ["pending", "pending", "pending", "pending"]

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
    s2_sessions_rows:    int        = 0
    s2_sessions_cols:    int        = 0
    s2_profiles_rows:    int        = 0
    s2_profiles_cols:    int        = 0
    s2_catalogue_rows:   int        = 0
    s2_catalogue_cols:   int        = 0
    s2_enriched_rows:    int        = 0
    s2_enriched_cols:    int        = 0
    s2_category_rows:    List[dict] = []
    s2_missing_features: List[str]  = []
    s2_coverage_pct:     float      = 0.0
    s2_gate_passed:      bool       = False
    s2_running:          bool       = False
    s2_error:            str        = ""
    s2_enrich_path:      str        = ""

    # ── Step 3: Health ──────────────────────────────────────────────────────────
    s3_total_checks: int        = 0
    s3_passed:       int        = 0
    s3_warned:       int        = 0
    s3_failed:       int        = 0
    s3_check_rows:   List[dict] = []
    s3_gate_passed:  bool       = False
    s3_running:      bool       = False
    s3_error:        str        = ""
    s3_report_path:  str        = ""

    # ── Step 4: Feature Engineering ─────────────────────────────────────────────
    s4_total_features:    int        = 0
    s4_users:             int        = 0
    s4_assessment_rows:   List[dict] = []
    s4_all_passed:        bool       = False
    s4_gate_passed:       bool       = False
    s4_running:           bool       = False
    s4_error:             str        = ""
    s4_feature_store_path: str       = ""

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def go_to(self, page: str):
        self.active_page = page

    def _refresh_upload_flag(self):
        self.all_uploaded = bool(
            self.sessions_path and
            self.profiles_path and
            self.catalogue_path
        )

    # ─────────────────────────────────────────────────────────────────────────
    # UPLOAD HANDLERS
    # ─────────────────────────────────────────────────────────────────────────
    async def handle_sessions_upload(self, files: list[rx.UploadFile]):
        self.upload_error = ""
        if not files:
            return
        f = files[0]
        dest = UPLOAD_DIR / f.filename
        dest.write_bytes(await f.read())
        self.sessions_filename = f.filename
        self.sessions_path = str(dest)
        self._refresh_upload_flag()

    async def handle_profiles_upload(self, files: list[rx.UploadFile]):
        self.upload_error = ""
        if not files:
            return
        f = files[0]
        dest = UPLOAD_DIR / f.filename
        dest.write_bytes(await f.read())
        self.profiles_filename = f.filename
        self.profiles_path = str(dest)
        self._refresh_upload_flag()

    async def handle_catalogue_upload(self, files: list[rx.UploadFile]):
        self.upload_error = ""
        if not files:
            return
        f = files[0]
        dest = UPLOAD_DIR / f.filename
        dest.write_bytes(await f.read())
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
    # STEP 2 — INGESTION + FEATURE PRESENCE CHECK
    # ─────────────────────────────────────────────────────────────────────────
    async def run_ingestion(self):
        # Capture all state vars as locals BEFORE the first yield
        sessions_path  = self.sessions_path
        profiles_path  = self.profiles_path
        catalogue_path = self.catalogue_path
        enrich_out     = str(UPLOAD_DIR / "enriched_sessions.csv")

        self.s2_running = True
        self.s2_error   = ""
        self.step_status = ["complete", "running", "pending", "pending"]
        yield

        try:
            loop = asyncio.get_running_loop()

            # Load the three CSVs (blocking I/O → thread pool)
            sessions, profiles, catalogue = await loop.run_in_executor(
                None,
                lambda: (
                    _load_csv(sessions_path),
                    _load_csv(profiles_path),
                    _load_csv(catalogue_path),
                ),
            )
            self.s2_sessions_rows  = int(sessions.shape[0])
            self.s2_sessions_cols  = int(sessions.shape[1])
            self.s2_profiles_rows  = int(profiles.shape[0])
            self.s2_profiles_cols  = int(profiles.shape[1])
            self.s2_catalogue_rows = int(catalogue.shape[0])
            self.s2_catalogue_cols = int(catalogue.shape[1])
            yield

            # Run enrich_sessions.py
            enrich_cmd = [
                PYTHON, str(PIPELINE_DIR / "enrich_sessions.py"),
                "--sessions", sessions_path,
                "--metadata", catalogue_path,
                "--out",      enrich_out,
            ]
            res = await loop.run_in_executor(
                None, lambda: _run_cmd(enrich_cmd, timeout=120)
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"enrich_sessions.py failed (exit {res.returncode}):\n"
                    f"{res.stderr or res.stdout}"
                )

            enriched = await loop.run_in_executor(
                None, lambda: _load_csv(enrich_out)
            )
            self.s2_enriched_rows = int(enriched.shape[0])
            self.s2_enriched_cols = int(enriched.shape[1])
            self.s2_enrich_path   = enrich_out
            yield

            # Feature presence check
            all_cols = (
                set(sessions.columns) |
                set(profiles.columns)  |
                set(catalogue.columns) |
                set(enriched.columns)
            )
            cat_rows:    list[dict] = []
            missing_all: list[str]  = []
            total_exp = total_pre   = 0

            for cat, feats in FEATURE_CATEGORIES.items():
                n_exp  = len(feats)
                n_pre  = sum(1 for f in feats if f in all_cols)
                missed = [f for f in feats if f not in all_cols]
                total_exp += n_exp
                total_pre += n_pre
                missing_all.extend(missed)
                pct    = round(n_pre / n_exp * 100) if n_exp else 100
                status = "green" if pct == 100 else ("amber" if pct >= 80 else "red")
                cat_rows.append({
                    "category": cat,
                    "expected": n_exp,
                    "present":  n_pre,
                    "missing":  n_exp - n_pre,
                    "pct":      pct,
                    "status":   status,
                })

            coverage = round(total_pre / total_exp * 100, 1) if total_exp else 0.0
            gate     = len(missing_all) == 0

            self.s2_category_rows    = cat_rows
            self.s2_missing_features = missing_all
            self.s2_coverage_pct     = coverage
            self.s2_gate_passed      = gate
            self.s2_running          = False
            self.step_status = (
                ["complete", "complete", "pending", "pending"] if gate
                else ["complete", "error", "pending", "pending"]
            )
            yield

        except Exception as exc:
            self.s2_error   = str(exc)
            self.s2_running = False
            self.step_status = ["complete", "error", "pending", "pending"]
            yield

    def proceed_to_health(self):
        self.active_page = "health"
        return PipelineState.run_health_check

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — FEATURE HEALTH / QUALITY CHECK
    # ─────────────────────────────────────────────────────────────────────────
    async def run_health_check(self):
        enrich_path   = self.s2_enrich_path
        profiles_path = self.profiles_path
        report_out    = str(UPLOAD_DIR / "health_report_run.csv")

        self.s3_running = True
        self.s3_error   = ""
        self.step_status = ["complete", "complete", "running", "pending"]
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
            res = await loop.run_in_executor(
                None, lambda: _run_cmd(cmd, timeout=300)
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"health_report.py failed (exit {res.returncode}):\n"
                    f"{res.stderr or res.stdout}"
                )

            self.s3_report_path = report_out
            yield

            rdf = await loop.run_in_executor(
                None, lambda: _load_csv(report_out)
            )
            total  = len(rdf)
            passed = int((rdf["status"] == "PASS").sum())
            warned = int((rdf["status"] == "WARN").sum())
            failed = int((rdf["status"] == "FAIL").sum())

            rdf["group"] = rdf["check"].str.extract(r"^([A-Za-z_]+\d*)")
            summary: list[dict] = []
            for grp, sub in rdf.groupby("group"):
                summary.append({
                    "check_group": str(grp),
                    "passed": int((sub["status"] == "PASS").sum()),
                    "warned": int((sub["status"] == "WARN").sum()),
                    "failed": int((sub["status"] == "FAIL").sum()),
                    "detail": str(sub["check"].iloc[0]),
                })

            gate = failed == 0
            self.s3_total_checks = total
            self.s3_passed       = passed
            self.s3_warned       = warned
            self.s3_failed       = failed
            self.s3_check_rows   = summary
            self.s3_gate_passed  = gate
            self.s3_running      = False
            self.step_status = (
                ["complete", "complete", "complete", "pending"] if gate
                else ["complete", "complete", "error", "pending"]
            )
            yield

        except Exception as exc:
            self.s3_error   = str(exc)
            self.s3_running = False
            self.step_status = ["complete", "complete", "error", "pending"]
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
        self.step_status = ["complete", "complete", "complete", "running"]
        yield

        try:
            loop = asyncio.get_running_loop()

            cmd_fs = [
                PYTHON, str(PIPELINE_DIR / "feature_store.py"),
                "--events",   enrich_path,
                "--profiles", profiles_path,
                "--out",      fs_out,
            ]
            res = await loop.run_in_executor(
                None, lambda: _run_cmd(cmd_fs, timeout=600)
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"feature_store.py failed (exit {res.returncode}):\n"
                    f"{res.stderr or res.stdout}"
                )

            self.s4_feature_store_path = fs_out
            yield

            fsdf = await loop.run_in_executor(
                None, lambda: _load_csv(fs_out)
            )
            self.s4_users          = int(fsdf.shape[0])
            self.s4_total_features = int(fsdf.shape[1])
            yield

            cmd_assess = [
                PYTHON, str(PIPELINE_DIR / "feature_store_assessment.py"),
                "--input", fs_out,
            ]
            res2 = await loop.run_in_executor(
                None, lambda: _run_cmd(cmd_assess, timeout=120)
            )

            # Parse stdout: "[PASS]/[FAIL]/[WARN]/[SKIP] check_name"
            assess_rows: list[dict] = []
            tag_map = {"[PASS]": "PASS", "[FAIL]": "FAIL",
                       "[WARN]": "WARN", "[SKIP]": "SKIP"}
            lines = res2.stdout.splitlines()
            for i, line in enumerate(lines):
                for tag, status in tag_map.items():
                    if tag in line:
                        check_name = line.replace(tag, "").strip()
                        detail = ""
                        if i + 1 < len(lines):
                            nxt = lines[i + 1].strip()
                            if nxt and not any(
                                t in nxt for t in
                                ["[PASS]","[FAIL]","[WARN]","[SKIP]","===","OVERALL"]
                            ):
                                detail = nxt
                        assess_rows.append({"check": check_name,
                                            "status": status,
                                            "detail": detail})
                        break

            if not assess_rows:
                assess_rows = [{
                    "check":  "Assessment result",
                    "status": "PASS" if res2.returncode == 0 else "FAIL",
                    "detail": (res2.stdout or res2.stderr)[:400],
                }]

            all_pass = (res2.returncode == 0) and all(
                r["status"] in ("PASS", "WARN", "SKIP") for r in assess_rows
            )
            self.s4_assessment_rows = assess_rows
            self.s4_all_passed      = all_pass
            self.s4_gate_passed     = all_pass
            self.s4_running         = False
            self.step_status = (
                ["complete", "complete", "complete", "complete"] if all_pass
                else ["complete", "complete", "complete", "error"]
            )
            yield

        except Exception as exc:
            self.s4_error   = str(exc)
            self.s4_running = False
            self.step_status = ["complete", "complete", "complete", "error"]
            yield
