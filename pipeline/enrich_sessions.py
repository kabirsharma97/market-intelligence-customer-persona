"""
TwinSim — Session Enrichment Script  v1.0
==========================================
Merges slim session_events.csv with content_catalogue.csv to produce
enriched_sessions.csv — the input required by health_report, feature_store
and feature_store_assessment.

Three-file input model:
  session_events.csv    — user behaviour (36 cols, content_id as FK)
  user_profiles.csv     — user attributes (unchanged, passed through)
  content_catalogue.csv — content catalogue (16 cols, content_id as PK)

Output:
  enriched_sessions.csv — 43 cols (36 session + 7 content columns rejoined)

Columns added back from content_catalogue:
  content_type            — series / film / documentary / sport / shortform
  genre                   — Drama / Football / Cricket etc.
  content_duration_minutes— episode runtime in minutes (from episode_duration_mins)
  is_live_event           — 0 or 1
  is_exclusive            — 0 or 1
  franchise_flag          — 0 or 1
  episode_position        — recomputed per user per title from total_episodes

Geo-block validation:
  content_unavailable_flag and unavailable_reason are already present in
  session_events.csv (set by the user's upstream system or by generate_signals).
  This script cross-validates them against geo_availability in content_catalogue
  and reports any mismatches as warnings — it does NOT overwrite existing flags.

Usage:
  python enrich_sessions.py
  python enrich_sessions.py --sessions session_events.csv \
                             --metadata content_catalogue.csv \
                             --out enriched_sessions.csv

Exit codes: 0 = success, 1 = error
"""

import argparse
import sys
import pandas as pd

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Column name mapping: content_catalogue → enriched_sessions ───────────────
# content_catalogue.csv uses content_duration_minutes — matches session schema directly
METADATA_RENAME = {}   # no renames needed; column names already aligned

# These 7 columns are added to session_events from content_catalogue
CONTENT_COLS = [
    "content_type",
    "genre",
    "content_duration_minutes",
    "is_live_event",
    "is_exclusive",
    "franchise_flag",
]

# Content types where episode_position is meaningful
EPISODIC_TYPES = {"series", "documentary"}

# Final column order — matches original 44-col session_events schema
ENRICHED_COLUMN_ORDER = [
    "event_id", "user_id", "session_id", "content_id", "segment_id",
    "session_number", "event_type", "timestamp", "window_day",
    "completion_pct", "early_drop_rate", "mid_session_drop_rate",
    "early_drop_flag", "mid_drop_flag", "session_depth", "session_duration_mins",
    "days_since_last_session",
    "content_type", "genre", "content_duration_minutes",
    "is_live_event", "is_exclusive", "franchise_flag", "episode_position",
    "churn_flag", "reactivation_flag", "content_discovery_source",
    "device_type", "time_of_day", "is_weekend", "event_calendar_flag",
    "skip_intro_flag", "rewatch_flag", "casting_usage_flag",
    "network_type", "buffer_events", "avg_bitrate_mbps", "avg_network_jitter",
    "peak_hour_congestion_flag", "content_unavailable_flag", "unavailable_reason",
    "network_stress_flag", "content_satisfaction",
]


def load_sessions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    print(f"[INFO] session_events loaded : {len(df):,} rows × {len(df.columns)} cols")
    if "content_id" not in df.columns:
        raise ValueError("session_events.csv must contain a 'content_id' column.")
    # Confirm slim format — content columns should NOT be present
    already_present = [c for c in CONTENT_COLS + ["episode_position"] if c in df.columns]
    if already_present:
        print(f"[WARN] These content columns already exist in session_events and will be "
              f"overwritten by content_catalogue values: {already_present}")
    return df


def load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    print(f"[INFO] content_catalogue loaded: {len(df):,} rows × {len(df.columns)} cols")
    if "content_id" not in df.columns:
        raise ValueError("content_catalogue.csv must contain a 'content_id' column.")
    df = df.rename(columns=METADATA_RENAME)   # no-op; names already aligned
    return df


def compute_episode_position(df: pd.DataFrame) -> pd.Series:
    """
    Recompute episode_position per user per title as a categorical label.

    Labels (matching health_report valid value set):
      premiere        — user's 1st session of this title
      mid_season      — any middle session
      penultimate_arc — user's (total_episodes - 1)-th session of this title
      finale          — user's total_episodes-th (or beyond) session
      NULL            — non-episodic content (film, sport, shortform)

    Logic:
      - Sort sessions by user_id + content_id + timestamp
      - Within each (user_id, content_id) group assign cumulative watch rank
      - Map rank against total_episodes to produce the label
    """
    df = df.copy()
    df["_ts_sort"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Cumulative watch rank per user per title, sorted by timestamp
    df = df.sort_values(["user_id", "content_id", "_ts_sort"])
    df["_watch_rank"] = (
        df.groupby(["user_id", "content_id"]).cumcount() + 1
    )

    total_eps = pd.to_numeric(
        df.get("total_episodes", pd.Series([1] * len(df))),
        errors="coerce"
    ).fillna(1).clip(lower=1).astype(int)

    def _label(rank: int, total: int) -> str:
        if rank == 1:
            return "premiere"
        if total > 1 and rank >= total:
            return "finale"
        if total > 3 and rank >= int(total * 0.75):
            return "penultimate_arc"   # last quarter before finale (total must be > 3)
        return "mid_season"

    labels = pd.Series(
        [_label(int(r), int(t)) for r, t in zip(df["_watch_rank"], total_eps)],
        index=df.index
    )

    # NULL for non-episodic content types
    ct = df["content_type"].astype(str).str.lower()
    episode_position = labels.where(ct.isin(EPISODIC_TYPES), other=None)

    df["episode_position"] = episode_position
    df = df.drop(columns=["_ts_sort", "_watch_rank"])
    return df["episode_position"]


def validate_geo_blocks(df: pd.DataFrame) -> None:
    """
    Cross-validate content_unavailable_flag against geo_availability.
    Reports mismatches as warnings — does NOT overwrite existing flags.
    geo_availability is a pipe-separated string e.g. 'US|UK|AU'.
    """
    if "geo_availability" not in df.columns:
        return
    if "geo_country" not in df.columns:
        return
    if "content_unavailable_flag" not in df.columns:
        return

    def is_geo_blocked(row):
        avail = str(row.get("geo_availability", ""))
        if not avail or avail == "nan":
            return False
        countries = [c.strip() for c in avail.split("|")]
        return str(row.get("geo_country", "")) not in countries

    expected_block = df.apply(is_geo_blocked, axis=1)
    actual_flag    = pd.to_numeric(df["content_unavailable_flag"],
                                   errors="coerce").fillna(0).astype(bool)

    # Sessions that should be geo-blocked but are not flagged
    missed = int((expected_block & ~actual_flag).sum())
    # Sessions flagged as geo-block but geo_availability says they should be fine
    extra  = int((~expected_block & (
        df.get("unavailable_reason", pd.Series([""] * len(df)))
          .fillna("").str.lower() == "geo_block"
    )).sum())

    if missed > 0:
        print(f"[WARN] Geo-block validation: {missed:,} sessions should be geo-blocked "
              f"(user's country not in geo_availability) but content_unavailable_flag=0")
    if extra > 0:
        print(f"[WARN] Geo-block validation: {extra:,} sessions flagged as geo_block "
              f"but user's country IS in geo_availability")
    if missed == 0 and extra == 0:
        print(f"[INFO] Geo-block validation: all flags consistent with geo_availability ✓")


def enrich(sessions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join sessions with metadata on content_id.
    Adds content columns, computes episode_position, validates geo-blocks.
    """
    # Columns to bring in from metadata (exclude content_id itself and
    # fields not needed in enriched sessions)
    meta_cols = ["content_id"] + [
        c for c in CONTENT_COLS + ["total_episodes", "geo_availability"]
        if c in metadata.columns
    ]
    meta_slim = metadata[meta_cols].drop_duplicates(subset=["content_id"])

    # Check all content_ids in sessions are present in metadata
    session_ids  = set(sessions["content_id"].dropna().unique())
    metadata_ids = set(meta_slim["content_id"].dropna().unique())
    missing = session_ids - metadata_ids
    if missing:
        print(f"[WARN] {len(missing)} content_id(s) in session_events not found in "
              f"content_catalogue — those sessions will have NULL content columns: "
              f"{sorted(missing)[:10]}")

    # Left join — every session row is kept; unmatched get NULL content cols
    enriched = sessions.merge(meta_slim, on="content_id", how="left")
    print(f"[INFO] After join          : {len(enriched):,} rows × {len(enriched.columns)} cols")

    # Compute episode_position
    enriched["episode_position"] = compute_episode_position(enriched)
    print(f"[INFO] episode_position computed "
          f"(episodic rows: {enriched['episode_position'].notna().sum():,})")

    # Geo-block cross-validation
    validate_geo_blocks(enriched)

    # Cast boolean-like int columns cleanly
    for col in ["is_live_event", "is_exclusive", "franchise_flag"]:
        if col in enriched.columns:
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0).astype(int)

    # Drop helper columns not needed in output
    enriched = enriched.drop(columns=["total_episodes", "geo_availability"],
                             errors="ignore")

    # Reorder to match the original 44-col schema
    final_cols = [c for c in ENRICHED_COLUMN_ORDER if c in enriched.columns]
    # Append any extra columns not in the order list (future-proofing)
    extra_cols = [c for c in enriched.columns if c not in final_cols]
    enriched = enriched[final_cols + extra_cols]

    return enriched


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TwinSim Session Enrichment — merges session_events + content_catalogue",
    )
    ap.add_argument("--sessions", default="session_events.csv",
                    help="Slim session events CSV (default: session_events.csv)")
    ap.add_argument("--metadata", default="content_catalogue.csv",
                    help="Content catalogue CSV (default: content_catalogue.csv)")
    ap.add_argument("--out",      default="enriched_sessions.csv",
                    help="Output enriched sessions CSV (default: enriched_sessions.csv)")
    args = ap.parse_args()

    print(f"[INFO] Loading {args.sessions} ...")
    sessions = load_sessions(args.sessions)

    print(f"[INFO] Loading {args.metadata} ...")
    metadata = load_metadata(args.metadata)

    print(f"[INFO] Enriching sessions ...")
    enriched = enrich(sessions, metadata)

    enriched.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\n[DONE] enriched_sessions.csv → {args.out}")
    print(f"       {len(enriched):,} rows × {len(enriched.columns)} columns")

    print(f"\n[SCHEMA] Final columns ({len(enriched.columns)}):")
    for i, c in enumerate(enriched.columns, 1):
        source = "content_catalogue" if c in CONTENT_COLS + ["episode_position"] else "session_events"
        print(f"  {i:>2}. {c:<40} ← {source}")

    print(f"\n[NEXT] python health_report.py "
          f"--events {args.out} --profiles user_profiles.csv "
          f"--dict_events data_dictionary_session_events.json "
          f"--dict_profiles data_dictionary_user_profiles.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
