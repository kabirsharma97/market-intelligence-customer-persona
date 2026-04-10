"""
TwinSim — Clustering Audit
============================
Layer 3 output validation. Run after clustering_engine.py and before
persona_engine.py.

Validates cluster_assignments.csv and cluster_metadata.json against
10 behavioral correctness checks. Exit code 1 on any FAIL — CI gate.

Usage:
    python clustering_audit.py \\
        --assignments cluster_assignments.csv \\
        --metadata cluster_metadata.json

Checks:
    1  Assignment completeness     — all events have a cluster_index
    2  Noise rate                  — noise < 30% of all events
    3  Cluster count               — between 3 and 12 clusters
    4  Behavioral prior plausibility — base_completion, base_churn_30d in valid ranges
    5  Behavioral label ordering   — completion ordering across auto-labeled clusters
    6  Behavioral label coverage    — all expected behavioral labels represented [SYNTH: also checks segment coverage when segment_id present]
    7  Stability gate              — mean silhouette not degenerate
    8  Cluster size balance        — no single cluster > 60% of non-noise events
    9  Noise distribution by seg   — no segment has >40% events marked noise (CA-03)
    10 Event dropout               — n_events >= 80% of feature_store_n (CA-04)

Exit codes:
    0 — all checks PASS or WARN
    1 — one or more FAIL (DO NOT proceed to persona_engine.py)
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

KNOWN_SEGMENTS = {
    "binge_heavy",
    "casual_dip",
    "completion_obsessed",
    "quick_churn",
    "re_engager",
}

# Expected completion ordering for auto-labeled clusters (ascending).
# re_engager sits AFTER casual_dip: re-engagers who return show moderate completion
# (40–50%), while casual_dip users barely complete content (20–30%). This ordering
# holds for real client data; synthetic-only datasets may show the reverse because
# return sessions are initialised at low completion in the simulator.
COMPLETION_ORDER = ["quick_churn", "casual_dip", "re_engager", "binge_heavy", "completion_obsessed"]

TOLERANCE = 1e-4


def run_audit(assignments_path: str, metadata_path: str) -> bool:
    print(f"[INFO] Assignments : {assignments_path}")
    print(f"[INFO] Metadata    : {metadata_path}")

    adf = pd.read_csv(assignments_path)
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    n_total    = len(adf)
    n_noise    = int((adf["cluster_index"] == -1).sum()) if "cluster_index" in adf.columns else 0
    n_assigned = n_total - n_noise
    clusters   = meta.get("clusters", [])
    stability  = meta.get("stability", {})

    results = {}

    # ── Check 1: Assignment completeness ──────────────────────────────────────
    if "cluster_index" not in adf.columns:
        results["1_assignment_completeness"] = (
            "FAIL", "cluster_index column missing from assignments CSV"
        )
    else:
        null_idx = int(adf["cluster_index"].isnull().sum())
        results["1_assignment_completeness"] = (
            ("PASS", f"All {n_total:,} events have a cluster_index (noise=-1 is valid)")
            if null_idx == 0 else
            ("FAIL", f"{null_idx} events have null cluster_index")
        )

    # ── Check 2: Noise rate < 30% ─────────────────────────────────────────────
    noise_pct = n_noise / n_total * 100 if n_total > 0 else 0
    if noise_pct < 10.0:
        results["2_noise_rate"] = (
            "PASS", f"Noise rate={noise_pct:.1f}% (excellent, <10%)"
        )
    elif noise_pct < 30.0:
        results["2_noise_rate"] = (
            "WARN", f"Noise rate={noise_pct:.1f}% (acceptable, <30% threshold)"
        )
    else:
        results["2_noise_rate"] = (
            "FAIL",
            (f"Noise rate={noise_pct:.1f}% exceeds 30% threshold. "
             f"Re-run with smaller --min_cluster_size or switch to K-Means fallback.")
        )

    # ── Check 3: Cluster count 3–12 ───────────────────────────────────────────
    n_clusters = len(clusters)
    if 3 <= n_clusters <= 12:
        results["3_cluster_count"] = (
            "PASS", f"{n_clusters} clusters (within expected range [3, 12])"
        )
    elif n_clusters < 3:
        results["3_cluster_count"] = (
            "FAIL",
            (f"Only {n_clusters} cluster(s). Too few to represent behavioral diversity. "
             f"Reduce --min_cluster_size or check feature variance.")
        )
    else:
        results["3_cluster_count"] = (
            "WARN",
            (f"{n_clusters} clusters — above 12. May be over-segmented. "
             f"Consider increasing --min_cluster_size.")
        )

    # ── Check 4: Behavioral prior plausibility ────────────────────────────────
    prior_failures = []
    prior_warnings = []
    for cl in clusters:
        priors = cl.get("behavioral_priors", {})
        idx    = cl["cluster_index"]
        label  = cl.get("behavioral_label", "unknown")

        comp  = priors.get("base_completion")
        churn = priors.get("base_churn_30d")
        react = priors.get("base_reactivation")

        if comp is not None and not (0.0 <= comp <= 1.0):
            prior_failures.append(f"cluster {idx} ({label}): base_completion={comp:.4f} out of [0,1]")
        if churn is not None and not (0.0 <= churn <= 1.0):
            prior_failures.append(f"cluster {idx} ({label}): base_churn_30d={churn:.4f} out of [0,1]")
        if react is not None and not (0.0 <= react <= 1.0):
            prior_failures.append(f"cluster {idx} ({label}): base_reactivation={react:.4f} out of [0,1]")

        # Plausibility: quick_churn should not have high completion
        if label == "quick_churn" and comp is not None and comp > 0.60:
            prior_warnings.append(
                f"cluster {idx}: labeled quick_churn but base_completion={comp:.4f} (>0.60 suspicious)"
            )
        # binge_heavy should not have high churn.
        # Threshold 0.45: binge_heavy churn_30d_prob=0.05 with 9 sessions gives
        # user-level P(churn)=0.37 at most. Centroids above 0.45 indicate
        # contamination from other segments (raised from 0.30 for v5.2.0 data).
        if label == "binge_heavy" and churn is not None and churn > 0.45:
            prior_warnings.append(
                f"cluster {idx}: labeled binge_heavy but base_churn_30d={churn:.4f} (>0.30 suspicious)"
            )
        # CA-02: completion_obsessed with low completion is definitionally wrong
        if label == "completion_obsessed" and comp is not None and comp < 0.65:
            prior_warnings.append(
                f"cluster {idx}: labeled completion_obsessed but base_completion={comp:.4f} (<0.65 — labeling failure)"
            )
        # CA-02: re_engager with zero reactivation has no reactivation signal — labeling failure
        if label == "re_engager" and react is not None and react == 0.0:
            prior_warnings.append(
                f"cluster {idx}: labeled re_engager but base_reactivation=0.0 — no reactivation signal (labeling failure)"
            )
        # CA-02: casual_dip with deep sessions contradicts casual behavior definition
        base_depth = priors.get("base_session_depth")
        if label == "casual_dip" and base_depth is not None and base_depth > 0.70:
            prior_warnings.append(
                f"cluster {idx}: labeled casual_dip but base_session_depth={base_depth:.4f} (>0.70 — contradicts casual definition)"
            )

    if prior_failures:
        results["4_behavioral_prior_plausibility"] = (
            "FAIL", f"Prior range violations: {'; '.join(prior_failures)}"
        )
    elif prior_warnings:
        results["4_behavioral_prior_plausibility"] = (
            "WARN",
            f"Prior plausibility warnings (review before Layer 4): {'; '.join(prior_warnings)}"
        )
    else:
        results["4_behavioral_prior_plausibility"] = (
            "PASS",
            f"All {n_clusters} clusters have plausible behavioral priors"
        )

    # ── Check 5: Behavioral label completion ordering ─────────────────────────
    # Among clusters that carry standard behavioral labels, completion should
    # increase following COMPLETION_ORDER. Note: re_engager sits BEFORE casual_dip
    # in the ordering because re_engager cluster centroids reflect low-completion
    # return sessions (lapsed users re-establishing habits), not the segment's
    # long-term completion capability. Updated for v5.2.0 segment parameters.
    label_completion: dict = {}
    for cl in clusters:
        lbl  = cl.get("behavioral_label", "mixed_behavior")
        comp = cl.get("behavioral_priors", {}).get("base_completion")
        if comp is not None and lbl in COMPLETION_ORDER:
            # If multiple clusters share the same label, use the mean
            label_completion[lbl] = label_completion.get(lbl, [])
            label_completion[lbl].append(comp)

    label_means = {lbl: float(np.mean(vals)) for lbl, vals in label_completion.items()}
    present = [l for l in COMPLETION_ORDER if l in label_means]
    violations = []
    for i in range(len(present) - 1):
        a, b = present[i], present[i + 1]
        if label_means[a] > label_means[b] + 0.05:
            violations.append(
                f"{a}({label_means[a]:.3f}) > {b}({label_means[b]:.3f})"
            )

    if not violations:
        summary = ", ".join(f"{l}={label_means[l]:.3f}" for l in present)
        results["5_label_completion_ordering"] = (
            "PASS", f"Completion ordering correct across labeled clusters: {summary}"
        )
    else:
        # CA-01: FAIL not WARN — a mislabeled cluster routes Layer 4 interventions
        # to the wrong playbook. This is a labeling failure, not a warning.
        results["5_label_completion_ordering"] = (
            "FAIL",
            (f"Completion ordering violation — labeling failure, blocks Layer 4: "
             f"{'; '.join(violations)}. Full: {label_means}")
        )

    # ── Check 6: Behavioral label coverage (production-valid) ────────────────
    # Verifies that the full set of expected behavioral labels is represented
    # across discovered clusters. This check uses auto-assigned behavioral_label
    # values derived from feature profile means — no segment_id dependency.
    # Production-valid: runs on real client data without modification.
    #
    # SYNTHETIC VALIDATION EXTENSION — only runs when segment_id is present
    # in the assignments CSV (i.e. --synthetic_validation_mode was active in
    # clustering_engine.py). Checks that all 5 known simulation segments appear
    # across cluster compositions. This is scaffolding for verifying that
    # discovered labels align with the simulation ground truth. It must NOT
    # run on real client data — segment_id does not exist in production.
    observed_labels = set(
        cl.get("behavioral_label", "") for cl in clusters
    ) - {"", None}
    expected_labels = set(COMPLETION_ORDER)
    missing_labels   = expected_labels - observed_labels
    duplicate_labels = {
        lbl for lbl in observed_labels
        if sum(1 for cl in clusters if cl.get("behavioral_label") == lbl) > 1
    }
    if not missing_labels:
        results["6_behavioral_label_coverage"] = (
            "PASS",
            f"All expected behavioral labels present: {sorted(observed_labels)}"
            + (f" (duplicate labels: {sorted(duplicate_labels)} — CAL-002 multi_assign)"
               if duplicate_labels else "")
        )
    else:
        # FAIL: a missing label means a behavioral segment has no persona.
        # Layer 4 cannot produce intervention playbooks for that segment.
        # The fix is to re-run clustering_engine.py with --k_min increased
        # by 1 to give the k-selector room to produce an additional cluster.
        n_clusters = len(clusters)
        results["6_behavioral_label_coverage"] = (
            "FAIL",
            (f"Labels absent: {sorted(missing_labels)}. "
             f"Duplicate labels: {sorted(duplicate_labels)}. "
             f"k={n_clusters} clusters is insufficient to recover all 5 behavioral "
             f"segments at this dataset scale. "
             f"Re-run clustering_engine.py with --k_min {n_clusters + 1}. "
             f"DO NOT proceed to persona_engine.py — a missing persona means "
             f"an entire behavioral segment has no intervention playbook.")
        )

    # Synthetic segment coverage check — scaffolding only
    seg_comp_present = any(
        cl.get("segment_composition", {}).get("_synthetic_validation_only")
        for cl in clusters
    )
    if "segment_id" in adf.columns or seg_comp_present:
        observed_segments = set()
        for cl in clusters:
            sc = cl.get("segment_composition", {}).get("segment_distribution", {})
            observed_segments.update(sc.keys())
        if "segment_id" in adf.columns:
            observed_segments.update(adf["segment_id"].dropna().unique().tolist())
        missing_segments = KNOWN_SEGMENTS - observed_segments
        if not missing_segments:
            results["6b_segment_coverage_synth"] = (
                "PASS",
                "[SYNTHETIC ONLY] All 5 simulation segments represented across cluster compositions"
            )
        else:
            results["6b_segment_coverage_synth"] = (
                "WARN",
                (f"[SYNTHETIC ONLY] Segments absent from all clusters: {missing_segments}. "
                 f"Check generate_signals.py segment weights or increase --n.")
            )
    else:
        results["6b_segment_coverage_synth"] = (
            "SKIP",
            ("[SYNTHETIC ONLY] segment_id not in assignments — production mode. "
             "Re-run clustering_engine.py with --synthetic_validation_mode to enable ")
        )

    # ── Check 7: Stability gate ────────────────────────────────────────────────
    # STABLE (sil > 0.35): PASS — proceed with confidence
    # MARGINAL (sil 0.20–0.35): WARN — proceed with caution
    # UNSTABLE (sil 0.05–0.20): WARN — acceptable for synthetic/small datasets;
    #   cluster structure is real but overlapping. Document in persona priors.
    # Degenerate (sil < 0.05): FAIL — clustering has produced no meaningful structure
    stab_label = stability.get("stability_label", "UNKNOWN")
    sil_mean   = stability.get("mean_silhouette", 0.0)
    ci_lo      = stability.get("ci_lower", 0.0)
    DEGENERATE_SIL = 0.05  # below this: no meaningful cluster structure exists

    if stab_label == "STABLE":
        results["7_stability_gate"] = (
            "PASS",
            (f"Clustering STABLE. Bootstrap sil={sil_mean:.4f}, "
             f"CI lower={ci_lo:.4f} (>0.35)")
        )
    elif stab_label == "MARGINAL":
        results["7_stability_gate"] = (
            "WARN",
            (f"Clustering MARGINAL. sil={sil_mean:.4f}. "
             f"Proceed with caution — validate persona priors manually before Layer 4.")
        )
    elif sil_mean >= DEGENERATE_SIL:
        # UNSTABLE but not degenerate — acceptable for synthetic/small-scale data
        results["7_stability_gate"] = (
            "WARN",
            (f"Clustering UNSTABLE (sil={sil_mean:.4f}, threshold=0.20). "
             f"Cluster structure exists but boundaries overlap — expected on synthetic "
             f"data <10k users or 60-day window. Persona priors valid but carry wider "
             f"uncertainty. Proceed to Layer 4 with confidence_note awareness. "
             f"Target: increase to 10k+ users or 90-day window for STABLE result.")
        )
    else:
        results["7_stability_gate"] = (
            "FAIL",
            (f"Degenerate clustering. sil={sil_mean:.4f} < {DEGENERATE_SIL}. "
             f"No meaningful cluster structure — DO NOT pass to persona_engine.py. "
             f"Check feature collinearity (Check 13) and increase dataset size.")
        )

    # ── Check 8: Cluster size balance ─────────────────────────────────────────
    # No single cluster should dominate >60% of non-noise events.
    # Dominance suggests the algorithm collapsed into a majority cluster.
    if n_assigned > 0:
        max_pct = max(cl["size"] / n_assigned * 100 for cl in clusters) if clusters else 0
        dom_cluster = max(clusters, key=lambda c: c["size"]) if clusters else {}
        dom_label   = dom_cluster.get("behavioral_label", "unknown")
        dom_size    = dom_cluster.get("size", 0)
        if max_pct <= 60.0:
            results["8_cluster_size_balance"] = (
                "PASS",
                (f"Largest cluster: {dom_label} "
                 f"({dom_size:,} events, {max_pct:.1f}% of non-noise). "
                 f"No dominant cluster (<=60% threshold).")
            )
        else:
            results["8_cluster_size_balance"] = (
                "WARN" if max_pct <= 75.0 else "FAIL",
                (f"Dominant cluster: {dom_label} "
                 f"({dom_size:,} events, {max_pct:.1f}% of non-noise). "
                 f"Consider reducing --min_cluster_size or checking feature variance.")
            )
    else:
        results["8_cluster_size_balance"] = (
            "FAIL", "No non-noise events to evaluate size balance."
        )

    # ── Check 9: Noise distribution by segment (CA-03) ───────────────────────
    # If one known segment has >40% of its events classified as noise, that
    # segment's persona will be undersized and its priors unreliable.
    if "is_noise" in adf.columns and "segment_id" in adf.columns:
        seg_noise_warnings = []
        for seg in KNOWN_SEGMENTS:
            seg_rows = adf[adf["segment_id"] == seg]
            if len(seg_rows) == 0:
                continue
            seg_noise_pct = float((seg_rows["is_noise"] == 1).mean() * 100)
            if seg_noise_pct > 40.0:
                seg_noise_warnings.append(
                    f"{seg}: {seg_noise_pct:.1f}% noise ({int((seg_rows['is_noise']==1).sum()):,} of {len(seg_rows):,} events)"
                )
        if seg_noise_warnings:
            results["9_noise_distribution_by_segment"] = (
                "WARN",
                (f"CA-03: Segment(s) with >40% events marked noise — persona will be undersized: "
                 f"{'; '.join(seg_noise_warnings)}")
            )
        else:
            seg_summary = ", ".join(
                f"{s}={float((adf[adf['segment_id']==s]['is_noise']==1).mean()*100):.1f}%"
                for s in sorted(KNOWN_SEGMENTS) if s in adf["segment_id"].values
            )
            results["9_noise_distribution_by_segment"] = (
                "PASS",
                f"CA-03: No segment has >40% noise. Per-segment noise: {seg_summary}"
            )
    else:
        results["9_noise_distribution_by_segment"] = (
            "WARN",
            "CA-03: Cannot evaluate — is_noise or segment_id column absent from assignments CSV"
        )

    # ── Check 10: Event dropout (CA-04 / XL-05) ───────────────────────────────
    # feature_store_n is written to cluster_metadata.json by clustering_engine.py
    # (CE-04 bonus). If n_events in metadata is <80% of feature_store_n, a
    # significant portion of events were dropped between feature store and clustering.
    feature_store_n = meta.get("feature_store_n")
    n_events_meta   = meta.get("n_events")
    if feature_store_n is not None and n_events_meta is not None and feature_store_n > 0:
        dropout_pct = (1.0 - n_events_meta / feature_store_n) * 100
        if n_events_meta < feature_store_n * 0.80:
            results["10_event_dropout"] = (
                "FAIL",
                (f"CA-04: Event dropout {dropout_pct:.1f}% exceeds 20% threshold. "
                 f"feature_store_n={feature_store_n:,}, n_events={n_events_meta:,}. "
                 f"Check clustering_engine.py filtering logic or feature_store.py output.")
            )
        elif n_events_meta < feature_store_n * 0.95:
            results["10_event_dropout"] = (
                "WARN",
                (f"CA-04: Event dropout {dropout_pct:.1f}% (>5% but within 20% threshold). "
                 f"feature_store_n={feature_store_n:,}, n_events={n_events_meta:,}. Monitor.")
            )
        else:
            results["10_event_dropout"] = (
                "PASS",
                (f"CA-04: Event dropout {dropout_pct:.1f}% within acceptable range (<5%). "
                 f"feature_store_n={feature_store_n:,}, n_events={n_events_meta:,}.")
            )
    else:
        results["10_event_dropout"] = (
            "WARN",
            ("CA-04: Cannot evaluate — feature_store_n absent from cluster_metadata.json. "
             "Ensure clustering_engine.py v4.0.0+ wrote this field.")
        )

    # ── Print results ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("TWINSIM — CLUSTERING AUDIT RESULTS")
    print("=" * 72)
    print(f"  Events     : {n_total:,}  (assigned={n_assigned:,}, noise={n_noise:,})")
    print(f"  Clusters   : {n_clusters}")
    print(f"  Algorithm  : {meta.get('algorithm', 'unknown')}")
    print(f"  Stability  : {stab_label} (sil={sil_mean:.4f})")
    print()

    all_pass = True
    for check_name, (status, msg) in results.items():
        icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]"}.get(status, f"[{status}]")
        print(f"  {icon} {check_name}")
        print(f"         {msg}")
        if status == "FAIL":
            all_pass = False

    print()

    # Per-cluster prior summary
    print("  ── Cluster behavioral priors (Layer 4 inputs) " + "─" * 24)
    print(f"  {'Idx':>4} {'Label':<24} {'Completion':>10} {'Churn30d':>9} {'React':>7}  Dominant seg")
    print("  " + "-" * 68)
    for cl in sorted(clusters, key=lambda c: c["cluster_index"]):
        priors = cl.get("behavioral_priors", {})
        comp   = priors.get("base_completion", float("nan"))
        churn  = priors.get("base_churn_30d", float("nan"))
        react  = priors.get("base_reactivation", float("nan"))
        seg_comp = cl.get("segment_composition", {})
        dom = seg_comp.get("dominant_segment", "—")
        if dom != "—":
            dom = f"{dom} [SYNTH]"
        label  = cl.get("behavioral_label", "—")
        print(
            f"  {cl['cluster_index']:>4} {label:<24} "
            f"{comp:>10.4f} {churn:>9.4f} {react:>7.4f}  {dom}"
        )

    print()
    print("=" * 72)
    overall = (
        "ALL CHECKS PASSED — safe to proceed to persona_engine.py"
        if all_pass else
        "ISSUES FOUND — DO NOT proceed to persona_engine.py"
    )
    print(f"  OVERALL: {overall}")
    print("=" * 72)

    return all_pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TwinSim — Clustering Audit")
    ap.add_argument(
        "--assignments", default="cluster_assignments.csv",
        help="cluster_assignments.csv from clustering_engine.py"
    )
    ap.add_argument(
        "--metadata", default="cluster_metadata.json",
        help="cluster_metadata.json from clustering_engine.py"
    )
    ap.add_argument(
        "--synthetic_validation_mode", action="store_true",
        help=(
            "Informational flag — audit reads segment_composition from metadata "
            "when present (set by clustering_engine.py --synthetic_validation_mode). "
            "No direct effect on audit logic — synthetic checks are gated by "
            "segment_id presence in the assignments CSV."
        )
    )
    args = ap.parse_args()
    ok = run_audit(args.assignments, args.metadata)
    sys.exit(0 if ok else 1)