"""Screen 4 — Feature Engineering."""
import reflex as rx
from twinsim_ui.state import PipelineState
from twinsim_ui import styles


def _assess_row(row: dict) -> rx.Component:
    has_issue   = (row["status"] == "FAIL") | (row["status"] == "WARN")
    is_expanded = row["expanded"] == "true"

    return rx.vstack(
        # ── Main row ──────────────────────────────────────────────────────
        rx.hstack(
            # Status dot
            rx.cond(
                row["status"] == "FAIL",
                rx.box(width="8px", height="8px", border_radius="50%",
                       background=styles.RED, flex_shrink="0"),
                rx.cond(
                    row["status"] == "WARN",
                    rx.box(width="8px", height="8px", border_radius="50%",
                           background=styles.AMBER, flex_shrink="0"),
                    rx.box(width="8px", height="8px", border_radius="50%",
                           background=styles.GREEN, flex_shrink="0"),
                ),
            ),
            # Check name chip
            rx.box(
                rx.text(row["check"], font_size="10px", font_weight="700",
                        color="#374151", white_space="nowrap"),
                background="#F3F4F6",
                border="1px solid #E5E7EB",
                border_radius="4px",
                padding="2px 7px",
                min_width="160px",
                flex_shrink="0",
            ),
            # Plain English description
            rx.text(
                row["description"],
                font_size="12px", color="#374151", flex="1",
            ),
            # Status badge
            rx.cond(
                row["status"] == "PASS",
                rx.badge("✓ PASS", color_scheme="green", variant="soft", size="1"),
                rx.cond(
                    row["status"] == "WARN",
                    rx.badge("⚠ WARN", color_scheme="amber", variant="soft", size="1"),
                    rx.cond(
                        row["status"] == "SKIP",
                        rx.badge("· SKIP", color_scheme="gray", variant="soft", size="1"),
                        rx.badge("✕ FAIL", color_scheme="red", variant="soft", size="1"),
                    ),
                ),
            ),
            # Expand chevron — only for WARN / FAIL
            rx.cond(
                has_issue,
                rx.button(
                    rx.cond(is_expanded, "▲", "▼"),
                    on_click=PipelineState.toggle_assess(row["check"]),
                    background="transparent",
                    color="#6B7280",
                    font_size="10px",
                    padding="2px 6px",
                    cursor="pointer",
                    border="1px solid #E5E7EB",
                    border_radius="4px",
                    _hover={"background": "#F9FAFB"},
                    flex_shrink="0",
                ),
                rx.box(width="28px"),
            ),
            spacing="3", align="center", width="100%",
            padding="9px 0",
        ),
        # ── Expanded reason panel ─────────────────────────────────────────
        rx.cond(
            is_expanded & has_issue,
            rx.box(
                rx.hstack(
                    rx.box(
                        width="3px",
                        min_height="16px",
                        background=rx.cond(row["status"] == "FAIL", styles.RED, styles.AMBER),
                        border_radius="2px",
                        flex_shrink="0",
                    ),
                    rx.text(
                        row["detail"],
                        font_size="11px",
                        color=rx.cond(row["status"] == "FAIL", styles.RED, "#92400E"),
                        line_height="1.7",
                        word_break="break-word",
                        white_space="pre-wrap",
                    ),
                    spacing="2", align="start", width="100%",
                ),
                background=rx.cond(
                    row["status"] == "FAIL", styles.RED_BG, styles.AMBER_BG,
                ),
                border=rx.cond(
                    row["status"] == "FAIL",
                    f"1px solid {styles.RED}",
                    f"1px solid {styles.AMBER}",
                ),
                border_radius="6px",
                padding="10px 14px",
                width="100%",
                margin_bottom="4px",
            ),
            rx.box(),
        ),
        border_bottom="1px solid #F3F4F6",
        spacing="0",
        width="100%",
    )


def features_page() -> rx.Component:
    return rx.vstack(
        # ── Running ────────────────────────────────────────────────────────────
        rx.cond(
            PipelineState.s4_running,
            rx.box(
                rx.hstack(
                    rx.spinner(size="3"),
                    rx.vstack(
                        rx.text("Running feature engineering pipeline…",
                                font_size="14px", font_weight="600", color="#111827"),
                        rx.text(
                            "feature_store.py — engineering 50 ML-ready features · "
                            "then running 14-check quality assessment",
                            font_size="12px", color="#6B7280",
                        ),
                        spacing="0", align="start",
                    ),
                    spacing="4", align="center",
                ),
                padding="16px 20px",
                background="#EFF6FF",
                border="1px solid #BFDBFE",
                border_radius="12px",
                width="100%",
            ),
            rx.box(),
        ),

        # ── Error ──────────────────────────────────────────────────────────────
        rx.cond(
            PipelineState.s4_error != "",
            rx.box(
                rx.hstack(
                    rx.text("⚠", font_size="14px", color=styles.RED),
                    rx.text(PipelineState.s4_error, font_size="12px",
                            color=styles.RED, flex="1", word_break="break-word"),
                    spacing="2", align="start",
                ),
                padding="12px", background=styles.RED_BG,
                border=f"1px solid {styles.RED}", border_radius="8px", width="100%",
            ),
            rx.box(),
        ),

        # ── Summary tiles ──────────────────────────────────────────────────────
        rx.grid(
            rx.vstack(
                rx.text(PipelineState.s4_total_features.to_string(),
                        font_size="32px", font_weight="700", color=styles.ACCENT),
                rx.text("ML Features", font_size="12px", font_weight="600", color="#374151"),
                rx.text("ready for clustering", font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px", background=styles.ACCENT_BG,
                border=f"1px solid {styles.ACCENT}", border_radius="12px",
                flex="1",
            ),
            rx.vstack(
                rx.text(PipelineState.s4_users_fmt,
                        font_size="32px", font_weight="700", color="#111827"),
                rx.text("Records", font_size="12px", font_weight="600", color="#374151"),
                rx.text("user sessions analysed", font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px", background="white",
                border="1px solid #E5E7EB", border_radius="12px",
                flex="1",
            ),
            rx.vstack(
                rx.text(
                    PipelineState.s4_assessment_rows.length().to_string(),
                    font_size="32px", font_weight="700", color="#111827",
                ),
                rx.text("QA Checks", font_size="12px", font_weight="600", color="#374151"),
                rx.text("automated quality gates", font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px", background="white",
                border="1px solid #E5E7EB", border_radius="12px",
                flex="1",
            ),
            rx.vstack(
                rx.cond(
                    PipelineState.s4_all_passed,
                    rx.text("100%", font_size="32px", font_weight="700", color=styles.GREEN),
                    rx.text("—", font_size="32px", font_weight="700", color="#9CA3AF"),
                ),
                rx.text("Pass Rate", font_size="12px", font_weight="600", color="#374151"),
                rx.text("Gate: must be 0 failures", font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px",
                background=rx.cond(PipelineState.s4_all_passed,
                                   styles.GREEN_BG, "#F9FAFB"),
                border=rx.cond(PipelineState.s4_all_passed,
                               f"1px solid {styles.GREEN}", "1px solid #E5E7EB"),
                border_radius="12px",
                flex="1",
            ),
            columns="4", spacing="3", width="100%",
        ),

        # ── Assessment results ─────────────────────────────────────────────────
        rx.cond(
            PipelineState.s4_assessment_rows.length() > 0,
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.text("Feature Store Quality Assessment",
                                    font_size="14px", font_weight="600", color="#111827"),
                            rx.text("14-check gate — completeness, ranges, variance, "
                                    "behavioural ordering, trajectory quality",
                                    font_size="12px", color="#6B7280"),
                            spacing="0", align="start",
                        ),
                        rx.spacer(),
                        rx.cond(
                            PipelineState.s4_all_passed,
                            rx.badge("ALL PASS", color_scheme="green",
                                     variant="solid", size="2"),
                            rx.badge("FAILURES DETECTED", color_scheme="red",
                                     variant="solid", size="2"),
                        ),
                        width="100%", align="center",
                    ),
                    rx.box(height="1px", background="#E5E7EB", width="100%"),
                    rx.hstack(
                        rx.box(width="8px", flex_shrink="0"),    # dot spacer
                        rx.text("Check", font_size="11px", font_weight="600",
                                color="#9CA3AF", min_width="160px"),
                        rx.text("What is being checked", font_size="11px",
                                font_weight="600", color="#9CA3AF", flex="1"),
                        rx.text("Status", font_size="11px", font_weight="600",
                                color="#9CA3AF"),
                        rx.box(width="28px", flex_shrink="0"),   # chevron spacer
                        spacing="3", width="100%", padding="6px 0",
                    ),
                    rx.foreach(PipelineState.s4_assessment_rows, _assess_row),
                    spacing="1", width="100%",
                ),
                **styles.CARD,
                width="100%",
            ),
            rx.box(),
        ),

        # ── Gate / CTA ─────────────────────────────────────────────────────────
        rx.cond(
            ~PipelineState.s4_running & (PipelineState.s4_total_features > 0),
            rx.cond(
                PipelineState.s4_gate_passed,
                rx.box(
                    rx.hstack(
                        rx.vstack(
                            rx.hstack(
                                rx.text("🎉", font_size="20px"),
                                rx.text("Feature store ready — proceed for clustering",
                                        font_size="14px", font_weight="700", color="white"),
                                spacing="2", align="center",
                            ),
                            rx.text(
                                "feature_store.csv — " +
                                PipelineState.s4_total_features.to_string() +
                                " ML-ready features · " +
                                PipelineState.s4_users_fmt +
                                " records · all 14 quality checks passed.",
                                font_size="12px", color="rgba(255,255,255,0.85)",
                            ),
                            spacing="1", align="start",
                        ),
                        rx.spacer(),
                        rx.box(
                            rx.text("Proceed for Clustering  →",
                                    font_size="13px", font_weight="600", color=styles.ACCENT),
                            padding="10px 20px",
                            background="white",
                            border_radius="8px",
                            cursor="pointer",
                            on_click=PipelineState.proceed_to_clustering(),
                            _hover={"background": "#F3F4F6"},
                        ),
                        spacing="4", align="center", width="100%",
                    ),
                    background=f"linear-gradient(135deg, {styles.ACCENT}, #C83A18)",
                    border_radius="12px",
                    padding="20px 24px",
                    width="100%",
                ),
                rx.box(
                    rx.hstack(
                        rx.text("✕", color="white", font_size="16px"),
                        rx.vstack(
                            rx.text("Feature engineering gate FAILED",
                                    font_size="13px", font_weight="600", color="white"),
                            rx.text(
                                "Review failed assessment checks, fix the underlying data, "
                                "and restart the pipeline.",
                                font_size="11px", color="rgba(255,255,255,0.85)",
                            ),
                            spacing="0", align="start",
                        ),
                        rx.spacer(),
                        rx.button(
                            "↩ Revisit Health Check",
                            on_click=PipelineState.go_to("health"),
                            background="rgba(255,255,255,0.2)",
                            color="white",
                            border="1px solid rgba(255,255,255,0.4)",
                            border_radius="6px",
                            padding="6px 16px",
                            font_size="12px",
                            cursor="pointer",
                        ),
                        spacing="3", align="center", width="100%",
                    ),
                    background=styles.RED,
                    border_radius="10px",
                    padding="14px 18px",
                    width="100%",
                ),
            ),
            rx.box(),
        ),

        spacing="4", width="100%", align="start",
    )
