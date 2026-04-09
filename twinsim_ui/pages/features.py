"""Screen 4 — Feature Engineering."""
import reflex as rx
from twinsim_ui.state import PipelineState
from twinsim_ui import styles


def _assess_row(row: dict) -> rx.Component:
    return rx.hstack(
        rx.text(row["check"], font_size="12px", color="#374151", flex="3",
                word_break="break-word"),
        rx.cond(
            row["status"] == "PASS",
            rx.badge("✓ PASS", color_scheme="green", variant="soft", size="1"),
            rx.cond(
                row["status"] == "WARN",
                rx.badge("⚠ WARN", color_scheme="amber", variant="soft", size="1"),
                rx.badge("✕ FAIL", color_scheme="red", variant="soft", size="1"),
            ),
        ),
        rx.text(row["detail"], font_size="11px", color="#6B7280", flex="4",
                word_break="break-word"),
        spacing="3", align="center", width="100%",
        padding="8px 0",
        border_bottom="1px solid #F3F4F6",
    )


def features_page() -> rx.Component:
    return rx.vstack(
        # ── Running ───────────────────────────────────────────────────────────
        rx.cond(
            PipelineState.s4_running,
            rx.box(
                rx.hstack(
                    rx.spinner(size="3"),
                    rx.vstack(
                        rx.text("Running feature engineering pipeline…",
                                font_size="14px", font_weight="600", color="#111827"),
                        rx.text(
                            "feature_store 4.py — engineering 50 ML-ready features across "
                            "5,000 users · then running 12-check quality assessment",
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

        # ── Error ─────────────────────────────────────────────────────────────
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

        # ── Feature store summary ──────────────────────────────────────────────
        rx.grid(
            rx.vstack(
                rx.text(PipelineState.s4_total_features.to_string(),
                        font_size="32px", font_weight="700", color=styles.ACCENT),
                rx.text("ML Features", font_size="12px", font_weight="600", color="#374151"),
                rx.text("in feature_store.csv", font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px", background=styles.ACCENT_BG,
                border=f"1px solid {styles.ACCENT}", border_radius="12px",
                flex="1",
            ),
            rx.vstack(
                rx.text(PipelineState.s4_users.to_string(),
                        font_size="32px", font_weight="700", color="#111827"),
                rx.text("Users", font_size="12px", font_weight="600", color="#374151"),
                rx.text("one row per user_id", font_size="10px", color="#9CA3AF"),
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
                rx.text("feature_store_assessment", font_size="10px", color="#9CA3AF"),
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
                rx.text("gate: all must pass", font_size="10px", color="#9CA3AF"),
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

        # ── Feature categories breakdown ───────────────────────────────────────
        rx.box(
            rx.vstack(
                rx.text("Feature Store Overview", font_size="14px",
                        font_weight="600", color="#111827"),
                rx.grid(
                    rx.vstack(
                        rx.text("Attention & Engagement",
                                font_size="12px", font_weight="600", color="#374151"),
                        rx.text("attention_decay_curve · binge_velocity · "
                                "completion_momentum · session_volatility_index",
                                font_size="11px", color="#6B7280"),
                        padding="12px", background="#F9FAFB",
                        border="1px solid #E5E7EB", border_radius="8px",
                    ),
                    rx.vstack(
                        rx.text("Satisfaction & Churn",
                                font_size="12px", font_weight="600", color="#374151"),
                        rx.text("satisfaction_trend_slope · churn_risk_score · "
                                "reactivation_probability · ltv_retention_index",
                                font_size="11px", color="#6B7280"),
                        padding="12px", background="#F9FAFB",
                        border="1px solid #E5E7EB", border_radius="8px",
                    ),
                    rx.vstack(
                        rx.text("Network & Device Quality",
                                font_size="12px", font_weight="600", color="#374151"),
                        rx.text("network_resilience_score · peak_hour_stress_ratio · "
                                "bitrate_stability_index · device_loyalty_score",
                                font_size="11px", color="#6B7280"),
                        padding="12px", background="#F9FAFB",
                        border="1px solid #E5E7EB", border_radius="8px",
                    ),
                    rx.vstack(
                        rx.text("Content Affinity",
                                font_size="12px", font_weight="600", color="#374151"),
                        rx.text("genre_affinity_score · live_sports_engagement · "
                                "franchise_loyalty_score · exclusive_content_pull",
                                font_size="11px", color="#6B7280"),
                        padding="12px", background="#F9FAFB",
                        border="1px solid #E5E7EB", border_radius="8px",
                    ),
                    columns="2", spacing="3", width="100%",
                ),
                spacing="3", width="100%",
            ),
            **styles.CARD,
            width="100%",
        ),

        # ── Assessment results ─────────────────────────────────────────────────
        rx.cond(
            PipelineState.s4_assessment_rows.length() > 0,
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.text("Feature Store Assessment",
                                    font_size="14px", font_weight="600", color="#111827"),
                            rx.text("12-check quality gate — all must PASS to proceed",
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
                        rx.text("Check", font_size="11px", font_weight="600",
                                color="#9CA3AF", flex="3"),
                        rx.text("Status", font_size="11px", font_weight="600",
                                color="#9CA3AF"),
                        rx.text("Detail", font_size="11px", font_weight="600",
                                color="#9CA3AF", flex="4"),
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
                                f"feature_store.csv contains {PipelineState.s4_total_features.to_string()} ML-ready "
                                "features across " + PipelineState.s4_users.to_string() +
                                " users · all 12 quality checks passed.",
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
