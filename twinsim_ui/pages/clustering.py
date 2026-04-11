"""Screen 5 — Clustering Engine."""
import reflex as rx
from twinsim_ui.state import PipelineState
from twinsim_ui import styles


def _cluster_row(row: dict) -> rx.Component:
    """One row in the cluster comparison table."""
    churn_color = rx.cond(
        row["churn_level"] == "high",
        styles.RED,
        rx.cond(row["churn_level"] == "medium", styles.AMBER, styles.GREEN),
    )
    return rx.hstack(
        # Avatar + name
        rx.hstack(
            rx.text(row["avatar"], font_size="18px"),
            rx.vstack(
                rx.text(row["name"], font_size="12px", font_weight="600",
                        color="#111827"),
                rx.text(row["label"], font_size="10px", color="#9CA3AF",
                        font_family="monospace"),
                spacing="0", align="start",
            ),
            spacing="2", align="center", flex="3",
        ),
        # Size
        rx.vstack(
            rx.text(row["size"].to_string(),
                    font_size="13px", font_weight="700", color="#111827"),
            rx.text(row["size_pct"].to_string() + "%",
                    font_size="10px", color="#6B7280"),
            spacing="0", align="center", flex="1",
        ),
        # Churn
        rx.text(
            row["churn"].to_string() + "%",
            font_size="13px", font_weight="700", color=churn_color,
            flex="1", text_align="center",
        ),
        # Completion
        rx.text(
            row["completion"].to_string() + "%",
            font_size="13px", font_weight="700", color="#3B82F6",
            flex="1", text_align="center",
        ),
        # Reactivation
        rx.text(
            row["reactivation"].to_string() + "%",
            font_size="13px", font_weight="700", color=styles.PURPLE,
            flex="1", text_align="center",
        ),
        spacing="3", align="center", width="100%",
        padding="10px 12px",
        border_bottom="1px solid #F3F4F6",
        _hover={"background": "#F9FAFB"},
    )


def _audit_row(row: dict) -> rx.Component:
    return rx.hstack(
        rx.cond(
            row["status"] == "PASS",
            rx.badge("✓ PASS", color_scheme="green", variant="soft", size="1"),
            rx.cond(
                row["status"] == "WARN",
                rx.badge("⚠ WARN", color_scheme="amber", variant="soft", size="1"),
                rx.cond(
                    row["status"] == "SKIP",
                    rx.badge("— SKIP", color_scheme="gray", variant="soft", size="1"),
                    rx.badge("✕ FAIL", color_scheme="red", variant="solid", size="1"),
                ),
            ),
        ),
        rx.text(row["check"], font_size="12px", color="#374151", flex="1"),
        rx.text(row["detail"], font_size="11px", color="#6B7280", flex="2",
                word_break="break-word"),
        spacing="3", align="center", width="100%",
        padding="7px 0",
        border_bottom="1px solid #F3F4F6",
    )


def clustering_page() -> rx.Component:
    return rx.vstack(

        # ── Running spinner ────────────────────────────────────────────────────
        rx.cond(
            PipelineState.s5_running,
            rx.box(
                rx.hstack(
                    rx.spinner(size="3"),
                    rx.vstack(
                        rx.text("Running clustering pipeline…",
                                font_size="14px", font_weight="600",
                                color="#111827"),
                        rx.text(
                            "K-Means optimisation · silhouette scoring · validation gate",
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

        # ── Error banner ───────────────────────────────────────────────────────
        rx.cond(
            PipelineState.s5_error != "",
            rx.box(
                rx.hstack(
                    rx.text("⚠", font_size="14px", color=styles.RED),
                    rx.text(PipelineState.s5_error, font_size="12px",
                            color=styles.RED, flex="1", word_break="break-word"),
                    spacing="2", align="start",
                ),
                padding="12px", background=styles.RED_BG,
                border=f"1px solid {styles.RED}",
                border_radius="8px", width="100%",
            ),
            rx.box(),
        ),

        # ── KPI tiles ──────────────────────────────────────────────────────────
        rx.grid(
            # Clusters found
            rx.vstack(
                rx.text(PipelineState.s5_n_clusters.to_string(),
                        font_size="36px", font_weight="700", color=styles.ACCENT),
                rx.text("Clusters Found", font_size="12px", font_weight="600",
                        color="#374151"),
                rx.text("K-Means optimal k", font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px",
                background=styles.ACCENT_BG,
                border=f"1px solid {styles.ACCENT}",
                border_radius="12px", flex="1",
            ),
            # Total events
            rx.vstack(
                rx.text(PipelineState.s5_n_events_fmt,
                        font_size="36px", font_weight="700", color="#111827"),
                rx.text("Total Events", font_size="12px", font_weight="600",
                        color="#374151"),
                rx.text("records in feature_store.csv",
                        font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px", background="white",
                border="1px solid #E5E7EB",
                border_radius="12px", flex="1",
            ),
            # Noise rate
            rx.vstack(
                rx.text(PipelineState.s5_noise_pct.to_string() + "%",
                        font_size="36px", font_weight="700",
                        color=rx.cond(
                            PipelineState.s5_noise_pct > 5,
                            styles.AMBER, styles.GREEN,
                        )),
                rx.text("Noise Rate", font_size="12px", font_weight="600",
                        color="#374151"),
                rx.text(PipelineState.s5_noise_events_fmt + " unassigned events",
                        font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px",
                background=rx.cond(
                    PipelineState.s5_noise_pct > 5,
                    styles.AMBER_BG, styles.GREEN_BG,
                ),
                border=rx.cond(
                    PipelineState.s5_noise_pct > 5,
                    f"1px solid {styles.AMBER}",
                    f"1px solid {styles.GREEN}",
                ),
                border_radius="12px", flex="1",
            ),
            # Quality score
            rx.vstack(
                rx.text(PipelineState.s5_silhouette.to_string(),
                        font_size="36px", font_weight="700", color=styles.BLUE),
                rx.text("Silhouette Score", font_size="12px", font_weight="600",
                        color="#374151"),
                rx.text(PipelineState.s5_stability_label,
                        font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px",
                background=styles.BLUE_BG,
                border=f"1px solid {styles.BLUE}",
                border_radius="12px", flex="1",
            ),
            columns="4", spacing="3", width="100%",
        ),

        # ── Cluster comparison table ───────────────────────────────────────────
        rx.cond(
            PipelineState.s5_cluster_rows.length() > 0,
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.text("Cluster Comparison",
                                    font_size="14px", font_weight="600",
                                    color="#111827"),
                            rx.text(
                                "Auto-labelled behavioural segments — "
                                "churn, completion and reactivation rates per cluster",
                                font_size="12px", color="#6B7280",
                            ),
                            spacing="0", align="start",
                        ),
                        rx.spacer(),
                        rx.badge(
                            PipelineState.s5_algorithm,
                            color_scheme="blue", variant="soft", size="1",
                        ),
                        width="100%", align="center",
                    ),
                    rx.box(height="1px", background="#E5E7EB", width="100%"),
                    # Column headers
                    rx.hstack(
                        rx.text("Segment", font_size="11px", font_weight="600",
                                color="#9CA3AF", flex="3"),
                        rx.text("Size", font_size="11px", font_weight="600",
                                color="#9CA3AF", flex="1", text_align="center"),
                        rx.text("Churn", font_size="11px", font_weight="600",
                                color=styles.RED, flex="1", text_align="center"),
                        rx.text("Completion", font_size="11px", font_weight="600",
                                color=styles.BLUE, flex="1", text_align="center"),
                        rx.text("Reactivation", font_size="11px", font_weight="600",
                                color=styles.PURPLE, flex="1", text_align="center"),
                        spacing="3", width="100%", padding="6px 12px",
                    ),
                    rx.foreach(PipelineState.s5_cluster_rows, _cluster_row),
                    spacing="1", width="100%",
                ),
                **styles.CARD,
                width="100%",
            ),
            rx.box(),
        ),

        # ── Audit strip ───────────────────────────────────────────────────────
        rx.cond(
            PipelineState.s5_audit_rows.length() > 0,
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.text("Clustering Audit",
                                    font_size="14px", font_weight="600",
                                    color="#111827"),
                            rx.text(
                                "Verifies cluster balance, coverage, stability and structural integrity",
                                font_size="12px", color="#6B7280",
                            ),
                            spacing="0", align="start",
                        ),
                        rx.spacer(),
                        rx.cond(
                            PipelineState.s5_audit_passed,
                            rx.badge("AUDIT PASS", color_scheme="green",
                                     variant="solid", size="2"),
                            rx.badge("AUDIT FAIL", color_scheme="red",
                                     variant="solid", size="2"),
                        ),
                        width="100%", align="center",
                    ),
                    rx.box(height="1px", background="#E5E7EB", width="100%"),
                    rx.hstack(
                        rx.text("Status", font_size="11px", font_weight="600",
                                color="#9CA3AF"),
                        rx.text("Check", font_size="11px", font_weight="600",
                                color="#9CA3AF", flex="1"),
                        rx.text("Detail", font_size="11px", font_weight="600",
                                color="#9CA3AF", flex="2"),
                        spacing="3", width="100%", padding="6px 0",
                    ),
                    rx.foreach(PipelineState.s5_audit_rows, _audit_row),
                    spacing="1", width="100%",
                ),
                **styles.CARD,
                width="100%",
            ),
            rx.box(),
        ),

        # ── Gate / CTA ─────────────────────────────────────────────────────────
        rx.cond(
            ~PipelineState.s5_running & (PipelineState.s5_n_clusters > 0),
            rx.cond(
                PipelineState.s5_gate_passed,
                rx.box(
                    rx.hstack(
                        rx.vstack(
                            rx.hstack(
                                rx.text("🎉", font_size="20px"),
                                rx.text(
                                    "Clustering complete — proceed to Persona Intelligence",
                                    font_size="14px", font_weight="700",
                                    color="white",
                                ),
                                spacing="2", align="center",
                            ),
                            rx.text(
                                PipelineState.s5_n_clusters.to_string() +
                                " segments identified · audit passed · "
                                "ready for persona profiling",
                                font_size="12px",
                                color="rgba(255,255,255,0.85)",
                            ),
                            spacing="1", align="start",
                        ),
                        rx.spacer(),
                        rx.box(
                            rx.text("View Personas  →",
                                    font_size="13px", font_weight="600",
                                    color=styles.ACCENT),
                            padding="10px 20px",
                            background="white",
                            border_radius="8px",
                            cursor="pointer",
                            on_click=PipelineState.proceed_to_persona(),
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
                            rx.text("Clustering audit FAILED",
                                    font_size="13px", font_weight="600",
                                    color="white"),
                            rx.text(
                                "Review failed audit checks and revisit the "
                                "feature engineering step.",
                                font_size="11px",
                                color="rgba(255,255,255,0.85)",
                            ),
                            spacing="0", align="start",
                        ),
                        rx.spacer(),
                        rx.button(
                            "↩ Revisit Features",
                            on_click=PipelineState.go_to("features"),
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
