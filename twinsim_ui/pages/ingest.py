"""Screen 2 — Data Ingestion + Feature Presence Check."""
import reflex as rx
from twinsim_ui.state import PipelineState
from twinsim_ui import styles


def _metric_tile(label: str, value, sub: str = "") -> rx.Component:
    return rx.vstack(
        rx.text(value, font_size="26px", font_weight="700", color="#111827"),
        rx.text(label, font_size="12px", font_weight="600", color="#374151"),
        rx.text(sub, font_size="11px", color="#9CA3AF"),
        spacing="0", align="center",
        padding="16px",
        background="white",
        border="1px solid #E5E7EB",
        border_radius="12px",
        min_width="140px",
        flex="1",
    )


def _status_badge(status: str) -> rx.Component:
    return rx.cond(
        status == "green",
        rx.badge("✓ Full Coverage", color_scheme="green", variant="soft", size="1"),
        rx.cond(
            status == "amber",
            rx.badge("⚠ Partial", color_scheme="amber", variant="soft", size="1"),
            rx.badge("✕ Missing", color_scheme="red", variant="soft", size="1"),
        ),
    )


def _enrich_step_card() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.box(
                rx.text("⚙", font_size="16px"),
                background=styles.ACCENT_BG,
                border_radius="8px",
                width="36px", height="36px",
                display="flex", align_items="center", justify_content="center",
            ),
            rx.vstack(
                rx.text("Session Enrichment Merge", font_size="14px",
                        font_weight="600", color="#111827"),
                rx.text(
                    "enrich_sessions.py — LEFT JOIN session_events × content_catalogue on content_id",
                    font_size="11px", color="#6B7280",
                ),
                spacing="0", align="start",
            ),
            spacing="3", align="center", width="100%",
        ),
        rx.cond(
            PipelineState.s2_running,
            rx.hstack(
                rx.spinner(size="2"),
                rx.text("Running enrichment…", font_size="12px", color="#6B7280"),
                spacing="2", align="center",
            ),
            rx.cond(
                PipelineState.s2_enriched_rows > 0,
                rx.hstack(
                    rx.box(width="8px", height="8px", border_radius="50%",
                           background=styles.GREEN),
                    rx.text(
                        f"Enriched sessions ready — ",
                        font_size="12px", color="#374151",
                    ),
                    rx.text(
                        PipelineState.s2_enriched_rows.to_string() + " rows × " +
                        PipelineState.s2_enriched_cols.to_string() + " cols",
                        font_size="12px", font_weight="600", color=styles.GREEN,
                    ),
                    spacing="2", align="center",
                ),
                rx.box(),
            ),
        ),
        padding="16px",
        background="#F0FDF4",
        border="1px solid #BBF7D0",
        border_radius="12px",
        width="100%",
        spacing="3",
    )


def _category_row(row: dict) -> rx.Component:
    return rx.hstack(
        rx.text(row["category"], font_size="12px", color="#374151",
                font_weight="500", flex="2"),
        rx.text(
            row["present"].to_string() + " / " + row["expected"].to_string(),
            font_size="12px", color="#6B7280", flex="1", text_align="center",
        ),
        rx.box(
            rx.box(
                width=row["pct"].to_string() + "%",
                height="6px",
                background=rx.cond(
                    row["status"] == "green", styles.GREEN,
                    rx.cond(row["status"] == "amber", styles.AMBER, styles.RED),
                ),
                border_radius="3px",
                transition="width 0.5s ease",
            ),
            width="120px", height="6px",
            background="#E5E7EB", border_radius="3px", overflow="hidden",
            flex="2",
        ),
        _status_badge(row["status"]),
        spacing="3", align="center", width="100%",
        padding="8px 0",
        border_bottom="1px solid #F3F4F6",
    )


def ingest_page() -> rx.Component:
    return rx.vstack(
        # ── Running spinner overlay (top) ────────────────────────────────────
        rx.cond(
            PipelineState.s2_running,
            rx.box(
                rx.hstack(
                    rx.spinner(size="3"),
                    rx.vstack(
                        rx.text("Ingesting data…", font_size="14px",
                                font_weight="600", color="#111827"),
                        rx.text("Loading files, running enrichment, checking feature coverage",
                                font_size="12px", color="#6B7280"),
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
            PipelineState.s2_error != "",
            rx.box(
                rx.hstack(
                    rx.text("⚠", font_size="14px", color=styles.RED),
                    rx.text(PipelineState.s2_error, font_size="12px",
                            color=styles.RED, flex="1", word_break="break-word"),
                    spacing="2", align="start",
                ),
                padding="12px",
                background=styles.RED_BG,
                border=f"1px solid {styles.RED}",
                border_radius="8px",
                width="100%",
            ),
            rx.box(),
        ),

        # ── Row / Column counts ───────────────────────────────────────────────
        rx.box(
            rx.vstack(
                rx.text("File Summary", font_size="14px", font_weight="600",
                        color="#111827"),
                rx.grid(
                    _metric_tile(
                        "Session Events",
                        PipelineState.s2_sessions_rows.to_string(),
                        PipelineState.s2_sessions_cols.to_string() + " columns",
                    ),
                    _metric_tile(
                        "User Profiles",
                        PipelineState.s2_profiles_rows.to_string(),
                        PipelineState.s2_profiles_cols.to_string() + " columns",
                    ),
                    _metric_tile(
                        "Content Catalogue",
                        PipelineState.s2_catalogue_rows.to_string(),
                        PipelineState.s2_catalogue_cols.to_string() + " columns",
                    ),
                    _metric_tile(
                        "Enriched Sessions",
                        PipelineState.s2_enriched_rows.to_string(),
                        PipelineState.s2_enriched_cols.to_string() + " columns",
                    ),
                    columns="4", spacing="3", width="100%",
                ),
                spacing="3", width="100%",
            ),
            **styles.CARD,
            width="100%",
        ),

        # ── Enrichment step ───────────────────────────────────────────────────
        _enrich_step_card(),

        # ── Feature Coverage ──────────────────────────────────────────────────
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.vstack(
                        rx.text("Feature Presence Check", font_size="14px",
                                font_weight="600", color="#111827"),
                        rx.text(
                            "Expected features verified across all three source files",
                            font_size="12px", color="#6B7280",
                        ),
                        spacing="0", align="start",
                    ),
                    rx.spacer(),
                    rx.vstack(
                        rx.text(
                            PipelineState.s2_coverage_pct.to_string() + "%",
                            font_size="24px", font_weight="700",
                            color=rx.cond(
                                PipelineState.s2_coverage_pct >= 100,
                                styles.GREEN,
                                rx.cond(PipelineState.s2_coverage_pct >= 80,
                                        styles.AMBER, styles.RED),
                            ),
                        ),
                        rx.text("Overall Coverage", font_size="10px", color="#9CA3AF"),
                        spacing="0", align="center",
                    ),
                    width="100%", align="center",
                ),
                rx.box(height="1px", background="#E5E7EB", width="100%"),
                # Header row
                rx.hstack(
                    rx.text("Feature Category", font_size="11px",
                            font_weight="600", color="#9CA3AF", flex="2"),
                    rx.text("Present / Expected", font_size="11px",
                            font_weight="600", color="#9CA3AF", flex="1",
                            text_align="center"),
                    rx.text("Coverage", font_size="11px",
                            font_weight="600", color="#9CA3AF", flex="2"),
                    rx.text("Status", font_size="11px",
                            font_weight="600", color="#9CA3AF"),
                    spacing="3", width="100%", padding="6px 0",
                ),
                rx.foreach(PipelineState.s2_category_rows, _category_row),
                spacing="2", width="100%",
            ),
            **styles.CARD,
            width="100%",
        ),

        # ── Missing features (if any) ─────────────────────────────────────────
        rx.cond(
            PipelineState.s2_missing_features.length() > 0,
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.text("⚠", color=styles.AMBER, font_size="16px"),
                        rx.text("Missing Features", font_size="13px",
                                font_weight="600", color="#92400E"),
                        spacing="2", align="center",
                    ),
                    rx.text(
                        "The following expected features were not found in the uploaded files.",
                        font_size="12px", color="#92400E",
                    ),
                    rx.hstack(
                        rx.foreach(
                            PipelineState.s2_missing_features,
                            lambda f: rx.badge(f, color_scheme="amber",
                                               variant="soft", size="1"),
                        ),
                        flex_wrap="wrap", spacing="2",
                    ),
                    spacing="2", width="100%",
                ),
                padding="16px",
                background=styles.AMBER_BG,
                border=f"1px solid {styles.AMBER}",
                border_radius="10px",
                width="100%",
            ),
            rx.box(),
        ),

        # ── Gate result + action ──────────────────────────────────────────────
        rx.cond(
            ~PipelineState.s2_running & (PipelineState.s2_enriched_rows > 0),
            rx.cond(
                PipelineState.s2_gate_passed,
                rx.hstack(
                    rx.box(
                        rx.hstack(
                            rx.text("✓", color="white", font_size="14px"),
                            rx.text("All checks passed — ready to proceed",
                                    font_size="13px", font_weight="600", color="white"),
                            spacing="2", align="center",
                        ),
                        background=styles.GREEN,
                        border_radius="8px",
                        padding="10px 18px",
                        flex="1",
                        display="flex",
                        align_items="center",
                    ),
                    rx.button(
                        "Run Health Check  →",
                        on_click=PipelineState.proceed_to_health,
                        background=styles.ACCENT,
                        color="white",
                        border_radius="8px",
                        padding="10px 24px",
                        font_size="13px",
                        font_weight="600",
                        cursor="pointer",
                        _hover={"background": styles.ACCENT_HOVER},
                        height="44px",
                    ),
                    spacing="3", width="100%",
                ),
                rx.box(
                    rx.hstack(
                        rx.text("✕", color="white", font_size="14px"),
                        rx.vstack(
                            rx.text("Feature presence gate FAILED",
                                    font_size="13px", font_weight="600", color="white"),
                            rx.text(
                                "Fix missing columns and re-upload your files.",
                                font_size="11px", color="rgba(255,255,255,0.8)",
                            ),
                            spacing="0", align="start",
                        ),
                        rx.spacer(),
                        rx.button(
                            "↩ Re-upload Files",
                            on_click=PipelineState.go_to("upload"),
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
