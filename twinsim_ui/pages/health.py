"""Screen 3 — Feature Health & Quality Check."""
import reflex as rx
from twinsim_ui.state import PipelineState
from twinsim_ui import styles


def _status_dot(status: str) -> rx.Component:
    """Small coloured dot showing PASS / WARN / FAIL."""
    return rx.cond(
        status == "FAIL",
        rx.box(width="8px", height="8px", border_radius="50%",
               background=styles.RED, flex_shrink="0"),
        rx.cond(
            status == "WARN",
            rx.box(width="8px", height="8px", border_radius="50%",
                   background=styles.AMBER, flex_shrink="0"),
            rx.box(width="8px", height="8px", border_radius="50%",
                   background=styles.GREEN, flex_shrink="0"),
        ),
    )


def _check_row(row: dict) -> rx.Component:
    has_issue = (row["status"] == "FAIL") | (row["status"] == "WARN")
    is_expanded = row["expanded"] == "true"

    return rx.vstack(
        # ── Main row ──────────────────────────────────────────────────────
        rx.hstack(
            # Status dot
            _status_dot(row["status"]),
            # Code chip
            rx.box(
                rx.text(row["check_group"], font_size="10px",
                        font_weight="700", color="#374151"),
                background="#F3F4F6",
                border="1px solid #E5E7EB",
                border_radius="4px",
                padding="2px 7px",
                min_width="52px",
                text_align="center",
            ),
            # Plain English description
            rx.text(
                row["description"],
                font_size="12px", color="#374151",
                flex="1",
            ),
            # Result badges
            rx.hstack(
                rx.cond(
                    row["passed"].to_string() != "0",
                    rx.badge(row["passed"].to_string() + " pass",
                             color_scheme="green", variant="soft", size="1"),
                    rx.box(),
                ),
                rx.cond(
                    row["warned"].to_string() != "0",
                    rx.badge(row["warned"].to_string() + " warn",
                             color_scheme="amber", variant="soft", size="1"),
                    rx.box(),
                ),
                rx.cond(
                    row["failed"].to_string() != "0",
                    rx.badge(row["failed"].to_string() + " fail",
                             color_scheme="red", variant="soft", size="1"),
                    rx.box(),
                ),
                spacing="1", flex_shrink="0",
            ),
            # Expand chevron — only shown when there is a warning or failure
            rx.cond(
                has_issue,
                rx.button(
                    rx.cond(is_expanded, "▲", "▼"),
                    on_click=PipelineState.toggle_check(row["check_group"]),
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
                rx.box(width="28px"),  # placeholder to keep alignment
            ),
            spacing="3", align="center", width="100%",
            padding="9px 0",
        ),
        # ── Expanded reason panel ─────────────────────────────────────────
        rx.cond(
            is_expanded & has_issue,
            rx.box(
                rx.text(
                    row["reason"],
                    font_size="11px",
                    color=rx.cond(row["status"] == "FAIL", styles.RED, "#92400E"),
                    line_height="1.6",
                    word_break="break-word",
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


def health_page() -> rx.Component:
    return rx.vstack(
        # ── Running ────────────────────────────────────────────────────────────
        rx.cond(
            PipelineState.s3_running,
            rx.box(
                rx.hstack(
                    rx.spinner(size="3"),
                    rx.vstack(
                        rx.text("Running 239 quality checks…", font_size="14px",
                                font_weight="600", color="#111827"),
                        rx.text(
                            "Null checks · Range validity · Temporal consistency · "
                            "Distribution checks · Logical consistency",
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
            PipelineState.s3_error != "",
            rx.box(
                rx.hstack(
                    rx.text("⚠", font_size="14px", color=styles.RED),
                    rx.text(PipelineState.s3_error, font_size="12px",
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
                rx.text(PipelineState.s3_total_checks.to_string(),
                        font_size="32px", font_weight="700", color="#111827"),
                rx.text("Total Checks", font_size="12px", font_weight="600", color="#374151"),
                rx.text("A1–A19 · B15–B24 · C25–C26",
                        font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px", background="white",
                border="1px solid #E5E7EB", border_radius="12px",
                flex="1",
            ),
            rx.vstack(
                rx.text(PipelineState.s3_passed.to_string(),
                        font_size="32px", font_weight="700", color=styles.GREEN),
                rx.text("Passed", font_size="12px", font_weight="600", color="#374151"),
                rx.text("Zero anomalies", font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px", background=styles.GREEN_BG,
                border=f"1px solid {styles.GREEN}", border_radius="12px",
                flex="1",
            ),
            rx.vstack(
                rx.text(PipelineState.s3_warned.to_string(),
                        font_size="32px", font_weight="700", color=styles.AMBER),
                rx.text("Warnings", font_size="12px", font_weight="600", color="#374151"),
                rx.text("Non-blocking notices", font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px", background=styles.AMBER_BG,
                border=f"1px solid {styles.AMBER}", border_radius="12px",
                flex="1",
            ),
            rx.vstack(
                rx.text(PipelineState.s3_failed.to_string(),
                        font_size="32px", font_weight="700",
                        color=rx.cond(PipelineState.s3_failed > 0, styles.RED, "#9CA3AF")),
                rx.text("Failed", font_size="12px", font_weight="600", color="#374151"),
                rx.text("Gate: must be 0", font_size="10px", color="#9CA3AF"),
                spacing="0", align="center",
                padding="16px",
                background=rx.cond(
                    PipelineState.s3_failed > 0,
                    styles.RED_BG, "#F9FAFB",
                ),
                border=rx.cond(
                    PipelineState.s3_failed > 0,
                    f"1px solid {styles.RED}", "1px solid #E5E7EB",
                ),
                border_radius="12px",
                flex="1",
            ),
            columns="4", spacing="3", width="100%",
        ),

        # ── Check list ─────────────────────────────────────────────────────────
        rx.cond(
            PipelineState.s3_check_rows.length() > 0,
            rx.box(
                rx.vstack(
                    rx.text("Check-Group Summary", font_size="14px",
                            font_weight="600", color="#111827"),
                    rx.text(
                        "Each row represents one check group; numbers show per-group results",
                        font_size="12px", color="#6B7280",
                    ),
                    rx.box(height="1px", background="#E5E7EB", width="100%"),
                    # Column headers
                    rx.hstack(
                        rx.box(width="8px", flex_shrink="0"),  # dot spacer
                        rx.text("Code", font_size="11px", font_weight="600",
                                color="#9CA3AF", min_width="52px"),
                        rx.text("What is being checked", font_size="11px",
                                font_weight="600", color="#9CA3AF", flex="1"),
                        rx.text("Results", font_size="11px", font_weight="600",
                                color="#9CA3AF"),
                        rx.box(width="28px", flex_shrink="0"),  # chevron spacer
                        spacing="3", width="100%", padding="6px 0",
                    ),
                    rx.foreach(PipelineState.s3_check_rows, _check_row),
                    spacing="1", width="100%",
                ),
                **styles.CARD,
                width="100%",
            ),
            rx.box(),
        ),

        # ── Gate result ────────────────────────────────────────────────────────
        rx.cond(
            ~PipelineState.s3_running & (PipelineState.s3_total_checks > 0),
            rx.cond(
                PipelineState.s3_gate_passed,
                rx.hstack(
                    rx.box(
                        rx.hstack(
                            rx.text("✓", color="white", font_size="14px"),
                            rx.text("100% pass rate — 0 FAIL · health gate cleared",
                                    font_size="13px", font_weight="600", color="white"),
                            spacing="2", align="center",
                        ),
                        background=styles.GREEN,
                        border_radius="8px",
                        padding="10px 18px",
                        flex="1",
                        display="flex", align_items="center",
                    ),
                    rx.button(
                        "Run Feature Engineering  →",
                        on_click=PipelineState.proceed_to_features,
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
                        rx.text("✕", color="white", font_size="16px"),
                        rx.vstack(
                            rx.text("Health check gate FAILED",
                                    font_size="13px", font_weight="600", color="white"),
                            rx.text(
                                "One or more checks returned FAIL. "
                                "Fix the data quality issues and re-upload your files.",
                                font_size="11px", color="rgba(255,255,255,0.85)",
                            ),
                            spacing="0", align="start",
                        ),
                        rx.spacer(),
                        rx.button(
                            "↩ Back to Upload",
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
