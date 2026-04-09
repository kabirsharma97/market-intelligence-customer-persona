"""Screen 3 — Feature Health & Quality Check."""
import reflex as rx
from twinsim_ui.state import PipelineState
from twinsim_ui import styles

# Friendly names for check groups
CHECK_LABELS = {
    "A1":  "A1 — Event ID uniqueness",
    "A2":  "A2 — Null / missing values",
    "A3":  "A3 — Data type validation",
    "A4":  "A4 — Range validity",
    "A5":  "A5 — Timestamp consistency",
    "A6":  "A6 — Session triangle logic",
    "A7":  "A7 — Event type consistency",
    "A8":  "A8 — Session ordering",
    "A9":  "A9 — Segment distribution",
    "A10": "A10 — Completion ordering",
    "A11": "A11 — Sparse satisfaction",
    "A12": "A12 — Skip-intro guard",
    "A13": "A13 — Casting guard",
    "A14": "A14 — Bitrate & network",
    "A15": "A15 — Drop flag alignment",
    "A16": "A16 — Days since first session",
    "A17": "A17 — Depth caps",
    "A18": "A18 — Network stress",
    "A19": "A19 — Episode position",
    "A_u": "A_u — Content unavailability",
    "B15": "B15 — User-ID uniqueness",
    "B16": "B16 — Profile nullability",
    "B17": "B17 — Profile range checks",
    "B18": "B18 — Price validation",
    "B19": "B19 — Lifecycle trajectory",
    "B20": "B20 — New profile fields",
    "B21": "B21 — Sparse ticket data",
    "B22": "B22 — Ticket-payment cross",
    "B23": "B23 — Ticket-segment cross",
    "B24": "B24 — Trajectory segment",
    "C25": "C25 — Cross-file join integrity",
    "C26": "C26 — Geo-block consistency",
}


def _check_row(row: dict) -> rx.Component:
    return rx.hstack(
        rx.text(row["check_group"], font_size="11px", color="#6B7280",
                font_weight="500", min_width="40px"),
        rx.text(row["detail"], font_size="12px", color="#374151", flex="3",
                word_break="break-word"),
        rx.hstack(
            rx.badge(
                row["passed"].to_string() + " pass",
                color_scheme="green", variant="soft", size="1",
            ),
            rx.cond(
                row["warned"].to_string() != "0",
                rx.badge(
                    row["warned"].to_string() + " warn",
                    color_scheme="amber", variant="soft", size="1",
                ),
                rx.box(),
            ),
            rx.cond(
                row["failed"].to_string() != "0",
                rx.badge(
                    row["failed"].to_string() + " fail",
                    color_scheme="red", variant="soft", size="1",
                ),
                rx.box(),
            ),
            spacing="1",
        ),
        spacing="3", align="center", width="100%",
        padding="8px 0",
        border_bottom="1px solid #F3F4F6",
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
                        rx.text("Code", font_size="11px", font_weight="600",
                                color="#9CA3AF", min_width="40px"),
                        rx.text("Description", font_size="11px", font_weight="600",
                                color="#9CA3AF", flex="3"),
                        rx.text("Results", font_size="11px", font_weight="600",
                                color="#9CA3AF"),
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
