"""EXL-branded sidebar navigation."""
import reflex as rx
from twinsim_ui.state import PipelineState
from twinsim_ui import styles


def _step_dot(idx: int) -> rx.Component:
    status = PipelineState.step_status[idx]
    return rx.cond(
        status == "complete",
        rx.box(
            rx.text("✓", color="white", font_size="9px", font_weight="700"),
            width="18px", height="18px", border_radius="50%",
            background=styles.GREEN,
            display="flex", align_items="center", justify_content="center",
            flex_shrink="0",
        ),
        rx.cond(
            status == "running",
            rx.box(
                width="18px", height="18px", border_radius="50%",
                border=f"2px solid {styles.ACCENT}",
                background=styles.ACCENT_BG,
                flex_shrink="0",
                animation="pulse 1.5s infinite",
            ),
            rx.cond(
                status == "error",
                rx.box(
                    rx.text("✕", color="white", font_size="9px", font_weight="700"),
                    width="18px", height="18px", border_radius="50%",
                    background=styles.RED,
                    display="flex", align_items="center", justify_content="center",
                    flex_shrink="0",
                ),
                rx.box(
                    rx.text(str(idx + 1), color="#9ca3af", font_size="9px", font_weight="600"),
                    width="18px", height="18px", border_radius="50%",
                    border="1.5px solid #4b5563",
                    display="flex", align_items="center", justify_content="center",
                    flex_shrink="0",
                ),
            ),
        ),
    )


def _nav_item(label: str, page: str, step_idx: int) -> rx.Component:
    is_active = PipelineState.active_page == page
    return rx.hstack(
        rx.box(
            width="3px", height="36px",
            border_radius="0 2px 2px 0",
            background=rx.cond(is_active, styles.ACCENT, "transparent"),
            position="absolute", left="0", top="0",
        ),
        rx.hstack(
            _step_dot(step_idx),
            rx.text(
                label,
                font_size="12px",
                font_weight=rx.cond(is_active, "600", "400"),
                color=rx.cond(is_active, "white", "#9ca3af"),
                flex="1",
            ),
            spacing="2", align="center", width="100%",
        ),
        position="relative",
        padding="8px 12px 8px 16px",
        border_radius="6px",
        background=rx.cond(is_active, styles.SIDEBAR_ACTIVE_BG, "transparent"),
        cursor="pointer",
        on_click=PipelineState.go_to(page),
        width="100%",
        _hover={"background": rx.cond(is_active, styles.SIDEBAR_ACTIVE_BG, "rgba(255,255,255,0.04)")},
        transition="all 0.15s ease",
        min_height="36px",
    )


def sidebar() -> rx.Component:
    return rx.vstack(
        # ── EXL Logo Header ─────────────────────────────────────────────────
        rx.vstack(
            rx.text(
                "EXL",
                color="#E8461E",
                font_size="38px",
                font_weight="900",
                letter_spacing="-0.03em",
                line_height="1",
                font_family="'Arial Black', 'Arial Bold', Arial, sans-serif",
            ),
            rx.text(
                "Market Intelligence",
                color="white", font_size="12px", font_weight="600",
                line_height="1.3",
            ),
            rx.text(
                "Customer Persona",
                color="#9ca3af", font_size="10px", line_height="1.2",
            ),
            spacing="1", align="start",
            padding="18px 14px 14px",
            width="100%",
        ),

        rx.box(height="1px", background=styles.SIDEBAR_BORDER, width="100%"),

        # ── Navigation ───────────────────────────────────────────────────────
        rx.vstack(
            rx.text(
                "PIPELINE STEPS",
                font_size="9px", font_weight="700", color="#4b5563",
                letter_spacing="0.12em", padding="12px 12px 6px",
            ),
            _nav_item("01  Upload Dataset",            "upload",      0),
            _nav_item("02  Ingestion & Feature Check", "ingest",      1),
            _nav_item("03  Health & Quality Check",    "health",      2),
            _nav_item("04  Feature Engineering",       "features",    3),
            _nav_item("05  Clustering",                "clustering",  4),
            _nav_item("06  Persona Intelligence",      "persona",     5),
            spacing="0", width="100%", align="start",
        ),

        rx.spacer(),

        # ── Footer status ────────────────────────────────────────────────────
        rx.box(
            rx.hstack(
                rx.box(
                    width="6px", height="6px", border_radius="50%",
                    background=rx.cond(
                        PipelineState.step_status[5] == "complete",
                        styles.GREEN,
                        rx.cond(
                            PipelineState.step_status[4] == "complete",
                            styles.BLUE,
                            "#4b5563",
                        ),
                    ),
                ),
                rx.text(
                    rx.cond(
                        PipelineState.step_status[5] == "complete",
                        "Personas ready ✓",
                        rx.cond(
                            PipelineState.step_status[4] == "complete",
                            "Clustering done",
                            rx.cond(
                                PipelineState.step_status[3] == "complete",
                                "Features ready",
                                rx.cond(
                                    PipelineState.all_uploaded,
                                    "Files uploaded",
                                    "Awaiting input",
                                ),
                            ),
                        ),
                    ),
                    font_size="11px",
                    color=rx.cond(
                        PipelineState.step_status[5] == "complete",
                        styles.GREEN,
                        rx.cond(
                            PipelineState.step_status[4] == "complete",
                            styles.BLUE,
                            "#6b7280",
                        ),
                    ),
                    font_weight="500",
                ),
                spacing="2", align="center",
            ),
            padding="12px 14px",
            border_top=f"1px solid {styles.SIDEBAR_BORDER}",
            width="100%",
        ),

        width="220px", min_width="220px",
        height="100vh",
        background=styles.SIDEBAR_BG,
        align="start", spacing="0",
        overflow_y="auto",
    )
