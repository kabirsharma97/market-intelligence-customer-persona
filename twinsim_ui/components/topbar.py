"""Top bar — page title + breadcrumb."""
import reflex as rx
from twinsim_ui.state import PipelineState
from twinsim_ui import styles


def topbar() -> rx.Component:
    return rx.hstack(
        rx.vstack(
            rx.text(
                rx.cond(PipelineState.active_page == "upload",   "Upload Dataset",
                rx.cond(PipelineState.active_page == "ingest",   "Data Ingestion & Feature Presence Check",
                rx.cond(PipelineState.active_page == "health",   "Feature Health & Quality Check",
                rx.cond(PipelineState.active_page == "features", "Feature Engineering",
                "TwinSim Pipeline")))),
                font_size="17px", font_weight="600", color="#111827",
            ),
            rx.text(
                rx.cond(PipelineState.active_page == "upload",
                    "Upload your three source files to begin the pipeline",
                rx.cond(PipelineState.active_page == "ingest",
                    "Row & column validation · feature coverage · session enrichment",
                rx.cond(PipelineState.active_page == "health",
                    "239 automated quality checks across session & profile data",
                rx.cond(PipelineState.active_page == "features",
                    "50-feature ML store · 12-check quality assessment",
                "")))),
                font_size="12px", color="#6B7280",
            ),
            spacing="0", align="start",
        ),
        rx.spacer(),
        rx.hstack(
            rx.box(
                rx.text("EXL", color="white", font_size="10px", font_weight="800"),
                background=styles.ACCENT,
                border_radius="4px",
                padding="4px 8px",
            ),
            rx.text("TwinSim v5.0", font_size="11px", color="#9CA3AF", font_weight="500"),
            spacing="2", align="center",
        ),
        padding="14px 20px",
        background="white",
        border_bottom="1px solid #E5E7EB",
        width="100%",
        align="center",
    )
