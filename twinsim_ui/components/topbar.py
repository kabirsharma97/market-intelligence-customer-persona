"""Top bar — page title + subtitle only (no TwinSim badge)."""
import reflex as rx
from twinsim_ui.state import PipelineState


def topbar() -> rx.Component:
    return rx.hstack(
        rx.vstack(
            rx.text(
                rx.cond(PipelineState.active_page == "upload",      "Upload Dataset",
                rx.cond(PipelineState.active_page == "ingest",      "Data Ingestion & Feature Presence Check",
                rx.cond(PipelineState.active_page == "health",      "Feature Health & Quality Check",
                rx.cond(PipelineState.active_page == "features",    "Feature Engineering",
                rx.cond(PipelineState.active_page == "clustering",  "Clustering",
                rx.cond(PipelineState.active_page == "persona",     "Persona Intelligence",
                "Market Intelligence")))))),
                font_size="17px", font_weight="600", color="#111827",
            ),
            rx.text(
                rx.cond(PipelineState.active_page == "upload",
                    "Download the feature store template, then upload your three source files",
                rx.cond(PipelineState.active_page == "ingest",
                    "Row & column validation · feature coverage · session enrichment merge",
                rx.cond(PipelineState.active_page == "health",
                    "239 automated quality checks across session & profile data",
                rx.cond(PipelineState.active_page == "features",
                    "50-feature ML store · 14-check quality assessment · clustering ready",
                rx.cond(PipelineState.active_page == "clustering",
                    "K-Means behavioural segmentation · stability scoring · audit gate",
                rx.cond(PipelineState.active_page == "persona",
                    "Behavioural archetypes · intervention playbooks · feature fingerprints",
                "")))))),
                font_size="12px", color="#6B7280",
            ),
            spacing="0", align="start",
        ),
        padding="14px 20px",
        background="white",
        border_bottom="1px solid #E5E7EB",
        width="100%",
        align="center",
    )
