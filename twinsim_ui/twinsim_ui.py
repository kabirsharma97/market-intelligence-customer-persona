"""
Market Intelligence Driven by Customer Persona
TwinSim v5.0 — End-to-end OTT customer persona intelligence platform.

Pipeline:
  Screen 1: Upload Dataset (3 source CSV files)
  Screen 2: Data Ingestion + Feature Presence Check
  Screen 3: Feature Health & Quality Check (239 checks)
  Screen 4: Feature Engineering (50 features · 12-check QA)
"""
import reflex as rx

from twinsim_ui.state import PipelineState
from twinsim_ui.components.sidebar import sidebar
from twinsim_ui.components.topbar import topbar
from twinsim_ui.pages.upload import upload_page
from twinsim_ui.pages.ingest import ingest_page
from twinsim_ui.pages.health import health_page
from twinsim_ui.pages.features import features_page


def index() -> rx.Component:
    return rx.theme(
        rx.hstack(
            sidebar(),
            rx.vstack(
                topbar(),
                rx.box(
                    rx.cond(
                        PipelineState.active_page == "upload",
                        upload_page(),
                        rx.cond(
                            PipelineState.active_page == "ingest",
                            ingest_page(),
                            rx.cond(
                                PipelineState.active_page == "health",
                                health_page(),
                                rx.cond(
                                    PipelineState.active_page == "features",
                                    features_page(),
                                    upload_page(),
                                ),
                            ),
                        ),
                    ),
                    flex="1",
                    overflow_y="auto",
                    padding="24px",
                    width="100%",
                    background="#F9FAFB",
                ),
                width="100%",
                height="100vh",
                spacing="0",
            ),
            width="100%",
            height="100vh",
            overflow="hidden",
            spacing="0",
        ),
        appearance="light",
        has_background=True,
        radius="medium",
        accent_color="tomato",
    )


app = rx.App(
    theme=rx.theme(
        appearance="light",
        has_background=True,
        radius="medium",
        accent_color="tomato",
    ),
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    ],
    style={
        "font_family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
    },
)
app.add_page(index, route="/")
