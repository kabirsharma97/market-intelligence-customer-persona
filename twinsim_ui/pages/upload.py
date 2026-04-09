"""Screen 1 — Upload Dataset."""
import reflex as rx
from twinsim_ui.state import PipelineState
from twinsim_ui import styles


def _template_download_banner() -> rx.Component:
    """Download banner for the Feature Store template — shown at the very top."""
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.box(
                    rx.text("📥", font_size="20px"),
                    background=styles.ACCENT_BG,
                    border_radius="8px",
                    width="40px", height="40px",
                    display="flex", align_items="center", justify_content="center",
                    flex_shrink="0",
                ),
                rx.vstack(
                    rx.text(
                        "Feature Store Template — Layer 1",
                        font_size="13px", font_weight="600", color="#111827",
                    ),
                    rx.text(
                        "Download the Excel sheet to understand all 81 expected features "
                        "across session events, user profiles, and content catalogue.",
                        font_size="11px", color="#6B7280",
                    ),
                    spacing="0", align="start",
                ),
                spacing="3", align="center", flex="1",
            ),
            rx.link(
                rx.hstack(
                    rx.text("⬇", font_size="13px"),
                    rx.text(
                        "Download Template",
                        font_size="12px", font_weight="600",
                    ),
                    spacing="1", align="center",
                ),
                href="/feature_store_template.xlsx",
                color="white",
                background=styles.ACCENT,
                border_radius="8px",
                padding="8px 16px",
                _hover={"background": styles.ACCENT_HOVER},
                text_decoration="none",
                flex_shrink="0",
            ),
            spacing="4", align="center", width="100%",
        ),
        padding="14px 18px",
        background=f"linear-gradient(135deg, {styles.ACCENT_BG}, rgba(232,70,30,0.04))",
        border=f"1px solid rgba(232,70,30,0.25)",
        border_radius="12px",
        width="100%",
    )


def _file_drop_zone(
    label: str,
    description: str,
    upload_id: str,
    filename_var,
    handler,
    icon: str = "📄",
) -> rx.Component:
    csv_accept = {
        "text/csv": [".csv"],
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    }
    return rx.upload(
        rx.vstack(
            rx.cond(
                filename_var != "",
                rx.hstack(
                    rx.box(
                        rx.text("✓", color="white", font_size="12px", font_weight="700"),
                        background=styles.GREEN,
                        border_radius="50%",
                        width="28px", height="28px",
                        display="flex", align_items="center", justify_content="center",
                    ),
                    rx.vstack(
                        rx.text(filename_var, font_size="12px", font_weight="500",
                                color="#111827", word_break="break-all"),
                        rx.text("Uploaded successfully", font_size="11px", color=styles.GREEN),
                        spacing="0", align="start",
                    ),
                    spacing="3", align="center", width="100%",
                    padding="20px 16px",
                ),
                rx.vstack(
                    rx.text(icon, font_size="26px"),
                    rx.text(label, font_size="13px", font_weight="600", color="#111827"),
                    rx.text(description, font_size="11px", color="#6B7280",
                            text_align="center"),
                    rx.box(
                        rx.text("Browse or drop file",
                                font_size="11px", color=styles.ACCENT, font_weight="500"),
                        padding="5px 12px",
                        border=f"1.5px solid {styles.ACCENT}",
                        border_radius="6px",
                        cursor="pointer",
                        _hover={"background": styles.ACCENT_BG},
                    ),
                    spacing="2", align="center", width="100%",
                    padding="22px 16px",
                ),
            ),
            width="100%",
        ),
        id=upload_id,
        accept=csv_accept,
        max_files=1,
        on_drop=handler(rx.upload_files(upload_id=upload_id)),
        border=rx.cond(
            filename_var != "",
            f"2px solid {styles.GREEN}",
            "1.5px dashed #D1D5DB",
        ),
        border_radius="12px",
        background=rx.cond(filename_var != "", styles.GREEN_BG, "#FAFAFA"),
        width="100%",
        cursor="pointer",
        _hover={"border_color": styles.ACCENT, "background": styles.ACCENT_BG},
        transition="all 0.15s ease",
        padding="0",
    )


def upload_page() -> rx.Component:
    return rx.center(
        rx.vstack(
            # ── Feature Store Template download ──────────────────────────────
            _template_download_banner(),

            # ── Section header ───────────────────────────────────────────────
            rx.hstack(
                rx.box(
                    rx.text("01", color="white", font_size="11px", font_weight="700"),
                    background=styles.ACCENT,
                    border_radius="50%",
                    width="28px", height="28px",
                    display="flex", align_items="center", justify_content="center",
                ),
                rx.text(
                    "Upload three source files to begin",
                    font_size="15px", font_weight="600", color="#111827",
                ),
                spacing="3", align="center",
            ),

            # ── Source description chips ─────────────────────────────────────
            rx.grid(
                rx.box(
                    rx.hstack(
                        rx.box(width="10px", height="10px", border_radius="50%",
                               background="#22C55E", flex_shrink="0"),
                        rx.vstack(
                            rx.text("user_profiles.csv", font_size="12px",
                                    font_weight="600", color="#111827"),
                            rx.text("38 cols · subscription, device & engagement",
                                    font_size="11px", color="#6B7280"),
                            spacing="0", align="start",
                        ),
                        spacing="2", align="center",
                    ),
                    **styles.CARD_COMPACT,
                ),
                rx.box(
                    rx.hstack(
                        rx.box(width="10px", height="10px", border_radius="50%",
                               background="#3B82F6", flex_shrink="0"),
                        rx.vstack(
                            rx.text("session_events.csv", font_size="12px",
                                    font_weight="600", color="#111827"),
                            rx.text("36 cols · user watch events & quality signals",
                                    font_size="11px", color="#6B7280"),
                            spacing="0", align="start",
                        ),
                        spacing="2", align="center",
                    ),
                    **styles.CARD_COMPACT,
                ),
                rx.box(
                    rx.hstack(
                        rx.box(width="10px", height="10px", border_radius="50%",
                               background=styles.ACCENT, flex_shrink="0"),
                        rx.vstack(
                            rx.text("content_catalogue.csv", font_size="12px",
                                    font_weight="600", color="#111827"),
                            rx.text("9 cols · content type, genre & availability",
                                    font_size="11px", color="#6B7280"),
                            spacing="0", align="start",
                        ),
                        spacing="2", align="center",
                    ),
                    **styles.CARD_COMPACT,
                ),
                columns="3", spacing="3", width="100%",
            ),

            # ── Drop zones (order: profiles → sessions → catalogue) ──────────
            rx.grid(
                _file_drop_zone(
                    "User Profiles",
                    "user_profiles.csv · 38 columns",
                    "upload_profiles",
                    PipelineState.profiles_filename,
                    PipelineState.handle_profiles_upload,
                    "👤",
                ),
                _file_drop_zone(
                    "Session Events",
                    "session_events.csv · 36 columns",
                    "upload_sessions",
                    PipelineState.sessions_filename,
                    PipelineState.handle_sessions_upload,
                    "📊",
                ),
                _file_drop_zone(
                    "Content Catalogue",
                    "content_catalogue.csv · 9 columns",
                    "upload_catalogue",
                    PipelineState.catalogue_filename,
                    PipelineState.handle_catalogue_upload,
                    "🎬",
                ),
                columns="3", spacing="4", width="100%",
            ),

            # ── Upload progress dots ──────────────────────────────────────────
            rx.box(
                rx.hstack(
                    rx.hstack(
                        rx.box(width="8px", height="8px", border_radius="50%",
                               background=rx.cond(PipelineState.profiles_filename != "",
                                                  styles.GREEN, "#D1D5DB")),
                        rx.text("User Profiles", font_size="11px", color="#6B7280"),
                        spacing="2", align="center",
                    ),
                    rx.hstack(
                        rx.box(width="8px", height="8px", border_radius="50%",
                               background=rx.cond(PipelineState.sessions_filename != "",
                                                  styles.GREEN, "#D1D5DB")),
                        rx.text("Session Events", font_size="11px", color="#6B7280"),
                        spacing="2", align="center",
                    ),
                    rx.hstack(
                        rx.box(width="8px", height="8px", border_radius="50%",
                               background=rx.cond(PipelineState.catalogue_filename != "",
                                                  styles.GREEN, "#D1D5DB")),
                        rx.text("Content Catalogue", font_size="11px", color="#6B7280"),
                        spacing="2", align="center",
                    ),
                    spacing="5", justify="center",
                ),
                padding="10px",
                background="#F9FAFB",
                border_radius="8px",
                border="1px solid #E5E7EB",
                width="100%",
            ),

            # ── Error ─────────────────────────────────────────────────────────
            rx.cond(
                PipelineState.upload_error != "",
                rx.box(
                    rx.hstack(
                        rx.text("⚠", color=styles.RED, font_size="14px"),
                        rx.text(PipelineState.upload_error, font_size="12px",
                                color=styles.RED, flex="1"),
                        spacing="2", align="start",
                    ),
                    padding="12px",
                    border_radius="8px",
                    background=styles.RED_BG,
                    border=f"1px solid {styles.RED}",
                    width="100%",
                ),
                rx.box(),
            ),

            # ── Ingest button ─────────────────────────────────────────────────
            rx.button(
                rx.cond(
                    PipelineState.all_uploaded,
                    "Click to Ingest Data  →",
                    "Upload all 3 files to continue",
                ),
                on_click=PipelineState.ingest_data,
                disabled=~PipelineState.all_uploaded,
                background=rx.cond(PipelineState.all_uploaded, styles.ACCENT, "#D1D5DB"),
                color=rx.cond(PipelineState.all_uploaded, "white", "#9CA3AF"),
                border_radius="8px",
                padding="10px 28px",
                font_size="14px",
                font_weight="600",
                cursor=rx.cond(PipelineState.all_uploaded, "pointer", "not-allowed"),
                _hover=rx.cond(
                    PipelineState.all_uploaded,
                    {"background": styles.ACCENT_HOVER}, {},
                ),
                width="100%",
                height="44px",
            ),

            spacing="4", max_width="820px", width="100%",
            padding="20px 0",
        ),
        width="100%",
        padding="0 24px",
    )
