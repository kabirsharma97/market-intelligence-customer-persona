"""Screen 2 — Data Ingestion & Feature Presence Check."""
import reflex as rx
from twinsim_ui.state import PipelineState
from twinsim_ui import styles


# ── Helper: file summary tile ─────────────────────────────────────────────────
def _file_tile(source: str, rows_fmt, cols, color: str) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.box(width="10px", height="10px", border_radius="50%",
                   background=color, flex_shrink="0"),
            rx.text(source, font_size="12px", font_weight="600", color="#374151"),
            spacing="2", align="center",
        ),
        rx.text("Records", font_size="10px", font_weight="600",
                color="#9CA3AF", letter_spacing="0.06em"),
        rx.text(
            rows_fmt,
            font_size="26px", font_weight="700", color="#111827",
        ),
        rx.text(cols.to_string() + " columns",
                font_size="11px", color="#9CA3AF"),
        spacing="1", align="start",
        padding="14px 16px",
        background="white",
        border="1px solid #E5E7EB",
        border_radius="12px",
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




def _category_row(row: dict) -> rx.Component:
    is_expanded = PipelineState.s2_expanded_categories.contains(row["category"])
    return rx.vstack(
        # ── Main row (always visible) ──────────────────────────────────────
        rx.hstack(
            # Chevron toggle
            rx.box(
                rx.text(
                    rx.cond(is_expanded, "▾", "▸"),
                    font_size="12px",
                    color=rx.cond(is_expanded, styles.ACCENT, "#9CA3AF"),
                ),
                cursor="pointer",
                on_click=PipelineState.toggle_category(row["category"]),
                width="20px", flex_shrink="0",
            ),
            rx.text(row["category"], font_size="12px", color="#374151",
                    font_weight="500", flex="2", cursor="pointer",
                    on_click=PipelineState.toggle_category(row["category"])),
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
        ),
        # ── Expanded: feature list ─────────────────────────────────────────
        rx.cond(
            is_expanded,
            rx.box(
                rx.text(
                    row["features_str"],
                    font_size="11px",
                    color="#6B7280",
                    font_family="monospace",
                    line_height="1.8",
                    word_break="break-word",
                ),
                padding="8px 12px 10px 28px",
                background="#F9FAFB",
                border_radius="8px",
                border="1px solid #E5E7EB",
                width="100%",
            ),
            rx.box(),
        ),
        spacing="0", width="100%",
        border_bottom="1px solid #F3F4F6",
    )


def ingest_page() -> rx.Component:
    return rx.vstack(

        # ── Running spinner ──────────────────────────────────────────────────
        rx.cond(
            PipelineState.s2_running,
            rx.box(
                rx.hstack(
                    rx.spinner(size="2"),
                    rx.text("Loading files and checking feature coverage…",
                            font_size="13px", color="#374151"),
                    spacing="3", align="center",
                ),
                padding="14px 18px",
                background="#EFF6FF",
                border="1px solid #BFDBFE",
                border_radius="10px",
                width="100%",
            ),
            rx.box(),
        ),

        # ── Error ────────────────────────────────────────────────────────────
        rx.cond(
            PipelineState.s2_error != "",
            rx.box(
                rx.hstack(
                    rx.text("⚠", font_size="14px", color=styles.RED),
                    rx.text(PipelineState.s2_error, font_size="12px",
                            color=styles.RED, flex="1", word_break="break-word"),
                    spacing="2", align="start",
                ),
                padding="12px", background=styles.RED_BG,
                border=f"1px solid {styles.RED}", border_radius="8px", width="100%",
            ),
            rx.box(),
        ),

        # ── SECTION 1: File Counts ───────────────────────────────────────────
        rx.cond(
            (PipelineState.s2_profiles_rows > 0) | PipelineState.s2_feature_checked,
            rx.box(
                rx.vstack(
                    rx.text("Step 1 — File Summary",
                            font_size="13px", font_weight="600", color="#111827"),
                    rx.text("Records and columns loaded from each uploaded file",
                            font_size="11px", color="#6B7280"),
                    rx.grid(
                        _file_tile("User Profiles",    PipelineState.s2_profiles_rows_fmt,
                                   PipelineState.s2_profiles_cols,  "#22C55E"),
                        _file_tile("Session Events",   PipelineState.s2_sessions_rows_fmt,
                                   PipelineState.s2_sessions_cols,  "#3B82F6"),
                        _file_tile("Content Catalogue",PipelineState.s2_catalogue_rows_fmt,
                                   PipelineState.s2_catalogue_cols, styles.ACCENT),
                        columns="3", spacing="3", width="100%",
                    ),
                    spacing="3", width="100%",
                ),
                **styles.CARD,
                width="100%",
            ),
            rx.box(),
        ),

        # ── SECTION 2: Feature Presence Check ───────────────────────────────
        rx.cond(
            PipelineState.s2_feature_checked,
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.text("Step 2 — Feature Presence Check",
                                    font_size="13px", font_weight="600", color="#111827"),
                            rx.text(
                                "Expected features verified across all three source files",
                                font_size="11px", color="#6B7280",
                            ),
                            spacing="0", align="start",
                        ),
                        rx.spacer(),
                        rx.vstack(
                            rx.text(
                                PipelineState.s2_coverage_pct.to_string() + "%",
                                font_size="22px", font_weight="700",
                                color=rx.cond(
                                    PipelineState.s2_coverage_pct >= 100,
                                    styles.GREEN,
                                    rx.cond(PipelineState.s2_coverage_pct >= 80,
                                            styles.AMBER, styles.RED),
                                ),
                            ),
                            rx.text("Coverage", font_size="10px", color="#9CA3AF"),
                            spacing="0", align="center",
                        ),
                        width="100%", align="center",
                    ),
                    rx.box(height="1px", background="#E5E7EB", width="100%"),
                    rx.hstack(
                        rx.text("Feature Category", font_size="11px", font_weight="600",
                                color="#9CA3AF", flex="2"),
                        rx.text("Present / Expected", font_size="11px", font_weight="600",
                                color="#9CA3AF", flex="1", text_align="center"),
                        rx.text("Coverage Bar", font_size="11px", font_weight="600",
                                color="#9CA3AF", flex="2"),
                        rx.text("Status", font_size="11px", font_weight="600",
                                color="#9CA3AF"),
                        spacing="3", width="100%", padding="6px 0",
                    ),
                    rx.foreach(PipelineState.s2_category_rows, _category_row),
                    # Missing features
                    rx.cond(
                        PipelineState.s2_missing_features.length() > 0,
                        rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.text("⚠", color=styles.AMBER),
                                    rx.text("Missing Features", font_size="12px",
                                            font_weight="600", color="#92400E"),
                                    spacing="2", align="center",
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
                            padding="12px",
                            background=styles.AMBER_BG,
                            border=f"1px solid {styles.AMBER}",
                            border_radius="8px",
                            width="100%",
                        ),
                        rx.box(),
                    ),
                    spacing="2", width="100%",
                ),
                **styles.CARD,
                width="100%",
            ),
            rx.box(),
        ),

        # ── SECTION 3: Session Data Preparation (Merge step) ─────────────────
        rx.cond(
            PipelineState.s2_feature_checked,
            rx.box(
                rx.vstack(
                    # Header
                    rx.hstack(
                        rx.box(
                            rx.text("⚙", font_size="16px"),
                            background=styles.ACCENT_BG,
                            border_radius="8px",
                            width="38px", height="38px",
                            display="flex", align_items="center", justify_content="center",
                            flex_shrink="0",
                        ),
                        rx.vstack(
                            rx.text("Step 3 — Session Data Preparation",
                                    font_size="13px", font_weight="600", color="#111827"),
                            rx.text(
                                "Combines viewing activity with content details, "
                                "figures out where each episode sits in a series, "
                                "and produces a single enriched dataset ready for analysis",
                                font_size="11px", color="#6B7280",
                            ),
                            spacing="0", align="start",
                        ),
                        spacing="3", align="center", width="100%",
                    ),

                    rx.box(height="1px", background="#E5E7EB", width="100%"),

                    # State: not started yet → show Run button
                    rx.cond(
                        ~PipelineState.s2_merge_done & ~PipelineState.s2_merge_running,
                        rx.hstack(
                            rx.box(
                                rx.text("Ready to merge session and content data",
                                        font_size="12px", color="#6B7280"),
                                flex="1",
                            ),
                            rx.button(
                                "Run Session Data Preparation  →",
                                on_click=PipelineState.run_merge,
                                background=styles.ACCENT,
                                color="white",
                                border_radius="8px",
                                padding="8px 20px",
                                font_size="12px",
                                font_weight="600",
                                cursor="pointer",
                                _hover={"background": styles.ACCENT_HOVER},
                            ),
                            spacing="3", align="center", width="100%",
                        ),
                        rx.box(),
                    ),

                    # State: running → spinner
                    rx.cond(
                        PipelineState.s2_merge_running,
                        rx.hstack(
                            rx.spinner(size="2"),
                            rx.vstack(
                                rx.text("Preparing session data…",
                                        font_size="12px", font_weight="500", color="#374151"),
                                rx.text(
                                    "Joining content metadata · computing episode position labels",
                                    font_size="11px", color="#6B7280",
                                ),
                                spacing="0", align="start",
                            ),
                            spacing="3", align="center",
                        ),
                        rx.box(),
                    ),

                    # State: done → result
                    rx.cond(
                        PipelineState.s2_merge_done,
                        rx.hstack(
                            rx.box(
                                rx.text("✓", color="white", font_size="12px"),
                                background=styles.GREEN,
                                border_radius="50%",
                                width="26px", height="26px",
                                display="flex", align_items="center", justify_content="center",
                                flex_shrink="0",
                            ),
                            rx.vstack(
                                rx.text("Enriched session dataset ready",
                                        font_size="12px", font_weight="600", color="#374151"),
                                rx.hstack(
                                    rx.text(PipelineState.s2_enriched_rows_fmt,
                                            font_size="13px", font_weight="700",
                                            color=styles.GREEN),
                                    rx.text("records ·", color="#9CA3AF",
                                            font_size="13px"),
                                    rx.text(PipelineState.s2_enriched_cols.to_string() +
                                            " columns",
                                            font_size="13px", color="#6B7280"),
                                    spacing="2", align="center",
                                ),
                                spacing="0", align="start",
                            ),
                            spacing="3", align="center", width="100%",
                        ),
                        rx.box(),
                    ),

                    # Merge error
                    rx.cond(
                        PipelineState.s2_merge_error != "",
                        rx.box(
                            rx.text(PipelineState.s2_merge_error, font_size="11px",
                                    color=styles.RED, word_break="break-word"),
                            padding="10px", background=styles.RED_BG,
                            border=f"1px solid {styles.RED}", border_radius="6px",
                            width="100%",
                        ),
                        rx.box(),
                    ),

                    spacing="3", width="100%",
                ),
                **styles.CARD,
                width="100%",
            ),
            rx.box(),
        ),

        # ── Gate: proceed to health check ────────────────────────────────────
        rx.cond(
            PipelineState.s2_merge_done,
            rx.hstack(
                rx.box(
                    rx.hstack(
                        rx.text("✓", color="white", font_size="13px"),
                        rx.text("All 3 steps complete — ready to run quality checks",
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
            rx.box(),
        ),

        spacing="4", width="100%", align="start",
    )
