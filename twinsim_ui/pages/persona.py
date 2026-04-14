"""Screen 6 — Persona Intelligence (overview + detail view 6B)."""
import reflex as rx
from twinsim_ui.state import PipelineState
from twinsim_ui import styles


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: churn colour
# ─────────────────────────────────────────────────────────────────────────────

def _churn_color(level: rx.Var) -> rx.Var:
    """level is a string Var: 'high' | 'medium' | 'low'"""
    return rx.cond(level == "high", styles.RED,
                   rx.cond(level == "medium", styles.AMBER, styles.GREEN))


def _churn_bg(level: rx.Var) -> rx.Var:
    """level is a string Var: 'high' | 'medium' | 'low'"""
    return rx.cond(level == "high", styles.RED_BG,
                   rx.cond(level == "medium", styles.AMBER_BG, styles.GREEN_BG))


# ─────────────────────────────────────────────────────────────────────────────
#  Persona Summary Bar (thin strip at top — same on overview and detail)
# ─────────────────────────────────────────────────────────────────────────────

def _summary_chip(p: dict, idx: int) -> rx.Component:
    is_active = PipelineState.s6_active_persona_idx == idx
    return rx.box(
        rx.hstack(
            # Avatar circle
            rx.box(
                rx.text(p["avatar"], font_size="22px", line_height="1"),
                width="44px", height="44px",
                background=rx.cond(is_active, "rgba(255,255,255,0.20)", "#F3F4F6"),
                border_radius="12px",
                display="flex", align_items="center", justify_content="center",
                flex_shrink="0",
            ),
            rx.vstack(
                rx.text(p["name"], font_size="13px", font_weight="700",
                        color=rx.cond(is_active, "white", "#111827"),
                        white_space="nowrap"),
                rx.hstack(
                    rx.box(
                        width="6px", height="6px", border_radius="50%",
                        background=rx.cond(
                            is_active,
                            "rgba(255,255,255,0.80)",
                            _churn_color(p["churn_level"]),
                        ),
                        flex_shrink="0",
                    ),
                    rx.text(
                        p["churn"].to_string() + "% churn",
                        font_size="11px", font_weight="500",
                        color=rx.cond(
                            is_active,
                            "rgba(255,255,255,0.85)",
                            _churn_color(p["churn_level"]),
                        ),
                    ),
                    spacing="1", align="center",
                ),
                spacing="1", align="start",
            ),
            spacing="3", align="center",
        ),
        padding="10px 16px",
        border_radius="14px",
        cursor="pointer",
        background=rx.cond(is_active, styles.ACCENT, "white"),
        border=rx.cond(
            is_active,
            f"2px solid {styles.ACCENT}",
            "1.5px solid #E5E7EB",
        ),
        box_shadow=rx.cond(
            is_active,
            "0 4px 12px rgba(232,70,30,0.30)",
            "0 1px 3px rgba(0,0,0,0.06)",
        ),
        transition="all 0.15s ease",
        on_click=PipelineState.set_active_persona(idx),
        _hover={
            "box_shadow": rx.cond(
                is_active,
                "0 4px 12px rgba(232,70,30,0.40)",
                "0 3px 10px rgba(0,0,0,0.10)",
            ),
            "transform": "translateY(-1px)",
        },
    )


def _summary_bar() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(
                "Persona Summary",
                font_size="11px", font_weight="700", color="#9CA3AF",
                letter_spacing="0.08em",
            ),
            rx.hstack(
                rx.foreach(PipelineState.s6_personas, lambda p, i: _summary_chip(p, i)),
                rx.spacer(),
                rx.cond(
                    PipelineState.s6_active_persona_idx >= 0,
                    rx.button(
                        "← Back to Overview",
                        on_click=PipelineState.back_to_overview(),
                        background="transparent",
                        color=styles.ACCENT,
                        border=f"1px solid {styles.ACCENT}",
                        border_radius="8px",
                        font_size="12px",
                        font_weight="600",
                        padding="8px 16px",
                        cursor="pointer",
                        _hover={"background": styles.ACCENT_BG},
                    ),
                    rx.box(),
                ),
                spacing="3", align="center", wrap="wrap", width="100%",
            ),
            spacing="2", align="start", width="100%",
        ),
        padding="16px 20px",
        background="white",
        border="1px solid #E5E7EB",
        border_radius="14px",
        width="100%",
        box_shadow="0 1px 4px rgba(0,0,0,0.05)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Overview card (Screen 6)
# ─────────────────────────────────────────────────────────────────────────────

def _intervention_chip(iv: dict) -> rx.Component:
    return rx.badge(
        iv["name"],
        color_scheme="blue", variant="soft", size="1",
    )


def _persona_card(p: dict, idx: int) -> rx.Component:
    return rx.box(
        rx.vstack(
            # ── Header ──────────────────────────────────────────────────────
            rx.hstack(
                rx.box(
                    rx.text(p["avatar"], font_size="28px", line_height="1"),
                    width="48px", height="48px",
                    background="#F3F4F6", border_radius="12px",
                    display="flex", align_items="center",
                    justify_content="center",
                ),
                rx.vstack(
                    rx.text(p["name"], font_size="13px", font_weight="700",
                            color="#111827", line_height="1.3"),
                    rx.text(p["label"], font_size="10px", color="#9CA3AF",
                            font_family="monospace"),
                    spacing="0", align="start", flex="1",
                ),
                rx.vstack(
                    rx.box(
                        rx.hstack(
                            rx.icon("users", size=10, color="#6B7280"),
                            rx.text(p["size_pct"].to_string() + "%",
                                    font_size="11px", font_weight="700",
                                    color="#374151"),
                            spacing="1", align="center",
                        ),
                        padding="3px 8px",
                        background="#F3F4F6",
                        border_radius="10px",
                        border="1px solid #E5E7EB",
                    ),
                    rx.button(
                        "⬇ PDF",
                        on_click=PipelineState.download_persona_pdf(idx),
                        background="transparent",
                        color="#6B7280",
                        border="1px solid #E5E7EB",
                        border_radius="6px",
                        font_size="10px",
                        font_weight="600",
                        padding="3px 8px",
                        cursor="pointer",
                        _hover={"background": "#F3F4F6", "color": "#111827"},
                    ),
                    spacing="1", align="end",
                ),
                spacing="3", align="start", width="100%",
            ),

            # ── Description ─────────────────────────────────────────────────
            rx.text(
                p["description"],
                font_size="11px", color="#4B5563",
                line_height="1.55",
            ),

            # ── Metric strip ────────────────────────────────────────────────
            rx.hstack(
                rx.vstack(
                    rx.text(p["churn"].to_string() + "%",
                            font_size="16px", font_weight="700",
                            color=_churn_color(p["churn_level"])),
                    rx.tooltip(
                        rx.text("CHURN ⓘ", font_size="8px", font_weight="700",
                                color="#9CA3AF", letter_spacing="0.08em",
                                cursor="help"),
                        content="% of users in this group who stopped using the service in the last 30 days",
                    ),
                    spacing="0", align="center",
                ),
                rx.box(width="1px", height="32px", background="#E5E7EB"),
                rx.vstack(
                    rx.text(p["completion"].to_string() + "%",
                            font_size="16px", font_weight="700",
                            color=styles.BLUE),
                    rx.tooltip(
                        rx.text("COMPLETION ⓘ", font_size="8px", font_weight="700",
                                color="#9CA3AF", letter_spacing="0.08em",
                                cursor="help"),
                        content="% of sessions where users watched content to the end",
                    ),
                    spacing="0", align="center",
                ),
                rx.box(width="1px", height="32px", background="#E5E7EB"),
                rx.vstack(
                    rx.text(p["reactivation"].to_string() + "%",
                            font_size="16px", font_weight="700",
                            color=styles.PURPLE),
                    rx.tooltip(
                        rx.text("RE-ENGAGE ⓘ", font_size="8px", font_weight="700",
                                color="#9CA3AF", letter_spacing="0.08em",
                                cursor="help"),
                        content="Likelihood that a dormant user in this group comes back and watches again",
                    ),
                    spacing="0", align="center",
                ),
                spacing="4", align="center", justify="center",
                width="100%",
                padding="10px 0",
                border_top="1px solid #F3F4F6",
                border_bottom="1px solid #F3F4F6",
            ),

            # ── Risk flag ───────────────────────────────────────────────────
            rx.cond(
                p["high_churn"],
                rx.hstack(
                    rx.text("⚠", font_size="11px"),
                    rx.text("High churn risk", font_size="11px",
                            color=styles.RED, font_weight="500"),
                    spacing="1", align="center",
                    padding="4px 8px",
                    background=styles.RED_BG,
                    border_radius="6px",
                    border=f"1px solid {styles.RED}",
                ),
                rx.hstack(
                    rx.text("✓", font_size="11px"),
                    rx.text("Stable segment", font_size="11px",
                            color=styles.GREEN, font_weight="500"),
                    spacing="1", align="center",
                    padding="4px 8px",
                    background=styles.GREEN_BG,
                    border_radius="6px",
                    border=f"1px solid {styles.GREEN}",
                ),
            ),

            # ── CTA buttons ─────────────────────────────────────────────────
            rx.hstack(
                rx.button(
                    "Persona Details",
                    on_click=PipelineState.set_active_persona(idx),
                    background=styles.ACCENT,
                    color="white",
                    border_radius="7px",
                    font_size="11px",
                    font_weight="600",
                    padding="7px 14px",
                    cursor="pointer",
                    flex="1",
                    _hover={"background": styles.ACCENT_HOVER},
                ),
                rx.button(
                    "Interventions",
                    on_click=PipelineState.set_active_persona(idx),
                    background="transparent",
                    color=styles.ACCENT,
                    border=f"1px solid {styles.ACCENT}",
                    border_radius="7px",
                    font_size="11px",
                    font_weight="600",
                    padding="7px 14px",
                    cursor="pointer",
                    flex="1",
                    _hover={"background": styles.ACCENT_BG},
                ),
                rx.button(
                    "💬 Chat",
                    background="transparent",
                    color=styles.BLUE,
                    border=f"1px solid {styles.BLUE}",
                    border_radius="7px",
                    font_size="11px",
                    font_weight="600",
                    padding="7px 14px",
                    cursor="pointer",
                    flex="1",
                    _hover={"background": styles.BLUE_BG},
                    title="Digital Twin Chat — coming soon",
                ),
                spacing="2", width="100%",
            ),

            spacing="3", align="start", width="100%",
        ),
        background="white",
        border="1px solid #E5E7EB",
        border_radius="14px",
        padding="18px",
        _hover={"box_shadow": "0 4px 16px rgba(0,0,0,0.08)",
                "border_color": "#D1D5DB"},
        transition="all 0.2s ease",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Intervention card (Screen 6B)
# ─────────────────────────────────────────────────────────────────────────────

def _intervention_card(iv: dict) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(iv["name"], font_size="12px", font_weight="700",
                        color="#111827", flex="1"),
                rx.badge(iv["channel"], color_scheme="blue", variant="soft", size="1"),
                spacing="2", align="center", width="100%",
            ),
            rx.hstack(
                rx.text("Trigger:", font_size="10px", color="#9CA3AF",
                        font_weight="600"),
                rx.text(iv["trigger"], font_size="10px", color="#6B7280",
                        flex="1", word_break="break-word"),
                spacing="1", align="start",
            ),
            rx.hstack(
                rx.text("Retention lift:", font_size="10px", color="#9CA3AF",
                        font_weight="600"),
                rx.text(
                    "+" + iv["ret_lift"].to_string() + "%",
                    font_size="10px", color=styles.GREEN, font_weight="600",
                ),
                spacing="1", align="center",
            ),
            spacing="2", align="start",
        ),
        background="#F9FAFB",
        border="1px solid #E5E7EB",
        border_radius="10px",
        padding="12px 14px",
    )


def _feature_row(feat: dict) -> rx.Component:
    bar_width = feat["pct"].to_string() + "%"
    return rx.hstack(
        rx.text(feat["name"], font_size="11px", color="#374151",
                font_family="monospace", flex="2", word_break="break-all"),
        rx.box(
            rx.box(
                height="6px",
                border_radius="3px",
                background=styles.ACCENT,
                width=bar_width,
            ),
            width="100%",
            height="6px",
            background="#F3F4F6",
            border_radius="3px",
            flex="3",
        ),
        rx.text(feat["val"].to_string(), font_size="11px",
                color="#6B7280", text_align="right", width="40px"),
        spacing="3", align="center", width="100%",
        padding="4px 0",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Detail view (Screen 6B)
# ─────────────────────────────────────────────────────────────────────────────

def _detail_view() -> rx.Component:
    return rx.vstack(

        # ── Header card ───────────────────────────────────────────────────────
        rx.box(
            rx.hstack(
                rx.box(
                    rx.text(PipelineState.s6_active_avatar,
                            font_size="36px", line_height="1"),
                    width="64px", height="64px",
                    background="#F3F4F6", border_radius="16px",
                    display="flex", align_items="center",
                    justify_content="center",
                ),
                rx.vstack(
                    rx.text(PipelineState.s6_active_name,
                            font_size="20px", font_weight="800", color="#111827"),
                    rx.text(PipelineState.s6_active_label,
                            font_size="12px", color="#9CA3AF",
                            font_family="monospace"),
                    rx.text(PipelineState.s6_active_description,
                            font_size="12px", color="#6B7280",
                            line_height="1.55", max_width="600px"),
                    spacing="1", align="start",
                ),
                rx.spacer(),
                rx.cond(
                    PipelineState.s6_active_high_churn,
                    rx.badge("HIGH CHURN RISK", color_scheme="red",
                             variant="solid", size="2"),
                    rx.badge("STABLE", color_scheme="green",
                             variant="solid", size="2"),
                ),
                spacing="4", align="start", width="100%",
            ),
            **styles.CARD, width="100%",
        ),

        # ── KPI tiles ─────────────────────────────────────────────────────────
        rx.grid(
            rx.vstack(
                rx.text(PipelineState.s6_active_churn.to_string() + "%",
                        font_size="28px", font_weight="700",
                        color=_churn_color(PipelineState.s6_active_churn_level)),
                rx.text("Churn Rate", font_size="11px", font_weight="600",
                        color="#374151"),
                rx.text(
                    PipelineState.s6_active_churn_ci_lo.to_string() +
                    "–" +
                    PipelineState.s6_active_churn_ci_hi.to_string() +
                    "% CI",
                    font_size="9px", color="#9CA3AF",
                ),
                spacing="0", align="center", padding="14px",
                background=_churn_bg(PipelineState.s6_active_churn_level),
                border=rx.cond(
                    PipelineState.s6_active_churn_level == "high",
                    f"1px solid {styles.RED}",
                    rx.cond(PipelineState.s6_active_churn_level == "medium",
                            f"1px solid {styles.AMBER}",
                            f"1px solid {styles.GREEN}"),
                ),
                border_radius="12px", flex="1",
            ),
            rx.vstack(
                rx.text(PipelineState.s6_active_completion.to_string() + "%",
                        font_size="28px", font_weight="700", color=styles.BLUE),
                rx.text("Completion", font_size="11px", font_weight="600",
                        color="#374151"),
                rx.text("avg session completion rate",
                        font_size="9px", color="#9CA3AF"),
                spacing="0", align="center", padding="14px",
                background=styles.BLUE_BG,
                border=f"1px solid {styles.BLUE}",
                border_radius="12px", flex="1",
            ),
            rx.vstack(
                rx.text(PipelineState.s6_active_reactivation.to_string() + "%",
                        font_size="28px", font_weight="700", color=styles.PURPLE),
                rx.text("Reactivation", font_size="11px", font_weight="600",
                        color="#374151"),
                rx.text("re-engagement probability",
                        font_size="9px", color="#9CA3AF"),
                spacing="0", align="center", padding="14px",
                background=styles.PURPLE_BG,
                border=f"1px solid {styles.PURPLE}",
                border_radius="12px", flex="1",
            ),
            rx.vstack(
                rx.text(PipelineState.s6_active_size_fmt,
                        font_size="28px", font_weight="700", color="#111827"),
                rx.text("Segment Size", font_size="11px", font_weight="600",
                        color="#374151"),
                rx.text(PipelineState.s6_active_size_pct.to_string() + "% of base",
                        font_size="9px", color="#9CA3AF"),
                spacing="0", align="center", padding="14px",
                background="white",
                border="1px solid #E5E7EB",
                border_radius="12px", flex="1",
            ),
            columns="4", spacing="3", width="100%",
        ),

        # ── Two-column: Interventions + Feature Fingerprint ───────────────────
        rx.hstack(

            # Left: Intervention Playbook
            rx.box(
                rx.vstack(
                    rx.text("Intervention Playbook",
                            font_size="13px", font_weight="700", color="#111827"),
                    rx.text("Recommended actions ranked by expected lift",
                            font_size="11px", color="#6B7280"),
                    rx.box(height="1px", background="#E5E7EB", width="100%"),
                    rx.foreach(PipelineState.s6_active_interventions,
                               _intervention_card),
                    spacing="3", align="start", width="100%",
                ),
                **styles.CARD,
                flex="1",
            ),

            # Right: Feature Fingerprint
            rx.box(
                rx.vstack(
                    rx.text("Feature Fingerprint",
                            font_size="13px", font_weight="700", color="#111827"),
                    rx.text("Top behavioural signals — normalised importance scores",
                            font_size="11px", color="#6B7280"),
                    rx.box(height="1px", background="#E5E7EB", width="100%"),
                    rx.foreach(PipelineState.s6_active_features, _feature_row),
                    spacing="2", align="start", width="100%",
                ),
                **styles.CARD,
                flex="1",
            ),

            spacing="4", align="start", width="100%",
        ),

        # ── Narrative quote ───────────────────────────────────────────────────
        rx.cond(
            PipelineState.s6_active_narrative != "",
            rx.box(
                rx.hstack(
                    rx.box(
                        width="4px", min_height="40px",
                        background=styles.ACCENT,
                        border_radius="2px",
                        flex_shrink="0",
                    ),
                    rx.text(
                        PipelineState.s6_active_narrative,
                        font_size="13px", color="#374151",
                        line_height="1.65",
                        font_style="italic",
                    ),
                    spacing="3", align="start", width="100%",
                ),
                **styles.CARD,
                width="100%",
            ),
            rx.box(),
        ),

        spacing="4", align="start", width="100%",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Overview grid (Screen 6)
# ─────────────────────────────────────────────────────────────────────────────

def _overview_grid() -> rx.Component:
    return rx.vstack(
        # Persona count summary
        rx.hstack(
            rx.vstack(
                rx.text(
                    PipelineState.s6_personas.length().to_string() + " Personas Identified",
                    font_size="16px", font_weight="700", color="#111827",
                ),
                rx.text(
                    "Behavioural segmentation complete — click a card to explore",
                    font_size="12px", color="#6B7280",
                ),
                spacing="0", align="start",
            ),
            rx.spacer(),
            rx.badge(
                "PERSONAS READY",
                color_scheme="green", variant="solid", size="2",
            ),
            width="100%", align="center",
        ),

        # 2-column card grid
        rx.grid(
            rx.foreach(
                PipelineState.s6_personas,
                lambda p, i: _persona_card(p, i),
            ),
            columns="2",
            spacing="4",
            width="100%",
        ),

        spacing="4", width="100%", align="start",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Running / error states
# ─────────────────────────────────────────────────────────────────────────────

def _running_banner() -> rx.Component:
    return rx.cond(
        PipelineState.s6_running,
        rx.box(
            rx.hstack(
                rx.spinner(size="3"),
                rx.vstack(
                    rx.text("Building persona profiles…",
                            font_size="14px", font_weight="600", color="#111827"),
                    rx.text(
                        "persona_engine.py — building behavioural archetypes "
                        "and intervention playbooks",
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
    )


def _error_banner() -> rx.Component:
    return rx.cond(
        PipelineState.s6_error != "",
        rx.box(
            rx.hstack(
                rx.text("⚠", font_size="14px", color=styles.RED),
                rx.text(PipelineState.s6_error, font_size="12px",
                        color=styles.RED, flex="1", word_break="break-word"),
                spacing="2", align="start",
            ),
            padding="12px", background=styles.RED_BG,
            border=f"1px solid {styles.RED}",
            border_radius="8px", width="100%",
        ),
        rx.box(),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def persona_page() -> rx.Component:
    return rx.vstack(
        _running_banner(),
        _error_banner(),

        # Only show content when personas exist
        rx.cond(
            PipelineState.s6_personas.length() > 0,
            rx.vstack(
                _summary_bar(),
                rx.cond(
                    PipelineState.s6_active_persona_idx >= 0,
                    _detail_view(),
                    _overview_grid(),
                ),
                spacing="4", width="100%", align="start",
            ),
            rx.box(),
        ),

        spacing="4", width="100%", align="start",
    )
