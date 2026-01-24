"""
Welcome Tab Component
Landing page with tab descriptions - French Gen Z / ASAP Rocky style
"""

from dash import html
import dash_bootstrap_components as dbc
from .common_ui import COLORS, CHART_COLORS


def render_welcome_tab():
    """
    Render the welcome page with tab descriptions
    """
    return html.Div(
        [
            # Hero section
            html.Div(
                [
                    html.H1(
                        "METADASH PELOOOO",
                        style={
                            "fontSize": "48px",
                            "fontWeight": "800",
                            "letterSpacing": "4px",
                            "color": COLORS["text_dark"],
                            "marginBottom": "8px",
                        },
                    ),
                    html.P(
                        "dashboard malveillant au max",
                        style={
                            "fontSize": "18px",
                            "color": COLORS["text_medium"],
                            "fontStyle": "italic",
                        },
                    ),
                ],
                style={
                    "textAlign": "center",
                    "padding": "40px 20px 30px",
                    "borderBottom": f"2px solid {COLORS['border']}",
                    "marginBottom": "40px",
                },
            ),
            # Intro text
            html.Div(
                [
                    html.P(
                        [
                            "Wech pelo, bienvenue sur ",
                            html.Span("MetaDAsh", style={"fontWeight": "700"}),
                            " — le nouveau QG des crapules. ",
                            "Let's get it.",
                        ],
                        style={
                            "fontSize": "16px",
                            "color": COLORS["text_medium"],
                            "textAlign": "center",
                            "maxWidth": "700px",
                            "margin": "0 auto 40px",
                            "lineHeight": "1.7",
                        },
                    ),
                ]
            ),
            # Tab cards grid
            dbc.Row(
                [
                    # Status Monitor
                    dbc.Col(
                        _create_tab_card(
                            icon="🔴",
                            title="Status Monitor",
                            description="Check en temps réel si tes bots tournent ou s'ils font la sieste. Rouge = y'a un problème chef.",
                            color=CHART_COLORS[3],
                        ),
                        lg=4,
                        md=6,
                        sm=12,
                        className="mb-4",
                    ),
                    # Overview
                    dbc.Col(
                        _create_tab_card(
                            icon="📊",
                            title="Overview",
                            description="La vue d'ensemble du compte. Balance, equity, marge — tout est là, bien posé.",
                            color=CHART_COLORS[0],
                        ),
                        lg=4,
                        md=6,
                        sm=12,
                        className="mb-4",
                    ),
                    # Strategy Types
                    dbc.Col(
                        _create_tab_card(
                            icon="🎯",
                            title="Strategy Types",
                            description="Compare tes différentes strats. MetaFVG, MetaOB, MetaGO (le bot ne dans le gros crane du G)... qui performe le plus ?",
                            color=CHART_COLORS[4],
                        ),
                        lg=4,
                        md=6,
                        sm=12,
                        className="mb-4",
                    ),
                    # PnL Performance
                    dbc.Col(
                        _create_tab_card(
                            icon="💰",
                            title="PnL Performance",
                            description="L'argent parle et Thomas dort. Courbe d'equity, drawdown, win streaks — tout pour savoir si tu gères ou pas.",
                            color=CHART_COLORS[1],
                        ),
                        lg=4,
                        md=6,
                        sm=12,
                        className="mb-4",
                    ),
                    # Detailed Analysis
                    dbc.Col(
                        _create_tab_card(
                            icon="🔬",
                            title="Detailed Analysis",
                            description="Pour les vrais. Analyse par heure, durée des trades, metrics en profondeur.",
                            color=CHART_COLORS[5],
                        ),
                        lg=4,
                        md=6,
                        sm=12,
                        className="mb-4",
                    ),
                    # Trades
                    dbc.Col(
                        _create_tab_card(
                            icon="📋",
                            title="Trades",
                            description="La liste de tous tes trades. Filtre par bot, regarde les détails, c'est transparent.",
                            color=CHART_COLORS[2],
                        ),
                        lg=4,
                        md=6,
                        sm=12,
                        className="mb-4",
                    ),
                    # Logs
                    dbc.Col(
                        _create_tab_card(
                            icon="📝",
                            title="Logs",
                            description="Les logs de tes bots, jour par jour. Quand ça bug, c'est ici que tu check.",
                            color=CHART_COLORS[6],
                        ),
                        lg=4,
                        md=6,
                        sm=12,
                        className="mb-4",
                    ),
                    # Raw Data
                    dbc.Col(
                        _create_tab_card(
                            icon="🗃️",
                            title="Raw Data",
                            description="Les données brutes pour les data nerds. Export, analyse, fais ton délire casse pas les couilles.",
                            color=CHART_COLORS[7],
                        ),
                        lg=4,
                        md=6,
                        sm=12,
                        className="mb-4",
                    ),
                ],
                className="px-3",
            ),
            # Footer quote
            html.Div(
                [
                    html.P(
                        "« Le marché dort jamais, et toi non plus quand t'as MetaDAsh »",
                        style={
                            "fontSize": "14px",
                            "color": COLORS["text_light"],
                            "fontStyle": "italic",
                            "textAlign": "center",
                            "marginTop": "30px",
                            "paddingTop": "30px",
                            "borderTop": f"1px solid {COLORS['border']}",
                        },
                    ),
                ]
            ),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    )


def _create_tab_card(icon, title, description, color):
    """Create a styled card for tab description"""
    return html.Div(
        [
            # Icon
            html.Div(
                icon,
                style={
                    "fontSize": "32px",
                    "marginBottom": "12px",
                },
            ),
            # Title
            html.H3(
                title,
                style={
                    "fontSize": "16px",
                    "fontWeight": "700",
                    "color": COLORS["text_dark"],
                    "marginBottom": "8px",
                },
            ),
            # Description
            html.P(
                description,
                style={
                    "fontSize": "13px",
                    "color": COLORS["text_medium"],
                    "lineHeight": "1.5",
                    "margin": "0",
                },
            ),
        ],
        style={
            "padding": "24px 20px",
            "backgroundColor": "white",
            "borderRadius": "12px",
            "border": f"1px solid {COLORS['border']}",
            "borderTop": f"3px solid {color}",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
            "height": "100%",
            "transition": "transform 0.2s, box-shadow 0.2s",
        },
    )
