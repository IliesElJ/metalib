"""
Main layout — sidebar + content area.
"""
from dash import dcc, html

ACCENT = '#BF6A3D'
SIDEBAR_BG = '#EFECE6'
BORDER = 'rgba(38,34,29,0.10)'
TEXT_PRIMARY = '#26221D'
TEXT_SECONDARY = '#7A756C'

NAV = [
    ('overview', 'Overview'),
    ('trades', 'Trades'),
    ('vol-forecast', 'Vol Forecast'),
    ('system', 'System'),
]


def _nav_btn(nav_id, label):
    return html.Button(
        [
            html.Div(id=f'nav-dot-{nav_id}', style={
                'width': '7px', 'height': '7px', 'borderRadius': '50%', 'flex': 'none',
            }),
            label,
        ],
        id=f'nav-{nav_id}', n_clicks=0,
        style={
            'display': 'flex', 'alignItems': 'center', 'gap': '10px',
            'padding': '9px 10px', 'borderRadius': '8px', 'cursor': 'pointer',
            'fontSize': '13.5px', 'border': 'none', 'width': '100%',
            'textAlign': 'left', 'fontFamily': 'inherit',
            'background': 'transparent', 'color': TEXT_SECONDARY, 'fontWeight': '500',
        }
    )


def get_layout():
    sidebar = html.Div([
        # Brand
        html.Div([
            html.Div(style={
                'width': '26px', 'height': '26px', 'borderRadius': '7px',
                'background': ACCENT, 'flex': 'none',
            }),
            html.Div("Strategy Ops", style={
                'fontSize': '14.5px', 'fontWeight': '600', 'letterSpacing': '-0.01em',
            }),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'padding': '0 8px'}),

        # Nav
        html.Div(
            [_nav_btn(nav_id, label) for nav_id, label in NAV],
            id='nav-items',
            style={'display': 'flex', 'flexDirection': 'column', 'gap': '2px'},
        ),

        # Footer
        html.Div([
            html.Div("Account", style={'fontSize': '11px', 'color': TEXT_SECONDARY}),
            html.Div("Live · MT5 Bridge", style={'fontSize': '12.5px', 'fontWeight': '500'}),
            html.Div(id='connection-status',
                     style={'fontSize': '11px', 'color': TEXT_SECONDARY, 'marginTop': '2px'}),
        ], style={
            'marginTop': 'auto', 'padding': '12px 10px',
            'borderTop': f'1px solid {BORDER}',
            'display': 'flex', 'flexDirection': 'column', 'gap': '3px',
        }),
    ], style={
        'width': '224px', 'flex': 'none', 'background': SIDEBAR_BG,
        'borderRight': f'1px solid {BORDER}', 'padding': '24px 16px',
        'display': 'flex', 'flexDirection': 'column', 'gap': '28px',
        'position': 'fixed', 'top': '0', 'left': '0', 'height': '100vh',
        'zIndex': '100', 'overflowY': 'auto',
    })

    return html.Div([
        dcc.Store(id='nav-store', data='overview'),
        dcc.Store(id='selected-strategy-store', data=None),
        dcc.Store(id='timerange-store', data='1M'),
        dcc.Store(id='day-offset-store', data=0),
        dcc.Store(id='data-store'),
        dcc.Store(id='account-info-store'),
        dcc.Store(id='trade-modal-store', data=None),
        dcc.Store(id='vol-symbol-store', data='EURUSD'),
        dcc.Store(id='vol-period-store', data='3M'),
        dcc.Store(id='vol-is-r2-store', data=None),
        dcc.Interval(id='startup-trigger', interval=500, n_intervals=0, max_intervals=1),
        html.Div(id='trade-modal-container'),

        html.Div([
            sidebar,
            html.Div(
                html.Div(id='main-content'),
                style={'marginLeft': '224px', 'flex': '1', 'minWidth': '0', 'minHeight': '100vh'},
            ),
        ], style={
            'display': 'flex', 'minHeight': '100vh',
            'background': '#F7F5F2', 'color': TEXT_PRIMARY,
        }),
    ])
