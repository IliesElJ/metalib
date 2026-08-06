"""Config tab — view and edit strategy YAML parameters."""

from dash import html, dcc
from utils_config import load_config, list_prod_strategies, is_local_override

BORDER = 'rgba(38,34,29,0.10)'
TEXT_PRIMARY = '#26221D'
TEXT_SECONDARY = '#7A756C'
ACCENT = '#BF6A3D'
ACCENT_SOFT = '#F3E4D8'
GREEN = '#3D7A54'
RED = '#B5473A'
SIDEBAR_BG = '#EFECE6'

TIMEFRAME_OPTIONS = [
    {'label': 'M1',  'value': 'mt5.TIMEFRAME_M1'},
    {'label': 'M5',  'value': 'mt5.TIMEFRAME_M5'},
    {'label': 'M15', 'value': 'mt5.TIMEFRAME_M15'},
    {'label': 'M30', 'value': 'mt5.TIMEFRAME_M30'},
    {'label': 'H1',  'value': 'mt5.TIMEFRAME_H1'},
    {'label': 'H4',  'value': 'mt5.TIMEFRAME_H4'},
    {'label': 'D1',  'value': 'mt5.TIMEFRAME_D1'},
    {'label': 'W1',  'value': 'mt5.TIMEFRAME_W1'},
    {'label': 'MN1', 'value': 'mt5.TIMEFRAME_MN1'},
]

# ── Parameter documentation ────────────────────────────────────────────────────

PARAM_DOCS = {
    # Universal
    'strategy_type':       'Strategy class identifier (read-only)',
    'symbols':             'MT5 symbol name(s) this instance trades, comma-separated',
    'timeframe':           'Candlestick timeframe used for signal generation',
    'tag':                 'Unique run ID embedded in trade comments and PM2 logs',
    'active_hours':        'Hours 0–23 during which orders are placed; leave empty for 24/7',
    'size_position':       'Fixed position size in lots',
    # metaob
    'atr_period':          'ATR lookback for stop-loss and take-profit sizing (bars)',
    'breakout_lookback':   'Consecutive bars required to confirm a pivot breakout',
    'pivot_window':        'Rolling window for pivot high/low detection (bars)',
    'sl_atr_mult':         'Stop-loss width expressed as a multiple of ATR',
    'tp_atr_mult':         'Take-profit distance expressed as a multiple of ATR',
    'sma_long_hours':      'Long SMA period for trend-direction filter (hours)',
    'sma_short_hours':     'Short SMA period for trend-direction filter (hours)',
    'trend_t_threshold':   't-score threshold: trend regression must exceed this to allow entry',
    'trend_window':        'Lookback window for linear trend regression (bars)',
    # metamtou
    'horizon':             'OU mean-reversion forecast horizon (bars ahead)',
    'k_min':               'Minimum lag anchors that must agree in direction (quorum)',
    'max_hold_days':       'Maximum calendar days to hold before forced exit',
    'rr_min':              'Minimum required risk/reward ratio to open a trade',
    'stop_mult':           'Stop-loss width as a multiple of the OU residual std',
    'trail_threshold':     'Profit fraction (0–1) at which trailing stop activates',
    'check_interval':      'Minutes between strategy polling cycles',
    # metafvg / metafvg_v2
    'limit_number_position': 'Maximum simultaneous open positions per instance',
    'spearman_lookback':   'Lookback for Spearman rank-correlation mean-reversion gate (bars)',
    'spearman_threshold':  'Minimum Spearman correlation required to allow entry',
    # metahar
    'short_factor':        'HAR short-term component multiplier (daily component)',
    'long_factor':         'HAR long-term component multiplier (monthly component)',
    'predicted_symbol':    'Symbol used as regression target in the HAR model',
    # metaga
    'high_length':         'GA high-frequency component length (bars)',
    'mid_length':          'GA mid-frequency component length (bars)',
    'low_length':          'GA low-frequency component length (bars)',
    'prob_bound':          'Minimum predicted probability required to trigger entry',
    # metado
    'risk_factor':         'Position size scaling factor relative to account equity',
    'mode':                'Strategy mode: mean_rev (counter-trend) or trend (breakout)',
    # metamlp
    'retrain_interval':    'Hours between MLP model retraining cycles',
    'confidence_threshold': 'Minimum model confidence score to open a trade (0–1)',
    # generic
    'tp_pct':              'Take-profit as a percentage of entry price',
    'sl_pct':              'Stop-loss as a percentage of entry price',
}

# ── Strategy documentation ─────────────────────────────────────────────────────

STRATEGY_DOCS = {
    'metaob': (
        'Order Blocks',
        'Identifies institutional order blocks — zones where impulsive moves originated '
        '— and enters on confirmed breakout of a rolling pivot high/low. A dual-SMA trend '
        'filter and linear-regression t-score gate suppress counter-trend and low-conviction entries.',
    ),
    'metamtou': (
        'Monthly True Opens - Ornstein',
        'Fits an Ornstein-Uhlenbeck process to each symbol at multiple lag anchors and enters '
        'when price deviates beyond a calibrated band. A k-quorum vote (≥k_min anchors must agree) '
        'filters weak setups. Exits via trailing stop once a profit threshold is reached.',
    ),
    'metafvg': (
        'Fair Value Gaps',
        'Trades imbalance gaps (FVGs) left after impulsive candles. Enters a reversion trade '
        'when price re-enters the gap. A Spearman rank-correlation gate confirms a mean-reverting '
        'environment before entry.',
    ),
    'metafvg_v2': (
        'FVG Mean-Reversion v2',
        'Enhanced FVG strategy with tighter Spearman gating and updated parameter calibration '
        'from the 2026 backtest campaign. Same core gap-fill logic as metafvg but with improved '
        'position-sizing and a position-cap per instance.',
    ),
    'metaga': (
        'Genetic Algorithm',
        'Entry rules evolved by a genetic algorithm over rolling in-sample windows. '
        'The GA optimises a combination of moving-average crossover lengths and a probability '
        'bound for the entry signal. Exits use fixed ATR-based stops.',
    ),
    'metago': (
        'MetaGO',
        'Multi-strategy orchestrator that manages a portfolio of sub-strategies. '
        'Allocates lot sizes dynamically based on recent drawdown and equity curve slope.',
    ),
    'metamlp': (
        'Multi-Horizon MLP',
        'XGBoost / MLP ensemble trained on multi-timeframe engineered features. '
        'Retrains on a rolling window and enters only when predicted return clears a '
        'confidence threshold.',
    ),
    'metahar': (
        'HAR Volatility',
        'Heterogeneous AutoRegressive (HAR) model for realised volatility. Predicts '
        'next-period vol from daily, weekly, and monthly components and trades the spread '
        'between implied and predicted volatility.',
    ),
    'metane': (
        'MetaNE',
        'Neuroevolution strategy: evolves a small neural network policy using a '
        'population-based search over entry/exit signals derived from price and volume features.',
    ),
    'metado': (
        'MetaDO',
        'Mean-reversion strategy based on deviation from a dynamic oscillator. '
        'Entries trigger when the oscillator crosses calibrated bands; risk_factor controls '
        'position sizing relative to account equity.',
    ),
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _input_style():
    return {
        'fontSize': '13px', 'padding': '5px 8px', 'border': f'1px solid {BORDER}',
        'borderRadius': '6px', 'fontFamily': 'inherit', 'color': TEXT_PRIMARY,
        'background': '#fff', 'width': '100%', 'outline': 'none', 'boxSizing': 'border-box',
    }


def _field_input(instance_name, param, value):
    """Return the appropriate editable component for a config parameter."""
    field_id = {'type': 'cfg-field', 'instance': instance_name, 'param': param}

    if param == 'strategy_type':
        return dcc.Input(
            id=field_id, type='text', value=str(value or ''),
            disabled=True,
            style={**_input_style(), 'color': TEXT_SECONDARY, 'cursor': 'default'},
        )

    if param == 'timeframe':
        str_val = str(value) if value is not None else None
        return dcc.Dropdown(
            id=field_id, options=TIMEFRAME_OPTIONS, value=str_val,
            clearable=False, className='cfg-tf-dropdown',
            style={'width': '100%'},
        )

    if param == 'symbols':
        sv = ', '.join(value) if isinstance(value, list) else (str(value) if value else '')
        return dcc.Input(
            id=field_id, type='text', value=sv,
            placeholder='EURUSD, GBPUSD',
            debounce=True, style=_input_style(),
        )

    if param == 'active_hours':
        if value is None or str(value).strip().lower() in ('none', 'null', ''):
            sv = ''
        elif isinstance(value, list):
            sv = ', '.join(str(h) for h in value)
        else:
            sv = str(value)
        return dcc.Input(
            id=field_id, type='text', value=sv,
            placeholder='0, 1, …, 23  (empty = always active)',
            debounce=True, style=_input_style(),
        )

    if isinstance(value, bool):
        return dcc.Dropdown(
            id=field_id,
            options=[{'label': 'True', 'value': 'true'}, {'label': 'False', 'value': 'false'}],
            value='true' if value else 'false',
            clearable=False, className='cfg-tf-dropdown',
            style={'width': '100%'},
        )

    if isinstance(value, int):
        return dcc.Input(
            id=field_id, type='number', value=value, step=1,
            debounce=True, style=_input_style(),
        )

    if isinstance(value, float):
        return dcc.Input(
            id=field_id, type='number', value=value, step='any',
            debounce=True, style=_input_style(),
        )

    return dcc.Input(
        id=field_id, type='text',
        value=str(value) if value is not None else '',
        debounce=True, style=_input_style(),
    )


def _param_row(instance_name, param, value):
    doc = PARAM_DOCS.get(param, '')
    return html.Div([
        html.Div(param, style={
            'fontSize': '12.5px', 'fontWeight': '500', 'color': TEXT_PRIMARY,
            'fontFamily': 'ui-monospace, monospace',
            'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap',
        }),
        html.Div(_field_input(instance_name, param, value), style={'minWidth': 0}),
        html.Div(doc, style={
            'fontSize': '12px', 'color': TEXT_SECONDARY, 'lineHeight': '1.4',
        }),
    ], style={
        'display': 'grid',
        'gridTemplateColumns': '170px 200px 1fr',
        'gap': '0 16px',
        'alignItems': 'center',
        'padding': '5px 0',
        'borderBottom': f'1px solid {BORDER}',
    })


def _instance_card(instance_name, params):
    symbol_label = (
        params.get('symbols', [''])[0] if isinstance(params.get('symbols'), list)
        else str(params.get('symbols', ''))
    )
    rows = [_param_row(instance_name, k, v) for k, v in sorted(params.items())]
    # Remove button is positioned absolute so it sits in the visual header
    # but is NOT inside <summary>, avoiding accidental accordion toggles.
    return html.Div([
        html.Details([
            html.Summary([
                html.Span(instance_name, style={
                    'fontSize': '13px', 'fontWeight': '600', 'color': TEXT_PRIMARY,
                }),
                html.Span(symbol_label, style={
                    'fontSize': '12px', 'color': TEXT_SECONDARY, 'marginLeft': '10px',
                }),
            ], style={'cursor': 'pointer', 'padding': '11px 44px 11px 16px',
                      'userSelect': 'none', 'display': 'flex', 'alignItems': 'center'}),
            html.Div(rows, style={'padding': '4px 16px 12px'}),
        ], open=True),
        html.Button('×', id={'type': 'cfg-remove-btn', 'index': instance_name}, n_clicks=0,
                    title='Remove instance',
                    style={
                        'position': 'absolute', 'top': '8px', 'right': '10px',
                        'border': 'none', 'background': 'none', 'cursor': 'pointer',
                        'fontSize': '17px', 'color': TEXT_SECONDARY, 'padding': '2px 6px',
                        'borderRadius': '4px', 'lineHeight': '1', 'fontFamily': 'inherit',
                    }),
    ], style={
        'position': 'relative',
        'background': '#fff', 'border': f'1px solid {BORDER}',
        'borderRadius': '10px', 'overflow': 'hidden', 'marginBottom': '10px',
    })


def _new_instance_card(idx, template_params):
    inst_key = f'__new_{idx}__'
    name_row = html.Div([
        html.Div('instance name', style={
            'fontSize': '12.5px', 'fontWeight': '600', 'color': ACCENT,
            'fontFamily': 'ui-monospace, monospace',
        }),
        html.Div(
            dcc.Input(
                id={'type': 'cfg-field', 'instance': inst_key, 'param': '__name__'},
                type='text', value='',
                placeholder='e.g. EURUSD_M15_OB',
                debounce=True, style=_input_style(),
            ),
            style={'minWidth': 0},
        ),
        html.Div('Unique YAML key for this instance', style={
            'fontSize': '12px', 'color': TEXT_SECONDARY, 'lineHeight': '1.4',
        }),
    ], style={
        'display': 'grid', 'gridTemplateColumns': '170px 200px 1fr',
        'gap': '0 16px', 'alignItems': 'center',
        'padding': '5px 0', 'borderBottom': f'1px solid {BORDER}',
    })
    rows = [name_row] + [
        _param_row(inst_key, k, v)
        for k, v in sorted(template_params.items())
        if k != '__name__'
    ]
    return html.Div([
        html.Div('New instance', style={
            'padding': '9px 16px', 'borderBottom': f'1px solid {BORDER}',
            'fontSize': '11px', 'fontWeight': '600', 'color': ACCENT,
            'background': 'rgba(191,106,61,0.04)', 'letterSpacing': '0.03em',
            'textTransform': 'uppercase',
        }),
        html.Div(rows, style={'padding': '4px 16px 12px'}),
    ], style={
        'background': '#fff', 'border': f'1px dashed {ACCENT}',
        'borderRadius': '10px', 'overflow': 'hidden', 'marginBottom': '10px',
    })


def _strategy_btn(name, display_name, is_active, has_local):
    badge = html.Span('local', style={
        'fontSize': '10px', 'background': ACCENT_SOFT, 'color': ACCENT,
        'borderRadius': '4px', 'padding': '1px 5px', 'marginLeft': '6px',
        'fontWeight': '600',
    }) if has_local else None

    return html.Button(
        [display_name, badge] if badge else [display_name],
        id={'type': 'config-strategy-btn', 'index': name},
        n_clicks=0,
        style={
            'display': 'flex', 'alignItems': 'center',
            'width': '100%', 'textAlign': 'left',
            'padding': '8px 10px', 'borderRadius': '7px', 'border': 'none',
            'fontSize': '13px', 'fontFamily': 'inherit', 'cursor': 'pointer',
            'background': '#fff' if is_active else 'transparent',
            'color': TEXT_PRIMARY if is_active else TEXT_SECONDARY,
            'fontWeight': '600' if is_active else '400',
        },
    )


# ── Public render function ─────────────────────────────────────────────────────

def render_config(strategy_name=None, removed_instances=None, added_instances=None):
    strategies = list_prod_strategies()

    # Left strategy list
    strategy_list = html.Div([
        html.Div('Strategy', style={
            'fontSize': '11px', 'fontWeight': '600', 'color': TEXT_SECONDARY,
            'padding': '0 10px 8px', 'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        *[_strategy_btn(s, STRATEGY_DOCS.get(s, (s,))[0], s == strategy_name, is_local_override(s)) for s in strategies],
    ], style={
        'width': '180px', 'flex': 'none', 'display': 'flex', 'flexDirection': 'column',
        'gap': '2px', 'paddingTop': '4px',
    })

    # Right: detail panel
    if strategy_name is None:
        detail = html.Div([
            html.Div('Select a strategy on the left to view and edit its parameters.',
                     style={'fontSize': '14px', 'color': TEXT_SECONDARY, 'paddingTop': '60px',
                            'textAlign': 'center'}),
        ], style={'flex': '1'})
    else:
        title, blurb = STRATEGY_DOCS.get(strategy_name, (strategy_name, ''))
        try:
            config = load_config(strategy_name)
        except Exception as e:
            config = {}
            blurb = f'Error loading config: {e}'

        local_badge = html.Span(
            '● local override active', style={
                'fontSize': '11px', 'color': ACCENT, 'fontWeight': '600',
                'background': ACCENT_SOFT, 'borderRadius': '5px', 'padding': '2px 7px',
            }
        ) if is_local_override(strategy_name) else None

        removed = set(removed_instances or [])
        added = added_instances or []

        instance_cards = [
            _instance_card(inst, params)
            for inst, params in config.items()
            if isinstance(params, dict) and inst not in removed
        ]
        for idx, new_inst in enumerate(added):
            instance_cards.append(_new_instance_card(idx, new_inst))

        add_btn = html.Button('+ Add instance', id='add-instance-btn', n_clicks=0, style={
            'padding': '8px 16px', 'border': f'1px dashed {BORDER}',
            'borderRadius': '7px', 'background': 'transparent',
            'color': TEXT_SECONDARY, 'fontSize': '13px', 'cursor': 'pointer',
            'fontFamily': 'inherit', 'width': '100%', 'marginTop': '4px',
        })

        detail = html.Div([
            # Header
            html.Div([
                html.Div([
                    html.Div(title or strategy_name, style={
                        'fontSize': '22px', 'fontWeight': '600', 'letterSpacing': '-0.02em',
                        'color': TEXT_PRIMARY,
                    }),
                    html.Div(blurb, style={
                        'fontSize': '13px', 'color': TEXT_SECONDARY,
                        'marginTop': '6px', 'maxWidth': '680px', 'lineHeight': '1.5',
                    }),
                ]),
                local_badge or html.Div(),
            ], style={'display': 'flex', 'alignItems': 'flex-start',
                      'justifyContent': 'space-between', 'marginBottom': '24px'}),

            # Instance cards
            html.Div(instance_cards or [
                html.Div('No instances found in config.', style={
                    'fontSize': '13px', 'color': TEXT_SECONDARY, 'padding': '20px 0',
                }),
            ]),
            add_btn,

            # Save bar
            html.Div([
                html.Button('Save', id='config-save-btn', n_clicks=0, style={
                    'padding': '8px 20px', 'borderRadius': '7px', 'border': 'none',
                    'background': '#f0ece6', 'color': TEXT_PRIMARY, 'fontSize': '13px',
                    'fontWeight': '500', 'fontFamily': 'inherit', 'cursor': 'pointer',
                }),
                html.Button(f'Save & Restart {strategy_name}', id='config-save-restart-btn',
                            n_clicks=0, style={
                    'padding': '8px 20px', 'borderRadius': '7px', 'border': 'none',
                    'background': ACCENT, 'color': '#fff', 'fontSize': '13px',
                    'fontWeight': '500', 'fontFamily': 'inherit', 'cursor': 'pointer',
                }),
                html.Div(id='config-status-msg', style={
                    'fontSize': '13px', 'color': GREEN, 'alignSelf': 'center',
                }),
            ], style={
                'display': 'flex', 'alignItems': 'center', 'gap': '10px',
                'marginTop': '32px', 'paddingTop': '20px', 'borderTop': f'1px solid {BORDER}',
            }),
        ], style={'flex': '1', 'minWidth': '0'})

    return html.Div([
        html.Div([
            html.Div('Config', style={
                'fontSize': '26px', 'fontWeight': '600', 'letterSpacing': '-0.02em',
                'fontFamily': "'Source Serif 4', Georgia, serif",
            }),
            html.Div('Strategy parameters — edits saved locally, not overwritten by git pull',
                     style={'fontSize': '13px', 'color': TEXT_SECONDARY, 'marginTop': '2px'}),
        ], style={'marginBottom': '28px'}),

        html.Div([
            strategy_list,
            html.Div(style={
                'width': '1px', 'background': BORDER, 'flex': 'none', 'margin': '0 24px',
            }),
            detail,
        ], style={'display': 'flex', 'alignItems': 'flex-start', 'minHeight': '400px'}),

    ], style={'padding': '28px 40px 60px'})
