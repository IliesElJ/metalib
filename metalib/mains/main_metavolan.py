import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz

# Import your MetaVolAn class (adjust path as needed)
# from metalib.MetaVolAn import MetaVolAn
from metalib.metaanalyser import MetaVolAn  # Using MetaAnalyser as MetaVolAn for now

# Streamlit page config
st.set_page_config(
    page_title="MetaVolAn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 MetaVolAn - Volatility & Correlation Analytics")
st.markdown("Real-time volatility analysis and correlation monitoring dashboard")

# Initialize session state
if 'MetaVolAn_instance' not in st.session_state:
    st.session_state.MetaVolAn_instance = None
if 'fitted' not in st.session_state:
    st.session_state.fitted = False

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Symbol selection
st.sidebar.subheader("Symbols")
available_symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "GER40.cash", "BTCUSD"]
selected_symbols = st.sidebar.multiselect(
    "Select trading symbols:",
    available_symbols,
    default=["EURUSD", "GBPUSD", "USDJPY"]
)

# Timeframe selection
st.sidebar.subheader("Timeframes")
timeframe_options = {
    "1M": mt5.TIMEFRAME_M1,
    "5M": mt5.TIMEFRAME_M5,
    "15M": mt5.TIMEFRAME_M15,
    "30M": mt5.TIMEFRAME_M30,
    "1H": mt5.TIMEFRAME_H1,
}
selected_timeframe_str = st.sidebar.selectbox("Data Timeframe:", list(timeframe_options.keys()), index=1)
selected_timeframe = timeframe_options[selected_timeframe_str]

# Volatility timeframe
vol_tf_options = ["1h", "4h", "1D", "1W"]
vol_tf = st.sidebar.selectbox("Volatility Timeframe:", vol_tf_options, index=1)

# Parameters
st.sidebar.subheader("Parameters")
vol_window = st.sidebar.slider("Volatility Window:", min_value=6, max_value=48, value=24, step=6)
hist_length = st.sidebar.slider("Historical Length:", min_value=100, max_value=2000, value=1000, step=100)

# Active hours (optional)
active_hours = st.sidebar.multiselect(
    "Active Hours (optional):",
    list(range(24)),
    default=list(range(8, 18))
)

# Strategy tag
tag = st.sidebar.text_input("Strategy Tag:", value="MetaVolAnDashboard")

# Fit button
if st.sidebar.button("🔄 Fit Model", type="primary", use_container_width=True):
    if len(selected_symbols) < 1:
        st.sidebar.error("Please select at least one symbol!")
    else:
        with st.spinner("Fitting MetaVolAn model..."):
            try:
                # Initialize MetaVolAn instance
                st.session_state.MetaVolAn_instance = MetaVolAn(
                    symbols=selected_symbols,
                    timeframe=selected_timeframe,
                    tag=tag,
                    active_hours=active_hours if active_hours else None,
                    vol_window=vol_window,
                    hist_length=hist_length,
                    vol_tf=vol_tf
                )

                # Connect to MT5
                st.session_state.MetaVolAn_instance.connect()

                # Fit the model
                st.session_state.MetaVolAn_instance.fit()
                st.session_state.fitted = True

                st.sidebar.success("✅ Model fitted successfully!")

            except Exception as e:
                st.sidebar.error(f"❌ Error fitting model: {str(e)}")

# Run Signals button
if st.session_state.fitted and st.sidebar.button("📡 Run Signals", use_container_width=True):
    with st.spinner("Running signals..."):
        try:
            # Load current data and run signals
            utc = pytz.timezone('UTC')
            end_time = datetime.now(utc)
            start_time = end_time - timedelta(hours=24)

            st.session_state.MetaVolAn_instance.loadData(start_time, end_time)
            st.session_state.MetaVolAn_instance.signals()
            st.session_state.MetaVolAn_instance.check_conditions()

            st.sidebar.success("✅ Signals updated!")

        except Exception as e:
            st.sidebar.error(f"❌ Error running signals: {str(e)}")

# Main dashboard content
if st.session_state.fitted and st.session_state.MetaVolAn_instance:
    MetaVolAn = st.session_state.MetaVolAn_instance

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Volatilities", "🔗 Correlations", "📊 Covariances", "📋 Current Signals"])

    with tab1:
        st.header("📈 Historical Volatilities")

        if MetaVolAn.fitted_vols:
            # Create z-scored volatility heatmap
            vol_data = {}
            for symbol in selected_symbols:
                if symbol in MetaVolAn.fitted_vols:
                    vols = MetaVolAn.fitted_vols[symbol].dropna()
                    # Z-score the volatilities
                    z_scored_vols = (vols - vols.mean()) / vols.std()
                    vol_data[symbol] = z_scored_vols

            if vol_data:
                # Create DataFrame for heatmap
                vol_df = pd.DataFrame(vol_data)
                vol_df = vol_df.dropna()

                # Create heatmap
                fig = px.imshow(
                    vol_df.T,  # Transpose so symbols are on y-axis, time on x-axis
                    aspect="auto",
                    color_continuous_scale='RdYlBu_r',
                    title="Z-Scored Volatilities Heatmap",
                    labels=dict(x="Time Period", y="Symbol", color="Z-Score")
                )

                fig.update_layout(
                    height=400,
                    title_text="Z-Scored Historical Volatilities",
                    xaxis_title="Time Periods",
                    yaxis_title="Symbols"
                )

                st.plotly_chart(fig, use_container_width=True)

            # Volatility statistics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Volatility Statistics")
                vol_stats = []
                for symbol in selected_symbols:
                    if symbol in MetaVolAn.fitted_vols:
                        vols = MetaVolAn.fitted_vols[symbol].dropna()
                        vol_stats.append({
                            'Symbol': symbol,
                            'Mean': f"{vols.mean():.4f}",
                            'Std': f"{vols.std():.4f}",
                            'Min': f"{vols.min():.4f}",
                            'Max': f"{vols.max():.4f}"
                        })

                if vol_stats:
                    st.dataframe(pd.DataFrame(vol_stats), use_container_width=True)

            with col2:
                st.subheader("📈 Current vs Historical")
                if hasattr(MetaVolAn, 'signalData') and MetaVolAn.signalData is not None:
                    current_data = []
                    for _, row in MetaVolAn.signalData.iterrows():
                        current_data.append({
                            'Symbol': row['symbol'],
                            'Current Vol': f"{row['current_vol']:.4f}",
                            'Vol Rank': f"{row['vol_rank']:.2%}",
                            'Status': '🔴 High' if row['vol_rank'] > 0.8 else '🟡 Medium' if row[
                                                                                               'vol_rank'] > 0.2 else '🟢 Low'
                        })

                    if current_data:
                        st.dataframe(pd.DataFrame(current_data), use_container_width=True)

    with tab2:
        st.header("🔗 Correlation Analysis")

        if len(selected_symbols) > 1 and MetaVolAn.fitted_correlations:
            # Get latest correlation matrix
            latest_period = max(MetaVolAn.fitted_correlations.keys())
            latest_corr = MetaVolAn.fitted_correlations[latest_period]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"📅 Latest Correlation Matrix ({latest_period.strftime('%Y-%m-%d %H:%M')})")

                # Create correlation heatmap
                fig = px.imshow(
                    latest_corr,
                    text_auto='.3f',
                    aspect="auto",
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1
                )
                fig.update_layout(
                    title="Correlation Heatmap",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("🎯 Correlation Stats")

                # Extract upper triangle correlations (excluding diagonal)
                mask = np.triu(np.ones_like(latest_corr, dtype=bool), k=1)
                upper_corr = latest_corr.where(mask)

                corr_stats = {
                    'Max Correlation': f"{upper_corr.max().max():.3f}",
                    'Min Correlation': f"{upper_corr.min().min():.3f}",
                    'Mean Correlation': f"{upper_corr.mean().mean():.3f}",
                    'Std Correlation': f"{upper_corr.std().std():.3f}"
                }

                for stat, value in corr_stats.items():
                    st.metric(stat, value)

            # Historical correlation evolution
            st.subheader("📈 Correlation Evolution")

            # Calculate average correlation over time
            avg_corrs = []
            periods = []
            for period, corr_matrix in MetaVolAn.fitted_correlations.items():
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                upper_corr = corr_matrix.where(mask)
                avg_corr = upper_corr.mean().mean()
                avg_corrs.append(avg_corr)
                periods.append(period)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=periods,
                y=avg_corrs,
                mode='lines+markers',
                name='Average Correlation',
                line=dict(width=3)
            ))
            fig.update_layout(
                title="Average Portfolio Correlation Over Time",
                xaxis_title="Time",
                yaxis_title="Average Correlation",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("📊 Covariance Analysis")

        if len(selected_symbols) > 1 and MetaVolAn.fitted_covariances:
            # Get latest covariance matrix
            latest_period = max(MetaVolAn.fitted_covariances.keys())
            latest_cov = MetaVolAn.fitted_covariances[latest_period]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"📅 Latest Covariance Matrix ({latest_period.strftime('%Y-%m-%d %H:%M')})")

                # Create covariance heatmap
                fig = px.imshow(
                    latest_cov,
                    text_auto='.6f',
                    aspect="auto",
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    title="Covariance Heatmap",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📈 Risk Metrics")

                # Calculate portfolio risk metrics
                diagonal_vars = np.diag(latest_cov)
                portfolio_var = latest_cov.sum().sum() / len(selected_symbols) ** 2

                risk_metrics = {
                    'Avg Individual Variance': f"{diagonal_vars.mean():.6f}",
                    'Max Individual Variance': f"{diagonal_vars.max():.6f}",
                    'Min Individual Variance': f"{diagonal_vars.min():.6f}",
                    'Portfolio Variance': f"{portfolio_var:.6f}"
                }

                for metric, value in risk_metrics.items():
                    st.metric(metric, value)

            # Individual variance evolution
            st.subheader("📊 Individual Variances Over Time")

            fig = go.Figure()
            for symbol in selected_symbols:
                variances = []
                periods = []
                for period, cov_matrix in MetaVolAn.fitted_covariances.items():
                    if symbol in cov_matrix.index:
                        variances.append(cov_matrix.loc[symbol, symbol])
                        periods.append(period)

                if variances:
                    fig.add_trace(go.Scatter(
                        x=periods,
                        y=variances,
                        mode='lines',
                        name=f"{symbol} Variance",
                        line=dict(width=2)
                    ))

            fig.update_layout(
                title="Individual Symbol Variances Over Time",
                xaxis_title="Time",
                yaxis_title="Variance",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("📋 Current Signals & Alerts")

        if hasattr(MetaVolAn, 'signalData') and MetaVolAn.signalData is not None:
            st.subheader("🎯 Current Signal Data")

            # Display current signals
            display_data = MetaVolAn.signalData.copy()
            if 'timestamp' in display_data.columns:
                display_data['timestamp'] = pd.to_datetime(display_data['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Format numerical columns
            numeric_cols = ['current_vol', 'vol_rank']
            if 'avg_correlation' in display_data.columns:
                numeric_cols.append('avg_correlation')

            for col in numeric_cols:
                if col in display_data.columns:
                    if col == 'vol_rank':
                        display_data[col] = display_data[col].apply(lambda x: f"{x:.2%}")
                    else:
                        display_data[col] = display_data[col].apply(lambda x: f"{x:.4f}")

            st.dataframe(display_data, use_container_width=True)

            # Alert summary
            st.subheader("🚨 Alert Summary")

            high_vol_count = sum(MetaVolAn.signalData['vol_rank'] > 0.95)
            low_vol_count = sum(MetaVolAn.signalData['vol_rank'] < 0.05)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🔴 High Vol Alerts", high_vol_count)

            with col2:
                st.metric("🟢 Low Vol Alerts", low_vol_count)

            with col3:
                if MetaVolAn.correlations is not None:
                    mask = np.triu(np.ones_like(MetaVolAn.correlations, dtype=bool), k=1)
                    upper_corr = MetaVolAn.correlations.where(mask)
                    extreme_corr = sum((upper_corr > 0.9).sum()) + sum((upper_corr < -0.9).sum())
                    st.metric("⚠️ Extreme Correlations", extreme_corr)

        else:
            st.info("📡 Run signals to see current data and alerts")

else:
    # Welcome screen
    st.info("👆 Configure parameters in the sidebar and click 'Fit Model' to start!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📈 Volatility Analysis
        - Real-time volatility computation
        - Historical volatility ranking
        - Multi-symbol monitoring
        """)

    with col2:
        st.markdown("""
        ### 🔗 Correlation Tracking
        - Dynamic correlation matrices
        - Correlation evolution over time
        - Portfolio correlation alerts
        """)

    with col3:
        st.markdown("""
        ### 📊 Risk Management
        - Covariance matrix analysis
        - Portfolio risk metrics
        - Real-time alerts system
        """)

# Footer
st.markdown("---")
st.markdown("🚀 **MetaVolAn Dashboard** - Real-time volatility and correlation analytics powered by MetaTrader 5")