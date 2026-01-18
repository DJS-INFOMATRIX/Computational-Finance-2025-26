import streamlit as st
import pandas as pd
import numpy as np  
import yfinance as yf
import plotly.graph_objects as go
from backend import BlackScholes, Strategy
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="Glass Box Options v4", layout="wide", page_icon="📦")

# --- CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #dee2e6;
        color: #000000 !important;
        margin-bottom: 10px;
    }
    .big-font { font-size: 18px !important; }
</style>
""", unsafe_allow_html=True)

st.title("📦 The Glass Box Strategist")

# --- STATE ---
if "legs" not in st.session_state:
    st.session_state.legs = []
if "fetched_prices" not in st.session_state:
    st.session_state.fetched_prices = {} 

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🏗️ Build Strategy", "🧠 Explain (Glass Box)", "⏳ Time Machine"])

# ==========================================
# TAB 1: BUILD
# ==========================================
with tab1:
    col_input, col_summ = st.columns([1, 2])

    with col_input:
        st.subheader("1. Add Option Leg")
        
        # Ticker & Fetch Row
        t_col1, t_col2 = st.columns([2, 1])
        ticker_input = t_col1.text_input("Ticker", value="AAPL").upper()
        
        if t_col2.button("📡 Fetch Price"):
            try:
                data = yf.Ticker(ticker_input).history(period="1d")
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    st.session_state.fetched_prices[ticker_input] = price
                    st.success(f"${price:.2f}")
                else:
                    st.error("Invalid Ticker")
            except:
                st.error("Error")

        # Get current price from cache or default
        current_price = st.session_state.fetched_prices.get(ticker_input, 150.0)

        # Form
        with st.form("leg_form"):
            action = st.selectbox("Action", ["Buy", "Sell"])
            op_type = st.selectbox("Type", ["Call", "Put"])
            strike = st.number_input("Strike Price ($)", value=float(current_price))
            days = st.number_input("Days to Expiry", value=30)
            vol = st.slider("Implied Volatility (%)", 10, 200, 30) / 100.0
            
            if st.form_submit_button("➕ Add to Strategy"):
                st.session_state.legs.append({
                    "ticker": ticker_input,
                    "action": action.lower(),
                    "op_type": op_type.lower(),
                    "strike": strike,
                    "expiry": days / 365.0,
                    "vol": vol,
                    "ui_days": days
                })
                # Ensure price is saved for this leg
                st.session_state.fetched_prices[ticker_input] = current_price

        # Advanced Settings (Hidden by default)
        with st.expander("⚙️ Advanced Model Settings"):
            risk_free = st.number_input(
                "Risk-Free Rate (Decimal)", 
                value=0.045, 
                step=0.001, 
                help="The interest rate you earn with zero risk (like a US Treasury Bond). Black-Scholes needs this to price time value."
            )

    with col_summ:
        st.subheader("2. Portfolio Overview")
        if st.session_state.legs:
            # Show Table
            df = pd.DataFrame(st.session_state.legs)
            st.dataframe(df[["ticker", "action", "op_type", "strike", "ui_days"]], use_container_width=True)
            
            # Calculate Total Metrics
            strat = Strategy(st.session_state.legs)
            # Use saved prices for calculation
            metrics = strat.calculate_metrics(st.session_state.fetched_prices, risk_free)
            
            # Summary Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Net Cost (Debit/Credit)", f"${metrics['price']:.2f}")
            m2.metric("Net Delta", f"{metrics['delta']:.2f}")
            m3.metric("Legs", len(st.session_state.legs))
            
            if st.button("🗑️ Clear All"):
                st.session_state.legs = []
        else:
            st.info("Add legs on the left to start.")

# ==========================================
# TAB 2: EXPLAIN
# ==========================================
with tab2:
    if not st.session_state.legs:
        st.warning("Please build a strategy in Tab 1 first.")
    else:
        # Recalculate metrics
        strat = Strategy(st.session_state.legs)
        metrics = strat.calculate_metrics(st.session_state.fetched_prices, risk_free)
        
        # 1. Plain English Sentence [Feature A]
        st.subheader("🗣️ The 'Plain English' Analysis")
        
        # Logic for the sentence
        prob_profit = 50 + (metrics['delta'] * 100) # Simple proxy logic
        # Clamp between 1% and 99%
        prob_profit = max(1, min(99, prob_profit))
        
        direction = "up" if metrics['delta'] > 0 else "down"
        earnings = abs(metrics['delta'])
        
        st.markdown(f"""
        <div class="metric-card big-font">
            "This trade has roughly a <b>{prob_profit:.0f}%</b> chance of making money. 
            For every <b>$1</b> the stock goes {direction}, you make approximately <b>${earnings:.2f}</b>."
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Real vs Hope Chart [Improved Visuals]
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("💰 Price Breakdown")
            st.markdown(f"The strategy costs **${metrics['price']:.2f}**. Why?")
            
            # Calculate percentages for labels
            total = max(metrics['price'], 0.01) # Avoid div/0
            real_pct = (metrics['intrinsic'] / total) * 100
            hope_pct = (metrics['time_value'] / total) * 100
            
            # Text labels for inside the bars
            text_real = f"${metrics['intrinsic']:.2f}<br>({real_pct:.0f}%)" if metrics['intrinsic'] > 0 else ""
            text_hope = f"${metrics['time_value']:.2f}<br>({hope_pct:.0f}%)"
            
            fig_val = go.Figure(data=[
                go.Bar(
                    name='Real Value (Intrinsic)', 
                    x=['Total Cost'], 
                    y=[metrics['intrinsic']], 
                    marker_color='#27ae60', # Strong Green
                    text=text_real,
                    textposition='auto'
                ),
                go.Bar(
                    name='Hope Value (Time)', 
                    x=['Total Cost'], 
                    y=[metrics['time_value']], 
                    marker_color='#f39c12', # Orange/Gold
                    text=text_hope,
                    textposition='auto'
                )
            ])
            
            fig_val.update_layout(
                barmode='stack', 
                yaxis_title="Option Value ($)",
                height=350,
                showlegend=True,
                title="Is this trade an Investment or a Gamble?"
            )
            st.plotly_chart(fig_val, use_container_width=True)
            
            # Dynamic Explanation
            if real_pct > 50:
                explanation = "✅ **Conservative:** Most of your money is paying for real, existing value."
            else:
                explanation = "⚠️ **Aggressive:** Most of your money is paying for 'Hope' (Time). If the stock doesn't move, this value disappears every day."
            
            st.info(explanation)

        # 3. Payoff Diagram (Visualizing Profit Zones)
        with c2:
            st.subheader("📈 Profit Zone (at Expiry)")
            
            # Generate P&L data
            # Use the first ticker's price as baseline
            base_ticker = st.session_state.legs[0]['ticker']
            base_price = st.session_state.fetched_prices.get(base_ticker, 100)
            
            prices = np.linspace(base_price * 0.7, base_price * 1.3, 50)
            pnl_values = []
            
            cost = metrics['price']
            
            for p in prices:
                # Calculate value at expiry (Time = 0)
                val_at_expiry = 0
                for leg in st.session_state.legs:
                    # Intrinsic value at expiry
                    if leg['op_type'] == 'call':
                        iv = max(p - leg['strike'], 0)
                    else:
                        iv = max(leg['strike'] - p, 0)
                    
                    if leg['action'] == 'buy':
                        val_at_expiry += iv
                    else:
                        val_at_expiry -= iv
                
                pnl_values.append(val_at_expiry - cost)
                
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(x=prices, y=pnl_values, fill='tozeroy', name='P&L'))
            fig_pnl.add_hline(y=0, line_color="black", line_dash="dash")
            fig_pnl.update_layout(
                title=f"Profit/Loss at Expiry ({base_ticker})",
                xaxis_title="Stock Price",
                yaxis_title="Profit / Loss ($)",
                height=300
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
# ==========================================
# TAB 3: TIME MACHINE
# ==========================================
with tab3:
    st.subheader("🕰️ Time Machine (Leverage Test)")
    
    # --- NEW: Date Selection ---
    col_dates1, col_dates2 = st.columns(2)
    with col_dates1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col_dates2:
        end_date = st.date_input("End Date", value=datetime.now())
        
    # Validation
    if start_date >= end_date:
        st.error("Error: Start Date must be before End Date.")
        
    show_roi = st.toggle("Show % Return (See the Leverage)", value=True)
    
    if st.button("🚀 Run Simulation"):
        if not st.session_state.legs:
            st.error("No strategy to test. Please build one in Tab 1.")
        elif start_date >= end_date:
            st.error("Please fix the dates above.")
        else:
            try:
                unique_tickers = list(set([leg['ticker'] for leg in st.session_state.legs]))
                
                hist_data = {}
                common_index = None
                
                with st.spinner(f"Traveling back to {start_date}..."):
                    for t in unique_tickers:
                        # UPDATED: Fetch specific date range
                        df = yf.download(t, start=start_date, end=end_date)
                        
                        if df.empty:
                            st.error(f"No data found for {t} in this range.")
                            continue

                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        
                        hist_data[t] = df['Close']
                        
                        if common_index is None:
                            common_index = df.index
                        else:
                            common_index = common_index.intersection(df.index)
                    
                    if common_index is None or len(common_index) == 0:
                        st.error("No overlapping data found for these tickers in this date range.")
                    else:
                        dates = common_index
                        strategy_values = []
                        
                        # Simulation Loop
                        for date in dates:
                            daily_price_map = {}
                            for t in unique_tickers:
                                daily_price_map[t] = hist_data[t].loc[date]
                            
                            strat = Strategy(st.session_state.legs)
                            # We use risk_free from the global variable (sidebar/expander)
                            metrics = strat.calculate_metrics(daily_price_map, risk_free)
                            strategy_values.append(metrics['price'])
                        
                        # Plotting
                        fig = go.Figure()
                        
                        # Handle case where strategy starts at 0 or negative
                        initial_strat_val = strategy_values[0]
                        # prevent divide by zero for ROI calc
                        if initial_strat_val == 0: initial_strat_val = 0.01 
                        
                        # Plot Stocks
                        for t in unique_tickers:
                            series = hist_data[t].loc[dates]
                            if show_roi:
                                start_val = series.iloc[0]
                                series = ((series - start_val) / start_val) * 100
                                y_label = "Return (%)"
                            else:
                                y_label = "Price ($)"
                                
                            fig.add_trace(go.Scatter(x=dates, y=series, name=f"{t} (Stock)", line=dict(dash='dot')))

                        # Plot Strategy
                        strat_series = pd.Series(strategy_values, index=dates)
                        if show_roi:
                            strat_series = ((strat_series - initial_strat_val) / initial_strat_val) * 100
                            
                        fig.add_trace(go.Scatter(x=dates, y=strat_series, name="Your Strategy", line=dict(color='green', width=3)))
                        
                        fig.update_layout(
                            title=f"Performance ({start_date} to {end_date})", 
                            yaxis_title=y_label,
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Simulation Error: {e}")