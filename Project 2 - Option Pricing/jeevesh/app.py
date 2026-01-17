
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set API Key if not already in environment (User provided fallback)
if "FRED_API_KEY" not in os.environ:
    os.environ["FRED_API_KEY"] = "y387648cf869ab66f42b52a0c8afad09e"

# Internal Modules
from data.market_data import get_current_price, get_risk_free_rate, get_historical_volatility, fetch_price_history
from pricing.black_scholes import decompose_option_price
from strategies.payoff_engine import OptionLeg, analyze_strategy, calculate_payoff
from explain.greeks_aggregator import aggregate_greeks
from explain.risk_classifier import classify_risk
from explain.english_explainer import generate_explanation
from stress.scenario_engine import simulate_scenario, backtest_strategy_historical
import datetime

# --- Page Config ---
st.set_page_config(
    page_title="Glass Box Strategist",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .risk-low { border-left: 5px solid green; padding-left: 10px; }
    .risk-medium { border-left: 5px solid orange; padding-left: 10px; }
    .risk-high { border-left: 5px solid red; padding-left: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.title("Market Data")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()

if 'ticker' not in st.session_state or st.session_state.ticker != ticker:
    st.session_state.ticker = ticker
    # Trigger fetch
    curr_price = get_current_price(ticker)
    hist_vol = get_historical_volatility(ticker)
    rf_rate = get_risk_free_rate()
    st.session_state.market_data = {
        "price": curr_price,
        "vol": hist_vol,
        "rate": rf_rate
    }

md = st.session_state.market_data

st.sidebar.metric("Spot Price", f"${md['price']:.2f}")
st.sidebar.metric("Hist. Volatility", f"{md['vol']*100:.1f}%")
st.sidebar.metric("Risk-Free Rate", f"{md['rate']*100:.1f}%")

st.sidebar.markdown("---")
st.sidebar.header("Overrides (Stress)")
vol_override = st.sidebar.slider("Volatility Override (%)", 1.0, 150.0, md['vol']*100.0) / 100.0
rate_override = st.sidebar.number_input("Risk-Free Rate (%)", value=md['rate']*100.0, step=0.1) / 100.0

# --- Tab Layout ---
st.title("The Glass Box Option Strategist")
tab_build, tab_explain, tab_timetravel = st.tabs(["Build Strategy", "Explain", "Historical Backtest"])

# --- Helper to Build Legs ---
def build_strategy_ui(current_price):
    st.subheader("Strategy Configuration")
    strat_type = st.selectbox("Preset Strategy", 
        ["Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread", "Iron Condor", "Custom"])
    
    legs = []
    expiry = st.number_input("Days to Expiration", min_value=1, value=30)
    T = expiry / 365.0
    
    if strat_type == "Long Call":
        k = st.number_input("Strike Price", value=float(int(current_price)), step=1.0)
        prem = st.number_input("Premium Paid", value=5.0, step=0.1)
        legs.append(OptionLeg('call', 'long', k, prem, 1))
        
    elif strat_type == "Bull Call Spread":
        c1, c2 = st.columns(2)
        k1 = c1.number_input("Long Strike (Lower)", value=float(int(current_price)), step=1.0)
        prem1 = c1.number_input("Premium Paid", value=5.0, step=0.1)
        k2 = c2.number_input("Short Strike (Higher)", value=float(int(current_price)+5), step=1.0)
        prem2 = c2.number_input("Premium Received", value=2.0, step=0.1)
        legs.append(OptionLeg('call', 'long', k1, prem1, 1))
        legs.append(OptionLeg('call', 'short', k2, prem2, 1))

    elif strat_type == "Iron Condor":
        c1, c2, c3, c4 = st.columns(4)
        k1 = c1.number_input("Long Put Strike", value=float(int(current_price)-10))
        p1 = c1.number_input("Prem Paid (LP)", value=1.0)
        k2 = c2.number_input("Short Put Strike", value=float(int(current_price)-5))
        p2 = c2.number_input("Prem Rec (SP)", value=3.0)
        k3 = c3.number_input("Short Call Strike", value=float(int(current_price)+5))
        p3 = c3.number_input("Prem Rec (SC)", value=3.0)
        k4 = c4.number_input("Long Call Strike", value=float(int(current_price)+10))
        p4 = c4.number_input("Prem Paid (LC)", value=1.0)
        legs.append(OptionLeg('put', 'long', k1, p1, 1))
        legs.append(OptionLeg('put', 'short', k2, p2, 1))
        legs.append(OptionLeg('call', 'short', k3, p3, 1))
        legs.append(OptionLeg('call', 'long', k4, p4, 1))
        
    # TODO: Add more presets or custom leg builder
    
    return legs, T

# --- TAB 1: BUILD ---
with tab_build:
    active_legs, T_val = build_strategy_ui(md['price'])
    
    # Analyze
    # Widen range to capture tails
    spot_range = np.linspace(md['price'] * 0.5, md['price'] * 1.5, 200)
    analysis = analyze_strategy(active_legs, spot_range, md['price'], T_val, md['rate'], md['vol'])
    payoff_vals = calculate_payoff(active_legs, spot_range)
    
    # --- Enhanced Payoff Chart ---
    st.markdown("### Payoff Diagram (at Expiration)")
    
    # Create separate arrays for Profit (Green) and Loss (Red)
    # This allow using 'tozeroy' filling correctly
    
    fig = go.Figure()
    
    # Add main line
    fig.add_trace(go.Scatter(x=spot_range, y=payoff_vals, mode='lines', name='P&L', line=dict(color='black', width=2)))
    
    # Add Green Zone (Profit)
    fig.add_trace(go.Scatter(x=spot_range, y=np.where(payoff_vals >= 0, payoff_vals, 0), 
                             fill='tozeroy', fillcolor='rgba(0, 200, 0, 0.2)', 
                             line=dict(width=0), name='Profit', showlegend=False))
                             
    # Add Red Zone (Loss)
    fig.add_trace(go.Scatter(x=spot_range, y=np.where(payoff_vals < 0, payoff_vals, 0), 
                             fill='tozeroy', fillcolor='rgba(200, 0, 0, 0.2)', 
                             line=dict(width=0), name='Loss', showlegend=False))

    fig.add_vline(x=md['price'], line_dash="dot", line_color="blue", annotation_text="Spot")
    
    for be in analysis['break_evens']:
        fig.add_vline(x=be, line_dash="dash", line_color="gray", annotation_text=f"BE: {be}")
        
    fig.update_layout(xaxis_title="Spot Price", yaxis_title="Profit / Loss ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics Layout
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max Profit", f"{analysis['max_profit']:.2f}" if analysis['max_profit'] < 100000 else "Unlimited")
    m2.metric("Max Loss", f"{analysis['max_loss']:.2f}")
    m3.metric("Net Cost", f"{analysis['net_entry_cost']:.2f}")
    
    pop = analysis.get('probability_of_profit', 'N/A')
    m4.metric("Prob. of Profit", f"{pop}%" if pop is not None else "N/A")

# --- TAB 2: EXPLAIN ---
with tab_explain:
    st.header("Plain English Explanation")
    
    # Compute Aggregates
    net_greeks = aggregate_greeks(active_legs, md['price'], T_val, rate_override, vol_override)
    risk_profile = classify_risk(analysis, net_greeks)
    explanation = generate_explanation(analysis, net_greeks)
    
    # Risk Badge
    r_class = risk_profile['category']
    css_class = "risk-low" if "LOW" in r_class else ("risk-high" if "HIGH" in r_class else "risk-medium")
    
    st.markdown(f"""
    <div class="{css_class}">
        <h3>Risk Profile: {r_class}</h3>
        <p>{risk_profile['reason']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader(explanation['summary'])
    
    if analysis.get('probability_of_profit'):
        st.markdown(f"**Probability:** There is a **{analysis['probability_of_profit']}%** theoretical chance this strategy expires profitable.")
        
    st.info(explanation['pnl_context'])
    
    # New Split Interface
    col_simple, col_tech = st.columns(2)
    
    with col_simple:
        st.markdown("### Simple Reasoning (The 'What')")
        for line in explanation['simple_reasoning']:
            st.success(f"✓ {line}")
            
    with col_tech:
        st.markdown("### Technical Logic (The 'Why')")
        for line in explanation['technical_reasoning']:
            st.code(line, language="text")

# --- TAB 3: BACKTEST ---
with tab_timetravel:
    st.header("Historical Backtest (Time Travel)")
    st.write("Test how this strategy structure would have performed during past market conditions.")
    
    # Date Range Input
    today = datetime.date.today()
    start_col, end_col = st.columns(2)
    start_date = start_col.date_input("Start Date", today - datetime.timedelta(days=180))
    end_date = end_col.date_input("End Date", today - datetime.timedelta(days=150))
    
    if st.button("Run Historical Simulation"):
        if start_date >= end_date:
            st.error("Start Date must be before End Date.")
        else:
            with st.spinner("Travelling back in time..."):
                results = backtest_strategy_historical(
                    active_legs,
                    ticker,
                    str(start_date),
                    str(end_date),
                    md['price'],
                    T_val,
                    md['rate']
                )
                
                if "error" in results:
                    st.error(results['error'])
                else:
                    # Plot P&L Curve
                    df_res = pd.DataFrame({
                        "Date": results['dates'],
                        "Strategy P&L": results['pnl'],
                        "Stock Price": results['underlying']
                    })
                    
                    st.subheader(f"Simulation Results ({start_date} to {end_date})")
                    
                    # Double Axis Chart
                    fig_backtest = go.Figure()
                    
                    # P&L Line (Left Axis)
                    fig_backtest.add_trace(go.Scatter(
                        x=df_res['Date'], y=df_res['Strategy P&L'],
                        name="Strategy P&L", line=dict(color='green', width=3)
                    ))
                    
                    # Stock Price (Right Axis)
                    fig_backtest.add_trace(go.Scatter(
                        x=df_res['Date'], y=df_res['Stock Price'],
                        name="Stock Price", line=dict(color='gray', dash='dot'),
                        yaxis="y2"
                    ))
                    
                    fig_backtest.update_layout(
                        yaxis=dict(title="Strategy P&L ($)"),
                        yaxis2=dict(title="Stock Price ($)", overlaying="y", side="right"),
                        title="Strategy Performance vs Underlying"
                    )
                    
                    st.plotly_chart(fig_backtest, use_container_width=True)
                    
                    # Risk Report
                    max_dd = df_res['Strategy P&L'].min()
                    max_p = df_res['Strategy P&L'].max()
                    final_pnl = df_res['Strategy P&L'].iloc[-1]
                    
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Max Drawdown", f"${max_dd:.2f}")
                    r2.metric("Max Peak Profit", f"${max_p:.2f}")
                    r3.metric("Final P&L", f"${final_pnl:.2f}")
                    
                    if final_pnl > 0:
                        st.success("Result: The strategy was PROFITABLE in this period.")
                    else:
                        st.error("Result: The strategy yielded a LOSS in this period.")
                    
                    st.caption("Note: This simulation assumes you opened the strategy structure with the same 'Moneyness' relative to the Start Date's spot price.")
    
    st.markdown("---")
    st.subheader("What-If Stress Test")
    st.write("Instantaneous shock simulation.")
    
    sc1, sc2 = st.columns(2)
    shock_spot = sc1.slider("Spot Price Change (%)", -30.0, 30.0, 0.0) / 100.0
    shock_vol = sc2.slider("Volatility Change (%)", -50.0, 100.0, 0.0) / 100.0
    
    sim_simple = simulate_scenario(
        active_legs, md['price'], T_val, vol_override, rate_override, shock_spot, shock_vol, 0
    )
    st.metric("Theoretical Value Now", f"${sim_simple['new_value']:.2f}", f"{sim_simple['pnl_change']:.2f}")
