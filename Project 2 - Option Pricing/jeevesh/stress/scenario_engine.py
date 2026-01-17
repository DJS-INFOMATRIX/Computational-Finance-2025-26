
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from strategies.payoff_engine import OptionLeg
from pricing.black_scholes import call_price, put_price

"""
Scenario Engine Module
======================

Enables "What-If" analysis and Stress Testing.
Re-prices the entire strategy under different market conditions (Mark-to-Market).

Note:
- Payoff Engine calculates value at Expiration (T=0).
- Scenario Engine calculates value NOW (T=t) or at some future T (T=t-delta).
"""

def calculate_strategy_value(strategy: List[OptionLeg], S: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the current theoretical market value of the strategy package.
    """
    total_value = 0.0
    
    for leg in strategy:
        # Calculate single option price
        if leg.op_type == 'call':
            price = call_price(S, leg.strike, T, r, sigma)
        else:
            price = put_price(S, leg.strike, T, r, sigma)
            
        # Add to total value (Long = +Price, Short = -Price)
        # Note: This is the LIQUIDATION value.
        multiplier = leg.quantity if leg.side == 'long' else -leg.quantity
        total_value += price * multiplier
        
    return total_value

def simulate_scenario(
    strategy: List[OptionLeg], 
    current_spot: float,
    current_time_to_expiry: float, 
    current_vol: float,
    risk_free_rate: float,
    shock_spot_pct: float = 0.0,
    shock_vol_pct: float = 0.0,
    time_step_days: int = 0
) -> Dict[str, float]:
    """
    Simulates the strategy value after applying shocks.
    
    Args:
        strategy: List of legs.
        current_spot: Reference spot price.
        current_time_to_expiry: Reference T (years).
        current_vol: Reference Sigma.
        shock_spot_pct: Percentage move in spot (e.g., -0.10 for -10%).
        shock_vol_pct: Percentage relative change in Vol (e.g., +0.20 for +20% vol spike).
        time_step_days: Days to advance forward (Time Travel).
        
    Returns:
        Dict with 'new_value', 'pnl_change', 'new_spot', 'new_vol'.
    """
    
    # 1. Apply Shocks
    new_spot = current_spot * (1 + shock_spot_pct)
    new_vol = current_vol * (1 + shock_vol_pct) 
    
    # Ensure Vol doesn't go negative
    new_vol = max(0.01, new_vol)
    
    # 2. Advance Time
    # New T = Old T - (Days / 365)
    new_T = current_time_to_expiry - (time_step_days / 365.0)
    
    # If expired, we switch to Payoff logic (Intrinsic value only) handled by T=0 in BS
    if new_T < 0:
        new_T = 0.0
        
    # 3. Calculate New Value
    new_value = calculate_strategy_value(strategy, new_spot, new_T, risk_free_rate, new_vol)
    
    # 4. Calculate Old Value (Basis)
    # We need the value at the START of the scenario (Current Market) to know P&L change
    # Note: Strategy entry cost is fixed, but "PnL Change" usually refers to 
    # change from current Mark-to-Market.
    current_mtm_value = calculate_strategy_value(strategy, current_spot, current_time_to_expiry, risk_free_rate, current_vol)
    
    pnl_unrealized = new_value - current_mtm_value
    
    return {
        "new_value": new_value,
        "pnl_change": pnl_unrealized,
        "new_spot": new_spot,
        "new_vol": new_vol,
        "new_time": new_T
    }

from data.market_data import fetch_price_history

def backtest_strategy_historical(
    base_strategy: List[OptionLeg],
    ticker: str,
    start_date: str,
    end_date: str,
    current_spot_ref: float,
    current_T_ref: float,
    risk_free_rate: float
) -> Dict[str, Any]:
    """
    Simulates holding the equivalent strategy structure over a historical period.
    
    Logic:
    1. Calculate 'Moneyness' of current legs relative to Current Spot.
       (e.g., Strike 105 vs Spot 100 -> 105% Moneyness).
    2. Fetch Historical Stock Price for [start_date, end_date].
    3. On Start Date: Open new virtual legs with strikes adjusted to Start Date Spot * Moneyness.
    4. Iterate daily:
       - Update Spot (Close Price).
       - Update T (Remaining Time).
       - Use Historical Volatility (rolling std dev) or fixed? -> Use rolling 20d vol for realism.
       - Calculate Strategy Value.
    
    Returns:
        Dict with 'dates', 'portfolio_value', 'underlying_price'.
    """
    
    # 1. Fetch Data
    # Fetch slightly more to calculate rolling vol
    df = fetch_price_history(ticker, period="max")
    
    # Filter range
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index).tz_localize(None)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    history = df.loc[mask].copy()
    
    if history.empty:
        return {"error": "No data found for selected range."}
        
    # Calculate Rolling Volatility (20-day)
    history['Returns'] = history['Close'].pct_change()
    history['Vol'] = history['Returns'].rolling(window=20).std() * np.sqrt(252)
    history['Vol'] = history['Vol'].fillna(0.20) # Fallback
    
    # 2. Setup Virtual Strategy (Relative Scaling)
    initial_spot = history.iloc[0]['Close']
    virtual_legs = []
    
    for leg in base_strategy:
        # Moneyness ratio
        moneyness = leg.strike / current_spot_ref
        
        # Create new leg scaled to history
        new_strike = initial_spot * moneyness
        
        # Use same premium? No, we must Price it at entry to get accurate cost basis.
        # We rely on BS to price the theoretical entry.
        
        virtual_legs.append(OptionLeg(
            op_type=leg.op_type,
            side=leg.side,
            strike=new_strike,
            premium=0.0, # Will calculate
            quantity=leg.quantity
        ))
        
    # Calculate Initial Cost Basis at T=0 (virtual)
    # Use Vol at start
    initial_vol = history.iloc[0]['Vol']
    initial_value = calculate_strategy_value(virtual_legs, initial_spot, current_T_ref, risk_free_rate, initial_vol)
    
    dates = []
    portfolio_values = []
    underlying_prices = []
    
    # 3. Simulation Loop
    # We simulate holding for the duration of the Date Range OR until Expiry runs out.
    # The 'Time' in the strategy decreases from current_T_ref down to 0.
    
    days_passed = 0
    
    for date, row in history.iterrows():
        spot = row['Close']
        vol = row['Vol']
        
        # Remaining Time
        T_remaining = current_T_ref - (days_passed / 365.0)
        
        if T_remaining < 0:
            T_remaining = 0
            
        value = calculate_strategy_value(virtual_legs, spot, T_remaining, risk_free_rate, vol)
        
        # Track P&L relative to initial value? 
        # Usually Portfolio Value is just the Liquidation Value.
        # Net P&L = Current Value - Initial Cost.
        # Let's return just Value, and user can see spread.
        
        dates.append(date)
        portfolio_values.append(value - initial_value) # Cumulative P&L
        underlying_prices.append(spot)
        
        days_passed += 1
        
    return {
        "dates": dates,
        "pnl": portfolio_values,
        "underlying": underlying_prices,
        "initial_spot": initial_spot
    }
