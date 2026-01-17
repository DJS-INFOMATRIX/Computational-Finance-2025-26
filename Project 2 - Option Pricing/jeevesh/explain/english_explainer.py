
from typing import Dict, Any, List
from strategies.payoff_engine import OptionLeg

"""
English Explainer Module
========================

Translate financial metrics into deterministic, educational Plain English.
NO Generative AI methods allowed. Rules only.
"""

def generate_explanation(strategy_analysis: Dict[str, Any], net_greeks: Dict[str, float]) -> Dict[str, Any]:
    """
    Generates a structured explanation object.
    
    Structure:
    - summary: One sentence overview.
    - pnl_context: Explain the money at stake.
    - greeks_explanation: Explain the forces (Time, Vol, Direction).
    - risks: Bullet points of key risks.
    """
    
    max_loss = strategy_analysis['max_loss']
    max_profit = strategy_analysis['max_profit']
    entry_cost = strategy_analysis['net_entry_cost']
    break_evens = strategy_analysis['break_evens']
    
    delta = net_greeks['delta']
    theta = net_greeks['theta']
    vega = net_greeks['vega']
    
    # 1. P&L Context (Cleaner Logic)
    if entry_cost > 0:
        trade_type = "Debit (Net Cost)"
        pnl_text = f"You payoff upfront is a net debit of ${entry_cost:.2f}."
        if max_profit > 1e6:
             max_profit_text = "Unlimited"
        else:
             max_profit_text = f"${max_profit:.2f}"
    else:
        trade_type = "Credit (Net Receive)"
        pnl_text = f"You receive a net credit of ${abs(entry_cost):.2f} upfront."
        max_profit_text = f"${abs(entry_cost):.2f}"

    # 2. Directional Bias (Delta)
    if delta > 0.1:
        direction = "Bullish"
        simple_dir = "This strategy profits if the stock price rises above your breakeven."
        tech_dir = f"Positive Net Delta ({delta:.2f}) indicates long directional exposure."
    elif delta < -0.1:
        direction = "Bearish"
        simple_dir = "This strategy profits if the stock price falls below your breakeven."
        tech_dir = f"Negative Net Delta ({delta:.2f}) indicates short directional exposure."
    else:
        direction = "Neutral"
        simple_dir = "This strategy profits if the stock price stays within a specific range."
        tech_dir = f"Near-Zero Net Delta ({delta:.2f}) minimizes directional risk."

    # 3. Time Decay (Theta)
    if theta > 0.01:
        time_text = "Time is on your side."
        simple_time = "You earn money every day just by holding this trade, assuming the stock price doesn't move against you."
        tech_time = f"Positive Net Theta ({theta:.2f}) means the option premium decays in your favor."
    elif theta < -0.01:
        time_text = "Time works against you."
        simple_time = "You lose a small amount of value every day due to time decay."
        tech_time = f"Negative Net Theta ({theta:.2f}) means you are paying for the time value of the options."
    else:
        time_text = "Time decay is neutral."
        simple_time = "Time passing has minimal effect on your profit."
        tech_time = "Net Theta is negligible."

    # 4. Volatility (Vega)
    if vega > 0.1:
        simple_vol = "You benefit if market volatility increases (e.g., earnings surprise)."
        tech_vol = f"Positive Vega ({vega:.2f}): Long Volatility exposure."
    elif vega < -0.1:
        simple_vol = "You benefit if market volatility decreases (e.g., markets calm down)."
        tech_vol = f"Negative Vega ({vega:.2f}): Short Volatility exposure."
    else:
        simple_vol = "Volatility changes have little impact."
        tech_vol = "Vega is neutral."

    # 5. Summary Construction
    summary = f"**{direction} Strategy** ({trade_type})"
    
    return {
        "summary": summary,
        "simple_reasoning": [simple_dir, simple_time, simple_vol],
        "technical_reasoning": [tech_dir, tech_time, tech_vol],
        "pnl_context": pnl_text,
        "key_metrics": {
            "Max Profit": max_profit_text,
            "Max Loss": f"${max_loss:.2f}",
        }
    }
