
from typing import Dict, Any

"""
Risk Classifier Module
======================

Classifies the risk profile of an option strategy based on deterministic rules.
Categories:
- LOW: Defined risk, covers, or high probability setups.
- MEDIUM: Standard defined risk spreads, decent leverage.
- HIGH: Undefined risk (naked writing) or very low probability.
"""

def classify_risk(analysis: Dict[str, Any], net_greeks: Dict[str, float]) -> Dict[str, str]:
    """
    Determines the risk category and provides a reason.
    
    Args:
        analysis: Output from payoff_engine.analyze_strategy (Max Profit, Max Loss, etc.)
        net_greeks: Net aggregated Greeks.
        
    Returns:
        Dict with "category" (LOW, MEDIUM, HIGH) and "reason" (Text explanation).
    """
    max_loss = analysis['max_loss']
    max_profit = analysis['max_profit']
    
    # RULE 1: Undefined Risk is always HIGH
    # In our engine, undefined loss usually shows up as a very large negative number 
    # if we scanned a huge range, but theoretically it is -Infinity.
    # Since our analysis runs on a finite range, we check if the bounds are hit 
    # or if specific strategy types are detected. 
    # BETTER APPROACH: Check relative magnitude. if Max Loss is essentially infinite 
    # (e.g. > 5x strike implied), it's high risk. 
    # Alternatively, defined risk strategies have specific max loss caps.
    
    # Heuristic for implementation simplicity: 
    # If Max Loss is massive (indicating naked position logic in simulation), flag HIGH.
    # For this "Glass Box", let's assume if max_loss exceeds, say, 100% of underlying notional 
    # on a typical range, or just check the strategy structure implicitly via the loss value.
    # Actually, a safer way for a "Glass Box" is to check if it's "Uncapped".
    # Implementation detail: The analyze_strategy function in payoff_engine returns 
    # min(pnl) from the scanned range. If the range was wide, this is a proxy.
    # A true 'undefined risk' check would analyze the legs (e.g. net short calls).
    
    # Let's rely on the threshold of loss relative to entry cost for "Leverage Risk".
    # If Net Cost is 0 or credit, and Max Loss is high, that's high risk.
    
    # SIMPLIFIED RULES FOR SYSTEM:
    # 1. High Risk: Potential loss > 5000 (arbitrary high unit) OR undefined logic implied.
    #    (Since we don't have an 'undefined' flag in analysis, we use the magnitude).
    # 2. Medium Risk: Defined risk but Risk > 3x Reward.
    # 3. Low Risk: defined risk, Risk < 3x Reward.
    
    # We will refine 'Undefined' detection by the P&L curve slope at edges in a future iteration.
    # For now, let's use the magnitude.
    
    # Assume units are dollars. 
    # If Max Loss is very large negative number (e.g., < -10000 on a 100 stock), treat as high.
    
    THRESHOLD_HIGH_LOSS = -5000 
    
    if max_loss < THRESHOLD_HIGH_LOSS:
        return {
            "category": "HIGH",
            "reason": "This strategy has potentially unlimited or very high downside risk. You are exposed to catastrophic loss if the market moves significantly against you."
        }

    # If it's a credit strategy (Net Entry Cost < 0), Max Loss is the risk margin.
    # If Reward / Risk ratio is poor (e.g. risking $500 to make $10), it's "High Probability" but "High Tail Risk".
    # Let's call that MEDIUM-HIGH.
    
    if max_profit > 0:
        risk_reward_ratio = abs(max_loss / max_profit)
        if risk_reward_ratio > 5:
            return {
                "category": "MEDIUM",
                "reason": f"You are risking ${abs(max_loss):.2f} to make ${max_profit:.2f}. While probability of profit might be high, a loss would wipe out many wins."
            }
            
    return {
        "category": "LOW/MEDIUM",
        "reason": "This strategy appears to have a defined risk profile with a balanced risk-to-reward ratio."
    }
