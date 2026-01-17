
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.stats import norm
import math

"""
Strategy Payoff Engine
======================

This module simulates the P&L (Profit and Loss) of multi-leg option strategies at expiration.
It allows users to visualize risk/reward profiles before entering a trade.

Concepts:
- Option Leg: A single component of a strategy (e.g., "Long Call at Strike 100").
- Payoff at Expiration: The value of the strategy if held until the final day.
  (Note: This simple engine models value *at expiration*. Interim value requires Greeks).
"""

@dataclass
class OptionLeg:
    """
    Represents one leg of an option strategy.
    
    Attributes:
        type (str): 'call' or 'put'
        side (str): 'long' (buy) or 'short' (sell)
        strike (float): The strike price K
        premium (float): The price paid (if long) or received (if short) per share
        quantity (int): Number of contracts (1 contract = 100 shares usually, but here we treat quantity as unit multiplier)
    """
    op_type: str
    side: str
    strike: float
    premium: float
    quantity: int = 1

    def calculate_expiration_value(self, spot_price: float) -> float:
        """
        Calculates the value of this single leg at expiration for a given spot price.
        Does NOT include the initial premium paid/received.
        """
        if self.op_type == 'call':
            intrinsic = max(0, spot_price - self.strike)
        else: # put
            intrinsic = max(0, self.strike - spot_price)
            
        # If we are Short, we OWE this value (negative value to us).
        # If we are Long, we OWN this value (positive to us).
        if self.side == 'short':
             return -intrinsic * self.quantity
        else:
             return intrinsic * self.quantity

    def net_profit(self, spot_price: float) -> float:
        """
        Calculates net P&L including the initial premium.
        """
        expiration_val = self.calculate_expiration_value(spot_price)
        
        # Cost basis:
        # Long: We paid premium (-cost)
        # Short: We received premium (+credit)
        if self.side == 'long':
            initial_cashflow = -self.premium * self.quantity
        else:
            initial_cashflow = self.premium * self.quantity
            
        return expiration_val + initial_cashflow


def calculate_payoff(strategy: List[OptionLeg], spot_range: np.ndarray) -> np.ndarray:
    """
    Calculates the aggregate P&L for a list of option legs across a range of spot prices.
    
    Args:
        strategy: List of OptionLeg objects.
        spot_range: Numpy array of potential underlying prices at expiration.
        
    Returns:
        Numpy array of net profit/loss values corresponding to the spot_range.
    """
    total_pnl = np.zeros_like(spot_range)
    
    for leg in strategy:
        # Vectorized calculation for efficiency over the array
        leg_pnl = np.array([leg.net_profit(s) for s in spot_range])
        total_pnl += leg_pnl
        
    return total_pnl

def calculate_probability_of_profit(analysis: Dict, S: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the theoretical Probability of Profit (PoP).
    
    Assumption: Risk-Neutral Probability (using Black-Scholes d2 logic logic).
    
    Logic:
    - Identify profit zones using break-even points.
    - Calculate probability of ending up in those zones.
    """
    if T <= 0: return 0.0
    if sigma <= 0: return 0.0
    
    break_evens = sorted(analysis['break_evens'])
    if not break_evens:
        # Either always profit or always loss
        # Check one point (e.g. current Spot)
        # Note: We need a sample calculation to know which side is profitable.
        # But 'analysis' dict doesn't store the curve.
        # Heuristic: If net_entry_cost < max_profit (and defined), we might need checks.
        # Simplest: Check Max Profit. If Max Profit > 0 everywhere, 100%.
        if analysis['max_loss'] >= 0: return 1.0 # Always profit
        return 0.0 # Always loss
        
    prob_profit = 0.0
    
    # We need to test the segments between break-evens to see which are profitable.
    # Segments: (-inf, BE1), (BE1, BE2), ..., (BEn, +inf)
    
    test_points = []
    # Point before first BE
    test_points.append(break_evens[0] * 0.99)
    # Points between BEs
    for i in range(len(break_evens) - 1):
        test_points.append((break_evens[i] + break_evens[i+1]) / 2.0)
    # Point after last BE
    test_points.append(break_evens[-1] * 1.01)
    
    # We need to know if these points are profitable.
    # But we don't have the strategy leg objects here directly in this function signature... 
    # Wait, strict separation concern.
    pass # Replaced by integrated logic below in analyze_strategy wrapper logic

def probability_terminal_price_below(K: float, S: float, T: float, r: float, sigma: float) -> float:
    """
    Probability that S_T < K (Risk Neutral).
    Formula: N(-d2)
    """
    if T <= 0: return 1.0 if S < K else 0.0
    d2_val = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(-d2_val)

def analyze_strategy(strategy: List[OptionLeg], spot_range: np.ndarray, 
                     current_spot: float = None, T: float = None, r: float = None, sigma: float = None) -> Dict:
    """
    Analyzes key metrics: Max Profit, Max Loss, Break-evens, and optionally PoP.
    
    Args:
        strategy: List of OptionLeg objects.
        spot_range: Array covering a wide enough range to find break-evens.
        
    Returns:
        Dictionary containing analysis metrics.
    """
    pnl_curve = calculate_payoff(strategy, spot_range)
    
    max_profit = np.max(pnl_curve)
    max_loss = np.min(pnl_curve)
    
    # Calculate Net Entry Cost/Credit
    net_cost = 0.0
    for leg in strategy:
        if leg.side == 'long':
            net_cost += leg.premium * leg.quantity
        else:
            net_cost -= leg.premium * leg.quantity
            
    # Find Break-Even Points (where P&L crosses zero)
    # We look for sign changes in the P&L array
    break_evens = []
    for i in range(len(pnl_curve) - 1):
        if (pnl_curve[i] < 0 and pnl_curve[i+1] > 0) or (pnl_curve[i] > 0 and pnl_curve[i+1] < 0):
            # Linear interpolation for more precision
            y1, y2 = pnl_curve[i], pnl_curve[i+1]
            x1, x2 = spot_range[i], spot_range[i+1]
            # 0 = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            # x = x1 - y1 * (x2 - x1) / (y2 - y1)
            zero_cross = x1 - y1 * (x2 - x1) / (y2 - y1)
            break_evens.append(round(zero_cross, 2))
            
    break_evens = sorted(list(set([round(be, 2) for be in break_evens]))) # Dedup
    
    # Calculate Probability of Profit if market data is provided
    pop = None
    if current_spot and T and r is not None and sigma:
        pop = 0.0
        # Check intervals
        boundaries = [0.0] + break_evens + [float('inf')]
        
        for i in range(len(boundaries) - 1):
            lower = boundaries[i]
            upper = boundaries[i+1]
            
            # Test midpoint P&L
            if upper == float('inf'):
                test_spot = max(lower * 1.01, lower + 1.0)
            elif lower == 0.0:
                 test_spot = upper * 0.99
            else:
                test_spot = (lower + upper) / 2.0
                
            # Compute P&L at test spot
            val = sum([leg.net_profit(test_spot) for leg in strategy])
            
            if val > 0:
                # This segment is profitable. Add its probability mass.
                # Prob(lower < S_T < upper) = Prob(S_T < upper) - Prob(S_T < lower)
                # Prob(S_T < K) = N(-d2)
                
                if upper == float('inf'):
                    p_upper = 1.0
                else:
                    p_upper = probability_terminal_price_below(upper, current_spot, T, r, sigma)
                    
                if lower == 0.0:
                    p_lower = 0.0
                else:
                    p_lower = probability_terminal_price_below(lower, current_spot, T, r, sigma)
                    
                pop += max(0, p_upper - p_lower)
        
        pop = round(pop * 100, 1)

    return {
        "max_profit": max_profit,
        "max_loss": max_loss,
        "net_entry_cost": net_cost, # Positive means Debit, Negative means Credit
        "break_evens": break_evens,
        "probability_of_profit": pop
    }
