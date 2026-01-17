
from typing import List, Dict
from strategies.payoff_engine import OptionLeg
from pricing.greeks import delta_call, delta_put, gamma, vega, theta_call, theta_put

def aggregate_greeks(strategy: List[OptionLeg], S: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """
    Calculates the net Greeks for a multi-leg option strategy.
    
    Logic:
    - Greeks are additive. Net Delta = Sum(Leg Delta * Quantity * Side).
    - Long side = +1, Short side = -1.
    """
    net_greeks = {
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0
    }
    
    for leg in strategy:
        # Determine multiplier based on side (Long/Short) and Quantity
        # Note: Quantity in OptionLeg is integer number of contracts.
        # Standard contract size is 100 shares, but usually Greeks are reported per share 
        # or per contract. Here we stick to per-share unit summation consistent with BS model 
        # and then multiply by quantity.
        
        multiplier = leg.quantity if leg.side == 'long' else -leg.quantity
        
        if leg.op_type == 'call':
            d = delta_call(S, leg.strike, T, r, sigma)
            t = theta_call(S, leg.strike, T, r, sigma)
        else:
            d = delta_put(S, leg.strike, T, r, sigma)
            t = theta_put(S, leg.strike, T, r, sigma)
            
        # Gamma and Vega are same for Call/Put at same strike
        g = gamma(S, leg.strike, T, r, sigma)
        v = vega(S, leg.strike, T, r, sigma)
        
        net_greeks["delta"] += d * multiplier
        net_greeks["gamma"] += g * multiplier
        net_greeks["vega"]  += v * multiplier
        net_greeks["theta"] += t * multiplier
        
    return net_greeks
