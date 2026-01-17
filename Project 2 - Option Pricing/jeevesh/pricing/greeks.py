
import numpy as np
from scipy.stats import norm
from .black_scholes import d1, d2

"""
Greeks Calculation Module
=========================

This module calculates the "Greeks" - the partial derivatives of the Black-Scholes formula.
They measure the sensitivity of the option's price to various factors (price, time, volatility).

Definitions:
- Delta: Sensitivity to Spot Price (First derivative).
- Gamma: Sensitivity of Delta to Spot Price (Second derivative).
- Vega:  Sensitivity to Volatility.
- Theta: Sensitivity to Time decay.
- Rho:   Sensitivity to Interest Rates.
"""

def delta_call(S, K, T, r, sigma):
    """
    Calculates Delta for a Call Option.
    
    Formula: N(d1)
    
    Explanation:
    Delta estimates how much the option price will change for a $1 increase in the stock price.
    - Example: Delta of 0.50 means if stock goes up $1, option goes up $0.50.
    - It can also be interpreted roughly as the percentage probability of the option expiring ITM.
    """
    if T <= 0: return 0
    d1_val = d1(S, K, T, r, sigma)
    return norm.cdf(d1_val)

def delta_put(S, K, T, r, sigma):
    """
    Calculates Delta for a Put Option.
    
    Formula: N(d1) - 1
    
    Explanation:
    Put Delta is always negative because put values decrease as stock price rises.
    - Example: Delta of -0.40 means if stock goes up $1, put option goes DOWN $0.40.
    """
    if T <= 0: return 0
    d1_val = d1(S, K, T, r, sigma)
    return norm.cdf(d1_val) - 1

def gamma(S, K, T, r, sigma):
    """
    Calculates Gamma (Same for Call and Put).
    
    Formula: N'(d1) / (S * sigma * sqrt(T)) where N' is standard normal PDF.
    
    Explanation:
    Gamma measures the acceleration of Delta. It tells you how fast your directional risk (Delta) is changing.
    - High Gamma means your risk profile changes rapidly with small market moves.
    - At-the-money options have the highest Gamma (highest uncertainty/sensitivity).
    """
    if T <= 0 or S <= 0 or sigma <= 0: return 0
    d1_val = d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    """
    Calculates Vega (Same for Call and Put).
    
    Formula: S * sqrt(T) * N'(d1) / 100 
    (Divided by 100 to standard convention: change per 1% vol point change).
    
    Explanation:
    Vega measures sensitivity to volatility.
    - Example: Vega of 0.10 means if Implied Volatility increases by 1% (e.g., 20% to 21%), 
      the option price increases by $0.10.
    - Vega is highest for at-the-money options with longer time to expiration.
    """
    if T <= 0: return 0
    d1_val = d1(S, K, T, r, sigma)
    # Standard convention is usually reporting per 1% change, but raw formula is per unit.
    # We return standard unit sensitivity here (per 100% vol change)
    # Most traders prefer to see it scaled to 1%, so we explicitly note this.
    return S * np.sqrt(T) * norm.pdf(d1_val) * 0.01

def theta_call(S, K, T, r, sigma):
    """
    Calculates Theta for a Call Option (Daily decay).
    
    Explanation:
    Theta measures time decay: how much value the option loses simply because one day passes.
    - Theta is essentially "rent" paid by the option buyer.
    - Displayed as a negative number (e.g., -0.05 means losing 5 cents per day).
    """
    if T <= 0: return 0
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(d1_val, sigma, T)
    
    t1 = -(S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    t2 = -r * K * np.exp(-r * T) * norm.cdf(d2_val)
    
    # Divide by 365 to get daily theta
    return (t1 + t2) / 365.0

def theta_put(S, K, T, r, sigma):
    """
    Calculates Theta for a Put Option (Daily decay).
    
    Explanation:
    Similar to Call Theta, usually negative.
    Put Theta can be positive for deep ITM puts (where interest earned on strike cash > time decay),
    but generally it represents daily value erosion.
    """
    if T <= 0: return 0
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(d1_val, sigma, T)
    
    t1 = -(S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    t2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)
    
    # Divide by 365 to get daily theta
    return (t1 + t2) / 365.0
