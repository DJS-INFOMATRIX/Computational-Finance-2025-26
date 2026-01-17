
import numpy as np
from scipy.stats import norm

"""
Black-Scholes Pricing Module
============================

This module implements the Black-Scholes-Merton model for pricing European options.
It strictly uses basic math and scipy.stats.norm for the cumulative distribution function.
NO external financial libraries are used to ensure transparency ("Glass Box").

Core Formula:
-------------
Call = S * N(d1) - K * e^(-rT) * N(d2)
Put  = K * e^(-rT) * N(-d2) - S * N(-d1)

Where:
S = Spot Price of the underlying asset
K = Strike Price of the option
T = Time to expiration (in years)
r = Risk-free interest rate (annualized, decimal)
sigma = Volatility of the underlying asset (annualized, decimal)
N(x) = Cumulative distribution function of the standard normal distribution

Intuitively:
- N(d1) is roughly the delta of the call options (probability of the option finishing ITM adjusted for hedge).
- N(d2) is the probability that the option finishes in-the-money (ITM).
- e^(-rT) is the discount factor, bringing future cash flows to present value.
"""

def d1(S, K, T, r, sigma):
    """
    Calculates the d1 component of the Black-Scholes formula.
    
    d1 measures the "moneyness" of the option relative to volatility and time.
    It represents the number of standard deviations the log-spot price is away from the log-strike price.
    """
    if T <= 0 or sigma <= 0:
        return 0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(d1_val, sigma, T):
    """
    Calculates the d2 component using d1.
    
    d2 = d1 - sigma * sqrt(T)
    It is used to calculate the probability of the option expiring in-the-money.
    """
    if T <= 0:
        return 0
    return d1_val - sigma * np.sqrt(T)

def call_price(S, K, T, r, sigma):
    """
    Calculates the theoretical price of a European Call Option.
    
    Logic:
    1. Calculate d1 and d2 to understand probability and risk.
    2. N(d1) * S represents the expected value of receiving the stock.
    3. N(d2) * K * e^(-rT) represents the expected cost of paying the strike price, discounted to today.
    4. Call Price = (Asset Value) - (Strike Cost).
    """
    if T <= 0:
        return max(0, S - K)
    
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(d1_val, sigma, T)
    
    price = S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
    return max(0, price)  # Option price cannot be negative

def put_price(S, K, T, r, sigma):
    """
    Calculates the theoretical price of a European Put Option.
    
    Logic:
    1. Put Price is the reverse of Call: We want to sell at K, not buy.
    2. N(-d2) * K * e^(-rT) represents the expected value of receiving the strike price cash.
    3. N(-d1) * S represents the expected obligation to deliver the stock.
    4. Put Price = (Strike Value) - (Asset Cost).
    """
    if T <= 0:
        return max(0, K - S)
    
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(d1_val, sigma, T)
    
    price = K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
    return max(0, price)

def decompose_option_price(price, S, K, option_type='call'):
    """
    Splits an option's price into Intrinsic Value and Time (Extrinsic) Value.
    
    - Intrinsic Value: The profit if exercised immediately.
    - Time Value (Extrinsic): The premium paid for the *possibility* that the price moves further in your favor before expiration.
      This value decays as options approach expiration (Theta decay).
    """
    if option_type == 'call':
        intrinsic = max(0, S - K)
    else:
        intrinsic = max(0, K - S)
        
    time_value = max(0, price - intrinsic)
    
    return intrinsic, time_value
