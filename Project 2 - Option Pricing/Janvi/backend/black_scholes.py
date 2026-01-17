"""
Black-Scholes Option Pricing Engine

This module implements the Black-Scholes formula for European options.
All intermediate values are calculated and returned for educational transparency.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> Dict:
    """
    Calculate Call option price using Black-Scholes formula.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years (e.g., 30 days = 30/365)
    r : float
        Risk-free interest rate (e.g., 0.05 for 5%)
    sigma : float
        Volatility (annualized standard deviation, e.g., 0.20 for 20%)
    
    Returns:
    --------
    dict : Contains price and all intermediate values for transparency
    
    Formula Breakdown:
    ------------------
    C = S * N(d1) - K * e^(-rT) * N(d2)
    
    where:
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
    N(x) = Cumulative standard normal distribution
    """
    
    # Handle edge case: option already expired
    if T <= 0:
        intrinsic_value = max(S - K, 0)
        return {
            'price': intrinsic_value,
            'intrinsic_value': intrinsic_value,
            'time_value': 0,
            'd1': None,
            'd2': None,
            'N_d1': None,
            'N_d2': None,
            'explanation': 'Option has expired. Price equals intrinsic value only.'
        }
    
    # Calculate d1
    # d1 represents the standardized distance from current price to strike,
    # adjusted for drift (risk-free rate + volatility adjustment)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Calculate d2
    # d2 = d1 - volatility * sqrt(time)
    # Represents probability that option will be exercised
    d2 = d1 - sigma * np.sqrt(T)
    
    # N(d1): Probability measure related to delta (hedge ratio)
    N_d1 = norm.cdf(d1)
    
    # N(d2): Probability that option will expire in-the-money
    N_d2 = norm.cdf(d2)
    
    # Black-Scholes Call Price
    # Stock component: S * N(d1) - value of holding stock weighted by probability
    # Strike component: K * e^(-rT) * N(d2) - present value of strike payment
    call_price = S * N_d1 - K * np.exp(-r * T) * N_d2
    
    # Calculate intrinsic value (immediate exercise value)
    intrinsic_value = max(S - K, 0)
    
    # Time value is the premium beyond intrinsic value
    time_value = call_price - intrinsic_value
    
    # Generate plain-English explanation
    explanation = generate_call_explanation(S, K, T, r, sigma, call_price, intrinsic_value, N_d2)
    
    return {
        'price': round(call_price, 2),
        'intrinsic_value': round(intrinsic_value, 2),
        'time_value': round(time_value, 2),
        'd1': round(d1, 4),
        'd2': round(d2, 4),
        'N_d1': round(N_d1, 4),
        'N_d2': round(N_d2, 4),
        'explanation': explanation
    }


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> Dict:
    """
    Calculate Put option price using Black-Scholes formula.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility (annualized)
    
    Returns:
    --------
    dict : Contains price and all intermediate values
    
    Formula:
    --------
    P = K * e^(-rT) * N(-d2) - S * N(-d1)
    
    Put-Call Parity:
    ----------------
    P = C - S + K * e^(-rT)
    """
    
    # Handle edge case: option already expired
    if T <= 0:
        intrinsic_value = max(K - S, 0)
        return {
            'price': intrinsic_value,
            'intrinsic_value': intrinsic_value,
            'time_value': 0,
            'd1': None,
            'd2': None,
            'N_d1': None,
            'N_d2': None,
            'explanation': 'Option has expired. Price equals intrinsic value only.'
        }
    
    # Calculate d1 and d2 (same as call)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # For puts, we use N(-d1) and N(-d2)
    # N(-d2): Probability that put will be exercised
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)
    
    # Black-Scholes Put Price
    put_price = K * np.exp(-r * T) * N_neg_d2 - S * N_neg_d1
    
    # Intrinsic value for put
    intrinsic_value = max(K - S, 0)
    
    # Time value
    time_value = put_price - intrinsic_value
    
    # Generate explanation
    explanation = generate_put_explanation(S, K, T, r, sigma, put_price, intrinsic_value, N_neg_d2)
    
    return {
        'price': round(put_price, 2),
        'intrinsic_value': round(intrinsic_value, 2),
        'time_value': round(time_value, 2),
        'd1': round(d1, 4),
        'd2': round(d2, 4),
        'N_d1': round(N_neg_d1, 4),
        'N_d2': round(N_neg_d2, 4),
        'explanation': explanation
    }


def generate_call_explanation(S: float, K: float, T: float, r: float, sigma: float, 
                              price: float, intrinsic: float, prob: float) -> str:
    """Generate plain-English explanation for call option pricing."""
    
    days = int(T * 365)
    moneyness = "in-the-money" if S > K else ("at-the-money" if S == K else "out-of-the-money")
    
    explanation = f"This call option is currently {moneyness}. "
    explanation += f"The stock price is ${S:.2f} and the strike price is ${K:.2f}. "
    
    if intrinsic > 0:
        explanation += f"If exercised now, you'd make ${intrinsic:.2f} per share. "
    else:
        explanation += f"Exercising now would yield no profit. "
    
    explanation += f"The option has {days} days until expiration. "
    explanation += f"Black-Scholes estimates a {prob*100:.1f}% probability this option will expire in-the-money. "
    
    if price > intrinsic:
        time_val = price - intrinsic
        explanation += f"The ${time_val:.2f} time value represents the potential for the stock to rise further."
    
    return explanation


def generate_put_explanation(S: float, K: float, T: float, r: float, sigma: float,
                             price: float, intrinsic: float, prob: float) -> str:
    """Generate plain-English explanation for put option pricing."""
    
    days = int(T * 365)
    moneyness = "in-the-money" if S < K else ("at-the-money" if S == K else "out-of-the-money")
    
    explanation = f"This put option is currently {moneyness}. "
    explanation += f"The stock price is ${S:.2f} and the strike price is ${K:.2f}. "
    
    if intrinsic > 0:
        explanation += f"If exercised now, you'd make ${intrinsic:.2f} per share. "
    else:
        explanation += f"Exercising now would yield no profit. "
    
    explanation += f"The option has {days} days until expiration. "
    explanation += f"Black-Scholes estimates a {prob*100:.1f}% probability this option will expire in-the-money. "
    
    if price > intrinsic:
        time_val = price - intrinsic
        explanation += f"The ${time_val:.2f} time value represents the potential for the stock to fall further."
    
    return explanation


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict:
    """
    Calculate option Greeks for risk measurement.
    
    Greeks measure sensitivities to various factors:
    - Delta: Sensitivity to stock price changes
    - Gamma: Rate of change of delta
    - Theta: Time decay (per day)
    - Vega: Sensitivity to volatility changes
    - Rho: Sensitivity to interest rate changes
    """
    
    if T <= 0:
        return {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0,
            'rho': 0
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change
    else:  # put
        delta = -norm.cdf(-d1)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    # Gamma is same for calls and puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta (per day) - time decay
    theta_call = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    
    theta_put = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    theta = theta_call if option_type.lower() == 'call' else theta_put
    
    # Vega (per 1% volatility change)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 4),
        'theta': round(theta, 4),
        'vega': round(vega, 4),
        'rho': round(rho, 4)
    }
