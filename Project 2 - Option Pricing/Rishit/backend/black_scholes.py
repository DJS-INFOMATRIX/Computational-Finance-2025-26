import numpy as np
from scipy.stats import norm
import math

class BlackScholes:
    """
    Core math engine for pricing European options using the Black-Scholes-Merton model.
    Includes decomposition of value and plain-English explanations.
    """

    def __init__(self, S, K, T, r, sigma, div_yield=0.0):
        # Strict validation
        if S <= 0: raise ValueError("Underlying price must be positive.")
        if K <= 0: raise ValueError("Strike price must be positive.")
        if T < 0: raise ValueError("Time to expiration cannot be negative.")
        if sigma < 0: raise ValueError("Volatility cannot be negative.")
        
        self.S = float(S)
        self.K = float(K)
        self.T = float(T) if T > 0 else 0.00001
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(div_yield)
        
        self.d1 = self._calculate_d1()
        self.d2 = self._calculate_d2()

    def _calculate_d1(self):
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _calculate_d2(self):
        return self.d1 - self.sigma * np.sqrt(self.T)

    def price(self, type_):
        if type_.lower() == 'call':
            return (self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)) - (self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
        else:
            return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)) - (self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))

    def intrinsic_value(self, type_):
        if type_.lower() == 'call':
            return max(0, self.S - self.K)
        else:
            return max(0, self.K - self.S)

    def time_value(self, type_):
        return max(0, self.price(type_) - self.intrinsic_value(type_))

    # Greeks
    def delta(self, type_):
        if type_.lower() == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(self.d1)
        else:
            return -np.exp(-self.q * self.T) * norm.cdf(-self.d1)

    def gamma(self):
        return (np.exp(-self.q * self.T) * norm.pdf(self.d1)) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * np.sqrt(self.T) / 100

    def theta(self, type_):
        term1 = -(self.S * self.sigma * np.exp(-self.q * self.T) * norm.pdf(self.d1)) / (2 * np.sqrt(self.T))
        if type_.lower() == 'call':
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            term3 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            term3 = -self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
        return (term1 + term2 + term3) / 365

    def rho(self, type_):
        if type_.lower() == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100

    def get_analysis(self, type_):
        """Returns comprehensive analysis including value decomposition and plain English explanation."""
        type_ = type_.lower()
        price = self.price(type_)
        intrinsic = self.intrinsic_value(type_)
        extrinsic = self.time_value(type_)
        delta = self.delta(type_)
        gamma = self.gamma()
        theta = self.theta(type_)
        
        moneyness = "OTM"
        if intrinsic > 0: moneyness = "ITM"
        elif abs(self.S - self.K) < (self.S * 0.005): moneyness = "ATM"

        explanation = (
            f"This {moneyness} {type_.title()} option is priced at ${price:.2f}. "
            f"It represents a mix of ${intrinsic:.2f} 'real' value (Intrinsic) and "
            f"${extrinsic:.2f} 'hope' value (Time Value). "
            f"With a Delta of {delta:.2f}, for every $1 the stock moves, this option moves approx ${abs(delta):.2f}. "
            f"Time decay (Theta) is eating ${abs(theta):.2f} of value typically per day."
        )

        return {
            "price": price,
            "intrinsic_value": intrinsic,
            "time_value": extrinsic,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": self.vega(),
            "rho": self.rho(type_),
            "explanation": explanation
        }

    # Helper for legacy calls
    def get_all_greeks(self, type_):
        return self.get_analysis(type_)
