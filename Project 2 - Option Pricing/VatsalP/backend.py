import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, T, r, sigma, option_type="call"):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.option_type = option_type.lower()

    def d1(self):
        if self.S <= 0 or self.K <= 0 or self.T <= 0 or self.sigma <= 0:
            return 0
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        try:
            d1 = self.d1()
            d2 = self.d2()
            if self.option_type == "call":
                return (self.S * norm.cdf(d1)) - (self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
            else:
                return (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)) - (self.S * norm.cdf(-d1))
        except:
            return 0.0

    def delta(self):
        try:
            d1 = self.d1()
            if self.option_type == "call":
                return norm.cdf(d1)
            else:
                return norm.cdf(d1) - 1
        except:
            return 0.0

    def intrinsic_value(self):
        """Calculates 'Real Value' - what it's worth if exercised NOW."""
        if self.option_type == "call":
            return max(self.S - self.K, 0)
        else:
            return max(self.K - self.S, 0)

class Strategy:
    def __init__(self, legs):
        self.legs = legs

    def calculate_metrics(self, price_map, r):
        total_price = 0
        total_delta = 0
        total_intrinsic = 0
        
        for leg in self.legs:
            ticker = leg.get('ticker', 'UNKNOWN')
            S = price_map.get(ticker, 0)
            
            if S > 0:
                bs = BlackScholes(S, leg['strike'], leg['expiry'], r, leg['vol'], leg['op_type'])
                p = bs.price()
                d = bs.delta()
                iv = bs.intrinsic_value()
                
                if leg['action'] == 'buy':
                    total_price += p
                    total_delta += d
                    total_intrinsic += iv
                else: # sell
                    total_price -= p
                    total_delta -= d
                    total_intrinsic -= iv
        
        # Time Value = Total Option Price - Intrinsic Value
        total_time_value = total_price - total_intrinsic
        
        return {
            "price": total_price,
            "delta": total_delta,
            "intrinsic": total_intrinsic,
            "time_value": total_time_value
        }