from typing import List, Dict, Any
import numpy as np
from scipy.stats import norm
from .black_scholes import BlackScholes

class OptionLeg:
    def __init__(self, type: str, side: str, K: float, T: float, quantity: int = 1):
        self.type = type.lower()
        self.side = side.lower()
        self.K = float(K)
        self.T = float(T)
        self.quantity = int(quantity)

    def direction(self) -> int:
        return 1 if self.side == 'buy' else -1

class StrategyEngine:
    def __init__(self, S: float, r: float, sigma: float):
        self.S = S
        self.r = r
        self.sigma = sigma
        self.legs: List[OptionLeg] = []

    def add_leg(self, leg: OptionLeg):
        self.legs.append(leg)

    def calculate_strategy_greeks(self) -> Dict[str, float]:
        total_greeks = {
            "price": 0.0, "delta": 0.0, "gamma": 0.0, 
            "theta": 0.0, "vega": 0.0, "rho": 0.0,
            "intrinsic_value": 0.0, "time_value": 0.0
        }

        for leg in self.legs:
            quantity_mult = leg.quantity * leg.direction()
            
            bs = BlackScholes(self.S, leg.K, leg.T, self.r, self.sigma)
            # Use new get_analysis method
            leg_data = bs.get_analysis(leg.type)
            
            for key in total_greeks:
                if key in leg_data:
                    total_greeks[key] += leg_data[key] * quantity_mult

        return total_greeks

    def get_net_cost(self):
        """Returns net debit (positive) or credit (negative)."""
        cost = 0.0
        for leg in self.legs:
            bs = BlackScholes(self.S, leg.K, leg.T, self.r, self.sigma)
            price = bs.price(leg.type)
            # Cost = Price * Quantity * (1 for Buy, -1 for Sell)
            cost += price * leg.quantity * leg.direction()
        return cost

    def analyze_risk_profile(self):
        """
        Calculates Max Profit, Max Loss, Breakevens, and Probability of Profit.
        Uses a scan method over a wide range of prices at expiration.
        """
        net_cost = self.get_net_cost()
        
        # Scan range: 3 std devs down to 3 std devs up
        # If T is very small, minimal range is +/- 20%
        # If T is large, range expands.
        
        T_max = max([leg.T for leg in self.legs]) if self.legs else 0.1
        vol_range = self.sigma * np.sqrt(T_max) * 3
        scan_min = self.S * max(0.2, (1 - vol_range))
        scan_max = self.S * (1 + vol_range)
        
        # Create a dense grid of prices
        prices = np.linspace(scan_min, scan_max, 2000)
        payoffs = []
        
        for p in prices:
            gross_value = 0.0
            for leg in self.legs:
                if leg.type == 'call':
                    val = max(0, p - leg.K)
                else:
                    val = max(0, leg.K - p)
                gross_value += val * leg.quantity * leg.direction()
            
            # PnL = Value at Exp - Initial Cost
            pnl = gross_value - net_cost
            payoffs.append(pnl)
            
        payoffs = np.array(payoffs)
        
        # Max Profit/Loss
        max_p = np.max(payoffs)
        max_l = np.min(payoffs)
        
        # Check edges to see if unlimited
        # If the slope at the edges is non-zero, it's unlimited
        slope_low = payoffs[1] - payoffs[0]
        slope_high = payoffs[-1] - payoffs[-2]
        
        max_profit_desc = f"${max_p:.2f}"
        max_loss_desc = f"${max_l:.2f}"
        
        if slope_high > 0.01: max_profit_desc = "Unlimited"
        if slope_low < -0.01: max_loss_desc = "Unlimited" # Put side down
        if slope_high < -0.01: max_loss_desc = "Unlimited" # Call side short
        if slope_low > 0.01: pass # Put side profit usually capped by 0 unless short put
        
        # Breakevens (where sign changes)
        breakevens = []
        for i in range(len(payoffs)-1):
            if payoffs[i] * payoffs[i+1] <= 0:
                # Linear interp
                fraction = abs(payoffs[i]) / (abs(payoffs[i]) + abs(payoffs[i+1]))
                be_price = prices[i] + fraction * (prices[i+1] - prices[i])
                breakevens.append(be_price)
                
        # Probability of Profit (POP)
        # Integrate lognormal PDF over profitable ranges
        prof_prob = 0.0
        
        # We need to find ranges where PnL > 0
        is_profitable = payoffs > 0
        
        # Use Simpson's rule or just sum of rectangular areas since grid is dense
        # PDF of stock price at T:
        # ln(St) ~ N(ln(S) + (r - 0.5sigma^2)T, sigma^2 T)
        
        # For simplicity, we can sum the PDF * delta_x for all profitable x
        # This is a discrete approximation of the integral
        
        step = prices[1] - prices[0]
        mu = np.log(self.S) + (self.r - 0.5 * self.sigma**2) * T_max
        std = self.sigma * np.sqrt(T_max)
        
        # Vectorized probability density calculation
        # f(x) = 1/(x sigma sqrt(T) sqrt(2pi)) * exp(...)
        # We use log-normal pdf provided by scipy to be safe
        from scipy.stats import lognorm
        # Shape parameter s = sigma * sqrt(T)
        # Scale = exp(mu) ?? No, scale = exp(mean of log)
        # Scipy lognorm: s is shape (sigma), scale is exp(mu)
        
        pdf_values = lognorm.pdf(prices, s=std, scale=np.exp(mu))
        
        # Sum prob where profitable
        # Filter pdf_values where is_profitable is True
        prof_prob = np.sum(pdf_values[is_profitable]) * step
        
        prof_prob = min(max(prof_prob, 0.0), 1.0) # Clamp 0-1

        # Text Generation
        # Convert to plain English
        if max_p > 1e6:
            profit_sentence = "Upside potential is mathematically unlimited."
        else:
            profit_sentence = f"Maximum profit is capped at {max_profit_desc}. "
            
        if max_l < -1e6:
            loss_sentence = "Loss potential is theoretically unlimited (be careful)."
        else:
            loss_sentence = f"Maximum possible loss is limited to {max_loss_desc}. "

        risk_text = f"<strong>Strategy Overview:</strong> {profit_sentence} {loss_sentence}<br/>"
        
        if breakevens:
            be_strs = [f"${be:.2f}" for be in breakevens]
            risk_text += f"Market needs to be at {', '.join(be_strs)} at expiration to break even.<br/>"
        
        if prof_prob > 0.6:
            risk_text += f"There is a <strong>high probability ({prof_prob*100:.1f}%)</strong> of making at least $0.01."
        elif prof_prob < 0.4:
            risk_text += f"The trade has a <strong>low probability ({prof_prob*100:.1f}%)</strong> of profit, implying it is a high-risk or hedging play."
        else:
            risk_text += f"The probability of profit is roughly a coin flip ({prof_prob*100:.1f}%)."

        return {
            "max_profit": max_profit_desc,
            "max_loss": max_loss_desc,
            "breakevens": breakevens,
            "pop": prof_prob,
            "risk_summary": risk_text
        }

    def generate_payoff_diagram(self, range_percent=0.2):
        """Regenerated to match the robust scan logic but returned in the simple format for frontend."""
        # Reuse robust logic but map to requested format
        net_cost = self.get_net_cost()
        scan_min = self.S * (1 - range_percent)
        scan_max = self.S * (1 + range_percent)
        prices = np.linspace(scan_min, scan_max, 100) # Lower res for chart
        
        payoffs = []
        for p in prices:
            val = 0.0
            for leg in self.legs:
                if leg.type == 'call': v = max(0, p - leg.K)
                else: v = max(0, leg.K - p)
                val += v * leg.quantity * leg.direction()
            payoffs.append({"underlying_price": p, "pnl": val - net_cost})

        return payoffs
