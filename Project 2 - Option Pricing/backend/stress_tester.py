from typing import List, Dict
import numpy as np
from .strategy_engine import StrategyEngine, OptionLeg

class StressTester:
    def __init__(self, strategy_engine: StrategyEngine):
        self.engine = strategy_engine

    def run_stress_test(self, vol_shock: float, price_shock: float, time_decay_days: float, rate_shock: float):
        """
        Runs a stress test with specific shock parameters.
        vol_shock: Percentage change (e.g., 0.20 for +20%)
        price_shock: Percentage change (e.g., -0.10 for -10%)
        time_decay_days: Number of days to forward
        rate_shock: Absolute change in rate (e.g., 0.01 for +1%)
        """
        # Apply shocks
        shocked_S = self.engine.S * (1 + price_shock)
        shocked_sigma = self.engine.sigma * (1 + vol_shock)
        shocked_r = self.engine.r + rate_shock
        
        # Calculate new T for each leg
        shocked_legs = []
        for leg in self.engine.legs:
            new_T = max(0.0001, leg.T - (time_decay_days / 365.0))
            shocked_legs.append(OptionLeg(leg.type, leg.side, leg.K, new_T, leg.quantity))
            
        # Create new engine with startled parameters
        shocked_engine = StrategyEngine(shocked_S, shocked_r, shocked_sigma)
        for leg in shocked_legs:
            shocked_engine.add_leg(leg)
            
        greeks = shocked_engine.calculate_strategy_greeks()
        return greeks['price']

    def generate_risk_report(self):
        """
        Runs comprehensive scenarios and returns Best/Worst case and Risk Score.
        """
        scenarios = [
            {"name": "Base Case", "vol": 0, "price": 0, "time": 0, "rate": 0},
            {"name": "Crash (-20%)", "vol": 0.5, "price": -0.2, "time": 0, "rate": 0},
            {"name": "Rally (+20%)", "vol": -0.2, "price": 0.2, "time": 0, "rate": 0},
            {"name": "Vol Crush (-30%)", "vol": -0.3, "price": 0, "time": 0, "rate": 0},
            {"name": "Vol Spike (+50%)", "vol": 0.5, "price": 0, "time": 0, "rate": 0},
            {"name": "Time (30 days)", "vol": 0, "price": 0, "time": 30, "rate": 0},
        ]
        
        current_price = self.engine.calculate_strategy_greeks()['price']
        results = []
        pnl_values = []
        
        for sc in scenarios:
            new_price = self.run_stress_test(sc['vol'], sc['price'], sc['time'], sc['rate'])
            pnl = new_price - current_price
            pnl_values.append(pnl)
            results.append({
                "scenario": sc['name'],
                "new_value": new_price,
                "pnl": pnl
            })
            
        best_case = max(pnl_values)
        worst_case = min(pnl_values)
        
        # Simple Risk Score (0-10): proportional to worst case loss relative to some baseline? 
        # Or maybe volatility of PnL results? Let's normalize it arbitrarily for now.
        # Ensure risk score is positive 0-10. 
        # If worst case is large negative, high risk.
        risk_score = min(10, abs(worst_case) / (self.engine.S * 0.05) * 10) if self.engine.S > 0 else 5
        
        return {
            "current_value": current_price,
            "scenarios": results,
            "best_case": best_case,
            "worst_case": worst_case,
            "risk_score": int(risk_score)
        }
