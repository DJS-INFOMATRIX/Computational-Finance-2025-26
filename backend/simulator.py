import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .strategy_engine import StrategyEngine, OptionLeg

class MarketSimulator:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data = None
        self.stock = yf.Ticker(ticker)

    def fetch_history(self, period="1y"):
        """Fetches historical data for the ticker."""
        self.data = self.stock.history(period=period)
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        
        # Calculate historical volatility (approximate)
        self.data['Returns'] = self.data['Close'].pct_change()
        self.volatility = self.data['Returns'].std() * np.sqrt(252) # Annualized
        return self.data

    def compare_option_price(self, legs: list, current_model_price: float):
        """
        Attempts to fetch real-time option prices from Yahoo Finance and compare with model.
        Returns % Error.
        Note: This is best-effort as YF data might be delayed or unavailable for specific strikes.
        """
        try:
            real_price = 0.0
            options_dates = self.stock.options
            if not options_dates: return {"error": "No option chain available"}
            
            # Simple heuristic: try to find nearest expiration to the first leg's T
            # This is complex because we need to map T (years) to a YF date string.
            # For this MVP, we will skip precise mapping and just return a placeholder or 
            # simplistic check if exact match found.
            
            # Placeholder for reliable real-time fetching in this architectural scope
            # Returning None to indicate "Data Unavailable" rather than fake error.
            return None 

        except Exception as e:
            return {"error": str(e)}

    def run_simulation(self, strategy_legs: list, days: int = 30, user_volatility: float = None):
        """
        Simulates strategy performance over the last `days`.
        Returns Equity Curve, Drawdown, ROI, and daily P&L.
        """
        if self.data is None:
            self.fetch_history()
            
        subset = self.data.iloc[-days:].copy()
        if subset.empty: return {"error": "Not enough data"}

        initial_price = subset.iloc[0]['Close']
        risk_free_rate = 0.045
        
        # Use user volatility if provided (What-If Analysis), else use historical
        vol_to_use = user_volatility if user_volatility else self.volatility
        
        # 1. Calculate Initial Entry Cost
        engine_start = StrategyEngine(initial_price, risk_free_rate, vol_to_use)
        for leg_dict in strategy_legs:
            engine_start.add_leg(OptionLeg(**leg_dict))
        
        start_greeks = engine_start.calculate_strategy_greeks()
        entry_cost = start_greeks['price'] # Net Debit/Credit
        
        equity_curve = []
        pnl_table = []
        peak_equity = 0.0
        max_drawdown = 0.0
        
        for i in range(len(subset)):
            date = subset.index[i]
            price = subset.iloc[i]['Close']
            
            # Decay time
            remaining_days = max(1, days - i) 
            T_current = remaining_days / 365.0
            
            # Re-evaluate strategy value
            engine_now = StrategyEngine(price, risk_free_rate, vol_to_use)
            for leg_dict in strategy_legs:
                updated_leg_dict = leg_dict.copy()
                updated_leg_dict['T'] = T_current
                engine_now.add_leg(OptionLeg(**updated_leg_dict))
            
            current_greeks = engine_now.calculate_strategy_greeks()
            current_value = current_greeks['price']
            
            # PnT = Current Value - Initial Cost
            # Equity = Initial Capital (Assumed 10k? No, just track PnL accumulation)
            # Let's track PnL directly.
            
            total_pnl = current_value - entry_cost
            
            # ROI = Total PnL / Initial Cost (if Debit strategy).
            # If Credit strategy (negative cost), ROI is tricky, usually PnL / Margin.
            # We will use PnL as raw value.
            
            equity_curve.append(total_pnl)
            
            # Drawdown calculation (on PnL curve? usually on Account Value)
            # Let's assume a phantom account starts at $10,000 to visualize DD.
            account_val = 10000 + total_pnl
            peak_equity = max(peak_equity, account_val)
            dd = (peak_equity - account_val) / peak_equity if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, dd)
            
            pnl_table.append({
                "date": date.strftime("%Y-%m-%d"),
                "underlying_price": price,
                "strategy_value": current_value,
                "pnl": total_pnl,
                "roi_pct": (total_pnl / abs(entry_cost) * 100) if abs(entry_cost) > 0.01 else 0
            })
            
        final_pnl = equity_curve[-1]
        final_roi = (final_pnl / abs(entry_cost) * 100) if abs(entry_cost) > 0.01 else 0
        
        return {
            "equity_curve": equity_curve,
            "max_drawdown_pct": max_drawdown * 100,
            "final_roi_pct": final_roi,
            "total_pnl": final_pnl,
            "history": pnl_table,
            "volatility_used": self.volatility
        }
