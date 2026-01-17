
import yfinance as yf
import pandas as pd
import os
import requests
from datetime import datetime, timedelta

"""
Market Data Module
==================

Handles data fetching from external sources (yfinance, FRED).
Implements local caching to respect API limits and ensure offline capability.

Sources:
- Prices: Yahoo Finance (via yfinance)
- Risk-Free Rate: FRED (Federal Reserve Economic Data) or Hardcoded Fallback.
"""

CACHE_DIR = "e:\\optionCF\\data\\cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_current_price(ticker: str) -> float:
    """
    Fetches the latest available price for a ticker.
    Refuses to fail: if yfinance fails, tries to read last cached value.
    """
    try:
        # Fast fetch of just the last day
        ticker_obj = yf.Ticker(ticker)
        # 'history' is more reliable than 'info' for price
        data = ticker_obj.history(period="1d")
        if not data.empty:
            price = data['Close'].iloc[-1]
            return float(price)
    except Exception as e:
        print(f"Warning: Live fetch failed for {ticker}: {e}")
    
    # Fallback to cache if live fails
    print(f"Attempting cache fallback for {ticker}...")
    cached_path = os.path.join(CACHE_DIR, f"{ticker}_history.csv")
    if os.path.exists(cached_path):
        df = pd.read_csv(cached_path)
        if not df.empty:
            return float(df.iloc[-1]['Close'])
            
    print(f"Error: No price data available for {ticker}")
    return 100.0 # Ultimate fallback for UI skeleton to not crash

def fetch_price_history(ticker: str, period="1y") -> pd.DataFrame:
    """
    Fetches historical OHLCV data.
    Caches result to CSV.
    """
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_history.csv")
    
    # Try fetching live
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period)
        if not df.empty:
            # Save to cache
            df.to_csv(cache_path)
            return df
    except Exception as e:
        print(f"Warning: History fetch failed for {ticker}: {e}")
        
    # Load from cache if fetch failed or returned empty
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        
    return pd.DataFrame()

def get_risk_free_rate() -> float:
    """
    Fetches the 10-Year Treasury Rate from FRED.
    Requires FRED_API_KEY env var.
    Falls back to constant 4.0% if missing or failed.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("Note: FRED_API_KEY not found. Using fallback Risk-Free Rate.")
        return 0.04 # 4% default
        
    series_id = "DGS10" # 10-Year Treasury Constant Maturity Rate
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
    
    try:
        response = requests.get(url)
        data = response.json()
        # Get latest observation
        obs = data['observations'][-1]
        rate = float(obs['value']) / 100.0 # Convert 4.5 to 0.045
        return rate
    except Exception as e:
        print(f"Warning: FRED fetch failed: {e}. Using fallback.")
        return 0.04

def get_historical_volatility(ticker: str, window=252) -> float:
    """
    Calculates annualized historical volatility.
    """
    df = fetch_price_history(ticker)
    if df.empty:
        return 0.20 # Default 20%
        
    df['returns'] = df['Close'].pct_change()
    vol = df['returns'].std() * (252 ** 0.5)
    return float(vol)
