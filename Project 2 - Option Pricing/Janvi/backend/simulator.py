"""
Historical Simulator - "Time Machine"

Allows users to test option strategies on historical data.
Fetches stock prices and recalculates option values over time.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from black_scholes import black_scholes_call, black_scholes_put


def fetch_historical_data(ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Fetch historical stock price data.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format (default: today)
    
    Returns:
    --------
    pd.DataFrame : Historical price data with Date and Close columns
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")
        
        # Return simplified dataframe
        return hist[['Close']].reset_index()
    
    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")


def calculate_historical_volatility(prices: pd.Series, window: int = 30) -> float:
    """
    Calculate historical volatility from price series.
    
    Parameters:
    -----------
    prices : pd.Series
        Historical closing prices
    window : int
        Lookback window in days
    
    Returns:
    --------
    float : Annualized volatility
    """
    # Calculate daily returns
    returns = np.log(prices / prices.shift(1))
    
    # Calculate standard deviation of returns
    volatility = returns.tail(window).std()
    
    # Annualize (assuming 252 trading days)
    annualized_vol = volatility * np.sqrt(252)
    
    return annualized_vol


def simulate_strategy_over_time(ticker: str, start_date: str, strategy_config: Dict,
                                end_date: str = None) -> Dict:
    """
    Simulate how an option strategy would have performed historically.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker
    start_date : str
        Historical start date (when strategy was entered)
    strategy_config : dict
        Strategy configuration with legs
    end_date : str
        End date for simulation (default: expiration date)
    
    Returns:
    --------
    dict : Contains performance metrics and time series data
    """
    
    # Fetch historical data
    hist_data = fetch_historical_data(ticker, start_date, end_date)
    
    if len(hist_data) == 0:
        raise ValueError("No historical data available for the specified period")
    
    # Get initial stock price
    initial_price = hist_data.iloc[0]['Close']
    
    # Extract strategy parameters
    legs = strategy_config['legs']
    r = strategy_config.get('risk_free_rate', 0.05)
    
    # Calculate or use provided volatility
    if 'volatility' in strategy_config and strategy_config['volatility'] is not None:
        sigma = strategy_config['volatility']
    else:
        # Calculate historical volatility from available data
        sigma = calculate_historical_volatility(hist_data['Close'])
    
    # Find maximum expiration days to determine simulation period
    max_expiration = max(leg['expiration_days'] for leg in legs)
    
    # Calculate initial strategy cost
    initial_cost = 0
    initial_leg_prices = []
    
    for leg in legs:
        T = leg['expiration_days'] / 365.0
        
        if leg['option_type'].lower() == 'call':
            option_data = black_scholes_call(initial_price, leg['strike'], T, r, sigma)
        else:
            option_data = black_scholes_put(initial_price, leg['strike'], T, r, sigma)
        
        price = option_data['price']
        initial_leg_prices.append(price)
        
        # Calculate cost (negative for buy, positive for sell)
        multiplier = 100 * leg['quantity']
        if leg['position'].lower() == 'buy':
            initial_cost -= price * multiplier
        else:
            initial_cost += price * multiplier
    
    # Simulate strategy value over time
    dates = []
    stock_prices = []
    strategy_values = []
    pnl_values = []
    
    for idx, row in hist_data.iterrows():
        current_date = row['Date']
        current_price = row['Close']
        
        # Calculate days elapsed
        days_elapsed = idx
        
        # Calculate strategy value at this point
        strategy_value = 0
        
        for i, leg in enumerate(legs):
            days_remaining = max(0, leg['expiration_days'] - days_elapsed)
            T = days_remaining / 365.0
            
            if T <= 0:
                # Option has expired - calculate intrinsic value
                if leg['option_type'].lower() == 'call':
                    intrinsic = max(current_price - leg['strike'], 0)
                else:
                    intrinsic = max(leg['strike'] - current_price, 0)
                option_value = intrinsic
            else:
                # Option still has time value
                if leg['option_type'].lower() == 'call':
                    option_data = black_scholes_call(current_price, leg['strike'], T, r, sigma)
                else:
                    option_data = black_scholes_put(current_price, leg['strike'], T, r, sigma)
                option_value = option_data['price']
            
            # Calculate position value
            multiplier = 100 * leg['quantity']
            if leg['position'].lower() == 'buy':
                strategy_value += option_value * multiplier
            else:
                strategy_value -= option_value * multiplier
        
        # Calculate P&L
        pnl = strategy_value + initial_cost
        
        dates.append(current_date.strftime('%Y-%m-%d'))
        stock_prices.append(round(current_price, 2))
        strategy_values.append(round(strategy_value, 2))
        pnl_values.append(round(pnl, 2))
        
        # Stop if we've passed expiration
        if days_elapsed >= max_expiration:
            break
    
    # Calculate performance metrics
    final_pnl = pnl_values[-1] if pnl_values else 0
    max_pnl = max(pnl_values) if pnl_values else 0
    min_pnl = min(pnl_values) if pnl_values else 0
    max_drawdown = min_pnl
    
    # Generate explanation
    explanation = generate_simulation_explanation(
        ticker, initial_price, stock_prices[-1], initial_cost, 
        final_pnl, max_pnl, min_pnl
    )
    
    return {
        'ticker': ticker,
        'start_date': start_date,
        'initial_stock_price': round(initial_price, 2),
        'final_stock_price': round(stock_prices[-1], 2),
        'initial_cost': round(initial_cost, 2),
        'final_pnl': round(final_pnl, 2),
        'max_profit': round(max_pnl, 2),
        'max_drawdown': round(max_drawdown, 2),
        'return_pct': round((final_pnl / abs(initial_cost) * 100) if initial_cost != 0 else 0, 2),
        'time_series': {
            'dates': dates,
            'stock_prices': stock_prices,
            'strategy_values': strategy_values,
            'pnl': pnl_values
        },
        'explanation': explanation,
        'volatility_used': round(sigma, 4)
    }


def generate_simulation_explanation(ticker: str, initial_price: float, final_price: float,
                                    initial_cost: float, final_pnl: float, 
                                    max_profit: float, max_loss: float) -> str:
    """Generate plain-English explanation of simulation results."""
    
    price_change = ((final_price - initial_price) / initial_price) * 100
    
    explanation = f"Starting from a stock price of ${initial_price:.2f}, {ticker} "
    
    if price_change > 0:
        explanation += f"rose {price_change:.1f}% to ${final_price:.2f}. "
    elif price_change < 0:
        explanation += f"fell {abs(price_change):.1f}% to ${final_price:.2f}. "
    else:
        explanation += f"remained flat at ${final_price:.2f}. "
    
    if initial_cost < 0:
        explanation += f"You paid ${abs(initial_cost):.2f} to enter this strategy. "
    else:
        explanation += f"You received ${initial_cost:.2f} to enter this strategy. "
    
    if final_pnl > 0:
        explanation += f"Your final profit was ${final_pnl:.2f}. "
    elif final_pnl < 0:
        explanation += f"Your final loss was ${abs(final_pnl):.2f}. "
    else:
        explanation += f"You broke even. "
    
    if max_profit > final_pnl:
        explanation += f"At one point, your profit peaked at ${max_profit:.2f}. "
    
    if max_loss < 0:
        explanation += f"Your maximum drawdown was ${abs(max_loss):.2f}. "
    
    if final_pnl > 0:
        explanation += "This strategy would have been profitable. ✓"
    else:
        explanation += "This strategy would have resulted in a loss. ✗"
    
    return explanation


def get_available_date_range(ticker: str) -> Dict:
    """
    Get the available date range for historical data.
    
    Returns:
    --------
    dict : Contains earliest and latest available dates
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max")
        
        if hist.empty:
            return {
                'ticker': ticker,
                'available': False,
                'error': 'No historical data available'
            }
        
        return {
            'ticker': ticker,
            'available': True,
            'earliest_date': hist.index[0].strftime('%Y-%m-%d'),
            'latest_date': hist.index[-1].strftime('%Y-%m-%d'),
            'total_days': len(hist)
        }
    
    except Exception as e:
        return {
            'ticker': ticker,
            'available': False,
            'error': str(e)
        }
