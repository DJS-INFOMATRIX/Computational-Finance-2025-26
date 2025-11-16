"""
Stock Data Generator and Technical Indicators
Generates mock OHLCV data and calculates technical indicators
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

class StockDataGenerator:
    """Generate realistic mock stock data with OHLCV format"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def generate_stock_data(self, 
                          days: int = 1000, 
                          initial_price: float = 100.0,
                          volatility: float = 0.02) -> pd.DataFrame:
        """
        Generate mock OHLCV stock data using geometric Brownian motion
        
        Args:
            days: Number of days to generate
            initial_price: Starting stock price
            volatility: Daily volatility (standard deviation of returns)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate random returns using geometric Brownian motion
        returns = np.random.normal(0.001, volatility, days)  # Small positive drift
        
        # Calculate cumulative prices
        prices = [initial_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[1:])  # Remove initial price
        
        # Generate OHLC data
        data = []
        for i, close in enumerate(prices):
            # Generate intraday volatility
            daily_vol = np.random.uniform(0.005, 0.03)
            
            # Open price (previous close + gap)
            if i == 0:
                open_price = initial_price
            else:
                gap = np.random.normal(0, 0.005)
                open_price = prices[i-1] * (1 + gap)
            
            # High and Low based on close and open
            high_low_range = abs(close - open_price) + close * daily_vol
            high = max(open_price, close) + np.random.uniform(0, high_low_range * 0.5)
            low = min(open_price, close) - np.random.uniform(0, high_low_range * 0.5)
            
            # Volume (correlated with price movement)
            price_change = abs(close - open_price) / open_price
            base_volume = np.random.uniform(1000000, 5000000)
            volume = base_volume * (1 + price_change * 10)
            
            data.append({
                'Date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': int(volume)
            })
        
        return pd.DataFrame(data)

class TechnicalIndicators:
    """Calculate various technical indicators"""
    
    @staticmethod
    def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def exponential_moving_average(prices: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(prices: pd.Series, 
             fast_period: int = 12, 
             slow_period: int = 26, 
             signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line
        """
        ema_fast = TechnicalIndicators.exponential_moving_average(prices, fast_period)
        ema_slow = TechnicalIndicators.exponential_moving_average(prices, slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.exponential_moving_average(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        }
    
    @staticmethod
    def volatility(prices: pd.Series, window: int = 10) -> pd.Series:
        """Calculate rolling volatility (standard deviation of returns)"""
        returns = prices.pct_change()
        return returns.rolling(window=window).std()
    
    @staticmethod
    def price_volume_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Price Volume Trend (PVT)
        PVT = Previous PVT + (Volume * (Close - Previous Close) / Previous Close)
        """
        price_change_ratio = close.pct_change()
        pvt = (volume * price_change_ratio).cumsum()
        return pvt
    
    @staticmethod
    def z_score_normalized_price(prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate z-score normalized price over rolling window"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        z_score = (prices - rolling_mean) / rolling_std
        return z_score

def create_features_and_labels(df: pd.DataFrame, lookback_days: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create features and labels for stock prediction
    
    Args:
        df: DataFrame with OHLCV data
        lookback_days: Number of days to look back for returns
        
    Returns:
        Tuple of (features_df, labels_series)
    """
    features_df = df.copy()
    
    # Calculate daily returns
    features_df['Returns'] = features_df['Close'].pct_change()
    
    # Past daily returns for last N days
    for i in range(1, lookback_days + 1):
        features_df[f'Returns_lag_{i}'] = features_df['Returns'].shift(i)
    
    # Technical indicators
    features_df['SMA_5'] = TechnicalIndicators.simple_moving_average(features_df['Close'], 5)
    features_df['SMA_10'] = TechnicalIndicators.simple_moving_average(features_df['Close'], 10)
    features_df['RSI'] = TechnicalIndicators.rsi(features_df['Close'], 14)
    
    # MACD indicators
    macd_data = TechnicalIndicators.macd(features_df['Close'])
    features_df['MACD'] = macd_data['MACD']
    features_df['MACD_Signal'] = macd_data['MACD_Signal']
    features_df['MACD_Histogram'] = macd_data['MACD_Histogram']
    
    # Bonus features
    features_df['Volatility'] = TechnicalIndicators.volatility(features_df['Close'], 10)
    features_df['PVT'] = TechnicalIndicators.price_volume_trend(features_df['Close'], features_df['Volume'])
    features_df['Z_Score_Price'] = TechnicalIndicators.z_score_normalized_price(features_df['Close'], 20)
    
    # Normalize PVT (it can get very large)
    features_df['PVT'] = (features_df['PVT'] - features_df['PVT'].mean()) / features_df['PVT'].std()
    
    # Create binary labels (1 if next day's close > today's close, 0 otherwise)
    # Shift by -1 to avoid data leakage
    features_df['Next_Close'] = features_df['Close'].shift(-1)
    labels = (features_df['Next_Close'] > features_df['Close']).astype(int)
    
    # Select feature columns (exclude OHLCV and helper columns)
    feature_columns = [col for col in features_df.columns 
                      if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                   'Next_Close', 'Returns']]
    
    # Remove rows with NaN values
    features_df = features_df[feature_columns].dropna()
    labels = labels[features_df.index]
    
    return features_df, labels

if __name__ == "__main__":
    # Test the data generator
    generator = StockDataGenerator()
    stock_data = generator.generate_stock_data(days=1000)
    
    print("Generated stock data shape:", stock_data.shape)
    print("\nFirst 5 rows:")
    print(stock_data.head())
    
    # Create features and labels
    features, labels = create_features_and_labels(stock_data)
    print(f"\nFeatures shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"Label distribution: {labels.value_counts()}")
