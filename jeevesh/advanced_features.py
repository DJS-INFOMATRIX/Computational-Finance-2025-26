"""
Advanced Feature Engineering for Stock Prediction
Add more sophisticated technical indicators and features
"""

import numpy as np
import pandas as pd
from data_generator import StockDataGenerator, TechnicalIndicators

class AdvancedTechnicalIndicators:
    """Advanced technical indicators for better prediction"""
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        # Calculate position within bands
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        bb_width = (upper_band - lower_band) / sma
        
        return {
            'BB_Upper': upper_band,
            'BB_Lower': lower_band,
            'BB_Position': bb_position,
            'BB_Width': bb_width
        }
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'Stoch_K': k_percent,
            'Stoch_D': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    @staticmethod
    def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20):
        """Calculate Commodity Channel Index (CCI)"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    @staticmethod
    def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Calculate Average True Range (ATR)"""
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def momentum_indicators(close: pd.Series):
        """Calculate various momentum indicators"""
        return {
            'Momentum_5': close - close.shift(5),
            'Momentum_10': close - close.shift(10),
            'Rate_of_Change': (close / close.shift(10) - 1) * 100,
            'Price_Oscillator': ((close.rolling(12).mean() - close.rolling(26).mean()) / 
                               close.rolling(26).mean()) * 100
        }

def create_advanced_features(df: pd.DataFrame, lookback_days: int = 10) -> pd.DataFrame:
    """Create advanced feature set with more technical indicators"""
    
    features_df = df.copy()
    
    # Basic returns
    features_df['Returns'] = features_df['Close'].pct_change()
    
    # Past daily returns
    for i in range(1, lookback_days + 1):
        features_df[f'Returns_lag_{i}'] = features_df['Returns'].shift(i)
    
    # Original technical indicators
    features_df['SMA_5'] = TechnicalIndicators.simple_moving_average(features_df['Close'], 5)
    features_df['SMA_10'] = TechnicalIndicators.simple_moving_average(features_df['Close'], 10)
    features_df['SMA_20'] = TechnicalIndicators.simple_moving_average(features_df['Close'], 20)
    features_df['RSI'] = TechnicalIndicators.rsi(features_df['Close'], 14)
    
    # MACD
    macd_data = TechnicalIndicators.macd(features_df['Close'])
    features_df['MACD'] = macd_data['MACD']
    features_df['MACD_Signal'] = macd_data['MACD_Signal']
    features_df['MACD_Histogram'] = macd_data['MACD_Histogram']
    
    # Advanced indicators
    bb_data = AdvancedTechnicalIndicators.bollinger_bands(features_df['Close'])
    features_df['BB_Position'] = bb_data['BB_Position']
    features_df['BB_Width'] = bb_data['BB_Width']
    
    stoch_data = AdvancedTechnicalIndicators.stochastic_oscillator(
        features_df['High'], features_df['Low'], features_df['Close']
    )
    features_df['Stoch_K'] = stoch_data['Stoch_K']
    features_df['Stoch_D'] = stoch_data['Stoch_D']
    
    features_df['Williams_R'] = AdvancedTechnicalIndicators.williams_r(
        features_df['High'], features_df['Low'], features_df['Close']
    )
    
    features_df['CCI'] = AdvancedTechnicalIndicators.commodity_channel_index(
        features_df['High'], features_df['Low'], features_df['Close']
    )
    
    features_df['ATR'] = AdvancedTechnicalIndicators.average_true_range(
        features_df['High'], features_df['Low'], features_df['Close']
    )
    
    # Momentum indicators
    momentum_data = AdvancedTechnicalIndicators.momentum_indicators(features_df['Close'])
    for key, value in momentum_data.items():
        features_df[key] = value
    
    # Volume indicators
    features_df['Volume_SMA'] = features_df['Volume'].rolling(10).mean()
    features_df['Volume_Ratio'] = features_df['Volume'] / features_df['Volume_SMA']
    
    # Price patterns
    features_df['High_Low_Ratio'] = features_df['High'] / features_df['Low']
    features_df['Open_Close_Ratio'] = features_df['Open'] / features_df['Close']
    
    # Volatility measures
    features_df['Volatility_5'] = TechnicalIndicators.volatility(features_df['Close'], 5)
    features_df['Volatility_10'] = TechnicalIndicators.volatility(features_df['Close'], 10)
    features_df['Volatility_20'] = TechnicalIndicators.volatility(features_df['Close'], 20)
    
    # Price position indicators
    features_df['Price_vs_SMA5'] = (features_df['Close'] - features_df['SMA_5']) / features_df['SMA_5']
    features_df['Price_vs_SMA20'] = (features_df['Close'] - features_df['SMA_20']) / features_df['SMA_20']
    
    # Normalized price
    features_df['Z_Score_Price'] = TechnicalIndicators.z_score_normalized_price(features_df['Close'], 20)
    
    # Create binary labels (shifted to prevent data leakage)
    features_df['Next_Close'] = features_df['Close'].shift(-1)
    labels = (features_df['Next_Close'] > features_df['Close']).astype(int)
    
    # Select feature columns
    feature_columns = [col for col in features_df.columns 
                      if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                   'Next_Close', 'Returns']]
    
    # Remove rows with NaN values
    features_clean = features_df[feature_columns].dropna()
    labels_clean = labels[features_clean.index]
    
    print(f"Advanced features created: {len(feature_columns)} features")
    print("New features added:")
    new_features = [col for col in feature_columns if col not in [
        'Returns_lag_1', 'Returns_lag_2', 'Returns_lag_3', 'Returns_lag_4', 'Returns_lag_5',
        'Returns_lag_6', 'Returns_lag_7', 'Returns_lag_8', 'Returns_lag_9', 'Returns_lag_10',
        'SMA_5', 'SMA_10', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Z_Score_Price'
    ]]
    for i, feature in enumerate(new_features, 1):
        print(f"{i:2d}. {feature}")
    
    return features_clean, labels_clean

def test_advanced_features():
    """Test the advanced feature set with neural network"""
    from neural_network import FeedforwardNeuralNetwork, prepare_data
    
    print("Testing Advanced Feature Engineering...")
    print("=" * 50)
    
    # Generate data
    generator = StockDataGenerator(seed=42)
    stock_data = generator.generate_stock_data(days=1000)
    
    # Create advanced features
    features, labels = create_advanced_features(stock_data)
    
    print(f"\nDataset info:")
    print(f"Total features: {features.shape[1]}")
    print(f"Total samples: {features.shape[0]}")
    print(f"Label distribution: Up={labels.sum()}, Down={(1-labels).sum()}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(features, labels)
    
    # Test with neural network
    print(f"\nTraining Neural Network with {features.shape[1]} features...")
    nn = FeedforwardNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=[128, 64],
        learning_rate=0.001
    )
    
    history = nn.train(X_train, y_train, epochs=300, verbose=False)
    results = nn.evaluate(X_test, y_test)
    
    print(f"\nAdvanced Model Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    
    # Compare with basic features
    from data_generator import create_features_and_labels
    basic_features, basic_labels = create_features_and_labels(stock_data)
    X_train_basic, X_test_basic, y_train_basic, y_test_basic, scaler_basic = prepare_data(basic_features, basic_labels)
    
    nn_basic = FeedforwardNeuralNetwork(
        input_size=X_train_basic.shape[1],
        hidden_sizes=[128, 64],
        learning_rate=0.001
    )
    
    nn_basic.train(X_train_basic, y_train_basic, epochs=300, verbose=False)
    results_basic = nn_basic.evaluate(X_test_basic, y_test_basic)
    
    print(f"\nComparison with Basic Features:")
    print(f"Basic Features ({basic_features.shape[1]}): {results_basic['accuracy']:.4f} accuracy")
    print(f"Advanced Features ({features.shape[1]}): {results['accuracy']:.4f} accuracy")
    
    improvement = results['accuracy'] - results_basic['accuracy']
    print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    return features, labels, results

if __name__ == "__main__":
    features, labels, results = test_advanced_features()
    
    print(f"\n{'='*50}")
    print("ADVANCED FEATURE ENGINEERING COMPLETE!")
    print("Try experimenting with:")
    print("- Different indicator periods")
    print("- Feature selection techniques")
    print("- Feature scaling methods")
    print("- Ensemble models")
