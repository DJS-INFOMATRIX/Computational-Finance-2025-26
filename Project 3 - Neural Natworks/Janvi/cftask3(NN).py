"""
Stock Price Movement Prediction for Microsoft (MSFT)
Compares Feedforward Neural Network (FFNN) vs Logistic Regression
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

def compute_rsi(data, window=14):
    """Compute Relative Strength Index (RSI) - Optimized"""
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window, min_periods=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data, fast=12, slow=26, signal=9):
    """Compute MACD, Signal, and Histogram - Optimized"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def load_and_prepare_data():
    """Load 5 years of stock data and compute technical indicators - Optimized"""
    print("=" * 80)
    print("STOCK PRICE PREDICTION: MSFT")
    print("=" * 80)
    print("\n[Step 1] Loading 5 years of historical data...")
    
    # Download 5 years of data
    ticker = "MSFT"
    data = yf.download(ticker, period="5y", progress=False)
    
    print(f"✓ Downloaded {len(data)} rows of data")
    
    # Clean missing values and extract Close price
    data = data.dropna()
    close_prices = data['Close']
    print(f"✓ After cleaning: {len(close_prices)} rows")
    
    # Compute technical indicators efficiently
    print("\n[Step 2] Computing technical indicators...")
    
    # Create DataFrame with vectorized operations
    df = pd.DataFrame(index=close_prices.index)
    df['Close'] = close_prices
    
    # CRITICAL: Add 10 days of historical daily returns as separate features
    daily_returns = close_prices.pct_change()
    for i in range(1, 11):
        df[f'Return_T-{i}'] = daily_returns.shift(i)
    
    # Moving Averages
    df['MA_5'] = close_prices.rolling(window=5, min_periods=5).mean()
    df['MA_10'] = close_prices.rolling(window=10, min_periods=10).mean()
    
    # RSI-14
    df['RSI_14'] = compute_rsi(close_prices, window=14)
    
    # MACD computation
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = compute_macd(close_prices)
    
    # Create binary target: 1 if next day's close > today's close, else 0
    df['Target'] = (close_prices.shift(-1) > close_prices).astype(np.int8)
    
    # Remove rows with NaN values and reset index
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"✓ Technical indicators computed")
    print(f"✓ Final dataset size: {len(df)} rows")
    
    return df

def split_data(df):
    """Chronological 70/15/15 time-series split - Optimized"""
    print("\n[Step 3] Splitting data (70% train, 15% validation, 15% test)...")
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    # Features: 10 days of historical returns + MA + RSI + MACD indicators (exclude Close and Target)
    feature_cols = [
        'Return_T-1', 'Return_T-2', 'Return_T-3', 'Return_T-4', 'Return_T-5',
        'Return_T-6', 'Return_T-7', 'Return_T-8', 'Return_T-9', 'Return_T-10',
        'MA_5', 'MA_10', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram'
    ]
    X = df[feature_cols].values
    y = df['Target'].values
    
    # Chronological split using array slicing (most efficient)
    splits = {
        'train': (X[:train_end], y[:train_end]),
        'val': (X[train_end:val_end], y[train_end:val_end]),
        'test': (X[val_end:], y[val_end:])
    }
    
    print(f"✓ Train set: {len(splits['train'][0])} samples")
    print(f"✓ Validation set: {len(splits['val'][0])} samples")
    print(f"✓ Test set: {len(splits['test'][0])} samples")
    
    return splits['train'][0], splits['val'][0], splits['test'][0], \
           splits['train'][1], splits['val'][1], splits['test'][1], feature_cols

def scale_features(X_train, X_val, X_test):
    """Apply MinMaxScaler fitted only on training set - Optimized"""
    print("\n[Step 4] Scaling features (MinMaxScaler fitted on train set only)...")
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Features scaled successfully")
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def build_ffnn(input_dim):
    """Build Feedforward Neural Network with 2 hidden layers - Optimized"""
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim, kernel_initializer='he_uniform'),
        Dense(16, activation='relu', kernel_initializer='he_uniform'),
        Dense(1, activation='sigmoid')
    ], name='FFNN_Stock_Predictor')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_ffnn(model, X_train, y_train, X_val, y_val):
    """Train FFNN with early stopping - Optimized"""
    print("\n[Step 5] Training Feedforward Neural Network...")
    print("Architecture: Input → Dense(32,ReLU) → Dense(16,ReLU) → Dense(1,Sigmoid)")
    print("Optimizer: Adam | Loss: Binary Cross-Entropy | Early Stopping: Enabled")
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0,
        mode='min'
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
        shuffle=True  # Shuffle within training set for better convergence
    )
    
    print(f"✓ Training completed ({len(history.history['loss'])} epochs)")
    print("✓ Best weights restored")
    
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics - Optimized"""
    if model_name == "FFNN":
        y_pred_proba = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_pred_proba > 0.5).astype(np.int8)
    else:  # Logistic Regression
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression as baseline - Optimized"""
    print("\n[Step 6] Training Logistic Regression (Baseline)...")
    
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs',
        n_jobs=-1  # Use all CPU cores
    )
    lr_model.fit(X_train, y_train)
    
    print("✓ Logistic Regression trained")
    
    return lr_model

def predict_next_day(model, X_test_scaled):
    """Generate prediction for the most recent day - Optimized"""
    # Use the last available feature row
    probability = model.predict(X_test_scaled[-1:], verbose=0)[0, 0]
    prediction_label = int(probability > 0.5)
    
    return probability, prediction_label

def main():
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Validate dataset size
    if len(df) < 200:
        print(f"\n❌ ERROR: Dataset has only {len(df)} rows. Minimum 200 rows required.")
        return
    
    print(f"✓ Dataset validation passed ({len(df)} rows ≥ 200)")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = split_data(df)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)
    
    # Build and train FFNN
    ffnn_model = build_ffnn(input_dim=len(feature_cols))
    ffnn_model = train_ffnn(ffnn_model, X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    
    # Evaluate both models on test set
    print("\n[Step 7] Evaluating models on test set...")
    ffnn_metrics = evaluate_model(ffnn_model, X_test_scaled, y_test, "FFNN")
    lr_metrics = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    
    print("✓ Evaluation completed")
    
    # Create comparison DataFrame
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    
    comparison_df = pd.DataFrame({
        'FFNN': [f"{v:.4f}" for v in ffnn_metrics.values()],
        'Logistic Regression': [f"{v:.4f}" for v in lr_metrics.values()]
    }, index=list(ffnn_metrics.keys()))
    
    print("\n" + comparison_df.to_string())
    
    # Determine better model
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    roc_diff = abs(ffnn_metrics['ROC-AUC'] - lr_metrics['ROC-AUC']) * 100
    better_model = "FFNN" if ffnn_metrics['ROC-AUC'] > lr_metrics['ROC-AUC'] else "Logistic Regression"
    
    print(f"\n✓ Better Model: {better_model}")
    print(f"  The {better_model} model performs better with a ROC-AUC advantage of {roc_diff:.2f}%")
    
    acc_diff = abs(ffnn_metrics['Accuracy'] - lr_metrics['Accuracy']) * 100
    if acc_diff > 0.01:
        acc_winner = "FFNN" if ffnn_metrics['Accuracy'] > lr_metrics['Accuracy'] else "Logistic Regression"
        print(f"  {acc_winner} shows {acc_diff:.2f}% higher accuracy on the test set.")
    else:
        print(f"  Both models achieve similar accuracy on the test set.")
    
    # Generate prediction for the most recent day
    print("\n" + "=" * 80)
    print("NEXT-DAY PREDICTION (MOST RECENT DATA)")
    print("=" * 80)
    
    probability, prediction_label = predict_next_day(ffnn_model, X_test_scaled)
    
    # Display prediction with emoji
    if prediction_label == 1:
        direction = "UP"
        emoji = "📈"
        print(f"\n{emoji} The FFNN model predicts the stock will go {direction} tomorrow")
    else:
        direction = "DOWN"
        emoji = "📉"
        print(f"\n{emoji} The FFNN model predicts the stock will go {direction} tomorrow")
    
    # Display probability
    percentage = probability * 100
    if prediction_label == 1:
        print(f"   Prediction probability = {probability:.4f} ({percentage:.2f}% chance of UP)")
    else:
        print(f"   Prediction probability = {1-probability:.4f} ({(100-percentage):.2f}% chance of DOWN)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
