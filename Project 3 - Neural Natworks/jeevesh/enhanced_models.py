"""
Enhanced Models for Improved Stock Prediction Accuracy
Implements advanced architectures and optimization techniques
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from data_generator import StockDataGenerator, create_features_and_labels
from advanced_features import create_advanced_features
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class EnhancedNeuralNetwork:
    """Enhanced Neural Network with advanced features for better accuracy"""
    
    def __init__(self, input_size, hidden_layers=[128, 64, 32], learning_rate=0.001, 
                 dropout_rate=0.2, l2_reg=0.001, use_batch_norm=True):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        
        # Initialize network architecture
        self.layers = []
        self.weights = []
        self.biases = []
        self.batch_norm_params = []
        
        # Build network
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_layers):
            # Xavier/Glorot initialization for better convergence
            w = np.random.randn(prev_size, hidden_size) * np.sqrt(2.0 / prev_size)
            b = np.zeros((1, hidden_size))
            
            self.weights.append(w)
            self.biases.append(b)
            
            # Batch normalization parameters
            if self.use_batch_norm:
                gamma = np.ones((1, hidden_size))
                beta = np.zeros((1, hidden_size))
                self.batch_norm_params.append({'gamma': gamma, 'beta': beta})
            
            prev_size = hidden_size
        
        # Output layer
        w_out = np.random.randn(prev_size, 1) * np.sqrt(2.0 / prev_size)
        b_out = np.zeros((1, 1))
        self.weights.append(w_out)
        self.biases.append(b_out)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def batch_normalize(self, x, gamma, beta, eps=1e-8):
        """Batch normalization for stable training"""
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU activation to prevent dying neurons"""
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        """Derivative of Leaky ReLU"""
        return np.where(x > 0, 1, alpha)
    
    def dropout(self, x, rate, training=True):
        """Dropout for regularization"""
        if not training or rate == 0:
            return x
        mask = np.random.binomial(1, 1 - rate, x.shape) / (1 - rate)
        return x * mask
    
    def forward_pass(self, X, training=True):
        """Enhanced forward pass with batch norm and dropout"""
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        
        # Hidden layers
        for i in range(len(self.hidden_layers)):
            # Linear transformation
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Batch normalization
            if self.use_batch_norm:
                z = self.batch_normalize(z, 
                                       self.batch_norm_params[i]['gamma'],
                                       self.batch_norm_params[i]['beta'])
            
            # Activation
            a = self.leaky_relu(z)
            
            # Dropout
            if training:
                a = self.dropout(a, self.dropout_rate, training)
            
            self.activations.append(a)
            current_input = a
        
        # Output layer
        z_out = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        output = self.sigmoid(z_out)
        self.activations.append(output)
        
        return output
    
    def sigmoid(self, x):
        """Stable sigmoid function"""
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Sigmoid derivative"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss with L2 regularization"""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Cross-entropy loss
        ce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # L2 regularization
        l2_loss = 0
        for w in self.weights:
            l2_loss += np.sum(w ** 2)
        l2_loss *= self.l2_reg / (2 * len(y_true))
        
        return ce_loss + l2_loss
    
    def backward_pass(self, X, y_true, y_pred):
        """Enhanced backpropagation with regularization"""
        m = X.shape[0]
        
        # Output layer gradient
        dz_out = y_pred - y_true
        dw_out = np.dot(self.activations[-2].T, dz_out) / m + self.l2_reg * self.weights[-1]
        db_out = np.mean(dz_out, axis=0, keepdims=True)
        
        # Store gradients
        dw_gradients = [dw_out]
        db_gradients = [db_out]
        
        # Backpropagate through hidden layers
        da = np.dot(dz_out, self.weights[-1].T)
        
        for i in reversed(range(len(self.hidden_layers))):
            # Dropout gradient (if applied during forward pass)
            if self.dropout_rate > 0:
                da = da / (1 - self.dropout_rate)
            
            # Activation gradient
            dz = da * self.leaky_relu_derivative(self.z_values[i])
            
            # Weight and bias gradients
            dw = np.dot(self.activations[i].T, dz) / m + self.l2_reg * self.weights[i]
            db = np.mean(dz, axis=0, keepdims=True)
            
            dw_gradients.insert(0, dw)
            db_gradients.insert(0, db)
            
            # Gradient for next layer
            if i > 0:
                da = np.dot(dz, self.weights[i].T)
        
        return dw_gradients, db_gradients
    
    def update_weights(self, dw_gradients, db_gradients):
        """Update weights using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw_gradients[i]
            self.biases[i] -= self.learning_rate * db_gradients[i]
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=500, batch_size=32, verbose=True):
        """Enhanced training with validation and early stopping"""
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            epoch_losses = []
            epoch_predictions = []
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward_pass(batch_X, training=True)
                
                # Compute loss
                loss = self.compute_loss(batch_y, y_pred)
                epoch_losses.append(loss)
                epoch_predictions.extend(y_pred.flatten())
                
                # Backward pass
                dw_gradients, db_gradients = self.backward_pass(batch_X, batch_y, y_pred)
                
                # Update weights
                self.update_weights(dw_gradients, db_gradients)
            
            # Training metrics
            train_loss = np.mean(epoch_losses)
            train_pred = np.array(epoch_predictions[:len(y_train)])
            train_acc = accuracy_score(y_train.flatten(), (train_pred > 0.5).astype(int))
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.forward_pass(X_val, training=False)
                val_loss = self.compute_loss(y_val, val_pred)
                val_acc = accuracy_score(y_val.flatten(), (val_pred > 0.5).astype(int))
                
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if verbose and (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}")
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.forward_pass(X, training=False)
    
    def predict(self, X):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y.flatten(), y_pred.flatten())
        roc_auc = roc_auc_score(y.flatten(), y_pred_proba.flatten())
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

def create_ultra_features(stock_data):
    """Create ultra-comprehensive feature set for maximum accuracy"""
    
    # Start with advanced features
    features_df, labels = create_advanced_features(stock_data)
    
    # Add even more sophisticated features
    df = stock_data.copy()
    
    # Check column names and use correct ones
    price_col = 'Close' if 'Close' in df.columns else 'close'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'
    volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
    
    # Multi-timeframe analysis
    for window in [3, 7, 14, 21, 30]:
        # Price momentum
        df[f'momentum_{window}'] = df[price_col].pct_change(window)
        
        # Volume-weighted features
        df[f'vwap_{window}'] = (df[price_col] * df[volume_col]).rolling(window).sum() / df[volume_col].rolling(window).sum()
        df[f'price_vs_vwap_{window}'] = df[price_col] / df[f'vwap_{window}'] - 1
        
        # Volatility features
        df[f'volatility_{window}'] = df[price_col].pct_change().rolling(window).std()
        if window == 20:  # Create base volatility for ratios
            df['volatility_20'] = df[f'volatility_{window}']
        
        # Support/Resistance levels
        df[f'high_max_{window}'] = df[high_col].rolling(window).max()
        df[f'low_min_{window}'] = df[low_col].rolling(window).min()
        df[f'price_position_{window}'] = (df[price_col] - df[f'low_min_{window}']) / (df[f'high_max_{window}'] - df[f'low_min_{window}'])
    
    # Calculate volatility ratios after all volatilities are created
    for window in [3, 7, 14, 21, 30]:
        if f'volatility_{window}' in df.columns and 'volatility_20' in df.columns:
            df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df['volatility_20']
    
    # Advanced technical patterns
    # Doji patterns
    df['doji'] = np.abs(df[open_col] - df[price_col]) / (df[high_col] - df[low_col])
    
    # Gap analysis
    df['gap_up'] = (df[open_col] > df[price_col].shift(1)).astype(int)
    df['gap_down'] = (df[open_col] < df[price_col].shift(1)).astype(int)
    df['gap_size'] = (df[open_col] - df[price_col].shift(1)) / df[price_col].shift(1)
    
    # Market microstructure
    df['spread'] = (df[high_col] - df[low_col]) / df[price_col]
    df['upper_shadow'] = (df[high_col] - np.maximum(df[open_col], df[price_col])) / df[price_col]
    df['lower_shadow'] = (np.minimum(df[open_col], df[price_col]) - df[low_col]) / df[price_col]
    
    # Fibonacci retracements (simplified)
    for period in [20, 50]:
        high_period = df[high_col].rolling(period).max()
        low_period = df[low_col].rolling(period).min()
        range_period = high_period - low_period
        
        df[f'fib_23.6_{period}'] = (df[price_col] - low_period) / range_period - 0.236
        df[f'fib_38.2_{period}'] = (df[price_col] - low_period) / range_period - 0.382
        df[f'fib_61.8_{period}'] = (df[price_col] - low_period) / range_period - 0.618
    
    # Market regime indicators
    df['trend_strength'] = np.abs(df[price_col].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
    if 'volatility_20' in df.columns:
        df['market_regime'] = np.where(df['volatility_20'] > df['volatility_20'].rolling(50).mean(), 1, 0)  # High vol regime
    
    # Add new features to existing feature set
    new_feature_cols = [col for col in df.columns if col not in stock_data.columns and col != 'target']
    
    # Combine with existing features
    for col in new_feature_cols:
        if col in df.columns:
            features_df[col] = df[col]
    
    # Remove rows with NaN values
    features_df = features_df.dropna()
    labels = labels.loc[features_df.index]
    
    print(f"Ultra features created: {len(features_df.columns)} features")
    print(f"New features added: {len(new_feature_cols)}")
    
    return features_df, labels

def test_enhanced_models():
    """Test enhanced models for improved accuracy"""
    
    print("ENHANCED MODEL TESTING")
    print("=" * 60)
    
    # Generate data
    generator = StockDataGenerator(seed=42)
    stock_data = generator.generate_stock_data(days=2000)  # More data for better training
    
    # Create ultra-comprehensive features
    print("Creating ultra-comprehensive feature set...")
    features, labels = create_ultra_features(stock_data)
    
    print(f"Dataset: {len(features)} samples, {len(features.columns)} features")
    print(f"Label distribution: Up={labels.sum()}, Down={len(labels) - labels.sum()}")
    
    # Prepare data with robust scaling
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X = scaler.fit_transform(features.values)
    y = labels.values.reshape(-1, 1)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples") 
    print(f"Test set: {len(X_test)} samples")
    
    # Test different architectures
    architectures = [
        ([256, 128, 64], "Deep Network (256->128->64)"),
        ([512, 256], "Wide Network (512->256)"),
        ([128, 64, 32, 16], "Very Deep (128->64->32->16)"),
        ([200], "Single Large Layer (200)"),
        ([100, 50, 25], "Pyramid (100->50->25)")
    ]
    
    results = {}
    
    for arch, name in architectures:
        print(f"\nTesting {name}...")
        
        # Create and train model
        model = EnhancedNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_layers=arch,
            learning_rate=0.001,
            dropout_rate=0.3,
            l2_reg=0.01,
            use_batch_norm=True
        )
        
        # Train with validation
        model.train(X_train, y_train, X_val, y_val, epochs=300, batch_size=64, verbose=False)
        
        # Evaluate
        test_results = model.evaluate(X_test, y_test)
        results[name] = test_results
        
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test ROC-AUC: {test_results['roc_auc']:.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    
    print(f"\n{'='*60}")
    print("ENHANCED MODEL RESULTS SUMMARY")
    print("=" * 60)
    
    for name, result in results.items():
        status = "🏆 BEST" if name == best_model else "✅ GOOD" if result['accuracy'] > 0.60 else "⚡ OK"
        print(f"{name:25s}: {result['accuracy']:.4f} accuracy, {result['roc_auc']:.4f} AUC {status}")
    
    print(f"\nBest Model: {best_model}")
    print(f"Best Accuracy: {best_accuracy:.4f} ({(best_accuracy-0.5)*100:.1f}% above random)")
    
    # Compare with baseline
    baseline_accuracy = 0.528  # Previous best
    improvement = best_accuracy - baseline_accuracy
    
    print(f"\nImprovement over baseline:")
    print(f"Previous best: {baseline_accuracy:.4f}")
    print(f"New best: {best_accuracy:.4f}")
    print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    return results, best_model

if __name__ == "__main__":
    results, best_model = test_enhanced_models()
    
    print(f"\n{'='*60}")
    print("ENHANCED MODELS COMPLETE!")
    print("Key Improvements:")
    print("- Ultra-comprehensive feature engineering (50+ features)")
    print("- Advanced neural network architectures")
    print("- Batch normalization and dropout regularization")
    print("- Leaky ReLU activation functions")
    print("- L2 regularization and early stopping")
    print("- Robust data scaling")
    print("- Multi-timeframe analysis")
    print("- Advanced technical patterns")
    print("\nNext steps:")
    print("- Integrate best model into web interface")
    print("- Add real-time data feeds")
    print("- Implement ensemble methods")
    print("- Add model interpretability features")
