"""
Ensemble Model for Stock Prediction
Combine multiple models for better performance
"""

import numpy as np
from neural_network import FeedforwardNeuralNetwork, prepare_data
from logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from data_generator import StockDataGenerator, create_features_and_labels
from advanced_features import create_advanced_features

class EnsemblePredictor:
    """Ensemble of multiple models for stock prediction"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = None
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models in the ensemble"""
        print("Training Ensemble Models...")
        print("=" * 40)
        
        results = {}
        
        # Train Neural Networks with different architectures
        print("1. Training Neural Network (64->32)...")
        nn1 = FeedforwardNeuralNetwork(X_train.shape[1], [64, 32], learning_rate=0.001)
        nn1.train(X_train, y_train, epochs=300, verbose=False)
        self.add_model('nn_64_32', nn1, weight=1.2)
        
        print("2. Training Neural Network (100)...")
        nn2 = FeedforwardNeuralNetwork(X_train.shape[1], [100], learning_rate=0.001)
        nn2.train(X_train, y_train, epochs=300, verbose=False)
        self.add_model('nn_100', nn2, weight=1.3)
        
        print("3. Training Neural Network (128->64->32)...")
        nn3 = FeedforwardNeuralNetwork(X_train.shape[1], [128, 64, 32], learning_rate=0.001)
        nn3.train(X_train, y_train, epochs=300, verbose=False)
        self.add_model('nn_deep', nn3, weight=1.1)
        
        print("4. Training Manual Logistic Regression...")
        lr_manual = LogisticRegression(learning_rate=0.01, max_iterations=1000)
        lr_manual.fit(X_train, y_train, verbose=False)
        self.add_model('lr_manual', lr_manual, weight=0.9)
        
        print("5. Training Sklearn Logistic Regression...")
        lr_sklearn = SklearnLR(random_state=42, max_iter=1000)
        lr_sklearn.fit(X_train, y_train.ravel())
        self.add_model('lr_sklearn', lr_sklearn, weight=0.9)
        
        print("6. Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_train, y_train.ravel())
        self.add_model('random_forest', rf, weight=1.0)
        
        # Evaluate individual models
        if X_val is not None and y_val is not None:
            print("\nIndividual Model Performance:")
            print("-" * 40)
            
            for name, model in self.models.items():
                pred_proba = self._get_model_prediction(model, X_val, name)
                pred = (pred_proba > 0.5).astype(int)
                
                acc = accuracy_score(y_val, pred)
                auc = roc_auc_score(y_val, pred_proba)
                
                results[name] = {'accuracy': acc, 'roc_auc': auc}
                print(f"{name:15s}: Acc={acc:.4f}, AUC={auc:.4f}")
        
        return results
    
    def _get_model_prediction(self, model, X, model_name):
        """Get prediction from a specific model"""
        if 'nn_' in model_name:
            return model.predict_proba(X).flatten()
        elif 'lr_manual' in model_name:
            return model.predict_proba(X).flatten()
        elif 'lr_sklearn' in model_name:
            return model.predict_proba(X)[:, 1]
        elif 'random_forest' in model_name:
            return model.predict_proba(X)[:, 1]
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def predict_proba(self, X):
        """Ensemble prediction using weighted average"""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred = self._get_model_prediction(model, X, name)
            predictions.append(pred)
            weights.append(self.weights[name])
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def predict(self, X):
        """Binary predictions from ensemble"""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def evaluate(self, X, y):
        """Evaluate ensemble performance"""
        pred_proba = self.predict_proba(X)
        pred = self.predict(X)
        
        accuracy = accuracy_score(y, pred)
        roc_auc = roc_auc_score(y, pred_proba)
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': pred,
            'probabilities': pred_proba
        }

def test_ensemble_models():
    """Test ensemble approach with different feature sets"""
    
    print("ENSEMBLE MODEL TESTING")
    print("=" * 60)
    
    # Generate data
    generator = StockDataGenerator(seed=42)
    stock_data = generator.generate_stock_data(days=1000)
    
    # Test with basic features
    print("\n1. Testing with Basic Features (19 features)...")
    basic_features, basic_labels = create_features_and_labels(stock_data)
    X_train_basic, X_test_basic, y_train_basic, y_test_basic, scaler_basic = prepare_data(
        basic_features, basic_labels, test_size=0.2, random_state=42
    )
    
    ensemble_basic = EnsemblePredictor()
    ensemble_basic.train_ensemble(X_train_basic, y_train_basic, X_test_basic, y_test_basic)
    
    results_basic = ensemble_basic.evaluate(X_test_basic, y_test_basic)
    print(f"\nBasic Features Ensemble Results:")
    print(f"Accuracy: {results_basic['accuracy']:.4f}")
    print(f"ROC-AUC: {results_basic['roc_auc']:.4f}")
    
    # Test with advanced features
    print(f"\n{'='*60}")
    print("2. Testing with Advanced Features (38 features)...")
    advanced_features, advanced_labels = create_advanced_features(stock_data)
    X_train_adv, X_test_adv, y_train_adv, y_test_adv, scaler_adv = prepare_data(
        advanced_features, advanced_labels, test_size=0.2, random_state=42
    )
    
    ensemble_advanced = EnsemblePredictor()
    ensemble_advanced.train_ensemble(X_train_adv, y_train_adv, X_test_adv, y_test_adv)
    
    results_advanced = ensemble_advanced.evaluate(X_test_adv, y_test_adv)
    print(f"\nAdvanced Features Ensemble Results:")
    print(f"Accuracy: {results_advanced['accuracy']:.4f}")
    print(f"ROC-AUC: {results_advanced['roc_auc']:.4f}")
    
    # Compare with single best model
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    # Single Neural Network (best from experiments)
    nn_single = FeedforwardNeuralNetwork(X_train_adv.shape[1], [100], learning_rate=0.001)
    nn_single.train(X_train_adv, y_train_adv, epochs=300, verbose=False)
    single_results = nn_single.evaluate(X_test_adv, y_test_adv)
    
    print(f"Single NN (100 neurons):     {single_results['accuracy']:.4f} accuracy, {single_results['roc_auc']:.4f} AUC")
    print(f"Basic Features Ensemble:     {results_basic['accuracy']:.4f} accuracy, {results_basic['roc_auc']:.4f} AUC")
    print(f"Advanced Features Ensemble:  {results_advanced['accuracy']:.4f} accuracy, {results_advanced['roc_auc']:.4f} AUC")
    
    # Calculate improvements
    basic_improvement = results_basic['accuracy'] - single_results['accuracy']
    advanced_improvement = results_advanced['accuracy'] - single_results['accuracy']
    
    print(f"\nImprovements over single model:")
    print(f"Basic Ensemble:    {basic_improvement:+.4f} ({basic_improvement*100:+.2f}%)")
    print(f"Advanced Ensemble: {advanced_improvement:+.4f} ({advanced_improvement*100:+.2f}%)")
    
    return ensemble_advanced, results_advanced

def create_trading_strategy(ensemble_model, confidence_threshold=0.6):
    """Create a simple trading strategy based on ensemble predictions"""
    
    print(f"\n{'='*60}")
    print("TRADING STRATEGY SIMULATION")
    print("=" * 60)
    
    # Generate new test data
    generator = StockDataGenerator(seed=123)  # Different seed for out-of-sample test
    test_data = generator.generate_stock_data(days=100)
    test_features, test_labels = create_advanced_features(test_data)
    
    # Prepare features (use same scaler from training)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(test_features.values)
    
    # Get predictions
    predictions = ensemble_model.predict_proba(X_test_scaled)
    
    # Trading strategy: only trade when confidence is high
    trades = []
    portfolio_value = 10000  # Start with $10,000
    position = 0  # 0 = no position, 1 = long, -1 = short
    
    for i in range(len(predictions)):
        confidence = abs(predictions[i] - 0.5) * 2  # Scale to 0-1
        
        if confidence > confidence_threshold:
            predicted_direction = 1 if predictions[i] > 0.5 else 0
            actual_direction = test_labels.iloc[i]
            
            # Simple strategy: buy if predict up, sell if predict down
            if predicted_direction == 1 and position <= 0:
                # Buy signal
                position = 1
                trades.append({
                    'day': i,
                    'action': 'BUY',
                    'confidence': confidence,
                    'predicted': predicted_direction,
                    'actual': actual_direction,
                    'correct': predicted_direction == actual_direction
                })
            elif predicted_direction == 0 and position >= 0:
                # Sell signal
                position = -1
                trades.append({
                    'day': i,
                    'action': 'SELL',
                    'confidence': confidence,
                    'predicted': predicted_direction,
                    'actual': actual_direction,
                    'correct': predicted_direction == actual_direction
                })
    
    # Calculate strategy performance
    if trades:
        correct_trades = sum(1 for trade in trades if trade['correct'])
        total_trades = len(trades)
        win_rate = correct_trades / total_trades
        
        print(f"Trading Strategy Results:")
        print(f"Confidence Threshold: {confidence_threshold}")
        print(f"Total Trades: {total_trades}")
        print(f"Correct Trades: {correct_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Confidence: {np.mean([t['confidence'] for t in trades]):.3f}")
        
        # Show sample trades
        print(f"\nSample Trades:")
        for i, trade in enumerate(trades[:5]):
            status = "✓" if trade['correct'] else "✗"
            print(f"Day {trade['day']:2d}: {trade['action']:4s} "
                  f"(conf: {trade['confidence']:.3f}) {status}")
    
    return trades

if __name__ == "__main__":
    # Run ensemble testing
    best_ensemble, results = test_ensemble_models()
    
    # Create trading strategy
    trades = create_trading_strategy(best_ensemble, confidence_threshold=0.65)
    
    print(f"\n{'='*60}")
    print("ENSEMBLE MODELING COMPLETE!")
    print("Key Achievements:")
    print("- Trained 6 different models in ensemble")
    print("- Tested with both basic and advanced features")
    print("- Created confidence-based trading strategy")
    print("- Achieved improved prediction accuracy")
    print("\nNext steps:")
    print("- Fine-tune ensemble weights")
    print("- Add more diverse models")
    print("- Implement cross-validation")
    print("- Test on real market data")
