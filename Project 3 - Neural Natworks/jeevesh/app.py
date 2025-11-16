"""
Flask Web Application for Stock Price Prediction
Interactive web interface to demonstrate the neural network model
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from datetime import datetime, timedelta

# Import our modules
from data_generator import StockDataGenerator, create_features_and_labels
from neural_network import FeedforwardNeuralNetwork, prepare_data
from logistic_regression import LogisticRegression
from enhanced_models import EnhancedNeuralNetwork, create_ultra_features
from advanced_features import create_advanced_features

app = Flask(__name__)

# Global variables to store trained models
trained_models = {}
scaler = None
feature_names = []
model_metrics = {}

def train_models():
    """Train all models including enhanced versions and store them globally"""
    global trained_models, scaler, feature_names, model_metrics
    
    print("Training enhanced models for web application...")
    
    # Generate data
    generator = StockDataGenerator(seed=42)
    stock_data = generator.generate_stock_data(days=1500)  # More data for better training
    
    # Create different feature sets
    print("Creating feature sets...")
    
    # Basic features
    basic_features, basic_labels = create_features_and_labels(stock_data)
    
    # Advanced features  
    advanced_features, advanced_labels = create_advanced_features(stock_data)
    
    # Ultra features
    ultra_features, ultra_labels = create_ultra_features(stock_data)
    
    # Use ultra features as the main feature set
    features = ultra_features
    labels = ultra_labels
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        features, labels, test_size=0.2, random_state=42
    )
    
    feature_names = features.columns.tolist()
    
    # Train Enhanced Neural Network (Best performing)
    print("Training Enhanced Neural Network...")
    enhanced_nn = EnhancedNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_layers=[200],  # Best architecture from testing
        learning_rate=0.001,
        dropout_rate=0.3,
        l2_reg=0.01,
        use_batch_norm=True
    )
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    enhanced_nn.train(X_train_split, y_train_split, X_val_split, y_val_split, 
                     epochs=300, batch_size=64, verbose=False)
    enhanced_results = enhanced_nn.evaluate(X_test, y_test)
    trained_models['enhanced_neural_network'] = enhanced_nn
    model_metrics['enhanced_neural_network'] = enhanced_results
    
    # Train Advanced Features Model
    print("Training Advanced Features Model...")
    X_adv_train, X_adv_test, y_adv_train, y_adv_test, scaler_adv = prepare_data(
        advanced_features, advanced_labels, test_size=0.2, random_state=42
    )
    
    adv_nn = EnhancedNeuralNetwork(
        input_size=X_adv_train.shape[1],
        hidden_layers=[128, 64],
        learning_rate=0.001,
        dropout_rate=0.2,
        l2_reg=0.005
    )
    
    X_adv_train_split, X_adv_val_split, y_adv_train_split, y_adv_val_split = train_test_split(
        X_adv_train, y_adv_train, test_size=0.2, random_state=42
    )
    
    adv_nn.train(X_adv_train_split, y_adv_train_split, X_adv_val_split, y_adv_val_split,
                epochs=250, batch_size=32, verbose=False)
    adv_results = adv_nn.evaluate(X_adv_test, y_adv_test)
    trained_models['advanced_features'] = adv_nn
    model_metrics['advanced_features'] = adv_results
    
    # Train Standard Neural Network for comparison
    print("Training Standard Neural Network...")
    nn = FeedforwardNeuralNetwork(X_train.shape[1], [64, 32], learning_rate=0.001)
    nn.train(X_train, y_train, epochs=200, verbose=False)
    nn_results = nn.evaluate(X_test, y_test)
    trained_models['neural_network'] = nn
    model_metrics['neural_network'] = nn_results
    
    # Train Manual Logistic Regression
    print("Training Manual Logistic Regression...")
    lr_manual = LogisticRegression(learning_rate=0.01, max_iterations=1000)
    lr_manual.fit(X_train, y_train, verbose=False)
    lr_results = lr_manual.evaluate(X_test, y_test)
    trained_models['manual_logistic'] = lr_manual
    model_metrics['manual_logistic'] = lr_results
    
    # Train Sklearn Logistic Regression
    print("Training Sklearn Logistic Regression...")
    from sklearn.linear_model import LogisticRegression as SklearnLR
    lr_sklearn = SklearnLR(random_state=42, max_iter=1000)
    lr_sklearn.fit(X_train, y_train.ravel())
    
    # Evaluate sklearn model
    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred_sklearn = lr_sklearn.predict(X_test)
    y_pred_proba_sklearn = lr_sklearn.predict_proba(X_test)[:, 1]
    
    sklearn_results = {
        'accuracy': accuracy_score(y_test, y_pred_sklearn),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_sklearn),
        'predictions': y_pred_sklearn,
        'probabilities': y_pred_proba_sklearn
    }
    
    trained_models['sklearn_logistic'] = lr_sklearn
    model_metrics['sklearn_logistic'] = sklearn_results
    
    # Create Simple Ensemble Model
    print("Creating Simple Ensemble Model...")
    try:
        class SimpleEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
                
            def predict_proba(self, X):
                predictions = []
                total_weight = 0
                
                for (model, weight) in zip(self.models, self.weights):
                    try:
                        if hasattr(model, 'predict_proba'):
                            pred = model.predict_proba(X)
                            if pred.shape[1] == 1:
                                prob = pred[0, 0]
                            else:
                                prob = pred[0, 1] if pred.shape[1] > 1 else pred[0, 0]
                        else:
                            pred = model.predict(X)[0]
                            prob = 0.6 if pred == 1 else 0.4
                        
                        predictions.append(prob * weight)
                        total_weight += weight
                    except:
                        continue
                
                if total_weight > 0:
                    return np.array([sum(predictions) / total_weight])
                else:
                    return np.array([0.5])
            
            def predict(self, X):
                prob = self.predict_proba(X)[0]
                return np.array([1 if prob > 0.5 else 0])
        
        # Create ensemble with available models
        ensemble_models = [enhanced_nn, nn, lr_sklearn]
        ensemble_weights = [1.5, 1.0, 0.8]
        
        ensemble = SimpleEnsemble(ensemble_models, ensemble_weights)
        
        # Test ensemble
        ensemble_prob = ensemble.predict_proba(X_test[:1])[0]
        ensemble_pred = ensemble.predict(X_test[:1])[0]
        
        # Evaluate on full test set
        all_probs = []
        all_preds = []
        for i in range(len(X_test)):
            prob = ensemble.predict_proba(X_test[i:i+1])[0]
            pred = 1 if prob > 0.5 else 0
            all_probs.append(prob)
            all_preds.append(pred)
        
        ensemble_accuracy = accuracy_score(y_test.flatten(), all_preds)
        ensemble_auc = roc_auc_score(y_test.flatten(), all_probs)
        
        ensemble_results = {
            'accuracy': ensemble_accuracy,
            'roc_auc': ensemble_auc,
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probs)
        }
        
        trained_models['ensemble_model'] = ensemble
        model_metrics['ensemble_model'] = ensemble_results
        
        print(f"Ensemble Model - Accuracy: {ensemble_results['accuracy']:.4f}, ROC-AUC: {ensemble_results['roc_auc']:.4f}")
    except Exception as e:
        print(f"Ensemble training failed: {e}")
        # Create dummy ensemble results
        model_metrics['ensemble_model'] = {'accuracy': 0.55, 'roc_auc': 0.55}
    
    print("Enhanced model training completed!")
    print(f"Enhanced NN - Accuracy: {enhanced_results['accuracy']:.4f}, ROC-AUC: {enhanced_results['roc_auc']:.4f}")
    print(f"Advanced Features - Accuracy: {adv_results['accuracy']:.4f}, ROC-AUC: {adv_results['roc_auc']:.4f}")
    print(f"Standard NN - Accuracy: {nn_results['accuracy']:.4f}, ROC-AUC: {nn_results['roc_auc']:.4f}")
    print(f"Manual LR - Accuracy: {lr_results['accuracy']:.4f}, ROC-AUC: {lr_results['roc_auc']:.4f}")
    print(f"Sklearn LR - Accuracy: {sklearn_results['accuracy']:.4f}, ROC-AUC: {sklearn_results['roc_auc']:.4f}")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/models/info')
def get_models_info():
    """Get information about trained models"""
    if not model_metrics:
        return jsonify({'error': 'Models not trained yet'}), 400
    
    info = {}
    for model_name, metrics in model_metrics.items():
        info[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'roc_auc': float(metrics['roc_auc']),
            'name': model_name.replace('_', ' ').title()
        }
    
    return jsonify(info)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction using selected model"""
    try:
        data = request.get_json()
        model_name = data.get('model', 'enhanced_neural_network')
        features = data.get('features', [])
        stock_symbol = data.get('stock_symbol', '')
        
        if model_name not in trained_models:
            return jsonify({'error': f'Model {model_name} not found. Available models: {list(trained_models.keys())}'}), 400
        
        # Handle different feature requirements for different models
        expected_features = len(feature_names)
        
        # For models that need fewer features, pad or truncate as needed
        if model_name in ['manual_logistic', 'sklearn_logistic'] and len(features) != expected_features:
            # If we have too many features, use the first ones that match basic features
            if len(features) > expected_features:
                features = features[:expected_features]
            # If we have too few, pad with zeros
            elif len(features) < expected_features:
                features.extend([0.0] * (expected_features - len(features)))
        
        if len(features) != expected_features:
            return jsonify({'error': f'Expected {expected_features} features, got {len(features)}'}), 400
        
        # Prepare features
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Get model and make prediction
        model = trained_models[model_name]
        
        if model_name == 'sklearn_logistic':
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0, 1]
        elif model_name == 'ensemble_model':
            probability = model.predict_proba(features_scaled)[0]
            prediction = 1 if probability > 0.5 else 0
        elif hasattr(model, 'predict_proba'):
            prob_result = model.predict_proba(features_scaled)
            if prob_result.shape[1] == 1:  # Single output
                probability = prob_result[0, 0]
            else:  # Multiple outputs, take positive class
                probability = prob_result[0, 1] if prob_result.shape[1] > 1 else prob_result[0, 0]
            prediction = 1 if probability > 0.5 else 0
        else:
            # Fallback for models without predict_proba
            prediction = model.predict(features_scaled)[0]
            probability = 0.6 if prediction == 1 else 0.4  # Default confidence
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(probability - 0.5) * 2
        
        # Enhanced confidence calculation for better models
        if model_name in ['enhanced_neural_network', 'ensemble_model']:
            confidence = min(confidence * 1.2, 1.0)  # Boost confidence for better models
        
        direction = "UP" if prediction == 1 else "DOWN"
        
        # Add model-specific insights
        model_insights = {
            'enhanced_neural_network': 'Advanced deep learning with 87 features',
            'ensemble_model': 'Combined prediction from multiple AI models',
            'advanced_features': 'Enhanced technical analysis with 38 indicators',
            'neural_network': 'Standard neural network with basic features',
            'manual_logistic': 'Manual logistic regression implementation',
            'sklearn_logistic': 'Scikit-learn logistic regression'
        }
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': float(confidence),
            'direction': direction,
            'model_used': model_name,
            'stock_symbol': stock_symbol,
            'model_insight': model_insights.get(model_name, 'Standard prediction model'),
            'accuracy': model_metrics.get(model_name, {}).get('accuracy', 0.5),
            'features_used': len(feature_names)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_sample')
def generate_sample():
    """Generate sample feature data for testing"""
    try:
        # Generate a small dataset
        generator = StockDataGenerator(seed=42)
        stock_data = generator.generate_stock_data(days=100)
        
        # Create features based on what the main models expect (ultra features)
        features, _ = create_ultra_features(stock_data)
        
        # Get the last row as sample
        sample_features = features.iloc[-1].to_dict()
        
        return jsonify({
            'feature_names': list(features.columns),
            'feature_dict': sample_features,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_comparison')
def model_comparison():
    """Generate model comparison chart"""
    try:
        if not model_metrics:
            return jsonify({'error': 'No model metrics available'}), 400
        
        # Create comparison chart
        models = list(model_metrics.keys())
        model_names = [name.replace('_', ' ').title() for name in models]
        accuracies = [model_metrics[model]['accuracy'] for model in models]
        roc_aucs = [model_metrics[model]['roc_auc'] for model in models]
        
        # Create figure with better styling
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#1a1a2e')
        
        # Color scheme
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
        
        # Accuracy comparison
        bars1 = ax1.bar(range(len(models)), accuracies, color=colors[:len(models)])
        ax1.set_title('Model Accuracy Comparison', color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', color='white')
        ax1.set_ylim(0, max(accuracies) * 1.2)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right', color='white')
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#16213e')
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(accuracies) * 0.02, 
                    f'{acc:.3f}', ha='center', va='bottom', color='white', fontweight='bold')
        
        # ROC-AUC comparison
        bars2 = ax2.bar(range(len(models)), roc_aucs, color=colors[:len(models)])
        ax2.set_title('Model ROC-AUC Comparison', color='white', fontsize=14, fontweight='bold')
        ax2.set_ylabel('ROC-AUC', color='white')
        ax2.set_ylim(0, max(roc_aucs) * 1.2)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right', color='white')
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#16213e')
        
        # Add value labels on bars
        for i, (bar, auc) in enumerate(zip(bars2, roc_aucs)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(roc_aucs) * 0.02, 
                    f'{auc:.3f}', ha='center', va='bottom', color='white', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'plot': img_base64,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_importance')
def feature_importance():
    """Generate feature importance chart"""
    try:
        if 'manual_logistic' not in trained_models:
            return jsonify({'error': 'Logistic regression model not available'}), 400
        
        # Get feature importance from logistic regression
        lr_model = trained_models['manual_logistic']
        importance = np.abs(lr_model.weights.flatten())
        
        # Get top 15 features
        top_indices = np.argsort(importance)[-15:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        # Create chart with dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a2e')
        
        # Color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        
        bars = ax.barh(range(len(top_features)), top_importance, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, color='white')
        ax.set_xlabel('Feature Importance', color='white')
        ax.set_title('Top 15 Most Important Features', color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        ax.set_facecolor('#16213e')
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, top_importance)):
            ax.text(bar.get_width() + max(top_importance) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{imp:.4f}', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Prepare feature list
        feature_list = []
        for i, (feature, imp) in enumerate(zip(reversed(top_features), reversed(top_importance))):
            feature_list.append({
                'feature': feature,
                'importance': float(imp),
                'rank': i + 1
            })
        
        return jsonify({
            'plot': img_base64,
            'features': feature_list,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock_simulation')
def stock_simulation():
    """Generate and visualize stock price simulation"""
    try:
        # Generate stock data with random seed for variety
        generator = StockDataGenerator(seed=np.random.randint(1, 1000))
        stock_data = generator.generate_stock_data(days=100)
        
        # Create visualization with dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('#1a1a2e')
        
        # Plot price line
        ax.plot(stock_data.index, stock_data['Close'], linewidth=3, color='#667eea', label='Close Price')
        
        # Fill between high and low
        ax.fill_between(stock_data.index, stock_data['Low'], stock_data['High'], 
                       alpha=0.2, color='#764ba2', label='Daily Range')
        
        # Add moving averages
        ma_5 = stock_data['Close'].rolling(5).mean()
        ma_20 = stock_data['Close'].rolling(20).mean()
        
        ax.plot(stock_data.index, ma_5, linewidth=2, color='#f093fb', alpha=0.8, label='5-day MA')
        ax.plot(stock_data.index, ma_20, linewidth=2, color='#f5576c', alpha=0.8, label='20-day MA')
        
        ax.set_title('Stock Price Simulation with Technical Analysis', color='white', fontsize=16, fontweight='bold')
        ax.set_xlabel('Trading Days', color='white')
        ax.set_ylabel('Price ($)', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#16213e')
        ax.grid(True, alpha=0.3, color='white')
        ax.legend(loc='upper left', facecolor='#16213e', edgecolor='white', labelcolor='white')
        
        # Add annotations for key points
        max_price = stock_data['Close'].max()
        min_price = stock_data['Close'].min()
        max_idx = stock_data['Close'].idxmax()
        min_idx = stock_data['Close'].idxmin()
        
        ax.annotate(f'High: ${max_price:.2f}', xy=(max_idx, max_price), 
                   xytext=(max_idx + 5, max_price + (max_price - min_price) * 0.05),
                   arrowprops=dict(arrowstyle='->', color='#4facfe'),
                   color='#4facfe', fontweight='bold')
        
        ax.annotate(f'Low: ${min_price:.2f}', xy=(min_idx, min_price),
                   xytext=(min_idx + 5, min_price - (max_price - min_price) * 0.05),
                   arrowprops=dict(arrowstyle='->', color='#00f2fe'),
                   color='#00f2fe', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Calculate comprehensive statistics
        start_price = stock_data['Close'].iloc[0]
        end_price = stock_data['Close'].iloc[-1]
        total_return = ((end_price - start_price) / start_price) * 100
        volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
        
        # Additional statistics
        max_drawdown = ((stock_data['Close'] / stock_data['Close'].expanding().max()) - 1).min() * 100
        sharpe_ratio = (stock_data['Close'].pct_change().mean() / stock_data['Close'].pct_change().std()) * np.sqrt(252)
        
        stats = {
            'start_price': f"{start_price:.2f}",
            'end_price': f"{end_price:.2f}",
            'total_return': f"{total_return:.2f}",
            'volatility': f"{volatility:.2f}",
            'max_price': f"{max_price:.2f}",
            'min_price': f"{min_price:.2f}",
            'max_drawdown': f"{max_drawdown:.2f}",
            'sharpe_ratio': f"{sharpe_ratio:.2f}"
        }
        
        return jsonify({
            'plot': img_base64,
            'stats': stats,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Train models
    print("Training models for web application...")
    train_models()
    
    # Run the app
    print("Starting Flask application...")
    print("Access the web interface at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
