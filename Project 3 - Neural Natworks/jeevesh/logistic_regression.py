"""
Logistic Regression Implementation from Scratch
Manual implementation for comparison with neural network
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd

class LogisticRegression:
    """
    Logistic Regression implemented from scratch using gradient descent
    
    Mathematical foundation:
    - Hypothesis: h(x) = sigmoid(θ^T * x)
    - Cost function: J(θ) = -1/m * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
    - Gradient: ∇J(θ) = 1/m * X^T * (h(x) - y)
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 tolerance: float = 1e-6, regularization: float = 0.0):
        """
        Initialize Logistic Regression
        
        Args:
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            regularization: L2 regularization parameter (Ridge)
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        
        # Model parameters
        self.weights = None
        self.bias = None
        
        # Training history
        self.cost_history = []
        self.accuracy_history = []
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function with numerical stability
        
        σ(z) = 1 / (1 + e^(-z))
        """
        # Clip z to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add bias column to feature matrix"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _compute_cost(self, h: np.ndarray, y: np.ndarray) -> float:
        """
        Compute logistic regression cost function
        
        J(θ) = -1/m * Σ[y*log(h) + (1-y)*log(1-h)] + λ/2m * ||θ||²
        
        Args:
            h: Predicted probabilities
            y: True labels
            
        Returns:
            Cost value
        """
        m = y.shape[0]
        
        # Clip predictions to prevent log(0)
        h_clipped = np.clip(h, 1e-15, 1 - 1e-15)
        
        # Logistic regression cost
        cost = -1/m * np.sum(y * np.log(h_clipped) + (1 - y) * np.log(1 - h_clipped))
        
        # Add L2 regularization (excluding bias term)
        if self.regularization > 0:
            reg_cost = self.regularization / (2 * m) * np.sum(self.weights[1:] ** 2)
            cost += reg_cost
        
        return cost
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict:
        """
        Train the logistic regression model
        
        Args:
            X: Training features (m, n)
            y: Training labels (m, 1)
            verbose: Whether to print training progress
            
        Returns:
            Training history
        """
        # Add intercept term
        X_with_intercept = self._add_intercept(X)
        m, n = X_with_intercept.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, (n, 1))
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            z = X_with_intercept.dot(self.weights)
            h = self._sigmoid(z)
            
            # Compute cost
            cost = self._compute_cost(h, y)
            self.cost_history.append(cost)
            
            # Compute accuracy
            predictions = (h > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            self.accuracy_history.append(accuracy)
            
            # Compute gradients
            gradient = 1/m * X_with_intercept.T.dot(h - y)
            
            # Add L2 regularization to gradient (excluding bias)
            if self.regularization > 0:
                reg_gradient = np.copy(gradient)
                reg_gradient[1:] += self.regularization / m * self.weights[1:]
                gradient = reg_gradient
            
            # Update parameters
            self.weights -= self.learning_rate * gradient
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[i] - self.cost_history[i-1]) < self.tolerance:
                if verbose:
                    print(f"Converged after {i+1} iterations")
                break
            
            # Print progress
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.max_iterations} - "
                      f"Cost: {cost:.6f} - Accuracy: {accuracy:.4f}")
        
        return {
            'cost_history': self.cost_history,
            'accuracy_history': self.accuracy_history
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        X_with_intercept = self._add_intercept(X)
        return self._sigmoid(X_with_intercept.dot(self.weights))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions"""
        return (self.predict_proba(X) > 0.5).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Returns:
            Dictionary with accuracy, confusion matrix, and ROC-AUC
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y, y_pred_proba)
        except ValueError:
            roc_auc = 0.5
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

def compare_models(X_train: np.ndarray, X_test: np.ndarray, 
                  y_train: np.ndarray, y_test: np.ndarray,
                  feature_names: list = None) -> Dict:
    """
    Compare manual logistic regression, sklearn logistic regression, and neural network
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        feature_names: List of feature names for importance analysis
        
    Returns:
        Dictionary with comparison results
    """
    from neural_network import FeedforwardNeuralNetwork
    
    results = {}
    
    print("Training Manual Logistic Regression...")
    # Manual Logistic Regression
    manual_lr = LogisticRegression(learning_rate=0.01, max_iterations=1000)
    manual_history = manual_lr.fit(X_train, y_train, verbose=False)
    manual_results = manual_lr.evaluate(X_test, y_test)
    
    results['manual_logistic'] = {
        'model': manual_lr,
        'history': manual_history,
        'test_results': manual_results
    }
    
    print("Training Sklearn Logistic Regression...")
    # Sklearn Logistic Regression
    sklearn_lr = SklearnLogisticRegression(random_state=42, max_iter=1000)
    sklearn_lr.fit(X_train, y_train.ravel())
    
    sklearn_pred_proba = sklearn_lr.predict_proba(X_test)[:, 1].reshape(-1, 1)
    sklearn_pred = sklearn_lr.predict(X_test).reshape(-1, 1)
    
    sklearn_results = {
        'accuracy': accuracy_score(y_test, sklearn_pred),
        'confusion_matrix': confusion_matrix(y_test, sklearn_pred),
        'roc_auc': roc_auc_score(y_test, sklearn_pred_proba),
        'predictions': sklearn_pred,
        'probabilities': sklearn_pred_proba
    }
    
    results['sklearn_logistic'] = {
        'model': sklearn_lr,
        'test_results': sklearn_results
    }
    
    print("Training Neural Network...")
    # Neural Network
    nn = FeedforwardNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=[64, 32],
        learning_rate=0.001
    )
    nn_history = nn.train(X_train, y_train, epochs=500, verbose=False)
    nn_results = nn.evaluate(X_test, y_test)
    
    results['neural_network'] = {
        'model': nn,
        'history': nn_history,
        'test_results': nn_results
    }
    
    # Feature importance analysis (for logistic regression)
    if feature_names is not None:
        # Manual LR feature importance (absolute weights)
        manual_importance = np.abs(manual_lr.weights[1:].flatten())  # Exclude bias
        manual_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': manual_importance
        }).sort_values('importance', ascending=False)
        
        # Sklearn LR feature importance
        sklearn_importance = np.abs(sklearn_lr.coef_.flatten())
        sklearn_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': sklearn_importance
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = {
            'manual_lr': manual_importance_df,
            'sklearn_lr': sklearn_importance_df
        }
    
    return results

def plot_model_comparison(results: Dict, save_path: str = None):
    """Plot comparison of different models"""
    models = ['manual_logistic', 'sklearn_logistic', 'neural_network']
    model_names = ['Manual LR', 'Sklearn LR', 'Neural Network']
    
    # Extract metrics
    accuracies = [results[model]['test_results']['accuracy'] for model in models]
    roc_aucs = [results[model]['test_results']['roc_auc'] for model in models]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # ROC-AUC comparison
    bars2 = ax2.bar(model_names, roc_aucs, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax2.set_title('Model ROC-AUC Comparison')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, auc in zip(bars2, roc_aucs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_feature_importance(importance_df: pd.DataFrame, title: str = "Feature Importance", 
                          top_n: int = 10, save_path: str = None):
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    
    # Select top N features
    top_features = importance_df.head(top_n)
    
    # Create horizontal bar plot
    plt.barh(range(len(top_features)), top_features['importance'], 
             color='steelblue', alpha=0.7)
    
    # Customize plot
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance (Absolute Weight)')
    plt.title(title)
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Add value labels
    for i, (_, row) in enumerate(top_features.iterrows()):
        plt.text(row['importance'] + max(top_features['importance']) * 0.01, i,
                f'{row["importance"]:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_training_comparison(results: Dict, save_path: str = None):
    """Plot training curves for models that have training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Manual Logistic Regression
    manual_history = results['manual_logistic']['history']
    ax1.plot(manual_history['cost_history'])
    ax1.set_title('Manual Logistic Regression - Cost')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.grid(True)
    
    ax2.plot(manual_history['accuracy_history'])
    ax2.set_title('Manual Logistic Regression - Accuracy')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    # Neural Network
    nn_history = results['neural_network']['history']
    ax3.plot(nn_history['loss_history'])
    ax3.set_title('Neural Network - Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True)
    
    ax4.plot(nn_history['accuracy_history'])
    ax4.set_title('Neural Network - Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

if __name__ == "__main__":
    # Test logistic regression
    from data_generator import StockDataGenerator, create_features_and_labels
    from neural_network import prepare_data
    
    # Generate test data
    generator = StockDataGenerator()
    stock_data = generator.generate_stock_data(days=1000)
    features, labels = create_features_and_labels(stock_data)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(features, labels)
    
    # Test manual logistic regression
    lr = LogisticRegression(learning_rate=0.01, max_iterations=1000)
    history = lr.fit(X_train, y_train)
    
    # Evaluate
    test_results = lr.evaluate(X_test, y_test)
    print(f"Logistic Regression Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Logistic Regression Test ROC-AUC: {test_results['roc_auc']:.4f}")
    
    # Compare all models
    comparison_results = compare_models(X_train, X_test, y_train, y_test, 
                                      feature_names=list(features.columns))
    
    # Plot comparisons
    plot_model_comparison(comparison_results)
    plot_training_comparison(comparison_results)
