"""
Feedforward Neural Network Implementation from Scratch
Manual implementation using only NumPy for stock price prediction
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd

class ActivationFunctions:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability"""
        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

class FeedforwardNeuralNetwork:
    """
    Feedforward Neural Network implemented from scratch
    
    Architecture:
    - Input layer (n_features)
    - Hidden layer 1 (ReLU activation)
    - Hidden layer 2 (ReLU activation) [optional]
    - Output layer (Sigmoid activation for binary classification)
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_sizes: List[int] = [64, 32], 
                 output_size: int = 1,
                 learning_rate: float = 0.001,
                 random_seed: int = 42):
        """
        Initialize neural network
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons (1 for binary classification)
            learning_rate: Learning rate for gradient descent
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights = {}
        self.biases = {}
        
        # Create layer sizes list
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights using Xavier initialization
        for i in range(len(layer_sizes) - 1):
            layer_name = f'layer_{i+1}'
            # Xavier initialization: sqrt(6 / (fan_in + fan_out))
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            self.weights[layer_name] = np.random.uniform(
                -limit, limit, (layer_sizes[i], layer_sizes[i+1])
            )
            self.biases[layer_name] = np.zeros((1, layer_sizes[i+1]))
        
        # Store activations and z values for backpropagation
        self.activations = {}
        self.z_values = {}
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network
        
        Mathematical explanation:
        For each layer l:
        z^(l) = a^(l-1) * W^(l) + b^(l)  (linear transformation)
        a^(l) = activation_function(z^(l))  (activation)
        
        Args:
            X: Input data (batch_size, input_size)
            
        Returns:
            Output predictions (batch_size, output_size)
        """
        # Store input as first activation
        self.activations['layer_0'] = X
        
        current_activation = X
        
        # Forward pass through hidden layers
        for i in range(len(self.hidden_sizes)):
            layer_name = f'layer_{i+1}'
            
            # Linear transformation: z = aW + b
            z = np.dot(current_activation, self.weights[layer_name]) + self.biases[layer_name]
            self.z_values[layer_name] = z
            
            # ReLU activation for hidden layers
            current_activation = ActivationFunctions.relu(z)
            self.activations[layer_name] = current_activation
        
        # Output layer
        output_layer_name = f'layer_{len(self.hidden_sizes)+1}'
        z_output = np.dot(current_activation, self.weights[output_layer_name]) + self.biases[output_layer_name]
        self.z_values[output_layer_name] = z_output
        
        # Sigmoid activation for output layer
        output = ActivationFunctions.sigmoid(z_output)
        self.activations[output_layer_name] = output
        
        return output
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss
        
        Mathematical formula:
        Loss = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
        
        Args:
            y_true: True labels (batch_size, 1)
            y_pred: Predicted probabilities (batch_size, 1)
            
        Returns:
            Average loss
        """
        m = y_true.shape[0]
        
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Binary cross-entropy loss
        loss = -1/m * np.sum(
            y_true * np.log(y_pred_clipped) + 
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        
        return loss
    
    def backward_propagation(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Backward propagation to compute gradients
        
        Mathematical explanation:
        For output layer:
        dL/dz^(L) = (ŷ - y) / m  (for sigmoid + cross-entropy)
        
        For hidden layers:
        dL/dz^(l) = dL/da^(l) * da^(l)/dz^(l)
        dL/da^(l) = dL/dz^(l+1) * W^(l+1)
        da^(l)/dz^(l) = activation_derivative(z^(l))
        
        Weight and bias gradients:
        dL/dW^(l) = a^(l-1)^T * dL/dz^(l)
        dL/db^(l) = sum(dL/dz^(l), axis=0)
        
        Args:
            X: Input data
            y_true: True labels
            y_pred: Predicted probabilities
        """
        m = X.shape[0]
        
        # Initialize gradients dictionaries
        dW = {}
        db = {}
        
        # Output layer gradients
        output_layer_name = f'layer_{len(self.hidden_sizes)+1}'
        
        # For sigmoid + cross-entropy, derivative simplifies to (y_pred - y_true)
        dz_output = (y_pred - y_true) / m
        
        # Previous layer activation
        prev_layer_name = f'layer_{len(self.hidden_sizes)}'
        a_prev = self.activations[prev_layer_name]
        
        # Weight and bias gradients for output layer
        dW[output_layer_name] = np.dot(a_prev.T, dz_output)
        db[output_layer_name] = np.sum(dz_output, axis=0, keepdims=True)
        
        # Propagate error backwards through hidden layers
        dz_current = dz_output
        
        for i in range(len(self.hidden_sizes), 0, -1):
            layer_name = f'layer_{i}'
            prev_layer_name = f'layer_{i-1}'
            
            # Compute da (error w.r.t. activation)
            da = np.dot(dz_current, self.weights[f'layer_{i+1}'].T)
            
            # Compute dz (error w.r.t. pre-activation)
            # For ReLU: derivative is 1 if z > 0, else 0
            dz_current = da * ActivationFunctions.relu_derivative(self.z_values[layer_name])
            
            # Weight and bias gradients
            a_prev = self.activations[prev_layer_name]
            dW[layer_name] = np.dot(a_prev.T, dz_current)
            db[layer_name] = np.sum(dz_current, axis=0, keepdims=True)
        
        # Update weights and biases using gradient descent
        for layer_name in self.weights.keys():
            self.weights[layer_name] -= self.learning_rate * dW[layer_name]
            self.biases[layer_name] -= self.learning_rate * db[layer_name]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 1000, batch_size: int = 32, verbose: bool = True) -> Dict:
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            epoch_accuracy = 0
            n_batches = 0
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # Forward propagation
                y_pred = self.forward_propagation(X_batch)
                
                # Compute loss
                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                
                # Compute accuracy
                predictions = (y_pred > 0.5).astype(int)
                batch_accuracy = np.mean(predictions == y_batch)
                epoch_accuracy += batch_accuracy
                
                # Backward propagation
                self.backward_propagation(X_batch, y_batch, y_pred)
                
                n_batches += 1
            
            # Average loss and accuracy for the epoch
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(avg_accuracy)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                val_accuracy = self.evaluate(X_val, y_val)['accuracy']
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Acc: {avg_accuracy:.4f} - "
                          f"Val_Loss: {val_loss:.4f} - Val_Acc: {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Accuracy: {avg_accuracy:.4f}")
        
        return {
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        return self.forward_propagation(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
    
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
        
        # ROC-AUC (handle case where only one class is present)
        try:
            roc_auc = roc_auc_score(y, y_pred_proba)
        except ValueError:
            roc_auc = 0.5  # Default for single class
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

def prepare_data(features: pd.DataFrame, labels: pd.Series, 
                test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Prepare data for training: split and normalize
    
    Args:
        features: Feature DataFrame
        labels: Labels Series
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    # Convert to numpy arrays
    X = features.values
    y = labels.values.reshape(-1, 1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def plot_training_history(history: Dict, save_path: str = None):
    """Plot training loss and accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(history['loss_history'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(history['accuracy_history'])
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_confusion_matrix(conf_matrix: np.ndarray, save_path: str = None):
    """Plot confusion matrix"""
    plt.figure(figsize=(6, 5))
    
    # Create heatmap
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0, 1], ['Down', 'Up'])
    plt.yticks([0, 1], ['Down', 'Up'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: str = None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

if __name__ == "__main__":
    # Test the neural network with dummy data
    from data_generator import StockDataGenerator, create_features_and_labels
    
    # Generate test data
    generator = StockDataGenerator()
    stock_data = generator.generate_stock_data(days=1000)
    features, labels = create_features_and_labels(stock_data)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels distribution: {labels.value_counts()}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(features, labels)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Create and train neural network
    nn = FeedforwardNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=[64, 32],
        learning_rate=0.001
    )
    
    # Train the model
    history = nn.train(X_train, y_train, epochs=500, verbose=True)
    
    # Evaluate on test set
    test_results = nn.evaluate(X_test, y_test)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"ROC-AUC: {test_results['roc_auc']:.4f}")
    print(f"Confusion Matrix:\n{test_results['confusion_matrix']}")
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(test_results['confusion_matrix'])
    plot_roc_curve(y_test.flatten(), test_results['probabilities'].flatten())
