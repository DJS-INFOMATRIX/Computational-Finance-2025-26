"""
Experiment with different neural network configurations
"""

from data_generator import StockDataGenerator, create_features_and_labels
from neural_network import FeedforwardNeuralNetwork, prepare_data
import numpy as np

def test_different_architectures():
    """Test various neural network architectures"""
    
    # Generate data
    generator = StockDataGenerator(seed=42)
    stock_data = generator.generate_stock_data(days=500)
    features, labels = create_features_and_labels(stock_data)
    X_train, X_test, y_train, y_test, scaler = prepare_data(features, labels)
    
    # Different architectures to test
    architectures = [
        ([32], "Single Hidden Layer (32)"),
        ([64, 32], "Two Hidden Layers (64->32)"),
        ([128, 64, 32], "Three Hidden Layers (128->64->32)"),
        ([100], "Single Large Layer (100)"),
        ([50, 25], "Smaller Network (50->25)")
    ]
    
    results = []
    
    print("Testing Different Neural Network Architectures:")
    print("=" * 60)
    
    for hidden_sizes, description in architectures:
        print(f"\nTesting: {description}")
        
        # Create and train model
        nn = FeedforwardNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            learning_rate=0.001
        )
        
        # Train with fewer epochs for quick testing
        history = nn.train(X_train, y_train, epochs=200, verbose=False)
        
        # Evaluate
        results_dict = nn.evaluate(X_test, y_test)
        
        print(f"  Accuracy: {results_dict['accuracy']:.4f}")
        print(f"  ROC-AUC: {results_dict['roc_auc']:.4f}")
        print(f"  Final Loss: {history['loss_history'][-1]:.4f}")
        
        results.append({
            'architecture': description,
            'accuracy': results_dict['accuracy'],
            'roc_auc': results_dict['roc_auc'],
            'final_loss': history['loss_history'][-1]
        })
    
    # Find best architecture
    best_arch = max(results, key=lambda x: x['accuracy'])
    print(f"\n{'='*60}")
    print("BEST ARCHITECTURE:")
    print(f"  {best_arch['architecture']}")
    print(f"  Accuracy: {best_arch['accuracy']:.4f}")
    print(f"  ROC-AUC: {best_arch['roc_auc']:.4f}")
    
    return results

def test_different_learning_rates():
    """Test different learning rates"""
    
    generator = StockDataGenerator(seed=42)
    stock_data = generator.generate_stock_data(days=300)
    features, labels = create_features_and_labels(stock_data)
    X_train, X_test, y_train, y_test, scaler = prepare_data(features, labels)
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    
    print("\n" + "="*60)
    print("Testing Different Learning Rates:")
    print("="*60)
    
    for lr in learning_rates:
        print(f"\nLearning Rate: {lr}")
        
        nn = FeedforwardNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=[64, 32],
            learning_rate=lr
        )
        
        history = nn.train(X_train, y_train, epochs=100, verbose=False)
        results_dict = nn.evaluate(X_test, y_test)
        
        print(f"  Accuracy: {results_dict['accuracy']:.4f}")
        print(f"  Final Loss: {history['loss_history'][-1]:.4f}")

if __name__ == "__main__":
    # Run experiments
    arch_results = test_different_architectures()
    test_different_learning_rates()
    
    print(f"\n{'='*60}")
    print("EXPERIMENTATION COMPLETE!")
    print("Try modifying the code to test:")
    print("- Different activation functions")
    print("- Regularization techniques") 
    print("- Different feature engineering")
    print("- Ensemble methods")
