"""
Main script for Stock Price Prediction using Neural Networks
Comprehensive training, evaluation, and comparison of models
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Import our modules
from data_generator import StockDataGenerator, create_features_and_labels
from neural_network import FeedforwardNeuralNetwork, prepare_data, plot_training_history, plot_confusion_matrix, plot_roc_curve
from logistic_regression import LogisticRegression, compare_models, plot_model_comparison, plot_feature_importance, plot_training_comparison

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_output_directory():
    """Create output directory for saving results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def print_model_summary(results: dict, model_name: str):
    """Print detailed model performance summary"""
    test_results = results['test_results']
    
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Summary")
    print(f"{'='*50}")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"ROC-AUC: {test_results['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Down  Up")
    print(f"Actual Down     {test_results['confusion_matrix'][0,0]:4d} {test_results['confusion_matrix'][0,1]:4d}")
    print(f"Actual Up       {test_results['confusion_matrix'][1,0]:4d} {test_results['confusion_matrix'][1,1]:4d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = test_results['confusion_matrix'].ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

def analyze_feature_importance(results: dict, output_dir: str):
    """Analyze and visualize feature importance"""
    if 'feature_importance' not in results:
        return
    
    print(f"\n{'='*50}")
    print("Feature Importance Analysis")
    print(f"{'='*50}")
    
    # Manual Logistic Regression feature importance
    manual_importance = results['feature_importance']['manual_lr']
    print("\nTop 10 Most Important Features (Manual Logistic Regression):")
    print(manual_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plot_feature_importance(
        manual_importance, 
        title="Feature Importance - Manual Logistic Regression",
        save_path=os.path.join(output_dir, "feature_importance_manual_lr.png")
    )
    
    # Sklearn Logistic Regression feature importance
    sklearn_importance = results['feature_importance']['sklearn_lr']
    plot_feature_importance(
        sklearn_importance,
        title="Feature Importance - Sklearn Logistic Regression", 
        save_path=os.path.join(output_dir, "feature_importance_sklearn_lr.png")
    )

def create_comprehensive_report(stock_data: pd.DataFrame, features: pd.DataFrame, 
                              labels: pd.Series, results: dict, output_dir: str):
    """Create a comprehensive analysis report"""
    
    # Data summary
    print(f"\n{'='*60}")
    print("STOCK PRICE PREDICTION - COMPREHENSIVE ANALYSIS REPORT")
    print(f"{'='*60}")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{'='*50}")
    print("Dataset Summary")
    print(f"{'='*50}")
    print(f"Total trading days: {len(stock_data)}")
    print(f"Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
    print(f"Features created: {len(features.columns)}")
    print(f"Valid samples after feature engineering: {len(features)}")
    print(f"\nLabel distribution:")
    print(f"Down days (0): {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)")
    print(f"Up days (1): {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")
    
    print(f"\nFeature list:")
    for i, feature in enumerate(features.columns, 1):
        print(f"{i:2d}. {feature}")
    
    # Stock price statistics
    print(f"\nStock Price Statistics:")
    print(f"Starting price: ${stock_data['Close'].iloc[0]:.2f}")
    print(f"Ending price: ${stock_data['Close'].iloc[-1]:.2f}")
    print(f"Total return: {((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Average daily return: {stock_data['Close'].pct_change().mean() * 100:.3f}%")
    print(f"Daily volatility: {stock_data['Close'].pct_change().std() * 100:.3f}%")
    
    # Model performance comparison
    print(f"\n{'='*50}")
    print("Model Performance Comparison")
    print(f"{'='*50}")
    
    models = ['manual_logistic', 'sklearn_logistic', 'neural_network']
    model_names = ['Manual Logistic Regression', 'Sklearn Logistic Regression', 'Neural Network']
    
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': [results[model]['test_results']['accuracy'] for model in models],
        'ROC-AUC': [results[model]['test_results']['roc_auc'] for model in models]
    })
    
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Individual model summaries
    for model, name in zip(models, model_names):
        print_model_summary(results[model], name)
    
    # Feature importance analysis
    analyze_feature_importance(results, output_dir)
    
    # Insights and recommendations
    print(f"\n{'='*50}")
    print("Key Insights and Recommendations")
    print(f"{'='*50}")
    
    best_model = model_names[np.argmax([results[model]['test_results']['accuracy'] for model in models])]
    best_accuracy = max([results[model]['test_results']['accuracy'] for model in models])
    best_auc = max([results[model]['test_results']['roc_auc'] for model in models])
    
    print(f"1. Best performing model: {best_model}")
    print(f"   - Accuracy: {best_accuracy:.4f}")
    print(f"   - ROC-AUC: {best_auc:.4f}")
    
    if best_accuracy > 0.55:
        print(f"\n2. The model shows promising predictive ability above random chance (50%)")
    else:
        print(f"\n2. The model performance is close to random chance - consider:")
        print(f"   - Adding more sophisticated features")
        print(f"   - Increasing the lookback window")
        print(f"   - Using ensemble methods")
    
    print(f"\n3. Improvement suggestions:")
    print(f"   - Regularization: Add L1/L2 regularization to prevent overfitting")
    print(f"   - Optimization: Try Adam optimizer instead of SGD")
    print(f"   - Features: Add more technical indicators (Bollinger Bands, Stochastic, etc.)")
    print(f"   - Architecture: Experiment with different network architectures")
    print(f"   - Data: Use real market data with more samples")
    print(f"   - Ensemble: Combine multiple models for better performance")
    
    if 'feature_importance' in results:
        top_feature = results['feature_importance']['manual_lr'].iloc[0]['feature']
        print(f"\n4. Most important feature: {top_feature}")
        print(f"   - Consider engineering more features similar to this one")

def main():
    """Main execution function"""
    print("Starting Stock Price Prediction Analysis...")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Results will be saved to: {output_dir}")
    
    # Generate stock data
    print("\n1. Generating mock stock data...")
    generator = StockDataGenerator(seed=42)
    stock_data = generator.generate_stock_data(days=1000, initial_price=100.0, volatility=0.02)
    
    # Create features and labels
    print("2. Creating features and labels...")
    features, labels = create_features_and_labels(stock_data, lookback_days=10)
    
    print(f"   - Generated {len(features)} samples with {len(features.columns)} features")
    print(f"   - Label distribution: {labels.value_counts().to_dict()}")
    
    # Prepare data for training
    print("3. Preparing data for training...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(features, labels, test_size=0.2, random_state=42)
    
    print(f"   - Training set: {X_train.shape[0]} samples")
    print(f"   - Test set: {X_test.shape[0]} samples")
    
    # Train and compare models
    print("4. Training and comparing models...")
    print("   This may take a few minutes...")
    
    results = compare_models(
        X_train, X_test, y_train, y_test, 
        feature_names=list(features.columns)
    )
    
    # Create visualizations
    print("5. Creating visualizations...")
    
    # Model comparison plots
    plot_model_comparison(results, save_path=os.path.join(output_dir, "model_comparison.png"))
    
    # Training curves
    plot_training_comparison(results, save_path=os.path.join(output_dir, "training_curves.png"))
    
    # Individual model plots for Neural Network
    nn_results = results['neural_network']
    plot_training_history(nn_results['history'], save_path=os.path.join(output_dir, "nn_training_history.png"))
    plot_confusion_matrix(nn_results['test_results']['confusion_matrix'], 
                         save_path=os.path.join(output_dir, "nn_confusion_matrix.png"))
    plot_roc_curve(y_test.flatten(), nn_results['test_results']['probabilities'].flatten(),
                   save_path=os.path.join(output_dir, "nn_roc_curve.png"))
    
    # Stock price visualization
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Date'], stock_data['Close'], linewidth=1)
    plt.title('Generated Stock Price Data')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stock_price_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = features.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_correlation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comprehensive report
    print("6. Generating comprehensive report...")
    create_comprehensive_report(stock_data, features, labels, results, output_dir)
    
    # Save results to files
    print("7. Saving results...")
    
    # Save feature data
    features.to_csv(os.path.join(output_dir, "features.csv"), index=False)
    pd.Series(labels).to_csv(os.path.join(output_dir, "labels.csv"), index=False, header=['label'])
    stock_data.to_csv(os.path.join(output_dir, "stock_data.csv"), index=False)
    
    # Save model predictions
    for model_name, model_data in results.items():
        if model_name != 'feature_importance':
            predictions_df = pd.DataFrame({
                'true_label': y_test.flatten(),
                'predicted_label': model_data['test_results']['predictions'].flatten(),
                'predicted_probability': model_data['test_results']['probabilities'].flatten()
            })
            predictions_df.to_csv(os.path.join(output_dir, f"{model_name}_predictions.csv"), index=False)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"Check the generated plots and CSV files for detailed analysis.")
    
    return results, output_dir

if __name__ == "__main__":
    # Run the complete analysis
    results, output_dir = main()
    
    print("\nAnalysis completed successfully!")
