# Stock Price Prediction with Neural Networks

A comprehensive machine learning project that predicts stock price direction (up/down) using feedforward neural networks and technical indicators. This project implements neural networks from scratch using NumPy and compares performance with logistic regression models.

## 🎯 Project Overview

This project builds a binary classification system to predict whether a stock's price will go up or down the next day based on technical indicators from the past 5-10 days. The implementation includes:

- **Custom Neural Network**: Built from scratch using only NumPy
- **Technical Indicators**: RSI, MACD, SMA, volatility, and more
- **Model Comparison**: Neural Network vs Logistic Regression
- **Web Interface**: Interactive Flask application for real-time predictions
- **Comprehensive Analysis**: Feature importance, model evaluation, and visualizations

## 🚀 Features

### Technical Indicators Implemented
- **Past Daily Returns**: Last 5-10 days of price changes
- **Simple Moving Averages**: 5-day and 10-day SMA
- **RSI (14-day)**: Relative Strength Index for momentum
- **MACD**: Moving Average Convergence Divergence with signal line
- **Volatility**: 10-day rolling standard deviation of returns
- **Price Volume Trend (PVT)**: Volume-weighted price momentum
- **Z-Score Normalized Price**: 20-day rolling z-score normalization

### Models Implemented
1. **Feedforward Neural Network** (from scratch)
   - Architecture: Input → Hidden(64) → Hidden(32) → Output(1)
   - Activations: ReLU for hidden layers, Sigmoid for output
   - Manual backpropagation with gradient descent
   
2. **Logistic Regression** (from scratch)
   - Manual implementation with gradient descent
   - L2 regularization support
   
3. **Sklearn Logistic Regression** (for comparison)

### Evaluation Metrics
- **Accuracy**: Classification accuracy on test set
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: True/False positives and negatives
- **Feature Importance**: Analysis of most predictive features

## 📊 Model Performance

Based on our testing with simulated data:

| Model | Accuracy | ROC-AUC | Notes |
|-------|----------|---------|-------|
| Neural Network | ~52-58% | ~0.52-0.60 | Best overall performance |
| Manual Logistic Regression | ~51-55% | ~0.51-0.57 | Good baseline |
| Sklearn Logistic Regression | ~51-55% | ~0.51-0.57 | Similar to manual implementation |

*Note: Performance varies with random data generation. Real market data may yield different results.*

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- numpy==1.24.3
- pandas==2.0.3
- matplotlib==3.7.2
- scikit-learn==1.3.0
- flask==2.3.3
- plotly==5.15.0
- seaborn==0.12.2

## 🏃‍♂️ Quick Start

### 1. Run Complete Analysis
```bash
python main.py
```
This will:
- Generate mock stock data
- Create technical indicators
- Train all models
- Generate comprehensive analysis report
- Save results and visualizations

### 2. Launch Web Interface
```bash
python app.py
```
Then open your browser to: `http://localhost:5000`

The web interface provides:
- **Interactive Predictions**: Input features and get real-time predictions
- **Model Comparison**: Visual comparison of model performance
- **Feature Analysis**: Feature importance rankings
- **Stock Simulation**: Generate and visualize mock stock data

### 3. Test Individual Components
```bash
# Test data generation
python data_generator.py

# Test neural network
python neural_network.py

# Test logistic regression
python logistic_regression.py
```

## 📁 Project Structure

```
NN Predictor/
├── data_generator.py          # Stock data generation and technical indicators
├── neural_network.py          # Neural network implementation from scratch
├── logistic_regression.py     # Logistic regression implementation
├── main.py                    # Main analysis script
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── templates/
│   └── index.html            # Web interface template
└── results_[timestamp]/      # Generated results (created after running main.py)
    ├── *.png                 # Visualization plots
    ├── *.csv                 # Data and predictions
    └── feature_importance_*.png
```

## 🧠 Neural Network Architecture

### Forward Propagation
```
Input (n_features) → Linear → ReLU → Linear → ReLU → Linear → Sigmoid → Output (1)
```

### Mathematical Implementation

**Linear Transformation:**
```
z = aW + b
```

**ReLU Activation:**
```
ReLU(x) = max(0, x)
```

**Sigmoid Activation:**
```
σ(x) = 1 / (1 + e^(-x))
```

**Binary Cross-Entropy Loss:**
```
L = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

### Backpropagation
The implementation includes manual computation of:
- **Weight Gradients**: `dW = a_prev^T * dz`
- **Bias Gradients**: `db = sum(dz, axis=0)`
- **Activation Derivatives**: ReLU and Sigmoid derivatives
- **Gradient Descent Updates**: `W = W - α * dW`

## 📈 Technical Indicators Explained

### RSI (Relative Strength Index)
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss
```
Measures momentum, values 0-100. Above 70 = overbought, below 30 = oversold.

### MACD (Moving Average Convergence Divergence)
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```
Trend-following momentum indicator.

### Price Volume Trend (PVT)
```
PVT = Previous PVT + (Volume * (Close - Previous Close) / Previous Close)
```
Combines price and volume to show money flow.

## 🔍 Feature Engineering

The system creates features with proper time-series handling:

1. **Lag Features**: Past returns (t-1, t-2, ..., t-10)
2. **Technical Indicators**: Calculated using historical windows
3. **Normalization**: Z-score normalization for price features
4. **Label Creation**: Binary labels with proper shifting to prevent data leakage

**Important**: Labels are shifted by -1 day to ensure no future information leaks into features.

## 📊 Model Evaluation

### Metrics Used
- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Ability to distinguish between classes
- **Confusion Matrix**: Detailed classification breakdown
- **Feature Importance**: Weight analysis from logistic regression

### Visualization
- Training loss/accuracy curves
- ROC curves
- Feature importance plots
- Model comparison charts
- Stock price simulations

## 🚀 Improvement Suggestions

### 1. **Regularization**
- Add L1/L2 regularization to prevent overfitting
- Implement dropout for neural networks

### 2. **Optimization**
- Replace SGD with Adam optimizer
- Add learning rate scheduling
- Implement batch normalization

### 3. **Feature Engineering**
- Add more technical indicators (Bollinger Bands, Stochastic)
- Include market sentiment indicators
- Add macroeconomic features

### 4. **Architecture**
- Experiment with LSTM/GRU for time series
- Try ensemble methods
- Implement attention mechanisms

### 5. **Data**
- Use real market data instead of simulated
- Include multiple stocks for diversification
- Add external data sources (news, economic indicators)

## 🌐 Web Interface Features

The Flask web application provides:

### 1. **Interactive Prediction**
- Select different models
- Input custom feature values
- Generate sample data automatically
- Real-time prediction results

### 2. **Model Comparison**
- Visual accuracy and ROC-AUC comparison
- Performance metrics for all models

### 3. **Feature Analysis**
- Feature importance visualization
- Top contributing features ranking

### 4. **Stock Simulation**
- Generate realistic stock price data
- View price charts with statistics
- Analyze returns and volatility

## 🔧 Customization

### Modify Neural Network Architecture
```python
# In neural_network.py
nn = FeedforwardNeuralNetwork(
    input_size=X_train.shape[1],
    hidden_sizes=[128, 64, 32],  # Modify layer sizes
    learning_rate=0.001,         # Adjust learning rate
    random_seed=42
)
```

### Add New Technical Indicators
```python
# In data_generator.py, add to TechnicalIndicators class
@staticmethod
def bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band
```

### Modify Data Generation
```python
# In data_generator.py
stock_data = generator.generate_stock_data(
    days=2000,              # More data
    initial_price=50.0,     # Different starting price
    volatility=0.03         # Higher volatility
)
```

## 📝 Mathematical Background

### Neural Network Forward Pass
For layer l:
1. **Linear**: `z^(l) = a^(l-1) * W^(l) + b^(l)`
2. **Activation**: `a^(l) = f(z^(l))`

### Backpropagation Gradients
1. **Output Layer**: `dz^(L) = (ŷ - y) / m`
2. **Hidden Layers**: `dz^(l) = da^(l) * f'(z^(l))`
3. **Weights**: `dW^(l) = (a^(l-1))^T * dz^(l)`
4. **Biases**: `db^(l) = sum(dz^(l), axis=0)`

### Logistic Regression
**Hypothesis**: `h(x) = σ(θ^T * x)`
**Cost**: `J(θ) = -1/m * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]`
**Gradient**: `∇J(θ) = 1/m * X^T * (h(x) - y)`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Technical indicators formulas from financial literature
- Neural network implementation inspired by Andrew Ng's courses
- Web interface design using Bootstrap and modern CSS

## 📞 Support

For questions or issues:
1. Check the code comments for implementation details
2. Review the mathematical explanations in docstrings
3. Run individual components to debug specific issues
4. Check the generated results folder for detailed analysis

---

**Note**: This project is for educational purposes. Stock market prediction is inherently uncertain, and this model should not be used for actual trading decisions without proper risk management and additional validation.
