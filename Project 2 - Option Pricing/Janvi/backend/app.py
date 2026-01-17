"""
Flask API Server for Glass Box Option Strategist

Provides REST API endpoints for:
- Option pricing
- Strategy building
- Historical simulation
- Stress testing
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

from black_scholes import black_scholes_call, black_scholes_put, calculate_greeks
from strategy import build_strategy_from_json
from simulator import simulate_strategy_over_time, get_available_date_range, calculate_historical_volatility, fetch_historical_data
from stress_test import (
    stress_test_price_change, 
    stress_test_volatility_change, 
    stress_test_time_decay,
    comprehensive_stress_test
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Serve frontend files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

@app.route('/')
def serve_index():
    """Serve the main page."""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (HTML, CSS, JS)."""
    return send_from_directory(FRONTEND_DIR, path)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Glass Box Option Strategist API is running'
    })


@app.route('/api/price/call', methods=['POST'])
def price_call():
    """
    Price a single call option.
    
    Request body:
    {
        "stock_price": 100,
        "strike": 105,
        "expiration_days": 30,
        "risk_free_rate": 0.05,
        "volatility": 0.20
    }
    """
    try:
        data = request.json
        
        S = data['stock_price']
        K = data['strike']
        T = data['expiration_days'] / 365.0
        r = data.get('risk_free_rate', 0.05)
        sigma = data.get('volatility', 0.20)
        
        result = black_scholes_call(S, K, T, r, sigma)
        greeks = calculate_greeks(S, K, T, r, sigma, 'call')
        
        result['greeks'] = greeks
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/price/put', methods=['POST'])
def price_put():
    """
    Price a single put option.
    
    Request body: Same as /api/price/call
    """
    try:
        data = request.json
        
        S = data['stock_price']
        K = data['strike']
        T = data['expiration_days'] / 365.0
        r = data.get('risk_free_rate', 0.05)
        sigma = data.get('volatility', 0.20)
        
        result = black_scholes_put(S, K, T, r, sigma)
        greeks = calculate_greeks(S, K, T, r, sigma, 'put')
        
        result['greeks'] = greeks
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/strategy/build', methods=['POST'])
def build_strategy():
    """
    Build and price a complete option strategy.
    
    Request body:
    {
        "stock_price": 100,
        "ticker": "AAPL",
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "legs": [
            {
                "option_type": "call",
                "position": "buy",
                "strike": 105,
                "quantity": 1,
                "expiration_days": 30
            }
        ]
    }
    """
    try:
        data = request.json
        strategy = build_strategy_from_json(data)
        
        return jsonify(strategy.to_dict())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/simulate/historical', methods=['POST'])
def simulate_historical():
    """
    Simulate strategy on historical data.
    
    Request body:
    {
        "ticker": "AAPL",
        "start_date": "2023-01-01",
        "end_date": "2023-06-01",  // optional
        "strategy": {
            "stock_price": 150,  // Will be overridden with actual historical price
            "risk_free_rate": 0.05,
            "volatility": 0.25,
            "legs": [...]
        }
    }
    """
    try:
        data = request.json
        
        ticker = data['ticker']
        start_date = data['start_date']
        end_date = data.get('end_date', None)
        strategy_config = data['strategy']
        
        result = simulate_strategy_over_time(ticker, start_date, strategy_config, end_date)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/data/date-range/<ticker>', methods=['GET'])
def get_date_range(ticker):
    """Get available date range for a ticker."""
    try:
        result = get_available_date_range(ticker)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/data/volatility', methods=['POST'])
def calculate_volatility():
    """
    Calculate historical volatility for a stock.
    
    Request body:
    {
        "ticker": "AAPL",
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "window": 30  // optional, default 30 days
    }
    """
    try:
        data = request.json
        
        ticker = data['ticker']
        start_date = data['start_date']
        end_date = data.get('end_date', None)
        window = data.get('window', 30)
        
        hist_data = fetch_historical_data(ticker, start_date, end_date)
        volatility = calculate_historical_volatility(hist_data['Close'], window)
        
        return jsonify({
            'ticker': ticker,
            'volatility': round(volatility, 4),
            'annualized_pct': round(volatility * 100, 2),
            'window_days': window
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/stress/price', methods=['POST'])
def stress_price():
    """
    Stress test strategy against price changes.
    
    Request body:
    {
        "strategy": {...},
        "price_changes": [-0.20, -0.10, 0, 0.10, 0.20]
    }
    """
    try:
        data = request.json
        
        strategy_config = data['strategy']
        price_changes = data.get('price_changes', [-0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30])
        
        result = stress_test_price_change(strategy_config, price_changes)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/stress/volatility', methods=['POST'])
def stress_volatility():
    """
    Stress test strategy against volatility changes.
    
    Request body:
    {
        "strategy": {...},
        "vol_changes": [-0.50, -0.25, 0, 0.25, 0.50]
    }
    """
    try:
        data = request.json
        
        strategy_config = data['strategy']
        vol_changes = data.get('vol_changes', [-0.50, -0.25, 0, 0.25, 0.50])
        
        result = stress_test_volatility_change(strategy_config, vol_changes)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/stress/time', methods=['POST'])
def stress_time():
    """
    Stress test strategy against time decay.
    
    Request body:
    {
        "strategy": {...},
        "days_forward": 30
    }
    """
    try:
        data = request.json
        
        strategy_config = data['strategy']
        days_forward = data.get('days_forward', 30)
        
        result = stress_test_time_decay(strategy_config, days_forward)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/stress/comprehensive', methods=['POST'])
def stress_comprehensive():
    """
    Run all stress tests.
    
    Request body:
    {
        "strategy": {...}
    }
    """
    try:
        data = request.json
        
        strategy_config = data['strategy']
        
        result = comprehensive_stress_test(strategy_config)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Glass Box Option Strategist - Backend Server")
    print("=" * 60)
    print("\nAPI Server starting on http://localhost:5001")
    print("\nAvailable endpoints:")
    print("  - GET  /                           (Frontend)")
    print("  - GET  /api/health                 (Health check)")
    print("  - POST /api/price/call             (Price call option)")
    print("  - POST /api/price/put              (Price put option)")
    print("  - POST /api/strategy/build         (Build strategy)")
    print("  - POST /api/simulate/historical    (Historical simulation)")
    print("  - GET  /api/data/date-range/<ticker>")
    print("  - POST /api/data/volatility")
    print("  - POST /api/stress/price")
    print("  - POST /api/stress/volatility")
    print("  - POST /api/stress/time")
    print("  - POST /api/stress/comprehensive")
    print("\n" + "=" * 60)
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
