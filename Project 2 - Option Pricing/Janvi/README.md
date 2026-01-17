# The Glass Box Option Strategist

A transparent, educational option pricing and strategy website that demystifies option trading through clear explanations and interactive tools.

## Architecture

This project follows a clean separation between frontend and backend:

- **Frontend**: HTML, CSS, JavaScript served as static files
- **Backend**: Python Flask API server handling calculations and data
- **Communication**: JSON REST API

## Project Structure

```
/cf
  /frontend
    index.html              # Strategy Builder page
    explain.html            # Black-Scholes explanation page
    time_machine.html       # Historical simulator
    stress_test.html        # What-if stress tester
    /css
      styles.css           # Global styles
    /js
      strategy.js          # Strategy builder logic
      explain.js           # Explanation page logic
      time_machine.js      # Time machine logic
      stress_test.js       # Stress test logic
      utils.js            # Shared utilities
  /backend
    app.py                # Flask API server
    black_scholes.py      # Black-Scholes pricing engine
    strategy.py           # Strategy builder logic
    simulator.py          # Historical simulation
    stress_test.py        # Stress testing calculations
    requirements.txt      # Python dependencies
```

## Features

### 1. Strategy Builder
Build multi-leg option strategies with:
- Multiple call/put options
- Buy/sell positions
- Real-time pricing via Black-Scholes
- Payoff diagrams
- Plain-English risk explanations

### 2. Black-Scholes Pricing Engine
Transparent option pricing showing:
- d1 and d2 calculations
- Probability values N(d1) and N(d2)
- Step-by-step math

### 3. "Explain" Page (Glass Box Feature)
Educational content covering:
- Black-Scholes formula breakdown
- Input parameter explanations
- Intrinsic vs time value
- Greeks and risk metrics

### 4. "Time Machine" Historical Simulator
Test strategies on historical data:
- Select any past date
- See how your strategy would have performed
- Track profit/loss over time
- Maximum drawdown analysis

### 5. "What-If" Stress Tester
Interactive scenario analysis:
- Volatility shocks
- Price movements
- Time decay simulation
- Best/worst case scenarios

## Installation & Setup

### 1. Install Python Dependencies

```bash
cd /Users/janvishah/cf/backend
pip3 install -r requirements.txt
```

### 2. Start the Backend Server

```bash
cd /Users/janvishah/cf/backend
python3 app.py
```

The API server will start on `http://localhost:5001`

You should see output like:
```
============================================================
Glass Box Option Strategist - Backend Server
============================================================

API Server starting on http://localhost:5001
...
 * Running on http://0.0.0.0:5001
```

### 3. Open the Frontend

**Option A: Direct File Access**
Simply open `/Users/janvishah/cf/frontend/index.html` in your web browser.

**Option B: Local HTTP Server** (Recommended)
In a new terminal window:

```bash
cd /Users/janvishah/cf/frontend
python3 -m http.server 8080
```

Then navigate to `http://localhost:8080` in your browser.

## Usage Guide

### Strategy Builder
1. Enter stock ticker and current price
2. Add option legs (calls/puts, buy/sell)
3. Set strike prices and expiration
4. View pricing breakdown and payoff diagram
5. Read plain-English risk assessment

### Explain Page
- Learn how Black-Scholes works
- Understand each input parameter
- See intermediate calculations
- Build intuition about option pricing

### Time Machine
1. Select a stock and historical date
2. Build your strategy
3. See how it would have performed
4. Analyze results with clear explanations

### Stress Test
1. Build a strategy
2. Use sliders to test scenarios
3. See real-time price updates
4. Identify high-risk situations

## Design Principles

1. **Transparency**: Every calculation is explained
2. **Education**: Plain-English commentary throughout
3. **Clarity**: Simple UI prioritizing understanding
4. **Trust**: No black boxes - all math is visible

## Technical Notes

- **Black-Scholes Model**: Standard European option pricing
- **Historical Data**: Retrieved via yfinance (Yahoo Finance)
- **Risk-Free Rate**: Default 5% (adjustable)
- **Volatility**: User-specified or historical calculation

## Disclaimer

This is an educational tool for learning about options. Not financial advice. Options trading involves significant risk.

## License

Educational project - MIT License
