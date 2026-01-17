# Quick Start Guide

## 🚀 Getting Started with Glass Box Option Strategist

### Step 1: Start the Backend Server

Open a terminal and run:

```bash
cd /Users/janvishah/cf/backend
python3 app.py
```

Keep this terminal window open. You should see:
```
API Server starting on http://localhost:5001
 * Running on http://0.0.0.0:5001
```

### Step 2: Open the Frontend

**Easy Way:**
1. Open your web browser (Chrome, Firefox, Safari)
2. Press Cmd+O (or File → Open File)
3. Navigate to: `/Users/janvishah/cf/frontend/index.html`
4. Click Open

**Better Way (Recommended):**
Open a NEW terminal window and run:
```bash
cd /Users/janvishah/cf/frontend
python3 -m http.server 8080
```

Then open your browser and go to: `http://localhost:8080`

### Step 3: Start Using the Application

You should now see the Glass Box Option Strategist interface!

Try these examples:

#### Example 1: Price a Simple Call Option
1. Click "How Black-Scholes Works" in the navigation
2. Scroll down to the "Interactive Example" section
3. Leave the default values (Stock Price $100, Strike $105, 30 days)
4. Click "Calculate Option Price"
5. See the complete breakdown with explanations!

#### Example 2: Build a Bull Call Spread
1. Go to "Strategy Builder" (main page)
2. Set Stock Price: 100
3. Add first leg:
   - Type: Call
   - Position: Buy
   - Strike: 100
   - Quantity: 1
   - Days: 30
4. Click "Add Leg to Strategy"
5. Add second leg:
   - Type: Call
   - Position: Sell
   - Strike: 110
   - Quantity: 1
   - Days: 30
6. Click "Add Leg to Strategy"
7. Click "Calculate Strategy"
8. See the payoff diagram and risk analysis!

#### Example 3: Test on Historical Data
1. Click "Time Machine"
2. Enter ticker: AAPL
3. Click "Check Data Availability"
4. Choose a start date (e.g., 1 year ago)
5. Add option legs (same as above)
6. Click "Run Historical Simulation"
7. See how your strategy would have performed!

#### Example 4: Stress Test a Strategy
1. Click "Stress Test"
2. Build a strategy (add some option legs)
3. Click "Run Stress Tests"
4. Use the interactive sliders to test different scenarios
5. See worst-case and best-case outcomes!

## Troubleshooting

**Port 5000 already in use?**
The app now uses port 5001. If that's also in use, edit `/Users/janvishah/cf/backend/app.py` and change the port number at the bottom.

**Module not found errors?**
Run: `cd /Users/janvishah/cf/backend && pip3 install -r requirements.txt`

**Frontend can't connect to backend?**
Make sure the backend server is running in a terminal window. You should see "Running on http://0.0.0.0:5001"

**Charts not displaying?**
Make sure you're accessing the frontend through `http://localhost:8080` (using the Python HTTP server) rather than opening the HTML file directly.

## What Makes This "Glass Box"?

Unlike most option calculators:
- ✅ Every calculation shows intermediate steps (d1, d2, probabilities)
- ✅ Plain-English explanations for every result
- ✅ Complete formula breakdowns
- ✅ Transparent risk metrics
- ✅ No black boxes - you understand exactly how prices are calculated

## Features Overview

### 1. Strategy Builder
Build multi-leg option strategies and see:
- Individual option prices with Black-Scholes breakdown
- Total strategy cost (debit/credit)
- Payoff diagrams
- Risk analysis with plain-English explanations
- Greeks for each leg

### 2. How Black-Scholes Works
Educational page explaining:
- The Black-Scholes formula step-by-step
- What each variable means
- d1 and d2 calculations
- Intrinsic vs time value
- Why volatility affects prices
- Interactive calculator with full transparency

### 3. Time Machine
Historical backtesting:
- Test strategies on real historical data
- See actual performance over time
- Understand how strategies behave in real markets
- Maximum profit, drawdown, and final P&L

### 4. Stress Test
"What-if" analysis:
- Price movement scenarios
- Volatility shocks
- Time decay simulation
- Interactive sliders for real-time testing
- Comprehensive risk summary

## Next Steps

- Experiment with different strategies (long call, put spreads, iron condors)
- Test your strategies on historical data for different stocks
- Use stress testing to understand worst-case scenarios
- Read the explanations to build intuition about option pricing

## Remember

This is an educational tool. It helps you understand option pricing and strategy risk. Always do your own research and consider consulting a financial advisor before real trading.

Enjoy learning about options! 📊
