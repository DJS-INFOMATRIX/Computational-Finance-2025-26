# Glass Box Option Strategist

**Current Status:** V1.0 - Full Feature Release (Strategy Builder, Explanation Engine, Historic Simulation, Stress Testing)

The **Glass Box Option Strategist** is a locally-hosted web application designed to bring transparency to option pricing. Unlike black-box trading tools, this project emphasizes **explainable risk**, **open-source math**, and **historical verification**.

## Features
If opening directly (file://) causes CORS issues, run this in the root folder:
```bash
python -m http.server 3000
```
Then visit `http://localhost:3000/frontend/`

## Tech Stack
- **Backend**: Python, FastAPI, NumPy, SciPy
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), Vanilla JS
- **Data**: yfinance
- **Charts**: Plotly.js
