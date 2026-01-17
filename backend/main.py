from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Literal, Optional
import os
from .black_scholes import BlackScholes
from .strategy_engine import StrategyEngine, OptionLeg
from .simulator import MarketSimulator
from .report_generator import ReportGenerator

app = FastAPI(title="Glass Box Option Strategist", version="1.0.0")

# CORS Setup
origins = ["*"] # Allow all for local dev ease

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Frontend
# Assuming running from project root: uvicorn backend.main:app
if os.path.exists("frontend"):
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")
elif os.path.exists("../frontend"): # Fallback if running from inside backend
    app.mount("/app", StaticFiles(directory="../frontend", html=True), name="frontend")

@app.get("/")
def read_root():
    return {"message": "Glass Box API Ready. Go to /app/index.html to view the UI."}

class LegModel(BaseModel):
    type: Literal['call', 'put']
    side: Literal['long', 'short']
    K: float
    T: float
    quantity: int

class StrategyRequest(BaseModel):
    ticker: str
    underlying_price: float
    volatility: float
    risk_free_rate: float
    legs: List[LegModel]

class SimulationRequest(BaseModel):
    ticker: str
    legs: List[LegModel]
    days: int

@app.post("/api/analyze-strategy")
def analyze_strategy(request: StrategyRequest):
    try:
        engine = StrategyEngine(request.underlying_price, request.risk_free_rate, request.volatility)
        
        for leg in request.legs:
            engine.add_leg(OptionLeg(leg.type, leg.side, leg.K, leg.T, leg.quantity))
            
        greeks = engine.calculate_strategy_greeks()
        risk_analysis = engine.analyze_risk_profile()
        payoff = engine.generate_payoff_diagram(range_percent=0.2)
        
        return {
            "status": "success",
            "greeks": greeks,
            "risk_analysis": risk_analysis,
            "payoff_diagram": payoff
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from .stress_tester import StressTester

# ... (Previous imports)

class StressRequest(BaseModel):
    underlying_price: float
    volatility: float
    risk_free_rate: float
    legs: List[LegModel]

@app.post("/api/run-simulation")
def run_simulation(request: SimulationRequest):
    try:
        sim = MarketSimulator(request.ticker)
        # Convert pydantic models to dicts
        legs_data = [leg.dict() for leg in request.legs]
        # New run_simulation returns a dict with 'equity_curve', 'max_drawdown_pct', etc.
        results = sim.run_simulation(legs_data, days=request.days)
        return {"status": "success", "results": results}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stress-test")
def stress_test(request: StressRequest):
    try:
        engine = StrategyEngine(request.underlying_price, request.risk_free_rate, request.volatility)
        for leg in request.legs:
            engine.add_leg(OptionLeg(leg.type, leg.side, leg.K, leg.T, leg.quantity))
            
        tester = StressTester(engine)
        report = tester.generate_risk_report()
        
        return {
            "status": "success",
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-report")
def generate_report(request: StrategyRequest):
    try:
        # 1. Analyze Strategy
        engine = StrategyEngine(request.underlying_price, request.risk_free_rate, request.volatility)
        legs_data = [leg.dict() for leg in request.legs]
        for leg in request.legs:
            engine.add_leg(OptionLeg(leg.type, leg.side, leg.K, leg.T, leg.quantity))
        greeks = engine.calculate_strategy_greeks()
        
        # 2. Run Simulation (Optional for report, but let's include if ticker is valid)
        sim_results = []
        try:
            sim = MarketSimulator(request.ticker)
            sim_results = sim.run_simulation(legs_data, days=30)
        except:
            pass # Ignore simulation failures for report generation if offline/invalid ticker

        # 3. Generate PDF
        gen = ReportGenerator()
        pdf_bytes = gen.generate_strategy_report(
            strategy_name=f"{request.ticker} Custom Strategy",
            legs=legs_data,
            greeks=greeks,
            simulation_results=sim_results
        )
        
        return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=strategy_report.pdf"})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
