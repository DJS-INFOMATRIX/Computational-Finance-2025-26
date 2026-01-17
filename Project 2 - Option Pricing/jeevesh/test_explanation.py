
from strategies.payoff_engine import OptionLeg, analyze_strategy
from explain.greeks_aggregator import aggregate_greeks
from explain.risk_classifier import classify_risk
from explain.english_explainer import generate_explanation
import numpy as np
import json

def run_test_case(name, strategy, S, T, r, sigma):
    print(f"\n=== TEST CASE: {name} ===")
    
    # 1. Math Analysis
    spot_range = np.linspace(S*0.8, S*1.2, 50)
    analysis = analyze_strategy(strategy, spot_range)
    
    # 2. Greeks
    net_greeks = aggregate_greeks(strategy, S, T, r, sigma)
    
    # 3. Risk Classifier
    risk = classify_risk(analysis, net_greeks)
    print(f"Risk Profile: {risk['category']}")
    print(f"Reason: {risk['reason']}")
    
    # 4. English Explanation
    explanation = generate_explanation(analysis, net_greeks)
    print("\n--- Generated Plain English ---")
    print(f"Summary: {explanation['summary']}")
    print(f"Context: {explanation['pnl_context']}")
    print(f"Time:    {explanation['greeks_context']['theta_text']}")
    print(f"Vol:     {explanation['greeks_context']['vega_text']}")

def test_explanation_layer():
    S, T, r, sigma = 100, 30/365, 0.05, 0.2
    
    # CASE A: Bull Call Spread (Debit, Defined Risk, Directional)
    # Buy 100 Call, Sell 105 Call
    leg1 = OptionLeg('call', 'long', 100, 2.50, 1) # made up prices
    leg2 = OptionLeg('call', 'short', 105, 0.80, 1)
    run_test_case("Bull Call Spread", [leg1, leg2], S, T, r, sigma)

    # CASE B: Iron Condor (Credit, Defined Risk, Neutral)
    # Short 90 Put, Long 85 Put, Short 110 Call, Long 115 Call
    ic_legs = [
        OptionLeg('put', 'short', 90, 1.20, 1),
        OptionLeg('put', 'long', 85, 0.40, 1),
        OptionLeg('call', 'short', 110, 1.10, 1),
        OptionLeg('call', 'long', 115, 0.35, 1)
    ]
    run_test_case("Iron Condor", ic_legs, S, T, r, sigma)

    # CASE C: Naked Short Call (Credit, Undefined Risk, Bearish)
    # Short 100 Call
    naked_leg = [OptionLeg('call', 'short', 100, 2.50, 1)]
    # Note: Payoff engine scan range might not catch infinite loss if not wide enough,
    # but let's see if risk classifier catches the high loss within range.
    spot_range_wide = np.linspace(80, 150, 100) 
    # adjust scan for this test inside the helper if needed, but generic is fine for demo
    run_test_case("Naked Short Call", naked_leg, S, T, r, sigma)

if __name__ == "__main__":
    test_explanation_layer()
