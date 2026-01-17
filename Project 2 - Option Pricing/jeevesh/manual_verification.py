
from pricing.black_scholes import call_price, put_price, decompose_option_price
from pricing.greeks import delta_call, gamma, vega, theta_call
from strategies.payoff_engine import OptionLeg, analyze_strategy
import numpy as np

def test_pricing():
    print("\n--- Testing Black-Scholes Pricing ---")
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    call = call_price(S, K, T, r, sigma)
    put = put_price(S, K, T, r, sigma)
    print(f"Call Price (S=100, K=100, T=1, r=5%, vol=20%): {call:.4f} (Expected ~10.45)")
    print(f"Put Price  (S=100, K=100, T=1, r=5%, vol=20%): {put:.4f} (Expected ~5.57)")
    
    intrinsic, time_val = decompose_option_price(call, 105, 100, 'call')
    print(f"Call ITM (S=105, K=100): Intrinsic={intrinsic}, Time={time_val:.4f}")

def test_greeks():
    print("\n--- Testing Greeks ---")
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    print(f"Delta Call: {delta_call(S, K, T, r, sigma):.4f} (Expected ~0.63)")
    print(f"Gamma:      {gamma(S, K, T, r, sigma):.4f} (Expected ~0.018)")
    print(f"Vega:       {vega(S, K, T, r, sigma):.4f} (Expected ~0.39)")
    print(f"Theta Call: {theta_call(S, K, T, r, sigma):.4f} (Expected ~-0.017)")

def test_strategy():
    print("\n--- Testing Strategy Engine (Bull Call Spread) ---")
    # Long 100 Call, Short 105 Call
    leg1 = OptionLeg('call', 'long', 100, 10.45, 1)
    leg2 = OptionLeg('call', 'short', 105, 7.00, 1) # made up price for test
    
    strategy = [leg1, leg2]
    spot_range = np.linspace(80, 120, 41)
    
    analysis = analyze_strategy(strategy, spot_range)
    print(f"Net Entry Cost: {analysis['net_entry_cost']:.2f}")
    print(f"Max Profit:     {analysis['max_profit']:.2f}")
    print(f"Max Loss:       {analysis['max_loss']:.2f}")
    print(f"Break-Evens:    {analysis['break_evens']}")

if __name__ == "__main__":
    test_pricing()
    test_greeks()
    test_strategy()
