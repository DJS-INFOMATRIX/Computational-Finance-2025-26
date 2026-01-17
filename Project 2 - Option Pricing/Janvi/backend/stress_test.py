"""
Stress Testing Module - "What-If" Analysis

Allows users to test how option strategies perform under various scenarios:
- Stock price changes
- Volatility shocks
- Time decay
"""

import numpy as np
from typing import Dict, List
from black_scholes import black_scholes_call, black_scholes_put


def stress_test_price_change(strategy_config: Dict, price_changes: List[float]) -> Dict:
    """
    Test strategy performance across different stock price scenarios.
    
    Parameters:
    -----------
    strategy_config : dict
        Strategy configuration
    price_changes : list
        List of percentage changes (e.g., [-0.20, -0.10, 0, 0.10, 0.20])
    
    Returns:
    --------
    dict : Strategy values at each price level
    """
    base_price = strategy_config['stock_price']
    legs = strategy_config['legs']
    r = strategy_config.get('risk_free_rate', 0.05)
    sigma = strategy_config.get('volatility', 0.20)
    
    results = {
        'price_changes': price_changes,
        'stock_prices': [],
        'strategy_values': [],
        'pnl': []
    }
    
    # Calculate initial cost
    initial_cost = 0
    for leg in legs:
        T = leg['expiration_days'] / 365.0
        if leg['option_type'].lower() == 'call':
            option_data = black_scholes_call(base_price, leg['strike'], T, r, sigma)
        else:
            option_data = black_scholes_put(base_price, leg['strike'], T, r, sigma)
        
        multiplier = 100 * leg['quantity']
        if leg['position'].lower() == 'buy':
            initial_cost -= option_data['price'] * multiplier
        else:
            initial_cost += option_data['price'] * multiplier
    
    # Test each price scenario
    for pct_change in price_changes:
        new_price = base_price * (1 + pct_change)
        
        # Calculate strategy value at new price
        strategy_value = 0
        for leg in legs:
            T = leg['expiration_days'] / 365.0
            
            if leg['option_type'].lower() == 'call':
                option_data = black_scholes_call(new_price, leg['strike'], T, r, sigma)
            else:
                option_data = black_scholes_put(new_price, leg['strike'], T, r, sigma)
            
            multiplier = 100 * leg['quantity']
            if leg['position'].lower() == 'buy':
                strategy_value += option_data['price'] * multiplier
            else:
                strategy_value -= option_data['price'] * multiplier
        
        pnl = strategy_value + initial_cost
        
        results['stock_prices'].append(round(new_price, 2))
        results['strategy_values'].append(round(strategy_value, 2))
        results['pnl'].append(round(pnl, 2))
    
    results['initial_cost'] = round(initial_cost, 2)
    results['explanation'] = generate_price_stress_explanation(results)
    
    return results


def stress_test_volatility_change(strategy_config: Dict, vol_changes: List[float]) -> Dict:
    """
    Test strategy performance across different volatility scenarios.
    
    Parameters:
    -----------
    strategy_config : dict
        Strategy configuration
    vol_changes : list
        List of percentage changes to volatility (e.g., [-0.50, -0.25, 0, 0.25, 0.50])
    
    Returns:
    --------
    dict : Strategy values at each volatility level
    """
    stock_price = strategy_config['stock_price']
    legs = strategy_config['legs']
    r = strategy_config.get('risk_free_rate', 0.05)
    base_sigma = strategy_config.get('volatility', 0.20)
    
    results = {
        'vol_changes': vol_changes,
        'volatilities': [],
        'strategy_values': [],
        'pnl': []
    }
    
    # Calculate initial cost at base volatility
    initial_cost = 0
    for leg in legs:
        T = leg['expiration_days'] / 365.0
        if leg['option_type'].lower() == 'call':
            option_data = black_scholes_call(stock_price, leg['strike'], T, r, base_sigma)
        else:
            option_data = black_scholes_put(stock_price, leg['strike'], T, r, base_sigma)
        
        multiplier = 100 * leg['quantity']
        if leg['position'].lower() == 'buy':
            initial_cost -= option_data['price'] * multiplier
        else:
            initial_cost += option_data['price'] * multiplier
    
    # Test each volatility scenario
    for vol_change in vol_changes:
        new_sigma = base_sigma * (1 + vol_change)
        new_sigma = max(0.01, new_sigma)  # Ensure volatility stays positive
        
        # Calculate strategy value at new volatility
        strategy_value = 0
        for leg in legs:
            T = leg['expiration_days'] / 365.0
            
            if leg['option_type'].lower() == 'call':
                option_data = black_scholes_call(stock_price, leg['strike'], T, r, new_sigma)
            else:
                option_data = black_scholes_put(stock_price, leg['strike'], T, r, new_sigma)
            
            multiplier = 100 * leg['quantity']
            if leg['position'].lower() == 'buy':
                strategy_value += option_data['price'] * multiplier
            else:
                strategy_value -= option_data['price'] * multiplier
        
        pnl = strategy_value + initial_cost
        
        results['volatilities'].append(round(new_sigma * 100, 2))  # Convert to percentage
        results['strategy_values'].append(round(strategy_value, 2))
        results['pnl'].append(round(pnl, 2))
    
    results['initial_cost'] = round(initial_cost, 2)
    results['base_volatility'] = round(base_sigma * 100, 2)
    results['explanation'] = generate_vol_stress_explanation(results)
    
    return results


def stress_test_time_decay(strategy_config: Dict, days_forward: int = 30) -> Dict:
    """
    Test how strategy value changes over time (theta decay).
    
    Parameters:
    -----------
    strategy_config : dict
        Strategy configuration
    days_forward : int
        Number of days to simulate forward
    
    Returns:
    --------
    dict : Strategy values over time
    """
    stock_price = strategy_config['stock_price']
    legs = strategy_config['legs']
    r = strategy_config.get('risk_free_rate', 0.05)
    sigma = strategy_config.get('volatility', 0.20)
    
    # Find minimum expiration to limit simulation
    max_days = min(days_forward, min(leg['expiration_days'] for leg in legs))
    
    results = {
        'days': [],
        'strategy_values': [],
        'pnl': [],
        'time_decay_per_day': []
    }
    
    # Calculate initial cost
    initial_cost = 0
    for leg in legs:
        T = leg['expiration_days'] / 365.0
        if leg['option_type'].lower() == 'call':
            option_data = black_scholes_call(stock_price, leg['strike'], T, r, sigma)
        else:
            option_data = black_scholes_put(stock_price, leg['strike'], T, r, sigma)
        
        multiplier = 100 * leg['quantity']
        if leg['position'].lower() == 'buy':
            initial_cost -= option_data['price'] * multiplier
        else:
            initial_cost += option_data['price'] * multiplier
    
    previous_value = None
    
    # Simulate each day
    for day in range(0, max_days + 1):
        # Calculate strategy value
        strategy_value = 0
        for leg in legs:
            days_remaining = leg['expiration_days'] - day
            T = max(0, days_remaining) / 365.0
            
            if T <= 0:
                # Calculate intrinsic value at expiration
                if leg['option_type'].lower() == 'call':
                    intrinsic = max(stock_price - leg['strike'], 0)
                else:
                    intrinsic = max(leg['strike'] - stock_price, 0)
                option_value = intrinsic
            else:
                if leg['option_type'].lower() == 'call':
                    option_data = black_scholes_call(stock_price, leg['strike'], T, r, sigma)
                else:
                    option_data = black_scholes_put(stock_price, leg['strike'], T, r, sigma)
                option_value = option_data['price']
            
            multiplier = 100 * leg['quantity']
            if leg['position'].lower() == 'buy':
                strategy_value += option_value * multiplier
            else:
                strategy_value -= option_value * multiplier
        
        pnl = strategy_value + initial_cost
        
        # Calculate daily decay
        if previous_value is not None:
            decay = strategy_value - previous_value
        else:
            decay = 0
        
        results['days'].append(day)
        results['strategy_values'].append(round(strategy_value, 2))
        results['pnl'].append(round(pnl, 2))
        results['time_decay_per_day'].append(round(decay, 2))
        
        previous_value = strategy_value
    
    results['initial_cost'] = round(initial_cost, 2)
    results['total_decay'] = round(results['strategy_values'][-1] - results['strategy_values'][0], 2)
    results['explanation'] = generate_time_decay_explanation(results)
    
    return results


def comprehensive_stress_test(strategy_config: Dict) -> Dict:
    """
    Run all stress tests and provide comprehensive analysis.
    
    Returns:
    --------
    dict : All stress test results with risk assessment
    """
    # Price stress test
    price_changes = [-0.30, -0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.30]
    price_stress = stress_test_price_change(strategy_config, price_changes)
    
    # Volatility stress test
    vol_changes = [-0.50, -0.25, 0, 0.25, 0.50]
    vol_stress = stress_test_volatility_change(strategy_config, vol_changes)
    
    # Time decay test
    time_stress = stress_test_time_decay(strategy_config, days_forward=30)
    
    # Identify worst-case and best-case scenarios
    worst_price_pnl = min(price_stress['pnl'])
    best_price_pnl = max(price_stress['pnl'])
    worst_vol_pnl = min(vol_stress['pnl'])
    best_vol_pnl = max(vol_stress['pnl'])
    
    # Overall risk assessment
    risk_summary = {
        'worst_case_price_shock': round(worst_price_pnl, 2),
        'best_case_price_shock': round(best_price_pnl, 2),
        'worst_case_vol_shock': round(worst_vol_pnl, 2),
        'best_case_vol_shock': round(best_vol_pnl, 2),
        'total_time_decay': time_stress['total_decay'],
        'risk_level': assess_overall_risk(worst_price_pnl, worst_vol_pnl)
    }
    
    return {
        'price_stress': price_stress,
        'volatility_stress': vol_stress,
        'time_decay': time_stress,
        'risk_summary': risk_summary,
        'overall_explanation': generate_comprehensive_explanation(risk_summary)
    }


def generate_price_stress_explanation(results: Dict) -> str:
    """Generate explanation for price stress test."""
    worst_pnl = min(results['pnl'])
    best_pnl = max(results['pnl'])
    
    explanation = f"If the stock price changes, your P&L ranges from ${worst_pnl:.2f} "
    explanation += f"(worst case) to ${best_pnl:.2f} (best case). "
    
    if worst_pnl < -1000:
        explanation += "⚠️ Large price movements could result in significant losses."
    
    return explanation


def generate_vol_stress_explanation(results: Dict) -> str:
    """Generate explanation for volatility stress test."""
    worst_pnl = min(results['pnl'])
    best_pnl = max(results['pnl'])
    
    explanation = f"Volatility changes affect your position. "
    explanation += f"Your P&L could range from ${worst_pnl:.2f} to ${best_pnl:.2f}. "
    
    if abs(best_pnl - worst_pnl) > 1000:
        explanation += "This strategy is highly sensitive to volatility changes."
    
    return explanation


def generate_time_decay_explanation(results: Dict) -> str:
    """Generate explanation for time decay test."""
    total_decay = results['total_decay']
    avg_daily = total_decay / len(results['days']) if len(results['days']) > 0 else 0
    
    explanation = f"Over time, this strategy loses approximately ${avg_daily:.2f} per day "
    explanation += f"to time decay (theta). Total decay over the period: ${total_decay:.2f}. "
    
    if total_decay < -500:
        explanation += "⚠️ Significant time decay - this position loses value quickly."
    
    return explanation


def assess_overall_risk(worst_price: float, worst_vol: float) -> str:
    """Assess overall risk level."""
    worst_overall = min(worst_price, worst_vol)
    
    if worst_overall < -10000:
        return "EXTREME"
    elif worst_overall < -5000:
        return "HIGH"
    elif worst_overall < -1000:
        return "MODERATE"
    else:
        return "LOW"


def generate_comprehensive_explanation(risk_summary: Dict) -> str:
    """Generate comprehensive risk explanation."""
    explanation = "COMPREHENSIVE RISK ANALYSIS:\n\n"
    
    explanation += f"Risk Level: {risk_summary['risk_level']}\n\n"
    
    explanation += f"Worst-case from price shock: ${risk_summary['worst_case_price_shock']:.2f}\n"
    explanation += f"Best-case from price shock: ${risk_summary['best_case_price_shock']:.2f}\n\n"
    
    explanation += f"Worst-case from volatility shock: ${risk_summary['worst_case_vol_shock']:.2f}\n"
    explanation += f"Best-case from volatility shock: ${risk_summary['best_case_vol_shock']:.2f}\n\n"
    
    explanation += f"Time decay impact: ${risk_summary['total_time_decay']:.2f}\n\n"
    
    if risk_summary['risk_level'] in ['EXTREME', 'HIGH']:
        explanation += "⚠️ WARNING: This strategy has significant downside risk. "
        explanation += "Ensure you understand the potential losses before proceeding."
    
    return explanation
