"""
Option Strategy Builder

Combines multiple option legs into a complete strategy.
Calculates total cost, payoff diagrams, and risk metrics.
"""

import numpy as np
from black_scholes import black_scholes_call, black_scholes_put, calculate_greeks
from typing import List, Dict


class OptionLeg:
    """Represents a single option position in a strategy."""
    
    def __init__(self, option_type: str, position: str, strike: float, 
                 quantity: int, expiration_days: int):
        """
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        position : str
            'buy' or 'sell'
        strike : float
            Strike price
        quantity : int
            Number of contracts (1 contract = 100 shares)
        expiration_days : int
            Days until expiration
        """
        self.option_type = option_type.lower()
        self.position = position.lower()
        self.strike = strike
        self.quantity = quantity
        self.expiration_days = expiration_days
        self.price = 0
        self.details = {}
    
    def calculate_price(self, S: float, r: float, sigma: float):
        """Calculate the option price and store details."""
        T = self.expiration_days / 365.0
        
        if self.option_type == 'call':
            self.details = black_scholes_call(S, self.strike, T, r, sigma)
        else:
            self.details = black_scholes_put(S, self.strike, T, r, sigma)
        
        self.price = self.details['price']
        
        # Calculate greeks
        self.greeks = calculate_greeks(S, self.strike, T, r, sigma, self.option_type)
    
    def get_cost(self):
        """
        Get the initial cost/credit of this leg.
        Buying = negative (debit)
        Selling = positive (credit)
        """
        multiplier = 100  # 1 contract = 100 shares
        cost = self.price * self.quantity * multiplier
        
        if self.position == 'buy':
            return -cost  # You pay when buying
        else:
            return cost   # You receive when selling
    
    def payoff_at_expiration(self, stock_prices: np.ndarray) -> np.ndarray:
        """
        Calculate payoff at expiration for a range of stock prices.
        
        Returns:
        --------
        np.ndarray : Payoff values for each stock price
        """
        multiplier = 100
        
        if self.option_type == 'call':
            # Call payoff: max(S - K, 0)
            intrinsic = np.maximum(stock_prices - self.strike, 0)
        else:
            # Put payoff: max(K - S, 0)
            intrinsic = np.maximum(self.strike - stock_prices, 0)
        
        # If buying: gain from intrinsic, loss from premium paid
        # If selling: loss from intrinsic, gain from premium received
        if self.position == 'buy':
            payoff = (intrinsic - self.price) * self.quantity * multiplier
        else:
            payoff = (self.price - intrinsic) * self.quantity * multiplier
        
        return payoff
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'option_type': self.option_type,
            'position': self.position,
            'strike': self.strike,
            'quantity': self.quantity,
            'expiration_days': self.expiration_days,
            'price': round(self.price, 2),
            'cost': round(self.get_cost(), 2),
            'details': self.details,
            'greeks': self.greeks
        }


class OptionStrategy:
    """Combines multiple option legs into a strategy."""
    
    def __init__(self, stock_price: float, ticker: str = ""):
        """
        Parameters:
        -----------
        stock_price : float
            Current stock price
        ticker : str
            Stock ticker symbol (optional)
        """
        self.stock_price = stock_price
        self.ticker = ticker
        self.legs: List[OptionLeg] = []
    
    def add_leg(self, leg: OptionLeg):
        """Add an option leg to the strategy."""
        self.legs.append(leg)
    
    def calculate_strategy(self, r: float = 0.05, sigma: float = 0.20):
        """
        Calculate prices for all legs.
        
        Parameters:
        -----------
        r : float
            Risk-free rate (default 5%)
        sigma : float
            Volatility (default 20%)
        """
        for leg in self.legs:
            leg.calculate_price(self.stock_price, r, sigma)
    
    def get_total_cost(self) -> float:
        """
        Calculate total strategy cost.
        Negative = debit (you pay)
        Positive = credit (you receive)
        """
        return sum(leg.get_cost() for leg in self.legs)
    
    def get_payoff_diagram(self, price_range: float = 0.5) -> Dict:
        """
        Generate payoff diagram data.
        
        Parameters:
        -----------
        price_range : float
            Percentage range around current price (default 50%)
        
        Returns:
        --------
        dict : Contains stock prices and corresponding payoffs
        """
        # Generate range of stock prices
        min_price = self.stock_price * (1 - price_range)
        max_price = self.stock_price * (1 + price_range)
        stock_prices = np.linspace(min_price, max_price, 100)
        
        # Calculate total payoff across all legs
        total_payoff = np.zeros_like(stock_prices)
        for leg in self.legs:
            total_payoff += leg.payoff_at_expiration(stock_prices)
        
        return {
            'stock_prices': stock_prices.tolist(),
            'payoffs': total_payoff.tolist(),
            'breakeven_points': self._find_breakeven_points(stock_prices, total_payoff),
            'max_profit': round(float(np.max(total_payoff)), 2),
            'max_loss': round(float(np.min(total_payoff)), 2)
        }
    
    def _find_breakeven_points(self, prices: np.ndarray, payoffs: np.ndarray) -> List[float]:
        """Find stock prices where payoff crosses zero."""
        breakeven = []
        
        for i in range(len(payoffs) - 1):
            # Check if payoff crosses zero
            if payoffs[i] * payoffs[i + 1] < 0:
                # Linear interpolation to find exact crossing point
                price_be = prices[i] - payoffs[i] * (prices[i + 1] - prices[i]) / (payoffs[i + 1] - payoffs[i])
                breakeven.append(round(float(price_be), 2))
        
        return breakeven
    
    def get_risk_analysis(self) -> Dict:
        """
        Provide plain-English risk analysis.
        
        Returns:
        --------
        dict : Risk metrics and explanations
        """
        total_cost = self.get_total_cost()
        payoff_data = self.get_payoff_diagram()
        
        # Determine strategy type
        if total_cost < 0:
            cost_type = "debit"
            cost_desc = f"You paid ${abs(total_cost):.2f} to enter this strategy."
        elif total_cost > 0:
            cost_type = "credit"
            cost_desc = f"You received ${total_cost:.2f} to enter this strategy."
        else:
            cost_type = "zero-cost"
            cost_desc = "This strategy costs nothing to enter."
        
        # Max profit/loss analysis
        max_profit = payoff_data['max_profit']
        max_loss = payoff_data['max_loss']
        
        if max_profit == float('inf'):
            profit_desc = "Unlimited profit potential."
        else:
            profit_desc = f"Maximum profit is ${max_profit:.2f}."
        
        if abs(max_loss) == float('inf'):
            loss_desc = "WARNING: Unlimited loss potential!"
            risk_level = "EXTREME"
        elif abs(max_loss) > 10000:
            loss_desc = f"WARNING: Maximum loss is ${abs(max_loss):.2f}."
            risk_level = "HIGH"
        elif abs(max_loss) > 1000:
            loss_desc = f"Maximum loss is ${abs(max_loss):.2f}."
            risk_level = "MODERATE"
        else:
            loss_desc = f"Maximum loss is limited to ${abs(max_loss):.2f}."
            risk_level = "LOW"
        
        # Breakeven analysis
        breakeven_points = payoff_data['breakeven_points']
        if len(breakeven_points) == 0:
            be_desc = "No breakeven point - strategy is always profitable or always unprofitable."
        elif len(breakeven_points) == 1:
            be_desc = f"Breakeven point is ${breakeven_points[0]:.2f}."
        else:
            be_desc = f"Breakeven points are at ${', $'.join(map(str, breakeven_points))}."
        
        return {
            'total_cost': round(total_cost, 2),
            'cost_type': cost_type,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_level': risk_level,
            'breakeven_points': breakeven_points,
            'explanation': f"{cost_desc} {profit_desc} {loss_desc} {be_desc}",
            'recommendation': self._generate_recommendation(risk_level, cost_type)
        }
    
    def _generate_recommendation(self, risk_level: str, cost_type: str) -> str:
        """Generate educational recommendation based on risk."""
        if risk_level == "EXTREME":
            return "⚠️ EXTREME RISK: This strategy has unlimited loss potential. Only use if you fully understand the risks and have appropriate risk management in place."
        elif risk_level == "HIGH":
            return "⚠️ HIGH RISK: Significant losses are possible. Ensure you can afford the maximum loss before entering."
        elif risk_level == "MODERATE":
            return "Moderate risk strategy. Your maximum loss is defined, but still substantial."
        else:
            return "Lower risk strategy with limited downside. Your maximum loss is capped."
    
    def to_dict(self):
        """Convert entire strategy to dictionary."""
        return {
            'ticker': self.ticker,
            'stock_price': self.stock_price,
            'legs': [leg.to_dict() for leg in self.legs],
            'total_cost': round(self.get_total_cost(), 2),
            'risk_analysis': self.get_risk_analysis(),
            'payoff_diagram': self.get_payoff_diagram()
        }


def build_strategy_from_json(data: Dict) -> OptionStrategy:
    """
    Build a strategy from JSON request data.
    
    Expected format:
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
    strategy = OptionStrategy(
        stock_price=data['stock_price'],
        ticker=data.get('ticker', '')
    )
    
    for leg_data in data['legs']:
        leg = OptionLeg(
            option_type=leg_data['option_type'],
            position=leg_data['position'],
            strike=leg_data['strike'],
            quantity=leg_data['quantity'],
            expiration_days=leg_data['expiration_days']
        )
        strategy.add_leg(leg)
    
    # Calculate prices
    r = data.get('risk_free_rate', 0.05)
    sigma = data.get('volatility', 0.20)
    strategy.calculate_strategy(r, sigma)
    
    return strategy
