# Black-Scholes Option Pricing Model
# Analytical pricing and Greeks for European vanilla options
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# ---Core Helper---
# Compute d1 and d2 from the Black-Scholes formula
def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

# ---Pricing---
# Black-Scholes price for a European call option
def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Black-Scholes price for a European put option (via put-call parity)
def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    return call_price(S, K, T, r, sigma) - S + K * np.exp(-r * T)

# ---Greeks---
# Delta (dV/dS): sensitivity to the underlying price
def delta(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> float:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0

# Gamma (d^2V/dS^2): identical for calls and puts
def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Vega (dV/d\sigma): identical for calls and puts
def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

# Theta (dV/dt): per calendar day
def theta(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> float:
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option == "call":
        return (common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    return (common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0

# Rho (dV/dr): sensitivity to the risk-free rate
def rho(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> float:
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

# Return all Greeks and the option price in a single dictionary
def greeks(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> dict[str, float]:
    price_fn = call_price if option == "call" else put_price
    return {
        "price": price_fn(S, K, T, r, sigma),
        "delta": delta(S, K, T, r, sigma, option),
        "gamma": gamma(S, K, T, r, sigma),
        "vega":  vega(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option),
        "rho":   rho(S, K, T, r, sigma, option),
    }


# ---Implied volatility---
# Implied volatility via Brent's method. Returns np.nan if not found
def implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option: str = "call") -> float:
    price_fn = call_price if option == "call" else put_price
    intrinsic = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
    if market_price < intrinsic - 1e-6 or market_price < 1e-10:
        return np.nan
    try:
        return brentq(lambda sigma: price_fn(S, K, T, r, sigma) - market_price, 1e-6, 10.0, xtol=1e-10)
    except ValueError:
        return np.nan