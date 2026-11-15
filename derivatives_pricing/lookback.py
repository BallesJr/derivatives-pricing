# Lookback Option Pricing
# Analytical and Monte Carlo pricing for European floating-strike lookback options
import numpy as np
from scipy.stats import norm


# ---Analytical pricing (Black-Scholes framework)---
# Analytical price of a floating-strike lookback call
def analytical_call(S: float, S_min: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(S - S_min, 0.0)

    k  = 2 * r / sigma**2
    d1 = (np.log(S / S_min) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S_min / S) + (0.5 * sigma**2 - r) * T) / (sigma * np.sqrt(T))

    term1 = S * norm.cdf(d1) - S_min * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))
    term2 = (sigma**2 / (2 * r)) * (
        -S * norm.cdf(d1)
        + S_min * np.exp(-r * T) * (S / S_min)**(1 - k) * norm.cdf(d2)
    )
    return term1 + term2


# Analytical price of a floating-strike lookback put (thesis formula, p.19)
def analytical_put(S: float, S_max: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(S_max - S, 0.0)

    k  = 2 * r / sigma**2
    d5 = (np.log(S_max / S) + (0.5 * sigma**2 - r) * T) / (sigma * np.sqrt(T))
    d6 = (np.log(S / S_max) + (0.5 * sigma**2 - r) * T) / (sigma * np.sqrt(T))
    d7 = (np.log(S / S_max) + (0.5 * sigma**2 + r) * T) / (sigma * np.sqrt(T))

    term1 = S_max * np.exp(-r * T) * norm.cdf(d5)
    term2 = S_max * np.exp(-r * T) * (1 / k) * (S / S_max)**(1 - k) * norm.cdf(d6)
    term3 = S * ((1 + 1 / k) * norm.cdf(d7) - 1)

    return term1 - term2 + term3


# ---Monte Carlo pricing (batch vectorisation)---
# Price a floating-strike lookback option via Monte Carlo
def mc_price(S0: float, r: float, sigma: float, T: float, option: str = "put",
             steps: int = 10_000, simulations: int = 100_000,
             batch_size: int = 2_000, seed: int = None) -> float:
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    discount = np.exp(-r * T)
    all_payoffs = []

    for start in range(0, simulations, batch_size):
        n = min(batch_size, simulations - start)
        z = np.random.standard_normal((n, steps))
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        cum_log = np.cumsum(log_returns, axis=1)
        # Include S0 at t=0 to capture it as potential min/max
        paths = S0 * np.exp(np.hstack([np.zeros((n, 1)), cum_log]))

        if option == "put":
            payoffs = np.max(paths, axis=1) - paths[:, -1]
        else:
            payoffs = paths[:, -1] - np.min(paths, axis=1)

        all_payoffs.append(np.maximum(payoffs, 0.0) * discount)

    return float(np.mean(np.concatenate(all_payoffs)))#