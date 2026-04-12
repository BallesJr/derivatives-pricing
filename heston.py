# Heston Option Pricing Modoel
# Analytical pricing and Greeks for European vanilla options
import numpy as np
from numba import njit
from scipy.integrate import quad


# Simulate Heston model core loop (Numba-accelereted)
@njit(cache=True)
def _simulate_heston(S: np.ndarray, v: np.ndarray, Z1: float, Z_S: float, kappa: float, theta: float, xi: float, r: float, dt: float, steps: int):
    for t in range(1, steps + 1):
        v_pos = np.maximum(v[t-1], 0)
        v[t] = v[t-1] + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * Z1[t-1]
        S[t] = S[t-1] * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * Z_S[t-1])
    return S, v

# Simulte various options by Heston model
def simulate(S0: float, v0: float, kappa: float, theta: float, xi: float, rho: float, r: float, T: float, steps: int, simulations: int, seed=None) -> tuple:
    S = np.zeros((steps + 1, simulations))
    v = np.zeros((steps + 1, simulations))
    S[0] = S0
    v[0] = v0
    if seed is not None:
        np.random.seed(seed)
    Z1 = np.random.standard_normal((steps, simulations))
    Z2 = np.random.standard_normal((steps, simulations))
    Z_S = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    dt = T / steps
    _simulate_heston(S, v, Z1, Z_S, kappa, theta, xi, r, dt, steps)
    return S, v

# Create paths and calculate the mean payoffs
def mc_call_price(S0: float, v0: float, kappa: float, theta: float, xi: float, rho: float, r: float, T: float, steps: int, simulations: int, K: float) -> float:
    S, v = simulate(S0, v0, kappa, theta, xi, rho, r, T, steps, simulations)
    payoffs = np.maximum(S[-1] - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

# ---Analytical pricing (characteristic function + numerical integration)---
# Heston characteristic function for probability P_j (j=1 or j=2)
def _characteristic_function(phi: float, S0: float, v0: float, kappa: float, theta: float, xi: float, rho: float, r: float, T: float, j: int) -> float:
    i = complex(0, 1)
    u = 0.5 if j == 1 else -0.5
    b = (kappa - rho * xi) if j == 1 else kappa
    a = kappa * theta

    d = np.sqrt((rho * xi * i * phi - b)**2 - xi**2 * (2 * u * i * phi - phi**2))
    g = (b - rho * xi * i * phi + d) / (b - rho * xi * i * phi - d)

    C = (r * i * phi * T
         + (a / xi**2) * ((b - rho * xi * i * phi + d) * T
                          - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))
    D = ((b - rho * xi * i * phi + d) / xi**2
         * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T))))

    return np.exp(C + D * v0 + i * phi * np.log(S0))

# Compute risk-neutral probability P_j via numerical integration
def _heston_P(j: int, S0: float, v0: float, kappa: float, theta: float, xi: float, rho: float, r: float, T: float, K: float) -> float:
    integrand = lambda phi: np.real(
        np.exp(-complex(0, 1) * phi * np.log(K))
        * _characteristic_function(phi, S0, v0, kappa, theta, xi, rho, r, T, j)
        / (complex(0, 1) * phi)
    )
    integral, _ = quad(integrand, 1e-6, 500.0)
    return 0.5 + integral / np.pi

# Heston analytical call price via the Heston formula: S0*P1 - K*e^{-rT}*P2
def analytical_call_price(S0: float, v0: float, kappa: float, theta: float, xi: float, rho: float, r: float, T: float, K: float) -> float:
    P1 = _heston_P(1, S0, v0, kappa, theta, xi, rho, r, T, K)
    P2 = _heston_P(2, S0, v0, kappa, theta, xi, rho, r, T, K)
    return S0 * P1 - K * np.exp(-r * T) * P2

# Heston analytical put price via put-call parity
def analytical_put_price(S0: float, v0: float, kappa: float, theta: float, xi: float, rho: float, r: float, T: float, K: float) -> float:
    return analytical_call_price(S0, v0, kappa, theta, xi, rho, r, T, K) - S0 + K * np.exp(-r * T)

# ---Greeks via central finite differences---
# Return the correct pricing function based on option type
def _price_fn(option: str):
    return analytical_call_price if option == "call" else analytical_put_price

# Delta (dV/dS)
def delta(S0: float, v0: float, kappa: float, theta: float, xi: float, rho: float, r: float, T: float, K: float, option: str = "call") -> float: 
    fn = _price_fn(option)
    eps = 0.01 * S0
    return (fn(S0 + eps, v0, kappa, theta, xi, rho, r, T, K) - fn(S0 - eps, v0, kappa, theta, xi, rho, r, T, K)) / (2 * eps)

# Gamma (d^2V/dS^2)
def gamma(S0: float, v0: float, kappa: float, theta: float, xi: float, rho: float, r: float, T: float, K: float, option: str = "call") -> float:
    fn = _price_fn(option)
    eps = 0.01 * S0
    price = fn(S0, v0, kappa, theta, xi, rho, r, T, K)
    price_up = fn(S0 + eps, v0, kappa, theta, xi, rho, r, T, K)
    price_dn = fn(S0 - eps, v0, kappa, theta, xi, rho, r, T, K)
    return (price_up - 2 * price + price_dn) / eps**2

# Vega (dV/d\sigma)
def vega(S0: float, v0: float, kappa: float, theta: float, xi: float, rho: float, r: float, T: float, K: float, option: str = "call") -> float:
    fn = _price_fn(option)
    eps = 0.01 * v0
    price_up = fn(S0, v0 + eps, kappa, theta, xi, rho, r, T, K)
    price_dn = fn(S0, v0 - eps, kappa, theta, xi, rho, r, T, K)
    return (price_up - price_dn) / (2 * eps) * 2 * np.sqrt(v0)

# Theta (dV/dt) per calendar day
def theta(S0: float, v0: float, kappa: float, theta_param: float, xi: float, rho: float, r: float, T: float, K: float, option: str = "call") -> float:
    fn = _price_fn(option)
    eps = 1 / 365
    if T <= eps:
        return np.nan
    price    = fn(S0, v0, kappa, theta_param, xi, rho, r, T, K)
    price_dn = fn(S0, v0, kappa, theta_param, xi, rho, r, T - eps, K)
    return (price_dn - price) / eps / 365

# Rho (dV/dr)
def rho(S0: float, v0: float, kappa: float, theta: float, xi: float, rho_param: float, r: float, T: float, K: float, option: str = "call") -> float:
    fn = _price_fn(option)
    eps = 1e-4
    price_up = fn(S0, v0, kappa, theta, xi, rho_param, r + eps, T, K)
    price_dn = fn(S0, v0, kappa, theta, xi, rho_param, r - eps, T, K)
    return (price_up - price_dn) / (2 * eps)

# Return all Greeks and price in a single dictionary
def greeks(S0: float, v0: float, kappa: float, theta: float, xi: float, rho: float, r: float, T: float, K: float, option: str = "call") -> dict:
    fn = _price_fn(option)
    return {
        "price": fn(S0, v0, kappa, theta, xi, rho, r, T, K),
        "delta": delta(S0, v0, kappa, theta, xi, rho, r, T, K, option),
        "gamma": gamma(S0, v0, kappa, theta, xi, rho, r, T, K, option),
        "vega":  vega(S0, v0, kappa, theta, xi, rho, r, T, K, option),
        "theta": theta(S0, v0, kappa, theta, xi, rho, r, T, K, option),
        "rho":   rho(S0, v0, kappa, theta, xi, rho, r, T, K, option),
    }