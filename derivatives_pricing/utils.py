# Quantitative Finance Utilities
# Risk metrics, performance analysis and market data helpers
import numpy as np

# ---Sharpe ratio---
# Compute annualised Sharpe ratio with 95% confidence interval
def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252,
                 risk_free_rate: float = 0.0) -> dict:
    rf = risk_free_rate / periods_per_year
    excess = returns - rf
    mu    = float(np.mean(excess))
    sigma = float(np.std(excess, ddof=1))

    if sigma == 0:
        return {"sharpe": np.nan, "se": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan}

    daily_sr = mu / sigma
    annual_sr = daily_sr * np.sqrt(periods_per_year)

    # Jobson-Korkie standard error
    T = len(returns)
    daily_se  = np.sqrt((1 + daily_sr**2 / 2) / T)
    annual_se = daily_se * np.sqrt(periods_per_year)

    return {
        "sharpe":   annual_sr,
        "se":       annual_se,
        "ci_lower": annual_sr - 1.96 * annual_se,
        "ci_upper": annual_sr + 1.96 * annual_se,
    }

# ---Drawdown---
# Compute percentage drawdown from peak at each point in time
def drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity_curve)
    return (equity_curve - peak) / peak

# Maximum peak-to-trough decline
def max_drawdown(equity_curve: np.ndarray) -> float:
    return float(np.min(drawdown_series(equity_curve)))

# ---Equity curve---
# Build cumulative equity curve from arithmetic returns
def equity_curve(returns: np.ndarray, initial_capital: float = 10_000.0) -> np.ndarray:
    return initial_capital * np.cumprod(1 + returns)

# ---Market data---
# Download historical returns from Yahoo Finance
def fetch_returns(ticker: str, period: str = "1y", log: bool = False) -> tuple:
    import yfinance as yf
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    close = data["Close"].squeeze()
    if log:
        returns = np.log(close / close.shift(1)).dropna().values
    else:
        returns = close.pct_change().dropna().values
    return returns, data