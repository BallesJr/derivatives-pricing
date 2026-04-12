"""
derivatives_pricing
====================
A professional Python library for derivatives pricing and quantitative risk analytics

Modules
-------
black_scholes : BS analytical pricing + full Greeks + implied volatility
heston        : Heston SV model — MC, analytical pricing, Greeks, calibration
lookback      : Floating-strike lookback options — analytical + MC
surface       : Implied volatility surface computation and visualisation
utils         : Sharpe ratio, drawdown, equity curve, market data
"""

from . import black_scholes, heston, lookback, surface, utils

__version__ = "1.0.0"
__author__  = "Joan Ballés Jr."