"""
MonteStrike: Monte Carlo US Option Probability of Touch Estimator

A CUDA-accelerated library for calculating option touch probabilities using 
Geometric Brownian Motion Monte Carlo simulation.

⚠️  FINANCIAL DISCLAIMER:
This library is for educational and research purposes only. 
It does not constitute financial advice. Trading options involves substantial 
risk and may not be suitable for all investors. Past performance does not 
guarantee future results. Always consult with a qualified financial advisor 
before making investment decisions.

Example usage:
    import montestrike as ms
    
    # Create calculator instance
    calc = ms.MonteCarloPoT()
    
    # Set up parameters
    params = ms.Parameters()
    params.current_price = 100.0
    params.strike_price = 105.0
    params.time_to_expiration = 30.0 / 365.0  # 30 days
    params.drift = 0.05  # 5% annual return
    params.volatility = 0.20  # 20% volatility
    params.steps_per_day = 50
    params.num_paths = 100000
    
    # Calculate probability of touch
    results = calc.calculate_pot(params)
    
    if results.computation_successful:
        print(f"Probability of Touch: {results.probability_of_touch:.4f}")
        print(f"Computation time: {results.metrics.computation_time_ms:.2f} ms")
    else:
        print(f"Error: {results.error_message}")
"""

# Import the compiled module
from .montestrike import *

# Re-export main classes for convenience
__all__ = [
    'MonteCarloPoT',
    'Parameters', 
    'Results',
    'DeviceAnalyzer',
    'DeviceInfo',
    'MemoryInfo',
    'ComputationMetrics',
    'ValidationResult',
    'VersionInfo',
    'ErrorCode',
    'validate_monte_carlo_parameters',
    'error_code_to_string',
    'get_version_info',
    'get_version_string'
]

# Version information
__version__ = get_version_string()
__author__ = "MonteStrike Contributors"
__license__ = "MIT"

def get_device_list():
    """
    Convenience function to get a list of all available CUDA devices.
    
    Returns:
        list: List of DeviceInfo objects for all detected devices
    """
    analyzer = DeviceAnalyzer()
    return analyzer.enumerate_devices()

def get_compatible_devices():
    """
    Get a list of only the compatible CUDA devices.
    
    Returns:
        list: List of DeviceInfo objects for compatible devices only
    """
    devices = get_device_list()
    return [device for device in devices if device.is_compatible]

def check_cuda_available():
    """
    Check if CUDA is available and there are compatible devices.
    
    Returns:
        bool: True if CUDA is available with compatible devices
    """
    try:
        compatible_devices = get_compatible_devices()
        return len(compatible_devices) > 0
    except:
        return False

def quick_pot_calculation(current_price, strike_price, days_to_expiration, 
                         annual_volatility, annual_drift=0.05, num_paths=100000):
    """
    Quick probability of touch calculation with sensible defaults.
    
    Args:
        current_price (float): Current stock price
        strike_price (float): Strike price to test for touch
        days_to_expiration (int): Days until expiration
        annual_volatility (float): Annual volatility (e.g., 0.20 for 20%)
        annual_drift (float): Annual expected return (default: 5%)
        num_paths (int): Number of simulation paths (default: 100K)
    
    Returns:
        Results: Calculation results object
    """
    calc = MonteCarloPoT()
    
    params = Parameters()
    params.current_price = current_price
    params.strike_price = strike_price
    params.time_to_expiration = days_to_expiration / 365.0
    params.drift = annual_drift
    params.volatility = annual_volatility
    params.steps_per_day = 50
    params.num_paths = num_paths
    
    return calc.calculate_pot(params)

# Display warning on import
import warnings
warnings.warn(
    "⚠️  FINANCIAL DISCLAIMER: MonteStrike is for educational purposes only. "
    "It does not constitute financial advice. Trading options involves substantial risk.",
    UserWarning,
    stacklevel=2
)