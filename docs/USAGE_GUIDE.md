# MonteStrike Usage Guide

This guide provides detailed information on using the MonteStrike library effectively for Monte Carlo probability of touch estimation.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Parameter Selection](#parameter-selection)
3. [Performance Optimization](#performance-optimization)
4. [Common Use Cases](#common-use-cases)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

## Getting Started

### Device Requirements

MonteStrike supports multiple backends:

**GPU Backend (Recommended)**:
- NVIDIA GPU with Compute Capability 6.0 or higher
- CUDA Toolkit 11.0+ installed
- Sufficient GPU memory (4GB+ recommended)

**CPU Backend (Fallback)**:
- Multi-core x86-64 processor
- No special requirements - works on any system

### Quick Device Check

```python
import montestrike as ms

# Check if CUDA is available
if ms.check_cuda_available():
    devices = ms.get_compatible_devices()
    print(f"Found {len(devices)} compatible device(s)")
    for device in devices:
        print(f"  {device.name} - {device.global_memory_bytes // (1024**3)} GB")
else:
    print("No compatible CUDA devices found - CPU backend will be used")
```

## CPU Backend Usage

MonteStrike includes a multi-threaded CPU backend that runs on any system without requiring CUDA.

### Backend Selection

```python
import montestrike as ms

params = ms.Parameters()
# ... set other parameters ...

# Force CPU backend
params.backend = ms.ComputeBackend.CPU
params.cpu_threads = 8  # Use 8 threads (auto-detect if 0)

calc = ms.MonteCarloPoT()
results = calc.estimate_pot(params)
```

### Performance Expectations

| Backend | Typical Performance | Hardware Example |
|---------|-------------------|------------------|
| CUDA | ~20M paths/sec | RTX 4060 |
| CPU | ~1.6M paths/sec | Intel Core Ultra 9 185H |

### Thread Configuration

```python
# Auto-detect thread count (recommended)
params.cpu_threads = 0

# Manual thread count
params.cpu_threads = 4  # Use 4 threads
```

### Backend Fallback Chain

MonteStrike automatically falls back through backends:
1. **CUDA** (if available and requested)
2. **CPU** (always available)

**Note**: AVX2 CPU optimizations are under investigation due to thread affinity limitations in WSL2/hypervisor environments.

## Parameter Selection

### Core Parameters

#### Current Price (S)
The current market price of the underlying asset.

```python
params.current_price = 220.50  # Current stock price
```

**Guidelines:**
- Use real-time or recent market data
- Ensure price reflects current market conditions
- Consider bid-ask spread for liquid markets

#### Strike Price (K)
The price level you want to test for probability of touch.

```python
params.strike_price = 225.00  # Target price level
```

**Guidelines:**
- Can be above (calls) or below (puts) current price
- Consider realistic price levels based on historical ranges
- Very distant strikes may have very low probabilities

#### Time to Expiration (T)
Time remaining until expiration, expressed in years.

```python
# Common conversions
params.time_to_expiration = 7 / 365.0      # 7 days
params.time_to_expiration = 30 / 365.0     # 30 days  
params.time_to_expiration = 0.25           # 3 months
```

**Guidelines:**
- Use trading days for more accuracy: `trading_days / 252`
- Very short expirations (< 1 day) may be unreliable
- Consider weekend and holiday effects

#### Drift (μ)
Expected annual return rate of the underlying asset.

```python
params.drift = 0.05    # 5% annual expected return
params.drift = 0.08    # 8% annual expected return
params.drift = -0.02   # -2% annual expected return (bear market)
```

**Estimation Methods:**
- Historical average returns over 1-3 years
- Risk-free rate + equity risk premium
- Analyst consensus estimates
- Market-implied forward rates

#### Volatility (σ)
Annual volatility (standard deviation of returns).

```python
params.volatility = 0.20   # 20% annual volatility
params.volatility = 0.35   # 35% annual volatility (high vol)
```

**Estimation Methods:**
- Historical volatility from price data
- Implied volatility from option prices
- GARCH or other volatility models
- VIX-based estimates for broad market

#### Steps per Day
Number of time steps per trading day.

```python
params.steps_per_day = 50   # Default: good balance
params.steps_per_day = 100  # Higher precision, slower
params.steps_per_day = 24   # Hourly steps
```

**Guidelines:**
- 50 steps/day is usually sufficient
- Increase for very short expirations
- Diminishing returns beyond 100 steps/day

#### Number of Paths
Number of Monte Carlo simulation paths.

```python
params.num_paths = 100000   # 100K: fast, basic accuracy
params.num_paths = 1000000  # 1M: balanced
params.num_paths = 4000000  # 4M: high accuracy, slower
```

**Accuracy vs Speed:**
- 100K paths: ±1-2% accuracy, fast
- 1M paths: ±0.5% accuracy, medium speed
- 4M paths: ±0.25% accuracy, slower

### Monte Carlo Error Rate Convergence

As path count increases, Monte Carlo simulation accuracy improves following statistical convergence patterns. Expected standard error ranges:

| Path Count | Standard Error | Confidence Interval (95%) |
|------------|---------------|---------------------------|
| 50K        | ±2.5%         | ±4.9%                     |
| 100K       | ±1.8%         | ±3.5%                     |
| 200K       | ±1.2%         | ±2.4%                     |
| 500K       | ±0.8%         | ±1.6%                     |
| 1M         | ±0.6%         | ±1.2%                     |
| 2M         | ±0.4%         | ±0.8%                     |
| 4M         | ±0.3%         | ±0.6%                     |

**Usage Guidelines:**
- **Development/Testing**: 50K-100K paths provide quick estimates with ±2-4% accuracy
- **Production (Fast)**: 500K-1M paths balance speed with ±1-2% accuracy  
- **Production (Precise)**: 2M-4M paths for critical calculations requiring ±0.5-1% accuracy
- **Statistical Significance**: Use antithetic variates to reduce error by ~30% at same path count

**Note**: Error rates assume normal market conditions. High volatility periods or extreme parameter values may require higher path counts for stable convergence.

## Performance Optimization

### Memory Management

MonteStrike pre-allocates GPU memory for optimal performance:

```python
# Check memory requirements
memory_needed = calc.estimate_memory_requirements(params.num_paths)
memory_info = calc.get_memory_info()

if memory_needed > memory_info.free_bytes:
    print("Warning: Insufficient GPU memory")
    # Reduce path count or use multiple runs
```

### Path Count Selection

Choose path counts based on your needs:

| Use Case | Recommended Paths | Typical Time (RTX 4060) |
|----------|------------------|-------------------------|
| Quick estimates | 50K-100K | < 50ms |
| Development/testing | 500K | ~100ms |
| Production (fast) | 1M | ~200ms |
| Production (accurate) | 2M-4M | 500ms-1s |

### Variance Reduction

#### Antithetic Variates

Enable antithetic variates for improved accuracy:

```python
params.use_antithetic_variates = True
```

**What are Antithetic Variates?**
Antithetic variates is a variance reduction technique that pairs each random path with its "opposite" path using negated random numbers. For example, if one path uses random values [0.5, -1.2, 0.8], the antithetic path uses [-0.5, 1.2, -0.8]. This pairing reduces Monte Carlo variance because extreme outcomes in opposite directions tend to cancel out.

**Benefits:**
- Reduces standard error by ~30% with same computational cost
- Particularly effective for near-the-money options and symmetric scenarios
- Minimal performance overhead (processes two paths per random sequence)
- Provides better convergence for most probability calculations

**When to Use:**
- **Recommended**: For most production calculations and important decisions
- **Optional**: Quick estimates where speed is more important than precision
- **Always**: When comparing different scenarios (ensures consistent variance reduction)

### Reproducible Results

Set a fixed random seed for consistent results:

```python
params.random_seed = 12345  # Any positive integer
```

## Common Use Cases

### Case 1: Covered Call Writing

Estimate probability of assignment for covered calls:

```python
params = ms.Parameters()
params.current_price = 100.0
params.strike_price = 105.0      # Call strike (above current)
params.time_to_expiration = 30 / 365.0
params.drift = 0.06              # Market return assumption
params.volatility = 0.25         # Stock's volatility
params.num_paths = 1000000

results = calc.estimate_pot(params)
assignment_prob = results.probability_of_touch

print(f"Probability of assignment: {assignment_prob:.2%}")
```

### Case 2: Put Protection

Estimate probability of put exercise:

```python
params.strike_price = 95.0       # Put strike (below current)  
# ... other parameters same as above

results = calc.estimate_pot(params)
exercise_prob = results.probability_of_touch

print(f"Probability of put exercise: {exercise_prob:.2%}")
```

### Case 3: Earnings Play

Model probability around earnings events:

```python
# Higher volatility around earnings
params.volatility = 0.45         # Elevated IV
params.time_to_expiration = 7 / 365.0  # Week to earnings
params.steps_per_day = 100       # Higher precision for short term
```

### Case 4: Multi-Strike Analysis

Compare probabilities across multiple strikes:

```python
strikes = [95, 100, 105, 110, 115]
probabilities = []

for strike in strikes:
    params.strike_price = strike
    results = calc.estimate_pot(params)
    probabilities.append(results.probability_of_touch)
    
# Plot or analyze the probability curve
```

## Troubleshooting

### Common Error Messages

#### "No CUDA device available"
- **Cause**: No NVIDIA GPU or CUDA not installed
- **Solution**: Install CUDA Toolkit, check GPU compatibility

#### "Incompatible device"
- **Cause**: GPU compute capability < 6.0
- **Solution**: Use newer GPU (GTX 10-series or later)

#### "Memory allocation failed"
- **Cause**: Insufficient GPU memory
- **Solution**: Reduce `num_paths` or close other GPU applications

#### "Invalid parameters"
- **Cause**: Parameter values outside valid ranges
- **Solution**: Check parameter validation errors, adjust values

### Performance Issues

#### Slow calculations
1. Check GPU temperature and throttling
2. Update NVIDIA drivers
3. Close other GPU-intensive applications
4. Reduce path count for faster results

#### Inconsistent results
1. Set fixed `random_seed` for reproducibility
2. Use sufficient path count (≥100K)
3. Check for parameter input errors
4. Verify driver stability

## Best Practices

### Parameter Estimation

1. **Use multiple data sources** for volatility and drift estimation
2. **Update parameters regularly** as market conditions change
3. **Consider market regime changes** (bull/bear markets)
4. **Validate with historical data** when possible

### Result Interpretation

1. **Consider confidence intervals** based on path count
2. **Compare with theoretical models** (Black-Scholes) for validation
3. **Account for model limitations** (no jumps, constant volatility)
4. **Use multiple time horizons** for sensitivity analysis

### Production Usage

1. **Cache calculator instances** to avoid re-initialization
2. **Batch similar calculations** for efficiency
3. **Monitor GPU memory usage** in long-running applications
4. **Implement error handling** for robust applications

### Code Organization

```python
class OptionAnalyzer:
    def __init__(self):
        self.calc = ms.MonteCarloPoT()
        self.calc.initialize()
    
    def analyze_probability(self, current_price, strike_price, 
                          days_to_expiry, volatility, drift=0.05):
        params = ms.Parameters()
        params.current_price = current_price
        params.strike_price = strike_price
        params.time_to_expiration = days_to_expiry / 365.0
        params.volatility = volatility
        params.drift = drift
        params.num_paths = 1000000
        
        # Validate before computing
        validation = self.calc.validate_parameters(params)
        if not validation.is_valid:
            raise ValueError(f"Invalid parameters: {validation.error_message}")
        
        return self.calc.estimate_pot(params)
```

### Testing and Validation

1. **Use known test cases** with analytical solutions
2. **Compare with market option prices** for implied probabilities
3. **Test edge cases** (ATM, very short/long expiry)
4. **Monitor computational consistency** across runs

### Memory Management

```python
# For memory-constrained environments
def calculate_large_simulation(params, target_paths=4000000):
    max_batch = 1000000  # 1M paths per batch
    batches = target_paths // max_batch
    
    total_touched = 0
    
    for batch in range(batches):
        batch_params = params.copy()
        batch_params.num_paths = max_batch
        batch_params.random_seed = params.random_seed + batch
        
        results = calc.estimate_pot(batch_params)
        total_touched += results.paths_touched
    
    return total_touched / target_paths
```

---

## Additional Resources

- [API Reference](API_REFERENCE.md)
- [Performance Benchmarks](PERFORMANCE_BENCHMARKS.md)
- [Example Code](../examples/)
- [Test Data Generation](../test/generate_test_data.py)

For more help, see the GitHub issues or discussions page.