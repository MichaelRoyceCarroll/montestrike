# MonteStrike

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**Monte Carlo US Option Probability of Touch Estimator**

MonteStrike is a high-performance CUDA-accelerated library for calculating the probability that an option will touch its strike price before expiration using Geometric Brownian Motion Monte Carlo simulation. Designed for stock trading hobbyists and GPU computing enthusiasts.

## ⚠️ Financial Disclaimer

**This library is for educational and research purposes only. It does not constitute financial advice.** Trading options involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions. Review the [DISCLAIMER.md](docs/DISCLAIMER.md) before using this library.


## Features

- 🚀 **CUDA-Accelerated**: Leverages NVIDIA GPUs for high-performance Monte Carlo simulation
- 📊 **Geometric Brownian Motion**: Industry-standard model for stock price simulation  
- 🎯 **Probability of Touch**: Calculate the likelihood of an option reaching its strike price
- 🔧 **Dual Interface**: Both C++ and Python APIs available
- ⚡ **High Performance**: Process millions of paths in milliseconds
- 🎛️ **Configurable**: Adjustable parameters for precision vs speed trade-offs
- 🧪 **Variance Reduction**: Optional antithetic variates for improved accuracy
- 📈 **Real-world Ready**: Includes test data generation from live market data

## Quick Start

### Python Example

```python
import montestrike as ms

# Create calculator
calc = ms.MonteCarloPoT()

# Set up parameters
params = ms.Parameters()
params.current_price = 100.0
params.strike_price = 105.0
params.time_to_expiration = 30.0 / 365.0  # 30 days
params.drift = 0.05  # 5% annual return
params.volatility = 0.20  # 20% volatility
params.num_paths = 1000000

# Calculate probability of touch
results = calc.calculate_pot(params)

if results.computation_successful:
    print(f"Probability of Touch: {results.probability_of_touch:.4f}")
    print(f"Computation time: {results.metrics.computation_time_ms:.2f} ms")
```

### C++ Example

```cpp
#include "montestrike/montestrike.hpp"

montestrike::MonteCarloPoT calculator;
calculator.initialize();

montestrike::MonteCarloPoT::Parameters params;
params.current_price = 100.0f;
params.strike_price = 105.0f;
params.time_to_expiration = 30.0f / 365.0f;
params.drift = 0.05f;
params.volatility = 0.20f;
params.num_paths = 1000000;

auto results = calculator.calculate_pot(params);
```

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 6.0+ (GTX 10-series or newer)
- 4GB+ GPU memory recommended for large simulations

### Software
- **Windows**: Windows 10/11, Visual Studio 2019+
- **Linux**: Ubuntu 22.04+ LTS, GCC 9+
- **CUDA Toolkit**: 11.0 or later
- **Python**: 3.8+ (for Python bindings)
- **CMake**: 3.18+

MonteStrike currently targets NVIDIA GPUs via CUDA. Other GPU vendors may be supported based on community demand.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/MichaelRoyceCarroll/montestrike.git
cd montestrike

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### Python Package

```bash
# Install from wheel (when available)
pip install montestrike

# Or build Python bindings
pip install .
```

## Usage Guide

### Parameter Selection

**Current Price (S)**: The current stock price  
**Strike Price (K)**: The price level to test for touch  
**Time to Expiration (T)**: Time remaining in years (e.g., 30 days = 30/365)  
**Drift (μ)**: Expected annual return rate (e.g., 0.05 = 5%)  
**Volatility (σ)**: Annual volatility (e.g., 0.20 = 20%)  
**Steps per Day**: Time resolution (50-100 recommended)  
**Number of Paths**: Simulation paths (1M+ for accurate results)

### Performance Guidelines

| Path Count | Accuracy | Speed | Memory | Use Case |
|------------|----------|-------|---------|----------|
| 50K        | Basic    | Fast  | Low     | Quick estimates |
| 100K       | Good     | Fast  | Low     | Development/testing |
| 500K       | Better   | Medium| Medium  | Production (fast) |
| 1M         | High     | Medium| Medium  | Production (balanced) |
| 2M+        | Highest  | Slow  | High    | Research/validation |

## API Reference

### Core Classes

#### `MonteCarloPoT`
Main calculator class for probability of touch estimation.

**Methods:**
- `calculate_pot(parameters)` - Run Monte Carlo simulation
- `validate_parameters(parameters)` - Validate input parameters
- `initialize(device_id)` - Initialize GPU resources
- `get_device_analyzer()` - Access device information

#### `Parameters`
Configuration for Monte Carlo simulation.

**Properties:**
- `current_price` (float) - Current stock price
- `strike_price` (float) - Strike price to test
- `time_to_expiration` (float) - Time to expiration in years
- `drift` (float) - Expected annual return
- `volatility` (float) - Annual volatility
- `steps_per_day` (uint32_t) - Time resolution
- `num_paths` (uint32_t) - Number of simulation paths

#### `Results`
Simulation results and performance metrics.

**Properties:**
- `probability_of_touch` (float) - Calculated probability (0-1)
- `computation_successful` (bool) - Success status
- `paths_processed` (uint64_t) - Number of paths simulated
- `metrics.computation_time_ms` (double) - Total computation time
- `metrics.throughput_paths_per_sec` (double) - Performance metric

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.cpp` / `basic_usage.py` - Simple probability calculation
- `performance_benchmark.cpp` - Performance testing across path counts

Example output from the basic usage application can be found in [`examples/basic_usage_cpp_output.txt`](examples/basic_usage_cpp_output.txt). This provides a sample run with interpretations of the results.

## Testing

### Generate Test Data

```bash
# Generate IWM call option test data (4-10 days)
python test/generate_test_data.py --symbol IWM --min-days 4 --max-days 10

# Generate U put option test data (10-30 days)  
python test/generate_test_data.py --symbol U --min-days 10 --max-days 30 --option-type put
```

Sample output from the C++ tests can be found in [`test/cpp/test_montestrike_cpp_output.txt`](test/cpp/test_montestrike_cpp_output.txt). This demonstrates successful execution and validation of the core calculations.

### Run Tests

```bash
# C++ tests
./build/test_montestrike_cpp

# Python tests
python -m pytest test/python/
```

## Performance Benchmarks

Typical performance on RTX 4060:

| Paths | Time (ms) | Throughput (M paths/s) |
|-------|-----------|------------------------|
| 100K  | 15        | 6.7                    |
| 500K  | 45        | 11.1                   |
| 1M    | 85        | 11.8                   |
| 2M    | 165       | 12.1                   |

*Results may vary based on GPU model, driver version, and system configuration.*

## Building from Source

### Dependencies

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler
- Python 3.8+ (for Python bindings)
- pybind11 (automatically downloaded)

Build and test system:
  - Win 11 (WSL2/Ubuntu 22.04).
  - CUDA SDK 12.6.
  - Intel(R) Core(TM) Ultra 9 185H, 32GB RAM. NVIDIA GeForce RTX 4060 8GB.
  - Python 3.10.x

### Build Options

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_EXAMPLES=ON \
  -DCMAKE_CUDA_ARCHITECTURES="60;61;70;75;80;86;89"
```

### Integration with Existing Projects

#### CMake

```cmake
find_package(MonteStrike REQUIRED)
target_link_libraries(your_target MonteStrike::montestrike)
```

#### pkg-config

```bash
gcc $(pkg-config --cflags montestrike) -o app app.c $(pkg-config --libs montestrike)
```

## Contributing

Contributions appreaciated on a case-by-case basis! Contact me at <michael dot carroll ATSIGN alumni dot usc dot edu> first. Please follow these steps to contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

A macOS build in particular could be appreciated!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Attribution Request

While the MIT license only requires copyright notice retention, attribution in project documentation or credits is appreciated if you find this library useful in your work.

## Support

- 📖 Documentation: See `docs/` directory
- 🐛 Bug Reports: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 📧 Contact: michael dot carroll ATSIGN alumni dot usc dot edu 

## Acknowledgments

- CUDA SDK team
- This project uses [`yfinance`](https://github.com/ranaroussi/yfinance). Thanks to the yfinance team!
- This project was architected and initially implemented with assistance from [Claude](https://claude.ai) (Anthropic) Sonnet4. 

---

**Remember**: This software is for educational purposes only and does not constitute financial advice. Always do your own research and consult with qualified professionals before making investment decisions.