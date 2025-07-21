# MonteStrike

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**Monte Carlo US Option Probability of Touch Estimator**

MonteStrike is a high-performance CUDA-accelerated library for estimating the probability that an option will touch its strike price before expiration using Geometric Brownian Motion Monte Carlo simulation. Designed for stock trading hobbyists and GPU computing enthusiasts.

## ⚠️ Financial Disclaimer

**This library is for educational and research purposes only. It does not constitute financial advice.** Trading options involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions. Review the [DISCLAIMER.md](docs/DISCLAIMER.md) before using this library.


## Features

- 🚀 **Multi-Backend Support**: CUDA GPU acceleration + CPU fallback for universal compatibility
- 🖥️ **CPU Threading**: Multi-threaded CPU implementation when GPU unavailable
- 📊 **Geometric Brownian Motion**: Industry-standard model for stock price simulation  
- 🎯 **Probability of Touch**: Estimate the likelihood of an option reaching its strike price
- 🔧 **Dual Interface**: Both C++ and Python APIs available
- ⚡ **High Performance**: Process millions of paths in milliseconds (GPU) or seconds (CPU)
- 🎛️ **Configurable**: Adjustable parameters for precision vs speed trade-offs
- 🧪 **Variance Reduction**: Optional antithetic variates for improved accuracy
- 📈 **Real-world Ready**: Includes test data generation from live market data
- 🔄 **Smart Fallback**: Automatically falls back to CPU when CUDA unavailable

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

# Estimate probability of touch
results = calc.estimate_pot(params)

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

auto results = calculator.estimate_pot(params);
```

## Requirements

### Hardware
- **GPU (Recommended)**: NVIDIA GPU with Compute Capability 6.0+ (GTX 10-series or newer)
- **CPU (Fallback)**: Multi-core x86-64 processor for CPU backend
- 4GB+ GPU memory recommended for large GPU simulations

### Software
- **Windows**: Windows 10/11, Visual Studio 2019+
- **Linux**: Ubuntu 22.04+ LTS, GCC 9+
- **CUDA Toolkit**: 11.0 or later (optional - for GPU acceleration)
- **Python**: 3.8+ (for Python bindings)
- **CMake**: 3.18+

MonteStrike supports both GPU (CUDA) and CPU backends. CUDA provides accelerated performance but is not required - the library automatically falls back to multi-threaded CPU implementation when GPU unavailable.

**Note**: AVX2 CPU optimizations are under investigation due to thread affinity limitations in WSL2/hypervisor environments.

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

### Backend Selection

MonteStrike supports multiple compute backends:

```cpp
params.backend = montestrike::ComputeBackend::CUDA;  // GPU acceleration (default)
params.backend = montestrike::ComputeBackend::AVX2;  // CPU vectorization  
params.backend = montestrike::ComputeBackend::CPU;   // Standard CPU
params.strict_backend_mode = false;  // Allow fallback if requested unavailable
```

```bash
# Test specific backend
./basic_usage_cpp --backend cuda
./basic_usage_cpp --backend cpu

# Test all available backends
./basic_usage_cpp --backend all
```

### Key Parameters (Quick Reference)

**Current Price (S)**: Current market price of the underlying stock  
**Strike Price (K)**: Target price level to test for probability of touch  
**Time to Expiration (T)**: Days/weeks until option expires (e.g., 30 days = 30/365)  
**Drift (μ)**: Expected annual return rate - market assumption (typically 3-8%)  
**Volatility (σ)**: Annual price movement volatility (typically 15-50% for stocks)  
**Number of Paths**: Simulation runs - more paths = higher accuracy (50K-4M range)  

*For detailed parameter guidance, see [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)*

### Performance Guidelines

| Path Count | Accuracy | GPU Time (RTX 4060) | CPU Time (Core Ultra 9) |
|------------|----------|---------------------|-------------------------|
| 50K        | ±2-3%    | 6ms                 | ~47ms                   |
| 100K       | ±1.5%    | 10ms                | ~87ms                   |
| 500K       | ±0.7%    | 43ms                | ~421ms                  |
| 1M         | ±0.5%    | 84ms                | ~821ms                  |
| 2M         | ±0.3%    | ~170ms              | ~1.6s                   |
| 4M         | ±0.2%    | ~340ms              | ~3.3s                   |

## Getting Started

### Quick Setup

```bash
git clone https://github.com/MichaelRoyceCarroll/montestrike.git
cd montestrike && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
```

**Next Steps:**
- 🚀 **Quick Start**: See [QUICK_START.md](QUICK_START.md) for 5-minute setup
- 📚 **Detailed Usage**: See [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) for complete guide
- 🔧 **API Reference**: Full API documentation in [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)

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

### Backend Comparison

Measured on Intel Core Ultra 9 185H + RTX 4060 Laptop GPU (1M paths):

| Backend | Time (ms) | Throughput (M paths/s) | Relative Performance |
|---------|-----------|------------------------|---------------------|
| **CUDA** | 50        | 20.1                   | 12.4x               |
| **CPU**  | 617       | 1.6                    | 1x                  |

### CUDA Scaling (RTX 4060 Laptop GPU)

| Paths | Time (ms) | Throughput (M paths/s) |
|-------|-----------|------------------------|
| 50K   | 4.7       | 2.7                    |
| 100K  | 8.1       | 12.3                   |
| 500K  | 36.7      | 13.6                   |
| 1M    | 51.6      | 19.4                   |

*Results may vary based on hardware, driver version, and system configuration.*

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

While the MIT license only requires copyright notice retention, **attribution** in project documentation or credits is appreciated if you find this library useful in your work.

## Support

- 📖 Documentation: See `docs/` directory
- 🐛 Bug Reports: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 📧 Contact: michael dot carroll ATSIGN alumni dot usc dot edu 

## Acknowledgments

- CUDA SDK team
- This project uses [`yfinance`](https://github.com/ranaroussi/yfinance). Thanks to the yfinance team!
- This project was architected and initially implemented with assistance from [Claude](https://claude.ai) (Anthropic) Sonnet4.
- _How to Make Money in Stocks: A Winning System in Good Times and Bad, Fourth Edition_ by William J. O'Neil
- _Options as a Strategic Investment_ by Lawrence G. McMillan 

---

**Remember**: This software is for educational purposes only and does not constitute financial advice. Always do your own research and consult with qualified professionals before making investment decisions.