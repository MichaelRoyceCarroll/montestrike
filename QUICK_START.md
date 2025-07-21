# 🚀 MonteStrike v0.2.0 Quick Start Guide

Get MonteStrike running in under 5 minutes!

## 📋 Prerequisites

### Required
- **CMake 3.18+** and **C++17 compiler** (GCC 9+/MSVC 2019+)
- **Multi-core CPU** (for CPU backend)

### Optional (for GPU acceleration)
- **NVIDIA GPU** with Compute Capability 6.0+ (GTX 1000 series or newer)
- **CUDA Toolkit 11.0+**

### Python Support (optional)
- **Python 3.8+** (for Python bindings)

## ⚡ Quick Build & Test

```bash
# Clone and build
git clone https://github.com/MichaelRoyceCarroll/montestrike.git
cd montestrike
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# Test C++ examples
./basic_usage_cpp --backend cuda    # GPU (if available)
./basic_usage_cpp --backend cpu     # CPU (always works)
./test_montestrike_cpp              # Full test suite

# Test Python bindings
PYTHONPATH=$(pwd) python ../test_python_backends.py
```

## 📊 Expected Performance

| Backend | Hardware Example | Performance |
|---------|------------------|-------------|
| **CUDA** | RTX 4060 | ~20M paths/sec |
| **CPU** | Core Ultra 9 185H | ~1.6M paths/sec |

## 🐍 Python Example

```python
import sys
sys.path.insert(0, 'build')  # Add build directory to path
import montestrike as ms

# Create calculator
calc = ms.MonteCarloPoT()

# Set parameters
params = ms.Parameters()
params.current_price = 100.0
params.strike_price = 105.0
params.time_to_expiration = 30.0 / 365.0  # 30 days
params.drift = 0.05            # 5% annual return
params.volatility = 0.20       # 20% volatility  
params.num_paths = 1000000     # 1M paths
params.backend = ms.ComputeBackend.CPU  # Force CPU (or CUDA)

# Calculate probability of touch
results = calc.calculate_pot(params)
print(f"Probability of Touch: {results.probability_of_touch:.4f}")
print(f"Throughput: {results.metrics.throughput_paths_per_sec:,.0f} paths/sec")
```

## 🔧 C++ Example

```cpp
#include "montestrike/montestrike.hpp"

int main() {
    montestrike::MonteCarloPoT calc;
    
    montestrike::MonteCarloPoT::Parameters params;
    params.current_price = 100.0f;
    params.strike_price = 105.0f;
    params.time_to_expiration = 30.0f / 365.0f;
    params.drift = 0.05f;
    params.volatility = 0.20f;
    params.num_paths = 1000000;
    params.backend = montestrike::ComputeBackend::CPU;  // or CUDA
    
    auto results = calc.calculate_pot(params);
    
    if (results.computation_successful) {
        std::cout << "PoT: " << results.probability_of_touch << std::endl;
        std::cout << "Throughput: " << results.metrics.throughput_paths_per_sec 
                  << " paths/sec" << std::endl;
    }
    
    return 0;
}
```

## 🛠️ Build Options

```bash
# Standard build (CPU + CUDA)
cmake .. -DCMAKE_BUILD_TYPE=Release

# CPU-only build (no CUDA required)
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=OFF

# Debug build with symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Disable Python bindings
cmake .. -DBUILD_PYTHON_BINDINGS=OFF
```

## 🏆 Backend Selection

MonteStrike automatically chooses the best available backend:

1. **CUDA** (if GPU available and requested)
2. **CPU** (always available as fallback)

Force a specific backend:
```cpp
params.backend = montestrike::ComputeBackend::CPU;   // Force CPU
params.backend = montestrike::ComputeBackend::CUDA;  // Force CUDA
params.cpu_threads = 8;  // Use 8 threads for CPU backend
```

## ❗ Troubleshooting

### "No compatible CUDA devices"
- **Solution**: Use CPU backend - it works on any system
- **Alternative**: Install CUDA Toolkit if you have NVIDIA GPU

### "Memory allocation failed"  
- **Solution**: Reduce `num_paths` (try 100K instead of 1M)
- **Alternative**: Close other GPU applications

### Python import errors
- **Solution**: Set `PYTHONPATH` to build directory
- **Alternative**: Install package: `pip install build/` (if setup.py available)

### Build errors
- **Solution**: Check CMake/compiler versions meet requirements
- **Alternative**: Use Docker/container with dependencies pre-installed

## 📚 Next Steps

- **Documentation**: See `docs/USAGE_GUIDE.md` for detailed parameter guidance
- **Examples**: Explore `examples/` directory for more use cases  
- **Testing**: Run `./test_montestrike_cpp` for full validation
- **Performance**: Try `./performance_benchmark` for detailed metrics

## ⚠️ Financial Disclaimer

This software is for **educational purposes only** and does not constitute financial advice. Trading options involves substantial risk.

---

**Get started in under 5 minutes!** 🚀