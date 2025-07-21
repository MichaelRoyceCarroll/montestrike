# 🚀 MonteStrike v0.2.0 - CPU Backend Support

**Monte Carlo US Option Probability of Touch Estimator with Multi-Backend Support**

## ✨ New Features

* **Multi-Backend Architecture**: CUDA GPU + CPU fallback for universal compatibility
* **CPU Threading**: Multi-threaded CPU implementation with configurable thread count
* **Smart Fallback**: Automatically uses CPU when CUDA unavailable
* **Backend Selection**: Manual backend selection via API (`ComputeBackend::CPU`, `ComputeBackend::CUDA`)
* **Thread Configuration**: Configurable CPU thread count with auto-detection
* **Universal Deployment**: Runs on any x86-64 system, GPU not required

## 📦 What's Included

* Core library with CPU + CUDA backends
* C++ and Python APIs
* Updated examples demonstrating backend selection
* Comprehensive test suite covering all backends
* Updated documentation with CPU usage guide
* **Updated example output files** with corrected probability calculations

## 🎯 Performance Benchmarks

**Tested on Intel Core Ultra 9 185H + RTX 4060:**
* **CUDA Backend**: ~20M paths/sec 
* **CPU Backend**: ~1.6M paths/sec (~12x difference)
* **Memory Usage**: CPU backend uses significantly less memory than GPU

## ⚠️ Important Notes

* **API Name Change**: `calculate_pot(..)` function renamed to `estimate_pot(..)`
* **AVX2 Support**: Temporarily disabled due to thread affinity limitations in WSL2/hypervisor environments
* **Compatibility**: CUDA Toolkit still recommended for accelerated performance
* **Fallback Behavior**: Library automatically falls back to CPU when GPU unavailable
* **Thread Safety**: All backends are thread-safe and can be used concurrently

## 🔧 Build Requirements

* **Minimum**: GCC 9+/MSVC 2019+, CMake 3.18+, C++17
* **GPU Acceleration**: CUDA Toolkit 11.0+, Compute Capability 6.0+
* **CPU Fallback**: Works on any x86-64 system

## 📚 Quick Start

```cpp
// C++ - Automatic backend selection
montestrike::MonteCarloPoT calc;
auto results = calc.estimate_pot(params);  // Uses GPU if available, else CPU

// C++ - Force CPU backend
params.backend = montestrike::ComputeBackend::CPU;
params.cpu_threads = 8;  // Use 8 threads
```

```python
# Python - Automatic fallback
import montestrike as ms
calc = ms.MonteCarloPoT()
results = calc.estimate_pot(params)

# Python - Force CPU backend  
params.backend = ms.ComputeBackend.CPU
params.cpu_threads = 0  # Auto-detect thread count
```

## 🐛 Bug Fixes

* **Critical**: Fixed CUDA antithetic variates implementation causing ~2x incorrect probability calculations
  - CPU and CUDA backends now produce consistent results within Monte Carlo variance
  - Bug affected simulations with `use_antithetic_variates = true` (most examples)
  - Updated example output files to reflect corrected results
* Fixed unused parameter warnings in CPU analyzer
* Improved thread affinity function API
* Resolved CUDA architecture compatibility warnings
* Enhanced error messages for disabled backends

---

**Full Changelog**: [v0.1.0...v0.2.0](https://github.com/MichaelRoyceCarroll/montestrike/compare/v0.1.0...v0.2.0)

⚠️ **Financial Disclaimer**: This software is for educational purposes only and does not constitute financial advice.
