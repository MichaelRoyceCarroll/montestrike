#!/usr/bin/env python3
"""
Test Python bindings for MonteStrike v0.2.0 CPU and CUDA backends
"""

import sys
import os
sys.path.insert(0, '/home/mcarr/montestrike/build')

import montestrike as ms

def test_version():
    """Test version information"""
    print("🔍 Testing Version Info...")
    version = ms.get_version_string()
    print(f"   Version: {version}")
    assert "0.2.0" in version, f"Expected 0.2.0 in version, got: {version}"
    print("   ✅ Version test passed")

def test_cpu_backend():
    """Test CPU backend"""
    print("\n🖥️  Testing CPU Backend...")
    
    calc = ms.MonteCarloPoT()
    
    params = ms.Parameters()
    params.current_price = 100.0
    params.strike_price = 105.0
    params.time_to_expiration = 7.0 / 365.0
    params.drift = 0.05
    params.volatility = 0.20
    params.steps_per_day = 50
    params.num_paths = 50000  # Smaller for faster test
    params.backend = ms.ComputeBackend.CPU
    params.cpu_threads = 4
    
    results = calc.calculate_pot(params)
    
    assert results.computation_successful, f"CPU calculation failed: {results.error_message}"
    assert 0.0 <= results.probability_of_touch <= 1.0, f"Invalid probability: {results.probability_of_touch}"
    assert results.paths_processed == params.num_paths, f"Path count mismatch: {results.paths_processed} vs {params.num_paths}"
    
    print(f"   PoT: {results.probability_of_touch:.4f}")
    print(f"   Throughput: {results.metrics.throughput_paths_per_sec:,.0f} paths/sec")
    print("   ✅ CPU backend test passed")

def test_cuda_backend():
    """Test CUDA backend"""
    print("\n🚀 Testing CUDA Backend...")
    
    calc = ms.MonteCarloPoT()
    
    # Try to initialize CUDA
    init_result = calc.initialize()
    if init_result != ms.ErrorCode.SUCCESS:
        print(f"   ⚠️  CUDA not available: {ms.error_code_to_string(init_result)}")
        print("   ℹ️  Skipping CUDA test")
        return
    
    params = ms.Parameters()
    params.current_price = 100.0
    params.strike_price = 105.0
    params.time_to_expiration = 7.0 / 365.0
    params.drift = 0.05
    params.volatility = 0.20
    params.steps_per_day = 50
    params.num_paths = 100000  # Larger for GPU
    params.backend = ms.ComputeBackend.CUDA
    
    results = calc.calculate_pot(params)
    
    assert results.computation_successful, f"CUDA calculation failed: {results.error_message}"
    assert 0.0 <= results.probability_of_touch <= 1.0, f"Invalid probability: {results.probability_of_touch}"
    assert results.paths_processed == params.num_paths, f"Path count mismatch: {results.paths_processed} vs {params.num_paths}"
    
    print(f"   PoT: {results.probability_of_touch:.4f}")
    print(f"   Throughput: {results.metrics.throughput_paths_per_sec:,.0f} paths/sec")
    print("   ✅ CUDA backend test passed")

def test_fallback_behavior():
    """Test automatic fallback"""
    print("\n🔄 Testing Fallback Behavior...")
    
    calc = ms.MonteCarloPoT()
    
    params = ms.Parameters()
    params.current_price = 100.0
    params.strike_price = 95.0  # Put option
    params.time_to_expiration = 14.0 / 365.0
    params.drift = 0.05
    params.volatility = 0.25
    params.steps_per_day = 50
    params.num_paths = 50000
    # Don't specify backend - let it choose automatically
    
    results = calc.calculate_pot(params)
    
    assert results.computation_successful, f"Fallback calculation failed: {results.error_message}"
    assert 0.0 <= results.probability_of_touch <= 1.0, f"Invalid probability: {results.probability_of_touch}"
    
    print(f"   PoT: {results.probability_of_touch:.4f}")
    print(f"   Device: {results.device_used.name}")
    print("   ✅ Fallback test passed")

def main():
    print("🚀 MonteStrike v0.2.0 Python Backend Test")
    print("=" * 45)
    
    try:
        test_version()
        test_cpu_backend()
        test_cuda_backend()
        test_fallback_behavior()
        
        print("\n🎉 All Python backend tests passed!")
        print("\n💡 Build and Test Instructions:")
        print("   1. mkdir build && cd build")
        print("   2. cmake .. -DCMAKE_BUILD_TYPE=Release")
        print("   3. make -j8")
        print("   4. PYTHONPATH=$(pwd) python ../examples/test_python_backends.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())