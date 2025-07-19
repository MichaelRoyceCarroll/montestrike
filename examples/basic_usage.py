#!/usr/bin/env python3
"""
MonteStrike Basic Usage Example (Python)

Demonstrates how to use the MonteStrike library to calculate
probability of touch for options using Monte Carlo simulation.
"""

import montestrike as ms
import time

def main():
    print("🚀 MonteStrike Basic Usage Example (Python)")
    print("=" * 45)
    
    # Display version information
    version_info = ms.get_version_info()
    print(f"📦 MonteStrike Version: {version_info.to_string()}")
    print(f"⚡ CUDA Version: {version_info.cuda_version}")
    print()
    
    # Check CUDA availability
    if not ms.check_cuda_available():
        print("❌ No compatible CUDA devices found!")
        return 1
    
    # Display available devices
    devices = ms.get_device_list()
    compatible_devices = ms.get_compatible_devices()
    
    print("🎯 Available CUDA Devices:")
    for device in devices:
        status = "✅ Compatible" if device.is_compatible else "❌ Incompatible"
        print(f"  Device {device.device_id}: {device.name} {status} (CC {device.compute_capability:.1f})")
    
    print(f"🏆 Found {len(compatible_devices)} compatible device(s)")
    print()
    
    # Create calculator instance
    calc = ms.MonteCarloPoT()
    
    # Initialize with auto-selected device
    init_result = calc.initialize()
    if init_result != ms.ErrorCode.SUCCESS:
        print(f"❌ Failed to initialize calculator: {ms.error_code_to_string(init_result)}")
        return 1
    
    print("✅ Calculator initialized successfully")
    print()
    
    # Set up option parameters for an IWM call
    params = ms.Parameters()
    params.current_price = 220.50      # Current IWM price
    params.strike_price = 225.00       # Strike price to test
    params.time_to_expiration = 7.0 / 365.0  # 7 days to expiration
    params.drift = 0.08                # 8% annual expected return
    params.volatility = 0.25           # 25% annual volatility
    params.steps_per_day = 50          # 50 time steps per day
    params.num_paths = 1000000         # 1 million simulation paths
    params.use_antithetic_variates = True  # Use variance reduction
    params.random_seed = 42            # Fixed seed for reproducibility
    
    print("📊 Option Parameters:")
    print(f"  Current Price (S): ${params.current_price:.2f}")
    print(f"  Strike Price (K): ${params.strike_price:.2f}")
    print(f"  Days to Expiration: {params.time_to_expiration * 365:.1f}")
    print(f"  Annual Drift (μ): {params.drift * 100:.1f}%")
    print(f"  Annual Volatility (σ): {params.volatility * 100:.1f}%")
    print(f"  Steps per Day: {params.steps_per_day}")
    print(f"  Simulation Paths: {params.num_paths:,}")
    print(f"  Antithetic Variates: {'Yes' if params.use_antithetic_variates else 'No'}")
    print()
    
    # Validate parameters
    validation = calc.validate_parameters(params)
    if not validation.is_valid:
        print(f"❌ Invalid parameters: {validation.error_message}")
        return 1
    
    print("✅ Parameters validated successfully")
    
    # Check memory requirements
    memory_required = calc.estimate_memory_requirements(params.num_paths)
    memory_info = calc.get_memory_info()
    
    print(f"💾 Estimated GPU memory required: {memory_required // (1024*1024)} MB")
    print(f"💾 Available GPU memory: {memory_info.free_bytes // (1024*1024)} MB")
    
    if memory_info.free_bytes < memory_required:
        print(f"⚠️  Warning: May not have sufficient GPU memory!")
    
    print()
    
    # Run Monte Carlo simulation
    print("🎲 Running Monte Carlo simulation...")
    start_time = time.time()
    
    results = calc.calculate_pot(params)
    
    end_time = time.time()
    total_time = (end_time - start_time) * 1000  # Convert to ms
    
    # Display results
    print()
    print("📈 RESULTS:")
    print("=" * 10)
    
    if results.computation_successful:
        print("✅ Computation completed successfully!")
        print()
        
        print(f"🎯 Probability of Touch: {results.probability_of_touch:.4f} ({results.probability_of_touch * 100:.2f}%)")
        print()
        
        print("📊 Simulation Statistics:")
        print(f"  Paths Processed: {results.paths_processed:,}")
        print(f"  Paths Touched: {results.paths_touched:,}")
        print(f"  Touch Rate: {results.paths_touched / results.paths_processed:.4f}")
        print()
        
        print("⚡ Performance Metrics:")
        print(f"  Total Time: {results.metrics.computation_time_ms:.2f} ms")
        print(f"  Kernel Time: {results.metrics.kernel_time_ms:.2f} ms")
        print(f"  Throughput: {results.metrics.throughput_paths_per_sec:,.0f} paths/sec")
        print(f"  Memory Used: {results.metrics.memory_used_bytes // (1024*1024)} MB")
        print()
        
        print(f"🖥️  Device Used: {results.device_used.name} (Device {results.device_used.device_id})")
        print()
        
        # Interpretation
        print("💡 INTERPRETATION:")
        print("=" * 18)
        
        prob = results.probability_of_touch
        if prob > 0.7:
            risk_level = "🔥 HIGH"
        elif prob > 0.3:
            risk_level = "⚖️  MODERATE" 
        else:
            risk_level = "❄️  LOW"
            
        print(f"{risk_level} probability of touching ${params.strike_price:.2f} before expiration")
        
        # Calculate moneyness
        moneyness = (params.strike_price / params.current_price - 1.0) * 100
        if moneyness > 0:
            position_desc = f"{abs(moneyness):.1f}% above current price (OTM call)"
        elif moneyness < 0:
            position_desc = f"{abs(moneyness):.1f}% below current price (ITM call)"
        else:
            position_desc = "at current price (ATM call)"
            
        print(f"📍 Strike is {position_desc}")
        
        # Trading insights
        print()
        print("💼 TRADING INSIGHTS:")
        print("=" * 20)
        
        if prob > 0.5:
            print(f"• The option has a {prob*100:.1f}% chance of going in-the-money")
            print("• Consider this when evaluating option premiums")
        else:
            print(f"• The option has only a {prob*100:.1f}% chance of going in-the-money")
            print("• Premium sellers might find this attractive")
            
        print(f"• Time decay will accelerate as expiration approaches")
        print(f"• Volatility of {params.volatility*100:.0f}% suggests significant price movement potential")
        
    else:
        print("❌ Computation failed!")
        print(f"Error: {results.error_message}")
        print(f"Error Code: {ms.error_code_to_string(results.error_code)}")
        return 1
    
    print()
    print("⚠️  DISCLAIMER: This simulation is for educational purposes only.")
    print("   It does not constitute financial advice. Trading options involves")
    print("   substantial risk and may not be suitable for all investors.")
    
    return 0

def example_with_quick_calculation():
    """Demonstrate the quick calculation helper function"""
    print("\n" + "=" * 50)
    print("🏃 Quick Calculation Example")
    print("=" * 50)
    
    # Use the convenience function for quick calculations
    results = ms.quick_pot_calculation(
        current_price=100.0,
        strike_price=105.0,
        days_to_expiration=14,
        annual_volatility=0.30,
        annual_drift=0.06,
        num_paths=500000
    )
    
    if results.computation_successful:
        print(f"🎯 Quick PoT calculation: {results.probability_of_touch:.4f}")
        print(f"⚡ Computation time: {results.metrics.computation_time_ms:.2f} ms")
    else:
        print(f"❌ Quick calculation failed: {results.error_message}")

if __name__ == "__main__":
    exit_code = main()
    
    if exit_code == 0:
        # Run additional example if main succeeded
        example_with_quick_calculation()
    
    exit(exit_code)