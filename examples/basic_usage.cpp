#include "montestrike/montestrike.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "🚀 MonteStrike Basic Usage Example" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Display version and device information
    auto version = montestrike::get_version_info();
    std::cout << "📦 MonteStrike Version: " << version.to_string() << std::endl;
    std::cout << "⚡ CUDA Version: " << version.cuda_version << std::endl;
    std::cout << std::endl;
    
    // Check available CUDA devices
    montestrike::DeviceAnalyzer analyzer;
    auto devices = analyzer.enumerate_devices();
    
    if (devices.empty()) {
        std::cerr << "❌ No CUDA devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "🎯 Available CUDA Devices:" << std::endl;
    for (const auto& device : devices) {
        std::cout << "  Device " << device.device_id << ": " << device.name;
        if (device.is_compatible) {
            std::cout << " ✅ Compatible (CC " << std::fixed << std::setprecision(1) 
                     << device.compute_capability << ")";
        } else {
            std::cout << " ❌ Incompatible (CC " << std::fixed << std::setprecision(1) 
                     << device.compute_capability << ")";
        }
        std::cout << std::endl;
    }
    
    auto best_device = analyzer.get_best_device();
    if (!best_device.is_compatible) {
        std::cerr << "❌ No compatible CUDA devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "🏆 Using best device: " << best_device.name << std::endl;
    std::cout << std::endl;
    
    // Create Monte Carlo calculator
    montestrike::MonteCarloPoT calculator;
    
    // Initialize with best device
    auto init_result = calculator.initialize(best_device.device_id);
    if (init_result != montestrike::ErrorCode::SUCCESS) {
        std::cerr << "❌ Failed to initialize calculator: " 
                  << montestrike::error_code_to_string(init_result) << std::endl;
        return 1;
    }
    
    std::cout << "✅ Calculator initialized successfully" << std::endl;
    std::cout << std::endl;
    
    // Set up option parameters
    montestrike::MonteCarloPoT::Parameters params;
    
    // Example: IWM call option
    params.current_price = 220.50f;        // Current IWM price
    params.strike_price = 225.00f;         // Strike price to test
    params.time_to_expiration = 7.0f / 365.0f;  // 7 days to expiration
    params.drift = 0.08f;                  // 8% annual expected return
    params.volatility = 0.25f;             // 25% annual volatility
    params.steps_per_day = 50;             // 50 time steps per day
    params.num_paths = 1000000;            // 1 million simulation paths
    params.use_antithetic_variates = true; // Use variance reduction
    params.random_seed = 42;               // Fixed seed for reproducibility
    
    std::cout << "📊 Option Parameters:" << std::endl;
    std::cout << "  Current Price (S): $" << std::fixed << std::setprecision(2) << params.current_price << std::endl;
    std::cout << "  Strike Price (K): $" << std::fixed << std::setprecision(2) << params.strike_price << std::endl;
    std::cout << "  Days to Expiration: " << std::fixed << std::setprecision(1) << (params.time_to_expiration * 365) << std::endl;
    std::cout << "  Annual Drift (μ): " << std::fixed << std::setprecision(1) << (params.drift * 100) << "%" << std::endl;
    std::cout << "  Annual Volatility (σ): " << std::fixed << std::setprecision(1) << (params.volatility * 100) << "%" << std::endl;
    std::cout << "  Steps per Day: " << params.steps_per_day << std::endl;
    std::cout << "  Simulation Paths: " << params.num_paths << std::endl;
    std::cout << "  Antithetic Variates: " << (params.use_antithetic_variates ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
    
    // Validate parameters
    auto validation = calculator.validate_parameters(params);
    if (!validation.is_valid) {
        std::cerr << "❌ Invalid parameters: " << validation.error_message << std::endl;
        return 1;
    }
    
    std::cout << "✅ Parameters validated successfully" << std::endl;
    
    // Estimate memory requirements
    auto memory_required = calculator.estimate_memory_requirements(params.num_paths);
    std::cout << "💾 Estimated GPU memory required: " << (memory_required / (1024*1024)) << " MB" << std::endl;
    
    // Check if we have sufficient memory
    auto memory_info = calculator.get_memory_info();
    if (memory_info.free_bytes < memory_required) {
        std::cerr << "⚠️  Warning: Estimated memory (" << (memory_required / (1024*1024)) 
                  << " MB) exceeds available memory (" << (memory_info.free_bytes / (1024*1024)) << " MB)" << std::endl;
    }
    
    std::cout << std::endl;
    
    // Run Monte Carlo simulation
    std::cout << "🎲 Running Monte Carlo simulation..." << std::endl;
    
    auto results = calculator.calculate_pot(params);
    
    // Display results
    std::cout << std::endl;
    std::cout << "📈 RESULTS:" << std::endl;
    std::cout << "==========" << std::endl;
    
    if (results.computation_successful) {
        std::cout << "✅ Computation completed successfully!" << std::endl;
        std::cout << std::endl;
        
        std::cout << "🎯 Probability of Touch: " << std::fixed << std::setprecision(4) 
                  << results.probability_of_touch << " (" 
                  << std::fixed << std::setprecision(2) << (results.probability_of_touch * 100) << "%)" << std::endl;
        
        std::cout << "📊 Simulation Statistics:" << std::endl;
        std::cout << "  Paths Processed: " << results.paths_processed << std::endl;
        std::cout << "  Paths Touched: " << results.paths_touched << std::endl;
        std::cout << "  Touch Rate: " << std::fixed << std::setprecision(4) 
                  << (static_cast<double>(results.paths_touched) / results.paths_processed) << std::endl;
        
        std::cout << std::endl;
        std::cout << "⚡ Performance Metrics:" << std::endl;
        std::cout << "  Total Time: " << std::fixed << std::setprecision(2) 
                  << results.metrics.computation_time_ms << " ms" << std::endl;
        std::cout << "  Kernel Time: " << std::fixed << std::setprecision(2) 
                  << results.metrics.kernel_time_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(0) 
                  << results.metrics.throughput_paths_per_sec << " paths/sec" << std::endl;
        std::cout << "  Memory Used: " << (results.metrics.memory_used_bytes / (1024*1024)) << " MB" << std::endl;
        
        std::cout << std::endl;
        std::cout << "🖥️  Device Used: " << results.device_used.name 
                  << " (Device " << results.device_used.device_id << ")" << std::endl;
        
        // Interpretation
        std::cout << std::endl;
        std::cout << "💡 INTERPRETATION:" << std::endl;
        std::cout << "==================" << std::endl;
        
        if (results.probability_of_touch > 0.7) {
            std::cout << "🔥 HIGH probability of touching $" << params.strike_price 
                     << " before expiration" << std::endl;
        } else if (results.probability_of_touch > 0.3) {
            std::cout << "⚖️  MODERATE probability of touching $" << params.strike_price 
                     << " before expiration" << std::endl;
        } else {
            std::cout << "❄️  LOW probability of touching $" << params.strike_price 
                     << " before expiration" << std::endl;
        }
        
        // Financial context
        double moneyness = (params.strike_price / params.current_price - 1.0) * 100;
        std::cout << "📍 Strike is " << std::fixed << std::setprecision(1) << std::abs(moneyness) << "% ";
        if (moneyness > 0) {
            std::cout << "above current price (OTM call)" << std::endl;
        } else if (moneyness < 0) {
            std::cout << "below current price (ITM call)" << std::endl;
        } else {
            std::cout << "at current price (ATM call)" << std::endl;
        }
        
    } else {
        std::cerr << "❌ Computation failed!" << std::endl;
        std::cerr << "Error: " << results.error_message << std::endl;
        std::cerr << "Error Code: " << montestrike::error_code_to_string(results.error_code) << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    std::cout << "⚠️  DISCLAIMER: This simulation is for educational purposes only." << std::endl;
    std::cout << "   It does not constitute financial advice. Trading options involves" << std::endl;
    std::cout << "   substantial risk and may not be suitable for all investors." << std::endl;
    
    return 0;
}