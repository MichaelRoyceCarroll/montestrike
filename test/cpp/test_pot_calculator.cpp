#include "test_assertions.hpp"
#include "montestrike/montestrike.hpp"
#include <chrono>
#include <iomanip>

namespace montestrike_test {

bool test_parameter_validation() {
    print_test_header("Parameter Validation");
    
    // Test valid parameters
    auto valid_result = montestrike::validate_monte_carlo_parameters(
        100.0f, 105.0f, 0.0833f, 0.05f, 0.20f, 50, 100000
    );
    MONTESTRIKE_ASSERT_TRUE(valid_result.is_valid, "Valid parameters should pass validation");
    MONTESTRIKE_ASSERT_EQ(valid_result.error_code, montestrike::ErrorCode::SUCCESS, "Valid parameters should have SUCCESS error code");
    
    // Test invalid current price
    auto invalid_price = montestrike::validate_monte_carlo_parameters(
        -100.0f, 105.0f, 0.0833f, 0.05f, 0.20f, 50, 100000
    );
    MONTESTRIKE_ASSERT_FALSE(invalid_price.is_valid, "Negative current price should fail validation");
    
    // Test invalid volatility
    auto invalid_vol = montestrike::validate_monte_carlo_parameters(
        100.0f, 105.0f, 0.0833f, 0.05f, -0.20f, 50, 100000
    );
    MONTESTRIKE_ASSERT_FALSE(invalid_vol.is_valid, "Negative volatility should fail validation");
    
    // Test invalid path count
    auto invalid_paths = montestrike::validate_monte_carlo_parameters(
        100.0f, 105.0f, 0.0833f, 0.05f, 0.20f, 50, 500
    );
    MONTESTRIKE_ASSERT_FALSE(invalid_paths.is_valid, "Too few paths should fail validation");
    
    print_test_result("Parameter Validation", true);
    return true;
}

bool test_device_analyzer() {
    print_test_header("Device Analyzer");
    
    montestrike::DeviceAnalyzer analyzer;
    
    // Test device enumeration
    auto devices = analyzer.enumerate_devices();
    MONTESTRIKE_ASSERT_GT(devices.size(), 0, "Should find at least one CUDA device");
    
    // Test best device selection
    auto best_device = analyzer.get_best_device();
    MONTESTRIKE_ASSERT_TRUE(best_device.is_compatible, "Best device should be compatible");
    MONTESTRIKE_ASSERT_GT(best_device.compute_capability, 6.0f, "Best device should have compute capability >= 6.0");
    
    // Test memory info
    auto memory_info = analyzer.get_memory_info();
    MONTESTRIKE_ASSERT_GT(memory_info.total_bytes, 0, "Should report positive total memory");
    
    // Test CUDA version
    auto cuda_version = analyzer.get_cuda_version();
    MONTESTRIKE_ASSERT_GT(cuda_version.length(), 0, "Should report CUDA version");
    
    print_test_result("Device Analyzer", true);
    return true;
}

bool test_basic_monte_carlo() {
    print_test_header("Basic Monte Carlo Calculation");
    
    montestrike::MonteCarloPoT calculator;
    
    // Test initialization
    auto init_result = calculator.initialize();
    MONTESTRIKE_ASSERT_EQ(init_result, montestrike::ErrorCode::SUCCESS, "Calculator should initialize successfully");
    
    // Set up basic parameters
    montestrike::MonteCarloPoT::Parameters params;
    params.current_price = 100.0f;
    params.strike_price = 105.0f;
    params.time_to_expiration = 30.0f / 365.0f;  // 30 days
    params.drift = 0.05f;
    params.volatility = 0.20f;
    params.steps_per_day = 50;
    params.num_paths = 50000;  // Small for fast test
    params.random_seed = 12345;  // Fixed seed for reproducibility
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto results = calculator.estimate_pot(params);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Validate results
    MONTESTRIKE_ASSERT_TRUE(results.computation_successful, "Computation should succeed");
    MONTESTRIKE_ASSERT_EQ(results.error_code, montestrike::ErrorCode::SUCCESS, "Should have success error code");
    MONTESTRIKE_ASSERT_GT(results.probability_of_touch, 0.0f, "Probability should be positive");
    MONTESTRIKE_ASSERT_LT(results.probability_of_touch, 1.0f, "Probability should be less than 1");
    MONTESTRIKE_ASSERT_EQ(results.paths_processed, params.num_paths, "Should process all requested paths");
    MONTESTRIKE_ASSERT_GT(results.paths_touched, 0, "Some paths should touch the strike");
    
    // Performance checks
    MONTESTRIKE_ASSERT_GT(results.metrics.throughput_paths_per_sec, 1000.0, "Should achieve reasonable throughput");
    MONTESTRIKE_ASSERT_LT(execution_time, 10000.0, "Should complete within 10 seconds");
    
    std::cout << "   PoT: " << results.probability_of_touch << std::endl;
    std::cout << "   Paths touched: " << results.paths_touched << "/" << results.paths_processed << std::endl;
    std::cout << "   Throughput: " << results.metrics.throughput_paths_per_sec << " paths/sec" << std::endl;
    
    print_test_result("Basic Monte Carlo Calculation", true, execution_time);
    return true;
}

bool test_different_path_counts() {
    print_test_header("Different Path Counts");
    
    montestrike::MonteCarloPoT calculator;
    
    montestrike::MonteCarloPoT::Parameters base_params;
    base_params.current_price = 100.0f;
    base_params.strike_price = 102.0f;
    base_params.time_to_expiration = 7.0f / 365.0f;  // 1 week
    base_params.drift = 0.05f;
    base_params.volatility = 0.25f;
    base_params.steps_per_day = 50;
    base_params.random_seed = 42;
    
    std::vector<uint32_t> path_counts = {50000, 100000, 500000};
    
    for (uint32_t paths : path_counts) {
        auto params = base_params;
        params.num_paths = paths;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto results = calculator.estimate_pot(params);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        MONTESTRIKE_ASSERT_TRUE(results.computation_successful, "Computation should succeed for " + std::to_string(paths) + " paths");
        MONTESTRIKE_ASSERT_EQ(results.paths_processed, paths, "Should process all paths");
        
        std::cout << "   " << paths << " paths: PoT=" << results.probability_of_touch 
                  << ", time=" << execution_time << "ms" << std::endl;
    }
    
    print_test_result("Different Path Counts", true);
    return true;
}

bool test_antithetic_variates() {
    print_test_header("Antithetic Variates");
    
    montestrike::MonteCarloPoT calculator;
    
    montestrike::MonteCarloPoT::Parameters params;
    params.current_price = 100.0f;
    params.strike_price = 103.0f;
    params.time_to_expiration = 14.0f / 365.0f;  // 2 weeks
    params.drift = 0.05f;
    params.volatility = 0.30f;
    params.steps_per_day = 50;
    params.num_paths = 100000;
    params.random_seed = 777;
    
    // Test without antithetic variates
    params.use_antithetic_variates = false;
    auto results_normal = calculator.estimate_pot(params);
    
    // Test with antithetic variates
    params.use_antithetic_variates = true;
    auto results_antithetic = calculator.estimate_pot(params);
    
    MONTESTRIKE_ASSERT_TRUE(results_normal.computation_successful, "Normal computation should succeed");
    MONTESTRIKE_ASSERT_TRUE(results_antithetic.computation_successful, "Antithetic computation should succeed");
    
    // Results should be close but not identical
    double diff = std::abs(results_normal.probability_of_touch - results_antithetic.probability_of_touch);
    MONTESTRIKE_ASSERT_LT(diff, 0.1, "Results with/without antithetic variates should be similar");
    
    std::cout << "   Normal: " << results_normal.probability_of_touch << std::endl;
    std::cout << "   Antithetic: " << results_antithetic.probability_of_touch << std::endl;
    std::cout << "   Difference: " << diff << std::endl;
    
    print_test_result("Antithetic Variates", true);
    return true;
}

bool test_edge_cases() {
    print_test_header("Edge Cases");
    
    montestrike::MonteCarloPoT calculator;
    
    // Test strike at current price (should have high PoT)
    montestrike::MonteCarloPoT::Parameters params_at_money;
    params_at_money.current_price = 100.0f;
    params_at_money.strike_price = 100.0f;  // Exactly at money
    params_at_money.time_to_expiration = 30.0f / 365.0f;
    params_at_money.drift = 0.05f;
    params_at_money.volatility = 0.20f;
    params_at_money.steps_per_day = 50;
    params_at_money.num_paths = 50000;
    
    auto results_at_money = calculator.estimate_pot(params_at_money);
    MONTESTRIKE_ASSERT_TRUE(results_at_money.computation_successful, "At-the-money calculation should succeed");
    MONTESTRIKE_ASSERT_GT(results_at_money.probability_of_touch, 0.8f, "At-the-money should have high PoT");
    
    // Test very short expiration
    montestrike::MonteCarloPoT::Parameters params_short;
    params_short.current_price = 100.0f;
    params_short.strike_price = 105.0f;
    params_short.time_to_expiration = 1.0f / 365.0f;  // 1 day
    params_short.drift = 0.05f;
    params_short.volatility = 0.20f;
    params_short.steps_per_day = 24;  // Hourly steps
    params_short.num_paths = 50000;
    
    auto results_short = calculator.estimate_pot(params_short);
    MONTESTRIKE_ASSERT_TRUE(results_short.computation_successful, "Short expiration calculation should succeed");
    MONTESTRIKE_ASSERT_LT(results_short.probability_of_touch, 0.5f, "Short expiration OTM should have lower PoT");
    
    std::cout << "   At-the-money PoT: " << results_at_money.probability_of_touch << std::endl;
    std::cout << "   Short expiration PoT: " << results_short.probability_of_touch << std::endl;
    
    print_test_result("Edge Cases", true);
    return true;
}

bool test_cpu_backend() {
    print_test_header("CPU Backend");
    
    montestrike::MonteCarloPoT calculator;
    
    // Test CPU backend with fixed parameters
    montestrike::MonteCarloPoT::Parameters params;
    params.current_price = 100.0f;
    params.strike_price = 105.0f;
    params.time_to_expiration = 30.0f / 365.0f;
    params.drift = 0.05f;
    params.volatility = 0.20f;
    params.steps_per_day = 50;
    params.num_paths = 100000;  // Moderate size for CPU
    params.random_seed = 12345;
    params.backend = montestrike::ComputeBackend::CPU;
    params.strict_backend_mode = true;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto results = calculator.estimate_pot(params);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Validate results
    MONTESTRIKE_ASSERT_TRUE(results.computation_successful, "CPU computation should succeed");
    MONTESTRIKE_ASSERT_EQ(results.error_code, montestrike::ErrorCode::SUCCESS, "Should have success error code");
    MONTESTRIKE_ASSERT_GT(results.probability_of_touch, 0.0f, "Probability should be positive");
    MONTESTRIKE_ASSERT_LT(results.probability_of_touch, 1.0f, "Probability should be less than 1");
    MONTESTRIKE_ASSERT_EQ(results.paths_processed, params.num_paths, "Should process all requested paths");
    MONTESTRIKE_ASSERT_GT(results.paths_touched, 0, "Some paths should touch the strike");
    
    // Performance checks - more lenient for CPU
    MONTESTRIKE_ASSERT_GT(results.metrics.throughput_paths_per_sec, 100.0, "Should achieve reasonable CPU throughput");
    MONTESTRIKE_ASSERT_LT(execution_time, 30000.0, "Should complete within 30 seconds on CPU");
    
    std::cout << "   PoT: " << results.probability_of_touch << std::endl;
    std::cout << "   Paths touched: " << results.paths_touched << "/" << results.paths_processed << std::endl;
    std::cout << "   Throughput: " << results.metrics.throughput_paths_per_sec << " paths/sec" << std::endl;
    std::cout << "   Device: " << results.device_used.name << std::endl;
    
    print_test_result("CPU Backend", true, execution_time);
    return true;
}

bool test_backend_comparison() {
    print_test_header("Backend Comparison");
    
    montestrike::MonteCarloPoT calculator;
    
    // Shared test parameters
    montestrike::MonteCarloPoT::Parameters base_params;
    base_params.current_price = 100.0f;
    base_params.strike_price = 105.0f;
    base_params.time_to_expiration = 7.0f / 365.0f;
    base_params.drift = 0.05f;
    base_params.volatility = 0.25f;
    base_params.steps_per_day = 50;
    base_params.num_paths = 100000;  // Same path count for fair comparison
    base_params.random_seed = 42;    // Same seed for reproducibility
    base_params.strict_backend_mode = true;
    
    std::vector<uint32_t> path_counts = {50000, 100000, 500000, 1000000};
    
    for (uint32_t paths : path_counts) {
        std::cout << "\n   Testing " << paths << " paths:" << std::endl;
        
        // Test CPU backend
        auto cpu_params = base_params;
        cpu_params.backend = montestrike::ComputeBackend::CPU;
        cpu_params.num_paths = paths;
        
        auto cpu_start = std::chrono::high_resolution_clock::now();
        auto cpu_results = calculator.estimate_pot(cpu_params);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        
        if (cpu_results.computation_successful) {
            std::cout << "     CPU: " << cpu_time << "ms, " 
                      << cpu_results.metrics.throughput_paths_per_sec << " paths/sec" << std::endl;
        } else {
            std::cout << "     CPU: Failed - " << cpu_results.error_message << std::endl;
        }
        
        // Test CUDA backend if available
        montestrike::DeviceAnalyzer analyzer;
        auto devices = analyzer.enumerate_devices();
        if (!devices.empty()) {
            auto cuda_params = base_params;
            cuda_params.backend = montestrike::ComputeBackend::CUDA;
            cuda_params.num_paths = paths;
            
            auto cuda_start = std::chrono::high_resolution_clock::now();
            auto cuda_results = calculator.estimate_pot(cuda_params);
            auto cuda_end = std::chrono::high_resolution_clock::now();
            
            double cuda_time = std::chrono::duration<double, std::milli>(cuda_end - cuda_start).count();
            
            if (cuda_results.computation_successful) {
                std::cout << "     CUDA: " << cuda_time << "ms, " 
                          << cuda_results.metrics.throughput_paths_per_sec << " paths/sec";
                if (cpu_results.computation_successful) {
                    double speedup = cpu_time / cuda_time;
                    std::cout << " (" << std::fixed << std::setprecision(1) << speedup << "x speedup)";
                }
                std::cout << std::endl;
            } else {
                std::cout << "     CUDA: Failed - " << cuda_results.error_message << std::endl;
            }
        } else {
            std::cout << "     CUDA: Not available" << std::endl;
        }
    }
    
    print_test_result("Backend Comparison", true);
    return true;
}

} // namespace montestrike_test