#pragma once

#include "types.hpp"
#include "device_analyzer.hpp"
#include "version.hpp"
#include <memory>

namespace montestrike {

class MonteCarloPoT {
public:
    struct Parameters {
        float current_price;            // S(t) - Current stock price
        float strike_price;             // K - Strike price to test for touch
        float time_to_expiration;       // T - Time to expiration in years
        float drift;                    // μ - Expected return rate (from historical data)
        float volatility;               // σ - Volatility (standard deviation of returns)
        uint32_t steps_per_day;         // Time resolution (default: 50, max: 100)
        uint32_t num_paths;             // Total paths to simulate (50K - 4M)
        bool use_antithetic_variates;   // Variance reduction technique
        uint32_t random_seed;           // For reproducible results (0 = system time)
        
        // Backend selection
        ComputeBackend backend;         // CUDA, AVX2, or CPU backend
        uint32_t cpu_threads;           // CPU threads to use (0 = auto-detect cores)
        bool strict_backend_mode;       // If true, fail if requested backend unavailable
        
        // Optional progress reporting
        ProgressCallback progress_callback;
        void* callback_user_data;
        uint32_t progress_report_interval_ms;
        
        // Device selection
        int32_t device_id;              // -1 for auto-select best device
        
        Parameters();
    };
    
    struct Results {
        float probability_of_touch;
        uint64_t paths_processed;
        uint64_t paths_touched;
        bool computation_successful;
        ErrorCode error_code;
        std::string error_message;
        ComputationMetrics metrics;
        DeviceInfo device_used;
        
        Results();
    };

public:
    MonteCarloPoT();
    ~MonteCarloPoT();
    
    Results estimate_pot(const Parameters& params);
    
    ValidationResult validate_parameters(const Parameters& params);
    
    ErrorCode initialize(int32_t device_id = -1);
    
    void shutdown();
    
    bool is_initialized() const;
    
    DeviceAnalyzer& get_device_analyzer();
    
    MemoryInfo get_memory_info();
    
    uint64_t estimate_memory_requirements(uint32_t num_paths);
    
    double estimate_computation_time(uint32_t num_paths);
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

ValidationResult validate_monte_carlo_parameters(
    float current_price,
    float strike_price, 
    float time_to_expiration,
    float drift,
    float volatility,
    uint32_t steps_per_day,
    uint32_t num_paths
);

std::string error_code_to_string(ErrorCode code);

} // namespace montestrike