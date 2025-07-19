#include "montestrike/montestrike.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <ctime>

// Forward declarations for CUDA kernel functions
extern "C" {
    cudaError_t launch_setup_random_states(curandState* d_states, unsigned long seed, int num_paths, int block_size);
    cudaError_t launch_monte_carlo_kernel(curandState* d_random_states, float current_price, float strike_price, 
                                        float drift, float volatility, float dt, int steps_per_day, int total_steps,
                                        int num_paths, bool use_antithetic_variates, bool* d_touch_results, 
                                        float* d_final_prices, int block_size);
    cudaError_t launch_reduce_kernel(bool* d_touch_results, int num_paths, unsigned long long* d_touch_count, int block_size);
    montestrike::ErrorCode allocate_simulation_memory(uint32_t num_paths, void** buffers_handle);
    void free_simulation_memory();
    uint64_t estimate_memory_requirements(uint32_t num_paths);
}

namespace montestrike {

// Parameter defaults
MonteCarloPoT::Parameters::Parameters() 
    : current_price(0.0f)
    , strike_price(0.0f)
    , time_to_expiration(0.0f)
    , drift(0.0f)
    , volatility(0.0f)
    , steps_per_day(50)
    , num_paths(100000)
    , use_antithetic_variates(false)
    , random_seed(0)
    , progress_callback(nullptr)
    , callback_user_data(nullptr)
    , progress_report_interval_ms(100)
    , device_id(-1)
{}

// Results defaults
MonteCarloPoT::Results::Results()
    : probability_of_touch(0.0f)
    , paths_processed(0)
    , paths_touched(0)
    , computation_successful(false)
    , error_code(ErrorCode::UNKNOWN_ERROR)
    , error_message("")
{}

// Implementation class
class MonteCarloPoT::Impl {
public:
    Impl() : device_analyzer_(), initialized_(false) {}
    
    ~Impl() {
        shutdown();
    }
    
    ErrorCode initialize(int32_t device_id) {
        if (initialized_) {
            return ErrorCode::SUCCESS;
        }
        
        // Check if CUDA is available
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            return ErrorCode::NO_CUDA_DEVICE;
        }
        
        // Select device
        if (device_id < 0) {
            auto best_device = device_analyzer_.get_best_device();
            if (!best_device.is_compatible) {
                return ErrorCode::INCOMPATIBLE_DEVICE;
            }
            device_id = best_device.device_id;
        }
        
        ErrorCode result = device_analyzer_.set_device(device_id);
        if (result != ErrorCode::SUCCESS) {
            return result;
        }
        
        current_device_ = device_id;
        initialized_ = true;
        return ErrorCode::SUCCESS;
    }
    
    void shutdown() {
        if (initialized_) {
            free_simulation_memory();
            initialized_ = false;
        }
    }
    
    Results calculate_pot(const Parameters& params) {
        Results results;
        
        if (!initialized_) {
            ErrorCode init_result = initialize(params.device_id);
            if (init_result != ErrorCode::SUCCESS) {
                results.error_code = init_result;
                results.error_message = error_code_to_string(init_result);
                return results;
            }
        }
        
        // Validate parameters
        auto validation = validate_monte_carlo_parameters(
            params.current_price, params.strike_price, params.time_to_expiration,
            params.drift, params.volatility, params.steps_per_day, params.num_paths
        );
        
        if (!validation.is_valid) {
            results.error_code = validation.error_code;
            results.error_message = validation.error_message;
            return results;
        }
        
        // Check device memory
        if (!device_analyzer_.has_sufficient_memory(current_device_, params.num_paths)) {
            results.error_code = ErrorCode::MEMORY_ALLOCATION_FAILED;
            results.error_message = "Insufficient GPU memory for requested path count";
            return results;
        }
        
        // Route to appropriate implementation based on progress callback
        if (params.progress_callback != nullptr) {
            return calculate_pot_with_progress(params);
        } else {
            return calculate_pot_fast_path(params);
        }
    }
    
    Results calculate_pot_fast_path(const Parameters& params) {
        Results results;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Allocate GPU memory
            void* buffers_handle;
            ErrorCode alloc_result = allocate_simulation_memory(params.num_paths, &buffers_handle);
            if (alloc_result != ErrorCode::SUCCESS) {
                results.error_code = alloc_result;
                results.error_message = error_code_to_string(alloc_result);
                return results;
            }
            
            // Cast to actual buffer types
            struct AllocatedBuffers {
                curandState* random_states;
                bool* touch_results;
                float* final_prices;
                unsigned long long* touch_count;
                uint32_t allocated_paths;
            };
            AllocatedBuffers* buffers = static_cast<AllocatedBuffers*>(buffers_handle);
            
            // Calculate simulation parameters
            float dt = 1.0f / (365.0f * params.steps_per_day);
            int total_steps = static_cast<int>(params.time_to_expiration * 365.0f * params.steps_per_day);
            
            // Setup random number generation
            unsigned long seed = params.random_seed;
            if (seed == 0) {
                seed = static_cast<unsigned long>(std::time(nullptr));
            }
            
            uint32_t block_size = device_analyzer_.get_optimal_block_size(current_device_);
            
            auto kernel_start = std::chrono::high_resolution_clock::now();
            
            // Initialize random states
            cudaError_t err = launch_setup_random_states(buffers->random_states, seed, params.num_paths, block_size);
            if (err != cudaSuccess) {
                results.error_code = ErrorCode::KERNEL_LAUNCH_FAILED;
                results.error_message = "Failed to initialize random states: " + std::string(cudaGetErrorString(err));
                return results;
            }
            cudaDeviceSynchronize();
            
            // Launch Monte Carlo kernel
            err = launch_monte_carlo_kernel(
                buffers->random_states,
                params.current_price,
                params.strike_price,
                params.drift,
                params.volatility,
                dt,
                params.steps_per_day,
                total_steps,
                params.num_paths,
                params.use_antithetic_variates,
                buffers->touch_results,
                buffers->final_prices,
                block_size
            );
            
            if (err != cudaSuccess) {
                results.error_code = ErrorCode::KERNEL_LAUNCH_FAILED;
                results.error_message = "Monte Carlo kernel failed: " + std::string(cudaGetErrorString(err));
                return results;
            }
            cudaDeviceSynchronize();
            
            // Reduce results
            err = launch_reduce_kernel(buffers->touch_results, params.num_paths, buffers->touch_count, block_size);
            if (err != cudaSuccess) {
                results.error_code = ErrorCode::KERNEL_LAUNCH_FAILED;
                results.error_message = "Reduction kernel failed: " + std::string(cudaGetErrorString(err));
                return results;
            }
            cudaDeviceSynchronize();
            
            auto kernel_end = std::chrono::high_resolution_clock::now();
            
            // Copy result back to host
            unsigned long long touch_count;
            err = cudaMemcpy(&touch_count, buffers->touch_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                results.error_code = ErrorCode::MEMORY_COPY_FAILED;
                results.error_message = "Failed to copy results: " + std::string(cudaGetErrorString(err));
                return results;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            
            // Calculate results
            results.paths_processed = params.num_paths;
            results.paths_touched = touch_count;
            results.probability_of_touch = static_cast<float>(touch_count) / static_cast<float>(params.num_paths);
            results.computation_successful = true;
            results.error_code = ErrorCode::SUCCESS;
            
            // Calculate metrics
            double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            double kernel_time = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
            
            results.metrics.computation_time_ms = total_time;
            results.metrics.kernel_time_ms = kernel_time;
            results.metrics.memory_transfer_time_ms = total_time - kernel_time;
            results.metrics.throughput_paths_per_sec = (params.num_paths * 1000.0) / total_time;
            results.metrics.memory_used_bytes = ::estimate_memory_requirements(params.num_paths);
            
            // Get device info
            results.device_used = device_analyzer_.get_device_info(current_device_);
            
        } catch (const std::exception& e) {
            results.error_code = ErrorCode::UNKNOWN_ERROR;
            results.error_message = "Exception during computation: " + std::string(e.what());
        }
        
        return results;
    }
    
    Results calculate_pot_with_progress(const Parameters& params) {
        // For now, just call fast path - progress reporting can be added later
        return calculate_pot_fast_path(params);
    }
    
    ValidationResult validate_parameters(const Parameters& params) {
        return validate_monte_carlo_parameters(
            params.current_price, params.strike_price, params.time_to_expiration,
            params.drift, params.volatility, params.steps_per_day, params.num_paths
        );
    }
    
    DeviceAnalyzer device_analyzer_;
    bool initialized_;
    int32_t current_device_;
};

// MonteCarloPoT public interface
MonteCarloPoT::MonteCarloPoT() : pimpl(std::make_unique<Impl>()) {}

MonteCarloPoT::~MonteCarloPoT() = default;

MonteCarloPoT::Results MonteCarloPoT::calculate_pot(const Parameters& params) {
    return pimpl->calculate_pot(params);
}

ValidationResult MonteCarloPoT::validate_parameters(const Parameters& params) {
    return pimpl->validate_parameters(params);
}

ErrorCode MonteCarloPoT::initialize(int32_t device_id) {
    return pimpl->initialize(device_id);
}

void MonteCarloPoT::shutdown() {
    pimpl->shutdown();
}

bool MonteCarloPoT::is_initialized() const {
    return pimpl->initialized_;
}

DeviceAnalyzer& MonteCarloPoT::get_device_analyzer() {
    return pimpl->device_analyzer_;
}

MemoryInfo MonteCarloPoT::get_memory_info() {
    return pimpl->device_analyzer_.get_memory_info();
}

uint64_t MonteCarloPoT::estimate_memory_requirements(uint32_t num_paths) {
    return ::estimate_memory_requirements(num_paths);
}

double MonteCarloPoT::estimate_computation_time(uint32_t num_paths) {
    // Rough estimate based on typical performance
    // This would be refined based on actual benchmarks
    double paths_per_ms = 1000.0; // Assume 1M paths per second baseline
    return num_paths / paths_per_ms;
}

} // namespace montestrike