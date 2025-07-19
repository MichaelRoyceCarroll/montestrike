#include "montestrike/montestrike.hpp"
#include "montestrike/types.hpp"
#include "montestrike/aligned_memory.hpp"
#include "montestrike/cpu_analyzer.hpp"
#include <random>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace montestrike {

class CpuMonteCarloCalculator {
public:
    static MonteCarloPoT::Results calculate_pot_cpu(
        float current_price,
        float strike_price,
        float time_to_expiration,
        float drift,
        float volatility,
        uint32_t steps_per_day,
        uint32_t num_paths,
        bool use_antithetic_variates,
        uint32_t random_seed,
        uint32_t cpu_threads,
        ProgressCallback progress_callback = nullptr,
        void* callback_user_data = nullptr,
        uint32_t progress_report_interval_ms = 1000
    );

private:
    struct WorkerData {
        uint32_t worker_id;
        uint32_t start_path;
        uint32_t end_path;
        float current_price;
        float strike_price;
        float time_to_expiration;
        float drift;
        float volatility;
        uint32_t total_steps;
        bool use_antithetic_variates;
        uint32_t random_seed;
        std::atomic<uint64_t>* paths_touched;
        std::atomic<uint64_t>* paths_completed;
        std::atomic<bool>* should_stop;
    };
    
    static void worker_thread(WorkerData data);
    static bool simulate_path(
        float current_price,
        float strike_price,
        float drift_term,
        float vol_sqrt_dt,
        uint32_t total_steps,
        std::mt19937& rng,
        std::normal_distribution<float>& normal_dist,
        bool use_antithetic = false
    );
};

MonteCarloPoT::Results CpuMonteCarloCalculator::calculate_pot_cpu(
    float current_price,
    float strike_price,
    float time_to_expiration,
    float drift,
    float volatility,
    uint32_t steps_per_day,
    uint32_t num_paths,
    bool use_antithetic_variates,
    uint32_t random_seed,
    uint32_t cpu_threads,
    ProgressCallback progress_callback,
    void* callback_user_data,
    uint32_t progress_report_interval_ms) {
    
    MonteCarloPoT::Results results;
    results.computation_successful = false;
    results.error_code = ErrorCode::SUCCESS;
    results.probability_of_touch = 0.0f;
    results.paths_processed = 0;
    results.paths_touched = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Get CPU info
        CpuAnalyzer& cpu_analyzer = get_cpu_analyzer();
        results.device_used = cpu_analyzer.get_cpu_info();
        
        // Determine number of threads
        uint32_t num_threads = cpu_threads;
        if (num_threads == 0) {
            num_threads = cpu_analyzer.get_optimal_thread_count();
        }
        num_threads = std::min(num_threads, cpu_analyzer.get_logical_cores());
        num_threads = std::max(1u, num_threads);
        
        // Calculate simulation parameters
        float days_to_expiration = time_to_expiration * 365.0f;
        uint32_t total_steps = static_cast<uint32_t>(std::ceil(days_to_expiration * steps_per_day));
        total_steps = std::max(1u, total_steps);
        
        // Shared atomic counters
        std::atomic<uint64_t> paths_touched{0};
        std::atomic<uint64_t> paths_completed{0};
        std::atomic<bool> should_stop{false};
        
        // Create worker threads
        std::vector<std::thread> workers;
        uint32_t paths_per_thread = num_paths / num_threads;
        uint32_t remaining_paths = num_paths % num_threads;
        
        uint32_t current_start = 0;
        for (uint32_t i = 0; i < num_threads; i++) {
            uint32_t thread_paths = paths_per_thread + (i < remaining_paths ? 1 : 0);
            
            WorkerData data;
            data.worker_id = i;
            data.start_path = current_start;
            data.end_path = current_start + thread_paths;
            data.current_price = current_price;
            data.strike_price = strike_price;
            data.time_to_expiration = time_to_expiration;
            data.drift = drift;
            data.volatility = volatility;
            data.total_steps = total_steps;
            data.use_antithetic_variates = use_antithetic_variates;
            data.random_seed = random_seed + i;  // Different seed per thread
            data.paths_touched = &paths_touched;
            data.paths_completed = &paths_completed;
            data.should_stop = &should_stop;
            
            workers.emplace_back(worker_thread, data);
            current_start += thread_paths;
        }
        
        // Progress reporting
        std::thread progress_thread;
        if (progress_callback) {
            progress_thread = std::thread([&]() {
                while (!should_stop.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(progress_report_interval_ms));
                    uint64_t completed = paths_completed.load();
                    float progress = static_cast<float>(completed) / static_cast<float>(num_paths);
                    progress_callback(std::min(progress, 1.0f), callback_user_data);
                    
                    if (completed >= num_paths) {
                        break;
                    }
                }
            });
        }
        
        // Wait for all workers to complete
        for (auto& worker : workers) {
            worker.join();
        }
        
        should_stop.store(true);
        if (progress_thread.joinable()) {
            progress_thread.join();
        }
        
        // Calculate results
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        results.paths_processed = paths_completed.load();
        results.paths_touched = paths_touched.load();
        results.probability_of_touch = static_cast<float>(results.paths_touched) / 
                                      static_cast<float>(results.paths_processed);
        results.computation_successful = true;
        
        // Fill metrics
        results.metrics.computation_time_ms = duration.count() / 1000.0;
        results.metrics.kernel_time_ms = results.metrics.computation_time_ms; // No separate kernel for CPU
        results.metrics.memory_transfer_time_ms = 0.0; // No memory transfer for CPU
        results.metrics.throughput_paths_per_sec = 
            static_cast<double>(results.paths_processed) / (duration.count() / 1e6);
        
        // Estimate memory usage (rough calculation)
        size_t memory_per_thread = sizeof(std::mt19937) + sizeof(std::normal_distribution<float>) + 
                                   total_steps * sizeof(float) * 2; // Price array + random numbers
        results.metrics.memory_used_bytes = memory_per_thread * num_threads;
        
        // Final progress report
        if (progress_callback) {
            progress_callback(1.0f, callback_user_data);
        }
        
    } catch (const std::exception& e) {
        results.error_code = ErrorCode::CPU_THREAD_ERROR;
        results.error_message = std::string("CPU Monte Carlo error: ") + e.what();
    } catch (...) {
        results.error_code = ErrorCode::UNKNOWN_ERROR;
        results.error_message = "Unknown error in CPU Monte Carlo calculation";
    }
    
    return results;
}

void CpuMonteCarloCalculator::worker_thread(WorkerData data) {
    try {
        // Initialize random number generator
        std::mt19937 rng(data.random_seed);
        std::normal_distribution<float> normal_dist(0.0f, 1.0f);
        
        // Pre-calculate constants
        float dt = data.time_to_expiration / static_cast<float>(data.total_steps);
        float drift_term = (data.drift - 0.5f * data.volatility * data.volatility) * dt;
        float vol_sqrt_dt = data.volatility * std::sqrt(dt);
        
        uint64_t local_touched = 0;
        uint32_t paths_in_batch = data.end_path - data.start_path;
        
        // Process paths
        for (uint32_t path = data.start_path; path < data.end_path; path++) {
            if (data.should_stop->load()) {
                break;
            }
            
            bool touched = simulate_path(
                data.current_price,
                data.strike_price,
                drift_term,
                vol_sqrt_dt,
                data.total_steps,
                rng,
                normal_dist
            );
            
            if (touched) {
                local_touched++;
            }
            
            // Antithetic variate (if enabled and this isn't the last path for odd total)
            if (data.use_antithetic_variates && (path + 1) < data.end_path) {
                // Use negative random numbers for antithetic path
                bool antithetic_touched = simulate_path(
                    data.current_price,
                    data.strike_price,
                    drift_term,
                    vol_sqrt_dt,
                    data.total_steps,
                    rng,
                    normal_dist,
                    true  // Use negative randoms
                );
                
                if (antithetic_touched) {
                    local_touched++;
                }
                
                path++; // Skip next iteration since we processed two paths
                data.paths_completed->fetch_add(2);
            } else {
                data.paths_completed->fetch_add(1);
            }
        }
        
        // Update global counter
        data.paths_touched->fetch_add(local_touched);
        
    } catch (...) {
        data.should_stop->store(true);
    }
}

bool CpuMonteCarloCalculator::simulate_path(
    float current_price,
    float strike_price,
    float drift_term,
    float vol_sqrt_dt,
    uint32_t total_steps,
    std::mt19937& rng,
    std::normal_distribution<float>& normal_dist,
    bool use_antithetic) {
    
    float price = current_price;
    bool is_call = strike_price > current_price;
    
    // Check for immediate touch
    if (std::abs(price - strike_price) < 1e-6f) {
        return true;
    }
    
    // Simulate price path
    for (uint32_t step = 0; step < total_steps; step++) {
        float random_val = normal_dist(rng);
        if (use_antithetic) {
            random_val = -random_val;
        }
        
        // Geometric Brownian Motion step
        price *= std::exp(drift_term + vol_sqrt_dt * random_val);
        
        // Check for touch
        if (is_call) {
            if (price >= strike_price) {
                return true;
            }
        } else {
            if (price <= strike_price) {
                return true;
            }
        }
    }
    
    return false;
}

// External interface for CPU Monte Carlo
extern "C" {
    MonteCarloPoT::Results calculate_pot_cpu_impl(
        float current_price,
        float strike_price,
        float time_to_expiration,
        float drift,
        float volatility,
        uint32_t steps_per_day,
        uint32_t num_paths,
        bool use_antithetic_variates,
        uint32_t random_seed,
        uint32_t cpu_threads,
        ProgressCallback progress_callback,
        void* callback_user_data,
        uint32_t progress_report_interval_ms) {
        
        return CpuMonteCarloCalculator::calculate_pot_cpu(
            current_price, strike_price, time_to_expiration,
            drift, volatility, steps_per_day, num_paths,
            use_antithetic_variates, random_seed, cpu_threads,
            progress_callback, callback_user_data, progress_report_interval_ms
        );
    }
}

} // namespace montestrike