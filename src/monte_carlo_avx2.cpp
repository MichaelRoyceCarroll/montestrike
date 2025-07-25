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

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace montestrike {

class Avx2MonteCarloCalculator {
public:
    static MonteCarloPoT::Results estimate_pot_avx2(
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
    static constexpr uint32_t SIMD_WIDTH = 8;  // AVX2 processes 8 floats at once
    
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
    
    static void worker_thread_avx2(WorkerData data);
    
#ifdef __AVX2__
    static uint32_t simulate_paths_simd(
        float current_price,
        float strike_price,
        float drift_term,
        float vol_sqrt_dt,
        uint32_t total_steps,
        std::mt19937& rng,
        std::normal_distribution<float>& normal_dist,
        uint32_t num_paths_batch,
        bool use_antithetic = false
    );
    
    
    static void generate_normal_random_avx2(
        std::mt19937& rng,
        std::normal_distribution<float>& normal_dist,
        float* output,
        uint32_t count,
        bool use_antithetic = false
    );
#endif
    
    static uint32_t simulate_paths_scalar_fallback(
        float current_price,
        float strike_price,
        float drift_term,
        float vol_sqrt_dt,
        uint32_t total_steps,
        std::mt19937& rng,
        std::normal_distribution<float>& normal_dist,
        uint32_t num_paths_batch,
        bool use_antithetic = false
    );
};

MonteCarloPoT::Results Avx2MonteCarloCalculator::estimate_pot_avx2(
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
        // Get CPU info and validate AVX2 support
        CpuAnalyzer& cpu_analyzer = get_cpu_analyzer();
        if (!cpu_analyzer.supports_avx2()) {
            results.error_code = ErrorCode::AVX2_NOT_SUPPORTED;
            results.error_message = "AVX2 instructions not supported on this CPU";
            return results;
        }
        
        results.device_used = cpu_analyzer.get_cpu_info();
        results.device_used.backend = ComputeBackend::AVX2;
        
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
            
            workers.emplace_back(worker_thread_avx2, data);
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
        results.metrics.kernel_time_ms = results.metrics.computation_time_ms;
        results.metrics.memory_transfer_time_ms = 0.0;
        results.metrics.throughput_paths_per_sec = 
            static_cast<double>(results.paths_processed) / (duration.count() / 1e6);
        
        // Estimate memory usage (including aligned buffers)
        size_t memory_per_thread = sizeof(std::mt19937) + sizeof(std::normal_distribution<float>) + 
                                   total_steps * SIMD_WIDTH * sizeof(float) * 3; // Prices, randoms, masks
        results.metrics.memory_used_bytes = memory_per_thread * num_threads;
        
        // Final progress report
        if (progress_callback) {
            progress_callback(1.0f, callback_user_data);
        }
        
    } catch (const std::exception& e) {
        results.error_code = ErrorCode::CPU_THREAD_ERROR;
        results.error_message = std::string("AVX2 Monte Carlo error: ") + e.what();
    } catch (...) {
        results.error_code = ErrorCode::UNKNOWN_ERROR;
        results.error_message = "Unknown error in AVX2 Monte Carlo calculation";
    }
    
    return results;
}

void Avx2MonteCarloCalculator::worker_thread_avx2(WorkerData data) {
    try {
        // Initialize random number generator
        std::mt19937 rng(data.random_seed);
        std::normal_distribution<float> normal_dist(0.0f, 1.0f);
        
        // Pre-calculate constants
        float dt = data.time_to_expiration / static_cast<float>(data.total_steps);
        float drift_term = (data.drift - 0.5f * data.volatility * data.volatility) * dt;
        float vol_sqrt_dt = data.volatility * std::sqrt(dt);
        
        uint64_t local_touched = 0;
        uint32_t total_paths = data.end_path - data.start_path;
        uint32_t paths_processed = 0;
        uint32_t last_reported_paths = 0;
        
        // Keep simple RNG approach for better performance
        
        // Process paths in simple SIMD batches (reverted from complex nested approach)
        while (paths_processed < total_paths && !data.should_stop->load()) {
            uint32_t remaining_paths = total_paths - paths_processed;
            uint32_t batch_size = std::min(remaining_paths, SIMD_WIDTH);
            
#ifdef __AVX2__
            uint32_t touched_in_batch = simulate_paths_simd(
                data.current_price,
                data.strike_price,
                drift_term,
                vol_sqrt_dt,
                data.total_steps,
                rng,
                normal_dist,
                batch_size,
                false
            );
#else
            uint32_t touched_in_batch = simulate_paths_scalar_fallback(
                data.current_price,
                data.strike_price,
                drift_term,
                vol_sqrt_dt,
                data.total_steps,
                rng,
                normal_dist,
                batch_size,
                false
            );
#endif
            
            local_touched += touched_in_batch;
            paths_processed += batch_size;
            
            // Antithetic variates (process another batch with negative randoms)
            if (data.use_antithetic_variates && paths_processed < total_paths) {
                remaining_paths = total_paths - paths_processed;
                batch_size = std::min(remaining_paths, SIMD_WIDTH);
                
#ifdef __AVX2__
                touched_in_batch = simulate_paths_simd(
                    data.current_price,
                    data.strike_price,
                    drift_term,
                    vol_sqrt_dt,
                    data.total_steps,
                    rng,
                    normal_dist,
                    batch_size,
                    true  // Use antithetic variates
                );
#else
                touched_in_batch = simulate_paths_scalar_fallback(
                    data.current_price,
                    data.strike_price,
                    drift_term,
                    vol_sqrt_dt,
                    data.total_steps,
                    rng,
                    normal_dist,
                    batch_size,
                    true
                );
#endif
                
                local_touched += touched_in_batch;
                paths_processed += batch_size;
            }
            
            // Update paths completed periodically (not every batch)
            if (paths_processed - last_reported_paths >= SIMD_WIDTH * 8) {
                data.paths_completed->fetch_add(paths_processed - last_reported_paths);
                last_reported_paths = paths_processed;
            }
        }
        
        // Update global counter
        data.paths_touched->fetch_add(local_touched);
        
    } catch (...) {
        data.should_stop->store(true);
    }
}

#ifdef __AVX2__
// Structure-of-Arrays for optimal SIMD cache locality  
struct SimdPathBatch {
    static constexpr uint32_t SIMD_WIDTH = 8;  // AVX2 processes 8 floats at once
    alignas(32) float prices[SIMD_WIDTH];         // All prices contiguous for SIMD load
    alignas(32) float randoms[SIMD_WIDTH];        // All randoms contiguous for SIMD load  
    alignas(32) uint32_t touched_mask[SIMD_WIDTH]; // All touch flags contiguous
    
    void reset(float initial_price) {
        for (int i = 0; i < SIMD_WIDTH; i++) {
            prices[i] = initial_price;
            touched_mask[i] = 0;
        }
    }
};

// Thread-local buffer pool for better memory reuse and cache locality
struct ThreadLocalSimdBuffers {
    static constexpr uint32_t BUFFER_POOL_SIZE = 4; // Multiple buffers to reduce contention
    SimdPathBatch buffers[BUFFER_POOL_SIZE];
    uint32_t next_buffer_index = 0;
    
    SimdPathBatch* get_buffer() {
        uint32_t index = next_buffer_index % BUFFER_POOL_SIZE;
        next_buffer_index++;
        return &buffers[index];
    }
};

// Fast AVX2 exponential function with Intel SVML optimization
static inline __m256 fast_exp_avx2(__m256 x) {
    // Clamp input to reasonable range to avoid overflow
    const __m256 max_input = _mm256_set1_ps(10.0f);
    const __m256 min_input = _mm256_set1_ps(-10.0f);
    x = _mm256_min_ps(_mm256_max_ps(x, min_input), max_input);
    
    // Use Intel SVML optimized exponential if available
#if defined(__INTEL_COMPILER) || (defined(__GNUC__) && defined(__AVX2__) && defined(__FAST_MATH__))
    // Intel SVML provides highly optimized _mm256_exp_ps
    return _mm256_exp_ps(x);
#else
    // Fallback to polynomial approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 sixth = _mm256_set1_ps(1.0f/6.0f);
    const __m256 twentyfourth = _mm256_set1_ps(1.0f/24.0f);
    
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    __m256 x4 = _mm256_mul_ps(x3, x);
    
    __m256 term1 = x;
    __m256 term2 = _mm256_mul_ps(x2, half);
    __m256 term3 = _mm256_mul_ps(x3, sixth);
    __m256 term4 = _mm256_mul_ps(x4, twentyfourth);
    
    __m256 result = _mm256_add_ps(one, term1);
    result = _mm256_add_ps(result, term2);
    result = _mm256_add_ps(result, term3);
    result = _mm256_add_ps(result, term4);
    
    return result;
#endif
}

uint32_t Avx2MonteCarloCalculator::simulate_paths_simd(
    float current_price,
    float strike_price,
    float drift_term,
    float vol_sqrt_dt,
    uint32_t total_steps,
    std::mt19937& rng,
    std::normal_distribution<float>& normal_dist,
    uint32_t num_paths_batch,
    bool use_antithetic) {
    
    // Get thread-local buffer pool for better memory reuse
    static thread_local ThreadLocalSimdBuffers buffer_pool;
    SimdPathBatch* batch = buffer_pool.get_buffer();
    
    // Initialize batch with starting prices
    batch->reset(current_price);
    
    // Load constants into SIMD vectors
    __m256 strike_vec = _mm256_set1_ps(strike_price);
    __m256 drift_vec = _mm256_set1_ps(drift_term);
    __m256 vol_vec = _mm256_set1_ps(vol_sqrt_dt);
    __m256 touched_vec = _mm256_setzero_ps();
    
    bool is_call = strike_price > current_price;
    
    // Check for immediate touch
    if (std::abs(current_price - strike_price) < 1e-6f) {
        return num_paths_batch;
    }
    
    // Keep prices in registers throughout simulation - eliminate store/load bottleneck
    __m256 price_vec = _mm256_set1_ps(current_price);  // Initialize prices in register
    
    // Simulate all paths in parallel - prices stay in registers
    for (uint32_t step = 0; step < total_steps; step++) {
        // Generate 8 random numbers directly into SoA buffer
        generate_normal_random_avx2(rng, normal_dist, batch->randoms, SimdPathBatch::SIMD_WIDTH, use_antithetic);
        
        // Load randoms only (prices already in register)
        __m256 random_vec = _mm256_load_ps(batch->randoms);
        
        // Calculate exp(drift_term + vol_sqrt_dt * random)
        __m256 exponent = _mm256_fmadd_ps(vol_vec, random_vec, drift_vec);
        
        // Fast AVX2 exponential approximation for better performance
        __m256 exp_vec = fast_exp_avx2(exponent);
        
        // Update prices in register: price *= exp(...)
        price_vec = _mm256_mul_ps(price_vec, exp_vec);
        
        // Check for touch using current register values
        __m256 touch_check;
        if (is_call) {
            // For calls: price >= strike
            touch_check = _mm256_cmp_ps(price_vec, strike_vec, _CMP_GE_OQ);
        } else {
            // For puts: price <= strike
            touch_check = _mm256_cmp_ps(price_vec, strike_vec, _CMP_LE_OQ);
        }
        
        // Accumulate touches (once touched, always touched)
        touched_vec = _mm256_or_ps(touched_vec, touch_check);
    }
    
    // Count touched paths using SoA touched_mask array
    _mm256_store_ps(reinterpret_cast<float*>(batch->touched_mask), touched_vec);
    uint32_t total_touched = 0;
    for (uint32_t i = 0; i < num_paths_batch; i++) {
        if (batch->touched_mask[i] != 0) {
            total_touched++;
        }
    }
    
    return total_touched;
}


void Avx2MonteCarloCalculator::generate_normal_random_avx2(
    std::mt19937& rng,
    std::normal_distribution<float>& normal_dist,
    float* output,
    uint32_t count,
    bool use_antithetic) {
    
    for (uint32_t i = 0; i < count; i++) {
        float random_val = normal_dist(rng);
        if (use_antithetic) {
            random_val = -random_val;
        }
        output[i] = random_val;
    }
}
#endif

uint32_t Avx2MonteCarloCalculator::simulate_paths_scalar_fallback(
    float current_price,
    float strike_price,
    float drift_term,
    float vol_sqrt_dt,
    uint32_t total_steps,
    std::mt19937& rng,
    std::normal_distribution<float>& normal_dist,
    uint32_t num_paths_batch,
    bool use_antithetic) {
    
    uint32_t touched_count = 0;
    bool is_call = strike_price > current_price;
    
    // Check for immediate touch
    if (std::abs(current_price - strike_price) < 1e-6f) {
        return num_paths_batch;
    }
    
    // Simulate each path individually (scalar fallback)
    for (uint32_t path = 0; path < num_paths_batch; path++) {
        float price = current_price;
        bool touched = false;
        
        for (uint32_t step = 0; step < total_steps && !touched; step++) {
            float random_val = normal_dist(rng);
            if (use_antithetic) {
                random_val = -random_val;
            }
            
            // Geometric Brownian Motion step
            price *= std::exp(drift_term + vol_sqrt_dt * random_val);
            
            // Check for touch
            if (is_call) {
                if (price >= strike_price) {
                    touched = true;
                }
            } else {
                if (price <= strike_price) {
                    touched = true;
                }
            }
        }
        
        if (touched) {
            touched_count++;
        }
    }
    
    return touched_count;
}

// External interface for AVX2 Monte Carlo
extern "C" {
    MonteCarloPoT::Results estimate_pot_avx2_impl(
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
        
        return Avx2MonteCarloCalculator::estimate_pot_avx2(
            current_price, strike_price, time_to_expiration,
            drift, volatility, steps_per_day, num_paths,
            use_antithetic_variates, random_seed, cpu_threads,
            progress_callback, callback_user_data, progress_report_interval_ms
        );
    }
}

} // namespace montestrike