#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cmath>

extern "C" {

__device__ float generate_normal_random(curandState* state) {
    return curand_normal(state);
}

__global__ void setup_random_states(curandState* states, unsigned long seed, int num_paths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void monte_carlo_pot_kernel(
    curandState* random_states,     // Pre-initialized random states
    float current_price,            // S(t)
    float strike_price,             // K
    float drift,                    // μ 
    float volatility,               // σ
    float dt,                       // time step
    int steps_per_day,
    int total_steps,
    int effective_paths,            // Number of thread paths to process
    bool use_antithetic_variates,
    bool* touch_results,            // Output: did path touch strike?
    float* final_prices             // Optional: final prices for debugging
) {
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= effective_paths) return;
    
    // Pre-calculate constants for performance
    const float drift_term = (drift - 0.5f * volatility * volatility) * dt;
    const float vol_sqrt_dt = volatility * sqrtf(dt);
    
    if (use_antithetic_variates) {
        // Process pair: original + antithetic (2 paths per thread)
        bool touched1 = false, touched2 = false;
        
        // Process original path
        float price1 = current_price;
        if (fabsf(current_price - strike_price) < 1e-6f) {
            touched1 = true;
        } else {
            bool check_upward = (strike_price > current_price);
            for (int step = 0; step < total_steps && !touched1; step++) {
                float random_val = generate_normal_random(&random_states[path_id]);
                price1 *= expf(drift_term + vol_sqrt_dt * random_val);
                
                if (check_upward) {
                    if (price1 >= strike_price) touched1 = true;
                } else {
                    if (price1 <= strike_price) touched1 = true;
                }
            }
        }
        
        // Process antithetic path (reset RNG to same state)
        curandState temp_state = random_states[path_id];
        float price2 = current_price;
        if (fabsf(current_price - strike_price) < 1e-6f) {
            touched2 = true;
        } else {
            bool check_upward = (strike_price > current_price);
            for (int step = 0; step < total_steps && !touched2; step++) {
                float random_val = -generate_normal_random(&temp_state); // Negative for antithetic
                price2 *= expf(drift_term + vol_sqrt_dt * random_val);
                
                if (check_upward) {
                    if (price2 >= strike_price) touched2 = true;
                } else {
                    if (price2 <= strike_price) touched2 = true;
                }
            }
        }
        
        // Store results for both paths
        touch_results[path_id * 2] = touched1;
        touch_results[path_id * 2 + 1] = touched2;
        if (final_prices) {
            final_prices[path_id * 2] = price1;
            final_prices[path_id * 2 + 1] = price2;
        }
    } else {
        // Process single path
        float price = current_price;
        bool touched = false;
        
        if (fabsf(current_price - strike_price) < 1e-6f) {
            touched = true;
        } else {
            bool check_upward = (strike_price > current_price);
            for (int step = 0; step < total_steps && !touched; step++) {
                float random_val = generate_normal_random(&random_states[path_id]);
                price *= expf(drift_term + vol_sqrt_dt * random_val);
                
                if (check_upward) {
                    if (price >= strike_price) touched = true;
                } else {
                    if (price <= strike_price) touched = true;
                }
            }
        }
        
        touch_results[path_id] = touched;
        if (final_prices) {
            final_prices[path_id] = price;
        }
    }
}

__global__ void reduce_touch_results(
    bool* touch_results,
    int num_paths,
    unsigned long long* touch_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    unsigned long long local_count = 0;
    
    for (int i = idx; i < num_paths; i += stride) {
        if (touch_results[i]) {
            local_count++;
        }
    }
    
    // Use atomicAdd to safely accumulate across all threads
    atomicAdd(touch_count, local_count);
}

// Host functions for kernel management
cudaError_t launch_setup_random_states(
    curandState* d_states, 
    unsigned long seed, 
    int num_paths,
    int block_size
) {
    int grid_size = (num_paths + block_size - 1) / block_size;
    setup_random_states<<<grid_size, block_size>>>(d_states, seed, num_paths);
    return cudaGetLastError();
}

cudaError_t launch_monte_carlo_kernel(
    curandState* d_random_states,
    float current_price,
    float strike_price,
    float drift,
    float volatility,
    float dt,
    int steps_per_day,
    int total_steps,
    int num_paths,
    bool use_antithetic_variates,
    bool* d_touch_results,
    float* d_final_prices,
    int block_size
) {
    int effective_paths = use_antithetic_variates ? num_paths / 2 : num_paths;
    int grid_size = (effective_paths + block_size - 1) / block_size;
    
    monte_carlo_pot_kernel<<<grid_size, block_size>>>(
        d_random_states,
        current_price,
        strike_price,
        drift,
        volatility,
        dt,
        steps_per_day,
        total_steps,
        effective_paths,  // Pass effective_paths, not num_paths!
        use_antithetic_variates,
        d_touch_results,
        d_final_prices
    );
    
    return cudaGetLastError();
}

cudaError_t launch_reduce_kernel(
    bool* d_touch_results,
    int num_paths,
    unsigned long long* d_touch_count,
    int block_size
) {
    int grid_size = min(32, (num_paths + block_size - 1) / block_size);
    
    // Initialize count to zero
    cudaMemset(d_touch_count, 0, sizeof(unsigned long long));
    
    reduce_touch_results<<<grid_size, block_size>>>(
        d_touch_results,
        num_paths,  // This should be the total paths (original num_paths)
        d_touch_count
    );
    
    return cudaGetLastError();
}

// Memory estimation functions
size_t calculate_memory_requirements(int num_paths) {
    size_t random_states_size = num_paths * sizeof(curandState);
    size_t touch_results_size = num_paths * sizeof(bool);
    size_t final_prices_size = num_paths * sizeof(float);
    size_t touch_count_size = sizeof(unsigned long long);
    
    return random_states_size + touch_results_size + final_prices_size + touch_count_size;
}

// Optimal configuration functions
int get_optimal_block_size(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    // Use multiple of warp size, typically 256 or 512 works well
    int max_threads = prop.maxThreadsPerBlock;
    if (max_threads >= 512) return 512;
    if (max_threads >= 256) return 256;
    return 128;
}

int get_optimal_grid_size(int device_id, int num_paths, int block_size) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    int grid_size = (num_paths + block_size - 1) / block_size;
    int max_grid_size = prop.maxGridSize[0];
    
    return min(grid_size, max_grid_size);
}

} // extern "C"