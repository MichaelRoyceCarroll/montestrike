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
    int num_paths,
    bool use_antithetic_variates,
    bool* touch_results,            // Output: did path touch strike?
    float* final_prices             // Optional: final prices for debugging
) {
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;
    
    // Determine if this is an antithetic path
    bool is_antithetic = use_antithetic_variates && (path_id >= num_paths / 2);
    int base_path_id = is_antithetic ? path_id - num_paths / 2 : path_id;
    
    float price = current_price;
    bool touched = false;
    
    // Pre-calculate constants for performance
    const float drift_term = (drift - 0.5f * volatility * volatility) * dt;
    const float vol_sqrt_dt = volatility * sqrtf(dt);
    
    // Check if we start exactly at strike (immediate touch)
    if (fabsf(current_price - strike_price) < 1e-6f) {
        touched = true;
    } else {
        // Determine touch direction based on strike vs current price
        bool check_upward = (strike_price > current_price);
        
        for (int step = 0; step < total_steps && !touched; step++) {
            // Generate normal random number
            float random_val = generate_normal_random(&random_states[base_path_id]);
            
            // Apply antithetic variates if needed
            if (is_antithetic) {
                random_val = -random_val;
            }
            
            // Update price using Geometric Brownian Motion
            price *= expf(drift_term + vol_sqrt_dt * random_val);
            
            // Check for touch based on direction
            if (check_upward) {
                if (price >= strike_price) {
                    touched = true;
                }
            } else {
                if (price <= strike_price) {
                    touched = true;
                }
            }
        }
    }
    
    touch_results[path_id] = touched;
    if (final_prices) {
        final_prices[path_id] = price;
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
        num_paths,
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
        num_paths,
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