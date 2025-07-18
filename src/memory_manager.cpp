#include "montestrike/types.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include <mutex>

namespace montestrike {

class MemoryManager {
public:
    struct AllocatedBuffers {
        curandState* random_states;
        bool* touch_results;  
        float* final_prices;
        unsigned long long* touch_count;
        uint32_t allocated_paths;
        
        AllocatedBuffers() : random_states(nullptr), touch_results(nullptr), 
                           final_prices(nullptr), touch_count(nullptr), allocated_paths(0) {}
    };
    
    static MemoryManager& instance() {
        static MemoryManager instance;
        return instance;
    }
    
    ErrorCode allocate_buffers(uint32_t num_paths, AllocatedBuffers& buffers) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check if we can reuse existing buffers
        if (current_buffers_.allocated_paths >= num_paths && 
            current_buffers_.random_states != nullptr) {
            buffers = current_buffers_;
            return ErrorCode::SUCCESS;
        }
        
        // Free existing buffers if they're too small
        free_buffers();
        
        // Calculate required sizes
        size_t random_states_size = num_paths * sizeof(curandState);
        size_t touch_results_size = num_paths * sizeof(bool);
        size_t final_prices_size = num_paths * sizeof(float);
        size_t touch_count_size = sizeof(unsigned long long);
        
        // Allocate new buffers
        cudaError_t err;
        
        err = cudaMalloc(&current_buffers_.random_states, random_states_size);
        if (err != cudaSuccess) {
            free_buffers();
            return ErrorCode::MEMORY_ALLOCATION_FAILED;
        }
        
        err = cudaMalloc(&current_buffers_.touch_results, touch_results_size);
        if (err != cudaSuccess) {
            free_buffers();
            return ErrorCode::MEMORY_ALLOCATION_FAILED;
        }
        
        err = cudaMalloc(&current_buffers_.final_prices, final_prices_size);
        if (err != cudaSuccess) {
            free_buffers();
            return ErrorCode::MEMORY_ALLOCATION_FAILED;
        }
        
        err = cudaMalloc(&current_buffers_.touch_count, touch_count_size);
        if (err != cudaSuccess) {
            free_buffers();
            return ErrorCode::MEMORY_ALLOCATION_FAILED;
        }
        
        current_buffers_.allocated_paths = num_paths;
        buffers = current_buffers_;
        
        return ErrorCode::SUCCESS;
    }
    
    void free_buffers() {
        if (current_buffers_.random_states) {
            cudaFree(current_buffers_.random_states);
            current_buffers_.random_states = nullptr;
        }
        if (current_buffers_.touch_results) {
            cudaFree(current_buffers_.touch_results);
            current_buffers_.touch_results = nullptr;
        }
        if (current_buffers_.final_prices) {
            cudaFree(current_buffers_.final_prices);
            current_buffers_.final_prices = nullptr;
        }
        if (current_buffers_.touch_count) {
            cudaFree(current_buffers_.touch_count);
            current_buffers_.touch_count = nullptr;
        }
        current_buffers_.allocated_paths = 0;
    }
    
    uint64_t get_allocated_memory() const {
        if (current_buffers_.allocated_paths == 0) return 0;
        
        uint64_t total = 0;
        total += current_buffers_.allocated_paths * sizeof(curandState);
        total += current_buffers_.allocated_paths * sizeof(bool);
        total += current_buffers_.allocated_paths * sizeof(float);
        total += sizeof(unsigned long long);
        
        return total;
    }
    
    uint32_t get_allocated_paths() const {
        return current_buffers_.allocated_paths;
    }
    
    ~MemoryManager() {
        free_buffers();
    }
    
private:
    MemoryManager() = default;
    
    AllocatedBuffers current_buffers_;
    std::mutex mutex_;
};

// External interface functions
extern "C" {
    
ErrorCode allocate_simulation_memory(uint32_t num_paths, void** buffers_handle) {
    auto& manager = MemoryManager::instance();
    auto* buffers = new MemoryManager::AllocatedBuffers();
    
    ErrorCode result = manager.allocate_buffers(num_paths, *buffers);
    if (result == ErrorCode::SUCCESS) {
        *buffers_handle = buffers;
    } else {
        delete buffers;
        *buffers_handle = nullptr;
    }
    
    return result;
}

void free_simulation_memory() {
    auto& manager = MemoryManager::instance();
    manager.free_buffers();
}

uint64_t get_memory_usage() {
    auto& manager = MemoryManager::instance();
    return manager.get_allocated_memory();
}

uint64_t estimate_memory_requirements(uint32_t num_paths) {
    uint64_t random_states_size = static_cast<uint64_t>(num_paths) * sizeof(curandState);
    uint64_t touch_results_size = static_cast<uint64_t>(num_paths) * sizeof(bool);
    uint64_t final_prices_size = static_cast<uint64_t>(num_paths) * sizeof(float);
    uint64_t touch_count_size = sizeof(unsigned long long);
    
    return random_states_size + touch_results_size + final_prices_size + touch_count_size;
}

} // extern "C"

} // namespace montestrike