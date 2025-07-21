#include "montestrike/device_analyzer.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <algorithm>
#include <sstream>

namespace montestrike {

// Helper function to get cores per SM for different architectures
uint32_t get_cores_per_sm(int major, int minor) {
    // Based on CUDA compute capabilities
    if (major == 7) {
        if (minor == 5) return 64;  // RTX 20xx, GTX 16xx
        return 64;                  // V100, RTX Titan
    } else if (major == 8) {
        if (minor == 6) return 128; // RTX 30xx
        if (minor == 9) return 128; // RTX 40xx  
        return 64;                  // A100, others
    } else if (major == 9) {
        return 128;                 // Future architectures
    } else if (major == 6) {
        if (minor == 1) return 128; // GTX 10xx
        return 192;                 // GTX 9xx
    }
    return 64; // Fallback
}

class DeviceAnalyzer::Impl {
public:
    Impl() : current_device_(-1) {
        refresh_device_list();
    }
    
    void refresh_device_list() {
        devices_.clear();
        
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            return; // No devices or CUDA not available
        }
        
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp prop;
            err = cudaGetDeviceProperties(&prop, i);
            if (err != cudaSuccess) {
                continue;
            }
            
            DeviceInfo info;
            info.device_id = i;
            info.name = prop.name;
            info.streaming_multiprocessors = prop.multiProcessorCount;
            info.cores_per_sm = get_cores_per_sm(prop.major, prop.minor);
            info.global_memory_bytes = prop.totalGlobalMem;
            info.max_threads_per_block = prop.maxThreadsPerBlock;
            info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
            info.compute_capability = prop.major + prop.minor * 0.1f;
            info.is_compatible = info.compute_capability >= 6.0f;
            
            // Get available memory
            if (info.is_compatible) {
                size_t free_mem, total_mem;
                cudaSetDevice(i);
                cudaMemGetInfo(&free_mem, &total_mem);
                info.available_memory_bytes = free_mem;
            } else {
                info.available_memory_bytes = 0;
            }
            
            devices_.push_back(info);
        }
        
        // Restore original device if one was set
        if (current_device_ >= 0) {
            cudaSetDevice(current_device_);
        }
    }
    
    std::vector<DeviceInfo> devices_;
    int current_device_;
};

DeviceAnalyzer::DeviceAnalyzer() : pimpl(std::make_unique<Impl>()) {}

DeviceAnalyzer::~DeviceAnalyzer() = default;

std::vector<DeviceInfo> DeviceAnalyzer::enumerate_devices() {
    pimpl->refresh_device_list();
    return pimpl->devices_;
}

DeviceInfo DeviceAnalyzer::get_device_info(int32_t device_id) {
    auto devices = enumerate_devices();
    for (const auto& device : devices) {
        if (device.device_id == device_id) {
            return device;
        }
    }
    
    // Return invalid device info
    DeviceInfo invalid_info = {};
    invalid_info.device_id = -1;
    invalid_info.is_compatible = false;
    return invalid_info;
}

DeviceInfo DeviceAnalyzer::get_best_device() {
    auto devices = enumerate_devices();
    
    // Filter to compatible devices only
    std::vector<DeviceInfo> compatible_devices;
    for (const auto& device : devices) {
        if (device.is_compatible) {
            compatible_devices.push_back(device);
        }
    }
    
    if (compatible_devices.empty()) {
        DeviceInfo invalid_info = {};
        invalid_info.device_id = -1;
        invalid_info.is_compatible = false;
        return invalid_info;
    }
    
    // Rank by performance: compute capability first, then total cores
    std::sort(compatible_devices.begin(), compatible_devices.end(), 
        [](const DeviceInfo& a, const DeviceInfo& b) {
            if (a.compute_capability != b.compute_capability) {
                return a.compute_capability > b.compute_capability;
            }
            uint32_t cores_a = a.streaming_multiprocessors * a.cores_per_sm;
            uint32_t cores_b = b.streaming_multiprocessors * b.cores_per_sm;
            return cores_a > cores_b;
        });
    
    return compatible_devices[0];
}

ErrorCode DeviceAnalyzer::set_device(int32_t device_id) {
    if (!is_device_compatible(device_id)) {
        return ErrorCode::INCOMPATIBLE_DEVICE;
    }
    
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return ErrorCode::DEVICE_SELECTION_FAILED;
    }
    
    pimpl->current_device_ = device_id;
    return ErrorCode::SUCCESS;
}

int32_t DeviceAnalyzer::get_current_device() {
    int device_id;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err == cudaSuccess) {
        pimpl->current_device_ = device_id;
        return device_id;
    }
    return -1;
}

MemoryInfo DeviceAnalyzer::get_memory_info(int32_t device_id) {
    MemoryInfo info = {};
    
    int original_device = get_current_device();
    
    if (device_id >= 0) {
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            return info;
        }
    }
    
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err == cudaSuccess) {
        info.total_bytes = total_bytes;
        info.free_bytes = free_bytes;
        info.used_bytes = total_bytes - free_bytes;
        info.allocated_for_paths = 0; // Would need to track this separately
    }
    
    // Restore original device
    if (device_id >= 0 && original_device >= 0) {
        cudaSetDevice(original_device);
    }
    
    return info;
}

bool DeviceAnalyzer::is_device_compatible(int32_t device_id) {
    auto info = get_device_info(device_id);
    return info.is_compatible;
}

bool DeviceAnalyzer::has_sufficient_memory(int32_t device_id, uint32_t num_paths) {
    auto memory_info = get_memory_info(device_id);
    
    // Estimate memory requirements
    size_t curand_states = num_paths * sizeof(curandState);  // ~48 bytes per state
    size_t touch_results = num_paths * sizeof(bool);
    size_t final_prices = num_paths * sizeof(float);
    size_t overhead = 1024 * 1024; // 1MB overhead
    
    size_t total_required = curand_states + touch_results + final_prices + overhead;
    
    return memory_info.free_bytes > total_required;
}

uint32_t DeviceAnalyzer::get_optimal_block_size(int32_t device_id) {
    if (device_id < 0) {
        device_id = get_current_device();
    }
    
    auto info = get_device_info(device_id);
    if (!info.is_compatible) {
        return 256; // Safe fallback
    }
    
    // Use multiple of warp size (32), typically 256 or 512 works well
    if (info.max_threads_per_block >= 512) {
        return 512;
    } else if (info.max_threads_per_block >= 256) {
        return 256;
    } else {
        return 128;
    }
}

uint32_t DeviceAnalyzer::get_optimal_grid_size(int32_t device_id, uint32_t num_paths) {
    if (device_id < 0) {
        device_id = get_current_device();
    }
    
    uint32_t block_size = get_optimal_block_size(device_id);
    uint32_t grid_size = (num_paths + block_size - 1) / block_size;
    
    // Cap at reasonable grid size to avoid excessive memory usage
    return std::min(grid_size, 65535u);
}

std::string DeviceAnalyzer::get_cuda_version() {
    int driver_version, runtime_version;
    
    cudaError_t err1 = cudaDriverGetVersion(&driver_version);
    cudaError_t err2 = cudaRuntimeGetVersion(&runtime_version);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        return "Unknown";
    }
    
    std::ostringstream oss;
    oss << "Driver: " << (driver_version / 1000) << "." << ((driver_version % 1000) / 10)
        << ", Runtime: " << (runtime_version / 1000) << "." << ((runtime_version % 1000) / 10);
    
    return oss.str();
}

} // namespace montestrike