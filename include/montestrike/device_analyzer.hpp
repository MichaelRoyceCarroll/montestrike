#pragma once

#include "types.hpp"
#include <vector>
#include <memory>

namespace montestrike {

class DeviceAnalyzer {
public:
    DeviceAnalyzer();
    ~DeviceAnalyzer();

    std::vector<DeviceInfo> enumerate_devices();
    
    DeviceInfo get_device_info(int32_t device_id);
    
    DeviceInfo get_best_device();
    
    ErrorCode set_device(int32_t device_id);
    
    int32_t get_current_device();
    
    MemoryInfo get_memory_info(int32_t device_id = -1);
    
    bool is_device_compatible(int32_t device_id);
    
    bool has_sufficient_memory(int32_t device_id, uint32_t num_paths);
    
    uint32_t get_optimal_block_size(int32_t device_id = -1);
    
    uint32_t get_optimal_grid_size(int32_t device_id, uint32_t num_paths);
    
    std::string get_cuda_version();
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace montestrike