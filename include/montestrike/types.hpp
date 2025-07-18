#pragma once

#include <cstdint>
#include <string>

namespace montestrike {

enum class ErrorCode : int32_t {
    SUCCESS = 0,
    INVALID_PARAMETERS = -1,
    CUDA_ERROR = -2,
    MEMORY_ALLOCATION_FAILED = -3,
    NO_CUDA_DEVICE = -4,
    INCOMPATIBLE_DEVICE = -5,
    KERNEL_LAUNCH_FAILED = -6,
    MEMORY_COPY_FAILED = -7,
    DEVICE_SELECTION_FAILED = -8,
    COMPUTATION_TIMEOUT = -9,
    UNKNOWN_ERROR = -100
};

struct DeviceInfo {
    int32_t device_id;
    std::string name;
    uint32_t streaming_multiprocessors;
    uint32_t cores_per_sm;
    uint64_t global_memory_bytes;
    uint32_t max_threads_per_block;
    uint32_t max_threads_per_sm;
    float compute_capability;
    bool is_compatible;
    uint64_t available_memory_bytes;
};

struct MemoryInfo {
    uint64_t total_bytes;
    uint64_t free_bytes;
    uint64_t used_bytes;
    uint64_t allocated_for_paths;
};

struct ComputationMetrics {
    double computation_time_ms;
    double kernel_time_ms;
    double memory_transfer_time_ms;
    double throughput_paths_per_sec;
    uint64_t memory_used_bytes;
};

struct ValidationResult {
    bool is_valid;
    std::string error_message;
    ErrorCode error_code;
};

typedef void (*ProgressCallback)(float percentage, void* user_data);

} // namespace montestrike