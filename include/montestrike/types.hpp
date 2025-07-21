#pragma once

#include <cstdint>
#include <string>
#include <ostream>

namespace montestrike {

enum class ComputeBackend : int32_t {
    CUDA = 0,      // Primary GPU backend
    AVX2 = 1,      // Vectorized CPU backend  
    CPU = 2        // Standard CPU backend
};

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
    BACKEND_NOT_AVAILABLE = -10,
    AVX2_NOT_SUPPORTED = -11,
    CPU_THREAD_ERROR = -12,
    UNKNOWN_ERROR = -100
};

struct DeviceInfo {
    int32_t device_id;
    std::string name;
    ComputeBackend backend;
    uint32_t streaming_multiprocessors;    // GPU: SMs, CPU: logical cores
    uint32_t cores_per_sm;                 // GPU: cores per SM, CPU: 1  
    uint64_t global_memory_bytes;          // GPU: VRAM, CPU: system RAM
    uint32_t max_threads_per_block;        // GPU: threads per block, CPU: threads per core
    uint32_t max_threads_per_sm;           // GPU: threads per SM, CPU: total threads
    float compute_capability;              // GPU: CC version, CPU: 0.0
    bool is_compatible;
    uint64_t available_memory_bytes;
    
    // CPU-specific fields
    bool supports_avx2;
    bool supports_fma;
    uint32_t cpu_logical_cores;
    uint32_t cpu_physical_cores;
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

// Utility functions for ErrorCode
inline std::ostream& operator<<(std::ostream& os, ErrorCode code) {
    switch (code) {
        case ErrorCode::SUCCESS: return os << "SUCCESS";
        case ErrorCode::INVALID_PARAMETERS: return os << "INVALID_PARAMETERS";
        case ErrorCode::CUDA_ERROR: return os << "CUDA_ERROR";
        case ErrorCode::MEMORY_ALLOCATION_FAILED: return os << "MEMORY_ALLOCATION_FAILED";
        case ErrorCode::NO_CUDA_DEVICE: return os << "NO_CUDA_DEVICE";
        case ErrorCode::INCOMPATIBLE_DEVICE: return os << "INCOMPATIBLE_DEVICE";
        case ErrorCode::KERNEL_LAUNCH_FAILED: return os << "KERNEL_LAUNCH_FAILED";
        case ErrorCode::MEMORY_COPY_FAILED: return os << "MEMORY_COPY_FAILED";
        case ErrorCode::DEVICE_SELECTION_FAILED: return os << "DEVICE_SELECTION_FAILED";
        case ErrorCode::COMPUTATION_TIMEOUT: return os << "COMPUTATION_TIMEOUT";
        case ErrorCode::BACKEND_NOT_AVAILABLE: return os << "BACKEND_NOT_AVAILABLE";
        case ErrorCode::AVX2_NOT_SUPPORTED: return os << "AVX2_NOT_SUPPORTED";
        case ErrorCode::CPU_THREAD_ERROR: return os << "CPU_THREAD_ERROR";
        case ErrorCode::UNKNOWN_ERROR: return os << "UNKNOWN_ERROR";
        default: return os << "UNKNOWN_ERROR_CODE";
    }
}

} // namespace montestrike