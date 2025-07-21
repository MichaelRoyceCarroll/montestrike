#pragma once

#include "types.hpp"
#include <vector>
#include <string>

namespace montestrike {

class CpuAnalyzer {
public:
    CpuAnalyzer();
    ~CpuAnalyzer() = default;
    
    // Get CPU information
    DeviceInfo get_cpu_info() const;
    
    // Check if AVX2 is supported at runtime
    bool supports_avx2() const;
    
    // Check if FMA is supported at runtime  
    bool supports_fma() const;
    
    // Get number of logical CPU cores
    uint32_t get_logical_cores() const;
    
    // Get number of physical CPU cores
    uint32_t get_physical_cores() const;
    
    // Get total system memory in bytes
    uint64_t get_total_memory() const;
    
    // Get available system memory in bytes
    uint64_t get_available_memory() const;
    
    // Get CPU brand string
    std::string get_cpu_brand() const;
    
    // Get optimal thread count for Monte Carlo simulation
    uint32_t get_optimal_thread_count() const;
    
    // Validate that requested backend is available
    ErrorCode validate_backend(ComputeBackend backend) const;
    
    // Set thread affinity (if supported by platform)
    bool set_thread_affinity(uint32_t thread_id, uint32_t core_id) const;
    
private:
    void detect_cpu_features();
    void detect_core_count();
    void detect_memory_info();
    
    // CPU feature flags
    bool avx2_supported_;
    bool fma_supported_;
    bool sse2_supported_;
    bool sse4_1_supported_;
    
    // Core information
    uint32_t logical_cores_;
    uint32_t physical_cores_;
    
    // Memory information
    uint64_t total_memory_;
    uint64_t page_size_;
    
    // CPU identification
    std::string cpu_brand_;
    std::string cpu_vendor_;
    uint32_t cpu_family_;
    uint32_t cpu_model_;
    uint32_t cpu_stepping_;
};

// Global CPU analyzer instance
CpuAnalyzer& get_cpu_analyzer();

} // namespace montestrike