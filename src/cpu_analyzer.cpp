#include "montestrike/cpu_analyzer.hpp"
#include <algorithm>
#include <thread>
#include <cstring>

#ifdef _WIN32
    #include <windows.h>
    #include <intrin.h>
    #include <sysinfoapi.h>
#else
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/sysinfo.h>
    #include <cpuid.h>
    #include <pthread.h>
#endif

namespace montestrike {

CpuAnalyzer::CpuAnalyzer() 
    : avx2_supported_(false)
    , fma_supported_(false)
    , sse2_supported_(false)
    , sse4_1_supported_(false)
    , logical_cores_(0)
    , physical_cores_(0)
    , total_memory_(0)
    , page_size_(0)
    , cpu_family_(0)
    , cpu_model_(0)
    , cpu_stepping_(0) {
    
    detect_cpu_features();
    detect_core_count();
    detect_memory_info();
}

void CpuAnalyzer::detect_cpu_features() {
#ifdef _WIN32
    int cpu_info[4];
    
    // Get CPU vendor and basic info
    __cpuid(cpu_info, 0);
    char vendor[13];
    memcpy(vendor, &cpu_info[1], 4);
    memcpy(vendor + 4, &cpu_info[3], 4);
    memcpy(vendor + 8, &cpu_info[2], 4);
    vendor[12] = '\0';
    cpu_vendor_ = std::string(vendor);
    
    // Get CPU features
    __cpuid(cpu_info, 1);
    sse2_supported_ = (cpu_info[3] & (1 << 26)) != 0;
    sse4_1_supported_ = (cpu_info[2] & (1 << 19)) != 0;
    fma_supported_ = (cpu_info[2] & (1 << 12)) != 0;
    
    // Extended features
    __cpuid(cpu_info, 7);
    avx2_supported_ = (cpu_info[1] & (1 << 5)) != 0;
    
    // Get CPU brand string
    char brand[49];
    __cpuid(cpu_info, 0x80000002);
    memcpy(brand, cpu_info, 16);
    __cpuid(cpu_info, 0x80000003);
    memcpy(brand + 16, cpu_info, 16);
    __cpuid(cpu_info, 0x80000004);
    memcpy(brand + 32, cpu_info, 16);
    brand[48] = '\0';
    cpu_brand_ = std::string(brand);
    
#else
    uint32_t eax, ebx, ecx, edx;
    
    // Get CPU vendor
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        char vendor[13];
        memcpy(vendor, &ebx, 4);
        memcpy(vendor + 4, &edx, 4);
        memcpy(vendor + 8, &ecx, 4);
        vendor[12] = '\0';
        cpu_vendor_ = std::string(vendor);
    }
    
    // Get CPU features
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        sse2_supported_ = (edx & (1 << 26)) != 0;
        sse4_1_supported_ = (ecx & (1 << 19)) != 0;
        fma_supported_ = (ecx & (1 << 12)) != 0;
        
        cpu_family_ = (eax >> 8) & 0xF;
        cpu_model_ = (eax >> 4) & 0xF;
        cpu_stepping_ = eax & 0xF;
    }
    
    // Extended features
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
        avx2_supported_ = (ebx & (1 << 5)) != 0;
    }
    
    // Get CPU brand string
    char brand[49];
    if (__get_cpuid(0x80000002, &eax, &ebx, &ecx, &edx)) {
        memcpy(brand, &eax, 4);
        memcpy(brand + 4, &ebx, 4);
        memcpy(brand + 8, &ecx, 4);
        memcpy(brand + 12, &edx, 4);
        
        __get_cpuid(0x80000003, &eax, &ebx, &ecx, &edx);
        memcpy(brand + 16, &eax, 4);
        memcpy(brand + 20, &ebx, 4);
        memcpy(brand + 24, &ecx, 4);
        memcpy(brand + 28, &edx, 4);
        
        __get_cpuid(0x80000004, &eax, &ebx, &ecx, &edx);
        memcpy(brand + 32, &eax, 4);
        memcpy(brand + 36, &ebx, 4);
        memcpy(brand + 40, &ecx, 4);
        memcpy(brand + 44, &edx, 4);
        
        brand[48] = '\0';
        cpu_brand_ = std::string(brand);
    }
#endif
    
    // Clean up brand string (remove extra spaces)
    size_t start = cpu_brand_.find_first_not_of(" \t");
    if (start != std::string::npos) {
        size_t end = cpu_brand_.find_last_not_of(" \t");
        cpu_brand_ = cpu_brand_.substr(start, end - start + 1);
    }
}

void CpuAnalyzer::detect_core_count() {
    logical_cores_ = std::thread::hardware_concurrency();
    
#ifdef _WIN32
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    physical_cores_ = sys_info.dwNumberOfProcessors;
    
    // Try to get actual physical core count
    DWORD buffer_size = 0;
    GetLogicalProcessorInformation(nullptr, &buffer_size);
    if (buffer_size > 0) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
            buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
            physical_cores_ = 0;
            for (const auto& info : buffer) {
                if (info.Relationship == RelationProcessorCore) {
                    physical_cores_++;
                }
            }
        }
    }
    
    if (physical_cores_ == 0) {
        physical_cores_ = logical_cores_;
    }
    
#else
    // Linux: try to read from /proc/cpuinfo
    physical_cores_ = logical_cores_;  // Fallback
    
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        uint32_t core_count = 0;
        uint32_t physical_id = 0;
        uint32_t max_physical_id = 0;
        
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "physical id", 11) == 0) {
                sscanf(line, "physical id\t: %u", &physical_id);
                max_physical_id = std::max(max_physical_id, physical_id);
            } else if (strncmp(line, "cpu cores", 9) == 0) {
                sscanf(line, "cpu cores\t: %u", &core_count);
            }
        }
        fclose(fp);
        
        if (core_count > 0) {
            physical_cores_ = core_count * (max_physical_id + 1);
        }
    }
#endif
    
    // Ensure we have at least 1 core
    if (logical_cores_ == 0) logical_cores_ = 1;
    if (physical_cores_ == 0) physical_cores_ = 1;
}

void CpuAnalyzer::detect_memory_info() {
#ifdef _WIN32
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        total_memory_ = mem_status.ullTotalPhys;
    }
    
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    page_size_ = sys_info.dwPageSize;
    
#else
    total_memory_ = static_cast<uint64_t>(sysconf(_SC_PHYS_PAGES)) * 
                    static_cast<uint64_t>(sysconf(_SC_PAGE_SIZE));
    page_size_ = sysconf(_SC_PAGE_SIZE);
#endif
}

DeviceInfo CpuAnalyzer::get_cpu_info() const {
    DeviceInfo info;
    info.device_id = -1;  // CPU doesn't have device ID
    info.name = cpu_brand_;
    info.backend = ComputeBackend::CPU;
    info.streaming_multiprocessors = logical_cores_;
    info.cores_per_sm = 1;
    info.global_memory_bytes = total_memory_;
    info.max_threads_per_block = logical_cores_;
    info.max_threads_per_sm = logical_cores_;
    info.compute_capability = 0.0f;  // N/A for CPU
    info.is_compatible = true;
    info.available_memory_bytes = get_available_memory();
    info.supports_avx2 = avx2_supported_;
    info.supports_fma = fma_supported_;
    info.cpu_logical_cores = logical_cores_;
    info.cpu_physical_cores = physical_cores_;
    
    return info;
}

bool CpuAnalyzer::supports_avx2() const {
    return avx2_supported_;
}

bool CpuAnalyzer::supports_fma() const {
    return fma_supported_;
}

uint32_t CpuAnalyzer::get_logical_cores() const {
    return logical_cores_;
}

uint32_t CpuAnalyzer::get_physical_cores() const {
    return physical_cores_;
}

uint64_t CpuAnalyzer::get_total_memory() const {
    return total_memory_;
}

uint64_t CpuAnalyzer::get_available_memory() const {
#ifdef _WIN32
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        return mem_status.ullAvailPhys;
    }
    return total_memory_ * 3 / 4;  // Fallback estimate
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return static_cast<uint64_t>(info.freeram) * info.mem_unit;
    }
    return total_memory_ * 3 / 4;  // Fallback estimate
#endif
}

std::string CpuAnalyzer::get_cpu_brand() const {
    return cpu_brand_;
}

uint32_t CpuAnalyzer::get_optimal_thread_count() const {
    // For Monte Carlo simulation, use physical cores to avoid hyperthreading overhead
    return std::max(1u, physical_cores_);
}

ErrorCode CpuAnalyzer::validate_backend(ComputeBackend backend) const {
    switch (backend) {
        case ComputeBackend::CPU:
            return ErrorCode::SUCCESS;
            
        case ComputeBackend::AVX2:
            if (!avx2_supported_) {
                return ErrorCode::AVX2_NOT_SUPPORTED;
            }
            return ErrorCode::SUCCESS;
            
        case ComputeBackend::CUDA:
            // CUDA validation should be handled by DeviceAnalyzer
            return ErrorCode::BACKEND_NOT_AVAILABLE;
            
        default:
            return ErrorCode::BACKEND_NOT_AVAILABLE;
    }
}

bool CpuAnalyzer::set_thread_affinity(uint32_t thread_id, uint32_t core_id) const {
#ifdef _WIN32
    HANDLE thread_handle = GetCurrentThread();
    DWORD_PTR mask = 1ULL << core_id;
    return SetThreadAffinityMask(thread_handle, mask) != 0;
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
#endif
}

// Global instance
CpuAnalyzer& get_cpu_analyzer() {
    static CpuAnalyzer instance;
    return instance;
}

} // namespace montestrike