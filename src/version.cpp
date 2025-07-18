#include "montestrike/version.hpp"
#include <sstream>
#include <cuda_runtime.h>

namespace montestrike {

std::string VersionInfo::to_string() const {
    std::ostringstream oss;
    oss << major << "." << minor << "." << patch;
    if (!build_type.empty()) {
        oss << "-" << build_type;
    }
    return oss.str();
}

VersionInfo get_version_info() {
    VersionInfo info;
    info.major = VERSION_MAJOR;
    info.minor = VERSION_MINOR;
    info.patch = VERSION_PATCH;
    info.build_type = VERSION_BUILD_TYPE;
    
    // Get CUDA version
    int runtime_version;
    cudaError_t err = cudaRuntimeGetVersion(&runtime_version);
    if (err == cudaSuccess) {
        std::ostringstream cuda_oss;
        cuda_oss << (runtime_version / 1000) << "." << ((runtime_version % 1000) / 10);
        info.cuda_version = cuda_oss.str();
    } else {
        info.cuda_version = "Unknown";
    }
    
    // Build date - would be filled by build system
    info.build_date = __DATE__ " " __TIME__;
    
    // Git commit - would be filled by build system 
    info.git_commit = "dev";
    
    return info;
}

std::string get_version_string() {
    auto info = get_version_info();
    return info.to_string();
}

} // namespace montestrike