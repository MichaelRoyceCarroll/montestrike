#pragma once

#include <string>

namespace montestrike {

struct VersionInfo {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
    std::string build_type;
    std::string cuda_version;
    std::string build_date;
    std::string git_commit;
    
    std::string to_string() const;
};

constexpr uint32_t VERSION_MAJOR = 0;
constexpr uint32_t VERSION_MINOR = 2;
constexpr uint32_t VERSION_PATCH = 0;
const std::string VERSION_BUILD_TYPE = "Release";

VersionInfo get_version_info();
std::string get_version_string();

} // namespace montestrike