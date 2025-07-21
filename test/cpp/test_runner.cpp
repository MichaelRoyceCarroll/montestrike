#include "test_assertions.hpp"
#include "montestrike/montestrike.hpp"
#include <iostream>
#include <vector>
#include <functional>

namespace montestrike_test {

// Global test counters
int total_tests = 0;
int passed_tests = 0;
int failed_tests = 0;

// Test function declarations
bool test_parameter_validation();
bool test_device_analyzer();
bool test_basic_monte_carlo();
bool test_different_path_counts();
bool test_antithetic_variates();
bool test_edge_cases();
bool test_cpu_backend();
bool test_backend_comparison();

} // namespace montestrike_test

int main() {
    using namespace montestrike_test;
    
    std::cout << "🚀 MonteStrike C++ Test Suite" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Display version info
    auto version_info = montestrike::get_version_info();
    std::cout << "📦 MonteStrike Version: " << version_info.to_string() << std::endl;
    std::cout << "⚡ CUDA Version: " << version_info.cuda_version << std::endl;
    std::cout << "📅 Build Date: " << version_info.build_date << std::endl;
    std::cout << std::endl;
    
    // Check CUDA availability
    montestrike::DeviceAnalyzer analyzer;
    auto devices = analyzer.enumerate_devices();
    
    if (devices.empty()) {
        std::cout << "⚠️  No CUDA devices found. CPU backend tests will still run." << std::endl;
    } else {
        auto compatible_devices = std::count_if(devices.begin(), devices.end(),
            [](const montestrike::DeviceInfo& d) { return d.is_compatible; });
        
        if (compatible_devices == 0) {
            std::cout << "⚠️  No compatible CUDA devices found. CPU backend tests will still run." << std::endl;
        } else {
            std::cout << "🎯 Found " << compatible_devices << " compatible CUDA device(s)" << std::endl;
            auto best_device = analyzer.get_best_device();
            std::cout << "🏆 Using device: " << best_device.name << " (CC " << best_device.compute_capability << ")" << std::endl;
        }
    }
    std::cout << std::endl;
    
    // Test suite
    std::vector<std::pair<std::string, std::function<bool()>>> tests = {
        {"Parameter Validation", test_parameter_validation},
        {"Device Analyzer", test_device_analyzer},
        {"Basic Monte Carlo", test_basic_monte_carlo},
        {"Different Path Counts", test_different_path_counts},
        {"Antithetic Variates", test_antithetic_variates},
        {"Edge Cases", test_edge_cases},
        {"CPU Backend", test_cpu_backend},
        {"Backend Comparison", test_backend_comparison}
    };
    
    int suite_passed = 0;
    int suite_failed = 0;
    
    for (const auto& test : tests) {
        try {
            if (test.second()) {
                suite_passed++;
            } else {
                suite_failed++;
                std::cerr << "💥 Test suite '" << test.first << "' failed!" << std::endl;
            }
        } catch (const std::exception& e) {
            suite_failed++;
            std::cerr << "💥 Test suite '" << test.first << "' threw exception: " << e.what() << std::endl;
        } catch (...) {
            suite_failed++;
            std::cerr << "💥 Test suite '" << test.first << "' threw unknown exception!" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Print final summary
    print_test_summary();
    
    std::cout << "\n📊 SUITE SUMMARY" << std::endl;
    std::cout << "Test suites passed: " << suite_passed << std::endl;
    std::cout << "Test suites failed: " << suite_failed << std::endl;
    
    if (suite_failed == 0 && failed_tests == 0) {
        std::cout << "🎉 ALL TESTS PASSED! 🎉" << std::endl;
        return 0;
    } else {
        std::cout << "💥 SOME TESTS FAILED! 💥" << std::endl;
        return 1;
    }
}