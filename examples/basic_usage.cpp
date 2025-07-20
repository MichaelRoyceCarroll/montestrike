#include "montestrike/montestrike.hpp"
#include "montestrike/cpu_analyzer.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [--backend <cuda|avx2|cpu|all>]" << std::endl;
    std::cout << "  --backend cuda   Use CUDA backend only" << std::endl;
    std::cout << "  --backend avx2   Use AVX2 backend only" << std::endl;  
    std::cout << "  --backend cpu    Use CPU backend only" << std::endl;
    std::cout << "  --backend all    Test all available backends" << std::endl;
    std::cout << "  (default: cuda with fallback)" << std::endl;
}

montestrike::ComputeBackend parse_backend(const std::string& backend_str) {
    if (backend_str == "cuda") return montestrike::ComputeBackend::CUDA;
    if (backend_str == "avx2") return montestrike::ComputeBackend::AVX2;
    if (backend_str == "cpu") return montestrike::ComputeBackend::CPU;
    throw std::invalid_argument("Invalid backend: " + backend_str);
}

void test_backend(montestrike::ComputeBackend backend, const std::string& backend_name, bool strict_mode = false) {
    std::cout << "\n🧪 Testing " << backend_name << " Backend" << std::endl;
    std::cout << std::string(30 + backend_name.length(), '=') << std::endl;

    // Create calculator and set parameters
    montestrike::MonteCarloPoT calculator;
    montestrike::MonteCarloPoT::Parameters params;
    
    // Standard test parameters
    params.current_price = 220.50f;
    params.strike_price = 225.00f;
    params.time_to_expiration = 7.0f / 365.0f;
    params.drift = 0.08f;
    params.volatility = 0.25f;
    params.steps_per_day = 50;
    params.num_paths = 1000000;
    params.use_antithetic_variates = true;
    params.random_seed = 42;
    params.backend = backend;
    params.strict_backend_mode = strict_mode;

    auto results = calculator.calculate_pot(params);

    if (results.computation_successful) {
        std::cout << "✅ Computation successful!" << std::endl;
        std::cout << "🎯 Probability of Touch: " << std::fixed << std::setprecision(4) 
                  << results.probability_of_touch << " (" 
                  << std::setprecision(2) << (results.probability_of_touch * 100) << "%)" << std::endl;
        std::cout << "⚡ Computation Time: " << std::setprecision(2) 
                  << results.metrics.computation_time_ms << " ms" << std::endl;
        std::cout << "⚡ Throughput: " << std::fixed << std::setprecision(0) 
                  << results.metrics.throughput_paths_per_sec << " paths/sec" << std::endl;
        std::cout << "🖥️  Device: " << results.device_used.name << std::endl;
    } else {
        std::cout << "❌ " << backend_name << " backend failed: " << results.error_message << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "🚀 MonteStrike Basic Usage Example" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Parse command line arguments
    std::string backend_option = "default";
    if (argc > 1) {
        if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (argc >= 3 && std::string(argv[1]) == "--backend") {
            backend_option = argv[2];
        } else {
            std::cerr << "❌ Invalid arguments." << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Display version and device information
    auto version = montestrike::get_version_info();
    std::cout << "📦 MonteStrike Version: " << version.to_string() << std::endl;
    std::cout << "⚡ CUDA Version: " << version.cuda_version << std::endl;
    std::cout << std::endl;
    
    // Get system information
    montestrike::CpuAnalyzer& cpu_analyzer = montestrike::get_cpu_analyzer();
    auto cpu_info = cpu_analyzer.get_cpu_info();
    
    std::cout << "💻 System Information:" << std::endl;
    std::cout << "  CPU: " << cpu_info.name << std::endl;
    std::cout << "  Logical Cores: " << cpu_info.cpu_logical_cores << std::endl;
    std::cout << "  AVX2 Support: " << (cpu_info.supports_avx2 ? "Yes" : "No") << std::endl;
    
    // Check CUDA availability
    montestrike::DeviceAnalyzer analyzer;
    auto devices = analyzer.enumerate_devices();
    bool cuda_available = !devices.empty();
    if (cuda_available) {
        auto best_device = analyzer.get_best_device();
        std::cout << "  CUDA Device: " << best_device.name << std::endl;
    } else {
        std::cout << "  CUDA Device: Not available" << std::endl;
    }
    std::cout << std::endl;
    
    // Handle backend selection
    if (backend_option == "all") {
        // Test all available backends
        std::cout << "📊 Testing All Available Backends:" << std::endl;
        std::cout << "=================================" << std::endl;
        
        test_backend(montestrike::ComputeBackend::CPU, "CPU", true);
        
        if (cpu_info.supports_avx2) {
            test_backend(montestrike::ComputeBackend::AVX2, "AVX2", true);
        } else {
            std::cout << "\n⚠️  Skipping AVX2 - not supported on this CPU" << std::endl;
        }
        
        if (cuda_available) {
            test_backend(montestrike::ComputeBackend::CUDA, "CUDA", true);
        } else {
            std::cout << "\n⚠️  Skipping CUDA - no compatible devices found" << std::endl;
        }
        
        return 0;
    }
    
    // Handle specific backend or default
    if (backend_option == "default") {
        // Default behavior (CUDA with fallback)
        test_backend(montestrike::ComputeBackend::CUDA, "CUDA (with fallback)", false);
    } else {
        // Specific backend requested
        try {
            auto backend = parse_backend(backend_option);
            std::string backend_name = backend_option;
            backend_name[0] = std::toupper(backend_name[0]); // Capitalize first letter
            test_backend(backend, backend_name, true);
        } catch (const std::exception& e) {
            std::cerr << "❌ " << e.what() << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    std::cout << std::endl;
    std::cout << "⚠️  DISCLAIMER: This simulation is for educational purposes only." << std::endl;
    std::cout << "   It does not constitute financial advice. Trading options involves" << std::endl;
    std::cout << "   substantial risk and may not be suitable for all investors." << std::endl;
    
    return 0;
}
