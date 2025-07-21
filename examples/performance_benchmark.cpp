#include "montestrike/montestrike.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>

struct BenchmarkResult {
    uint32_t num_paths;
    double computation_time_ms;
    double throughput_paths_per_sec;
    uint64_t memory_used_mb;
    float probability_of_touch;
    bool success;
};

class PerformanceBenchmark {
public:
    PerformanceBenchmark() : calculator_() {
        // Initialize with best available device
        auto init_result = calculator_.initialize();
        if (init_result != montestrike::ErrorCode::SUCCESS) {
            throw std::runtime_error("Failed to initialize calculator");
        }
    }
    
    BenchmarkResult run_benchmark(uint32_t num_paths) {
        BenchmarkResult result = {};
        result.num_paths = num_paths;
        
        // Set up consistent parameters for benchmarking
        montestrike::MonteCarloPoT::Parameters params;
        params.current_price = 100.0f;
        params.strike_price = 105.0f;
        params.time_to_expiration = 14.0f / 365.0f;  // 2 weeks
        params.drift = 0.05f;
        params.volatility = 0.25f;
        params.steps_per_day = 50;
        params.num_paths = num_paths;
        params.use_antithetic_variates = false;  // Disable for consistent benchmarking
        params.random_seed = 12345;  // Fixed seed for consistency
        
        // Run multiple iterations for more accurate timing
        const int iterations = 3;
        double total_time = 0.0;
        double total_throughput = 0.0;
        uint64_t memory_used = 0;
        float pot_result = 0.0f;
        bool all_success = true;
        
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            auto results = calculator_.estimate_pot(params);
            auto end = std::chrono::high_resolution_clock::now();
            
            if (!results.computation_successful) {
                all_success = false;
                break;
            }
            
            double iteration_time = std::chrono::duration<double, std::milli>(end - start).count();
            total_time += iteration_time;
            total_throughput += results.metrics.throughput_paths_per_sec;
            memory_used = results.metrics.memory_used_bytes;
            pot_result = results.probability_of_touch;
        }
        
        if (all_success) {
            result.computation_time_ms = total_time / iterations;
            result.throughput_paths_per_sec = total_throughput / iterations;
            result.memory_used_mb = memory_used / (1024 * 1024);
            result.probability_of_touch = pot_result;
            result.success = true;
        } else {
            result.success = false;
        }
        
        return result;
    }
    
    void run_full_benchmark() {
        std::cout << "🏁 MonteStrike Performance Benchmark" << std::endl;
        std::cout << "====================================" << std::endl;
        
        // Display system information
        auto& device_analyzer = calculator_.get_device_analyzer();
        auto best_device = device_analyzer.get_best_device();
        auto memory_info = device_analyzer.get_memory_info();
        
        std::cout << "🖥️  Test Device: " << best_device.name << std::endl;
        std::cout << "⚡ Compute Capability: " << std::fixed << std::setprecision(1) << best_device.compute_capability << std::endl;
        std::cout << "🧠 Streaming Multiprocessors: " << best_device.streaming_multiprocessors << std::endl;
        std::cout << "💾 Global Memory: " << (best_device.global_memory_bytes / (1024*1024*1024)) << " GB" << std::endl;
        std::cout << "💾 Available Memory: " << (memory_info.free_bytes / (1024*1024)) << " MB" << std::endl;
        std::cout << std::endl;
        
        // Path count scenarios to test
        std::vector<uint32_t> path_counts = {
            50000,      // 50K
            100000,     // 100K
            500000,     // 500K
            1000000,    // 1M
            2000000,    // 2M
            4000000     // 4M
        };
        
        std::vector<BenchmarkResult> results;
        
        std::cout << "🎯 Running benchmark scenarios..." << std::endl;
        std::cout << std::endl;
        
        // Table header
        std::cout << "| Paths    | Time (ms) | Throughput (M paths/s) | Memory (MB) | PoT    | Status |" << std::endl;
        std::cout << "|----------|-----------|------------------------|-------------|--------|--------|" << std::endl;
        
        for (uint32_t paths : path_counts) {
            std::cout << "| " << std::setw(8) << paths << " | ";
            std::cout.flush();
            
            auto result = run_benchmark(paths);
            results.push_back(result);
            
            if (result.success) {
                std::cout << std::setw(9) << std::fixed << std::setprecision(1) << result.computation_time_ms << " | ";
                std::cout << std::setw(22) << std::fixed << std::setprecision(2) << (result.throughput_paths_per_sec / 1000000.0) << " | ";
                std::cout << std::setw(11) << result.memory_used_mb << " | ";
                std::cout << std::setw(6) << std::fixed << std::setprecision(4) << result.probability_of_touch << " | ";
                std::cout << "✅ OK   |" << std::endl;
            } else {
                std::cout << "    ERROR | ";
                std::cout << "               ERROR   | ";
                std::cout << "      ERROR | ";
                std::cout << " ERROR | ";
                std::cout << "❌ FAIL |" << std::endl;
            }
        }
        
        std::cout << std::endl;
        
        // Performance analysis
        analyze_performance(results);
        
        // Save results to file
        save_results_to_file(results, best_device);
    }
    
private:
    void analyze_performance(const std::vector<BenchmarkResult>& results) {
        std::cout << "📊 Performance Analysis:" << std::endl;
        std::cout << "========================" << std::endl;
        
        // Find successful results
        std::vector<BenchmarkResult> successful_results;
        for (const auto& result : results) {
            if (result.success) {
                successful_results.push_back(result);
            }
        }
        
        if (successful_results.empty()) {
            std::cout << "❌ No successful benchmark runs!" << std::endl;
            return;
        }
        
        // Find best throughput
        auto best_throughput = std::max_element(successful_results.begin(), successful_results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.throughput_paths_per_sec < b.throughput_paths_per_sec;
            });
        
        std::cout << "🏆 Best Throughput: " << std::fixed << std::setprecision(2) 
                  << (best_throughput->throughput_paths_per_sec / 1000000.0) << " M paths/sec" 
                  << " (" << best_throughput->num_paths << " paths)" << std::endl;
        
        // Calculate efficiency (throughput per core)
        auto& device_analyzer = calculator_.get_device_analyzer();
        auto device = device_analyzer.get_best_device();
        uint32_t total_cores = device.streaming_multiprocessors * device.cores_per_sm;
        
        double efficiency = best_throughput->throughput_paths_per_sec / total_cores;
        std::cout << "⚡ Efficiency: " << std::fixed << std::setprecision(0) 
                  << efficiency << " paths/sec per core" << std::endl;
        
        // Memory scaling analysis
        if (successful_results.size() >= 2) {
            auto largest = std::max_element(successful_results.begin(), successful_results.end(),
                [](const BenchmarkResult& a, const BenchmarkResult& b) {
                    return a.num_paths < b.num_paths;
                });
            
            auto smallest = std::min_element(successful_results.begin(), successful_results.end(),
                [](const BenchmarkResult& a, const BenchmarkResult& b) {
                    return a.num_paths < b.num_paths;
                });
            
            double memory_scaling = static_cast<double>(largest->memory_used_mb) / smallest->memory_used_mb;
            double path_scaling = static_cast<double>(largest->num_paths) / smallest->num_paths;
            
            std::cout << "💾 Memory Scaling: " << std::fixed << std::setprecision(2) 
                      << memory_scaling << "x for " << std::fixed << std::setprecision(1) 
                      << path_scaling << "x paths" << std::endl;
        }
        
        // Performance recommendations
        std::cout << std::endl;
        std::cout << "💡 Recommendations:" << std::endl;
        if (best_throughput->num_paths >= 1000000) {
            std::cout << "• Optimal performance at 1M+ paths" << std::endl;
            std::cout << "• GPU resources fully utilized" << std::endl;
        } else {
            std::cout << "• Consider using larger path counts for better GPU utilization" << std::endl;
        }
        
        if (best_throughput->throughput_paths_per_sec > 1000000) {
            std::cout << "• Excellent performance achieved" << std::endl;
        } else {
            std::cout << "• Performance may be limited by memory bandwidth or compute capability" << std::endl;
        }
    }
    
    void save_results_to_file(const std::vector<BenchmarkResult>& results, const montestrike::DeviceInfo& device) {
        std::ofstream file("benchmark_results.csv");
        if (!file.is_open()) {
            std::cerr << "⚠️  Could not save benchmark results to file" << std::endl;
            return;
        }
        
        // CSV header
        file << "device_name,compute_capability,paths,time_ms,throughput_paths_per_sec,memory_mb,probability_of_touch,success\n";
        
        for (const auto& result : results) {
            file << device.name << ","
                 << device.compute_capability << ","
                 << result.num_paths << ","
                 << result.computation_time_ms << ","
                 << result.throughput_paths_per_sec << ","
                 << result.memory_used_mb << ","
                 << result.probability_of_touch << ","
                 << (result.success ? "true" : "false") << "\n";
        }
        
        file.close();
        std::cout << "💾 Benchmark results saved to benchmark_results.csv" << std::endl;
    }
    
    montestrike::MonteCarloPoT calculator_;
};

int main() {
    try {
        PerformanceBenchmark benchmark;
        benchmark.run_full_benchmark();
        
        std::cout << std::endl;
        std::cout << "✅ Benchmark completed successfully!" << std::endl;
        std::cout << std::endl;
        std::cout << "⚠️  Note: Results may vary based on GPU temperature, driver version," << std::endl;
        std::cout << "   and system load. Run multiple times for consistent measurements." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}