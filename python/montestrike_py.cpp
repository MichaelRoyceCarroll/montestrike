#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "montestrike/montestrike.hpp"

namespace py = pybind11;

PYBIND11_MODULE(montestrike, m) {
    m.doc() = "MonteStrike: Monte Carlo US Option Probability of Touch Estimator";
    
    // ComputeBackend enum
    py::enum_<montestrike::ComputeBackend>(m, "ComputeBackend")
        .value("CUDA", montestrike::ComputeBackend::CUDA)
        .value("AVX2", montestrike::ComputeBackend::AVX2)
        .value("CPU", montestrike::ComputeBackend::CPU)
        .export_values();

    // Error codes enum
    py::enum_<montestrike::ErrorCode>(m, "ErrorCode")
        .value("SUCCESS", montestrike::ErrorCode::SUCCESS)
        .value("INVALID_PARAMETERS", montestrike::ErrorCode::INVALID_PARAMETERS)
        .value("CUDA_ERROR", montestrike::ErrorCode::CUDA_ERROR)
        .value("MEMORY_ALLOCATION_FAILED", montestrike::ErrorCode::MEMORY_ALLOCATION_FAILED)
        .value("NO_CUDA_DEVICE", montestrike::ErrorCode::NO_CUDA_DEVICE)
        .value("INCOMPATIBLE_DEVICE", montestrike::ErrorCode::INCOMPATIBLE_DEVICE)
        .value("KERNEL_LAUNCH_FAILED", montestrike::ErrorCode::KERNEL_LAUNCH_FAILED)
        .value("MEMORY_COPY_FAILED", montestrike::ErrorCode::MEMORY_COPY_FAILED)
        .value("DEVICE_SELECTION_FAILED", montestrike::ErrorCode::DEVICE_SELECTION_FAILED)
        .value("COMPUTATION_TIMEOUT", montestrike::ErrorCode::COMPUTATION_TIMEOUT)
        .value("BACKEND_NOT_AVAILABLE", montestrike::ErrorCode::BACKEND_NOT_AVAILABLE)
        .value("AVX2_NOT_SUPPORTED", montestrike::ErrorCode::AVX2_NOT_SUPPORTED)
        .value("CPU_THREAD_ERROR", montestrike::ErrorCode::CPU_THREAD_ERROR)
        .value("UNKNOWN_ERROR", montestrike::ErrorCode::UNKNOWN_ERROR)
        .export_values();
    
    // DeviceInfo struct
    py::class_<montestrike::DeviceInfo>(m, "DeviceInfo")
        .def(py::init<>())
        .def_readwrite("device_id", &montestrike::DeviceInfo::device_id)
        .def_readwrite("name", &montestrike::DeviceInfo::name)
        .def_readwrite("backend", &montestrike::DeviceInfo::backend)
        .def_readwrite("streaming_multiprocessors", &montestrike::DeviceInfo::streaming_multiprocessors)
        .def_readwrite("cores_per_sm", &montestrike::DeviceInfo::cores_per_sm)
        .def_readwrite("global_memory_bytes", &montestrike::DeviceInfo::global_memory_bytes)
        .def_readwrite("max_threads_per_block", &montestrike::DeviceInfo::max_threads_per_block)
        .def_readwrite("max_threads_per_sm", &montestrike::DeviceInfo::max_threads_per_sm)
        .def_readwrite("compute_capability", &montestrike::DeviceInfo::compute_capability)
        .def_readwrite("is_compatible", &montestrike::DeviceInfo::is_compatible)
        .def_readwrite("available_memory_bytes", &montestrike::DeviceInfo::available_memory_bytes)
        .def_readwrite("supports_avx2", &montestrike::DeviceInfo::supports_avx2)
        .def_readwrite("supports_fma", &montestrike::DeviceInfo::supports_fma)
        .def_readwrite("cpu_logical_cores", &montestrike::DeviceInfo::cpu_logical_cores)
        .def_readwrite("cpu_physical_cores", &montestrike::DeviceInfo::cpu_physical_cores)
        .def("__repr__", [](const montestrike::DeviceInfo& info) {
            return "<DeviceInfo device_id=" + std::to_string(info.device_id) + 
                   " name='" + info.name + "' compatible=" + (info.is_compatible ? "True" : "False") + ">";
        });
    
    // MemoryInfo struct
    py::class_<montestrike::MemoryInfo>(m, "MemoryInfo")
        .def(py::init<>())
        .def_readwrite("total_bytes", &montestrike::MemoryInfo::total_bytes)
        .def_readwrite("free_bytes", &montestrike::MemoryInfo::free_bytes)
        .def_readwrite("used_bytes", &montestrike::MemoryInfo::used_bytes)
        .def_readwrite("allocated_for_paths", &montestrike::MemoryInfo::allocated_for_paths)
        .def("__repr__", [](const montestrike::MemoryInfo& info) {
            return "<MemoryInfo total=" + std::to_string(info.total_bytes / (1024*1024)) + "MB" +
                   " free=" + std::to_string(info.free_bytes / (1024*1024)) + "MB>";
        });
    
    // ComputationMetrics struct
    py::class_<montestrike::ComputationMetrics>(m, "ComputationMetrics")
        .def(py::init<>())
        .def_readwrite("computation_time_ms", &montestrike::ComputationMetrics::computation_time_ms)
        .def_readwrite("kernel_time_ms", &montestrike::ComputationMetrics::kernel_time_ms)
        .def_readwrite("memory_transfer_time_ms", &montestrike::ComputationMetrics::memory_transfer_time_ms)
        .def_readwrite("throughput_paths_per_sec", &montestrike::ComputationMetrics::throughput_paths_per_sec)
        .def_readwrite("memory_used_bytes", &montestrike::ComputationMetrics::memory_used_bytes);
    
    // ValidationResult struct
    py::class_<montestrike::ValidationResult>(m, "ValidationResult")
        .def(py::init<>())
        .def_readwrite("is_valid", &montestrike::ValidationResult::is_valid)
        .def_readwrite("error_message", &montestrike::ValidationResult::error_message)
        .def_readwrite("error_code", &montestrike::ValidationResult::error_code)
        .def("__bool__", [](const montestrike::ValidationResult& v) { return v.is_valid; });
    
    // VersionInfo struct
    py::class_<montestrike::VersionInfo>(m, "VersionInfo")
        .def(py::init<>())
        .def_readwrite("major", &montestrike::VersionInfo::major)
        .def_readwrite("minor", &montestrike::VersionInfo::minor)
        .def_readwrite("patch", &montestrike::VersionInfo::patch)
        .def_readwrite("build_type", &montestrike::VersionInfo::build_type)
        .def_readwrite("cuda_version", &montestrike::VersionInfo::cuda_version)
        .def_readwrite("build_date", &montestrike::VersionInfo::build_date)
        .def_readwrite("git_commit", &montestrike::VersionInfo::git_commit)
        .def("to_string", &montestrike::VersionInfo::to_string)
        .def("__str__", &montestrike::VersionInfo::to_string)
        .def("__repr__", [](const montestrike::VersionInfo& v) { 
            return "<VersionInfo " + v.to_string() + ">"; 
        });
    
    // DeviceAnalyzer class
    py::class_<montestrike::DeviceAnalyzer>(m, "DeviceAnalyzer")
        .def(py::init<>())
        .def("enumerate_devices", &montestrike::DeviceAnalyzer::enumerate_devices,
             "Get list of all available CUDA devices")
        .def("get_device_info", &montestrike::DeviceAnalyzer::get_device_info,
             "Get detailed information about a specific device", py::arg("device_id"))
        .def("get_best_device", &montestrike::DeviceAnalyzer::get_best_device,
             "Get the most performant compatible device")
        .def("set_device", &montestrike::DeviceAnalyzer::set_device,
             "Set the active CUDA device", py::arg("device_id"))
        .def("get_current_device", &montestrike::DeviceAnalyzer::get_current_device,
             "Get the currently active device ID")
        .def("get_memory_info", &montestrike::DeviceAnalyzer::get_memory_info,
             "Get memory information for a device", py::arg("device_id") = -1)
        .def("is_device_compatible", &montestrike::DeviceAnalyzer::is_device_compatible,
             "Check if device meets minimum requirements", py::arg("device_id"))
        .def("has_sufficient_memory", &montestrike::DeviceAnalyzer::has_sufficient_memory,
             "Check if device has enough memory for path count", 
             py::arg("device_id"), py::arg("num_paths"))
        .def("get_optimal_block_size", &montestrike::DeviceAnalyzer::get_optimal_block_size,
             "Get optimal CUDA block size for device", py::arg("device_id") = -1)
        .def("get_optimal_grid_size", &montestrike::DeviceAnalyzer::get_optimal_grid_size,
             "Get optimal CUDA grid size for device and path count",
             py::arg("device_id"), py::arg("num_paths"))
        .def("get_cuda_version", &montestrike::DeviceAnalyzer::get_cuda_version,
             "Get CUDA driver and runtime version information");
    
    // MonteCarloPoT Parameters
    py::class_<montestrike::MonteCarloPoT::Parameters>(m, "Parameters")
        .def(py::init<>())
        .def_readwrite("current_price", &montestrike::MonteCarloPoT::Parameters::current_price,
                      "Current stock price (S)")
        .def_readwrite("strike_price", &montestrike::MonteCarloPoT::Parameters::strike_price,
                      "Strike price to test for touch (K)")
        .def_readwrite("time_to_expiration", &montestrike::MonteCarloPoT::Parameters::time_to_expiration,
                      "Time to expiration in years (T)")
        .def_readwrite("drift", &montestrike::MonteCarloPoT::Parameters::drift,
                      "Expected return rate (μ)")
        .def_readwrite("volatility", &montestrike::MonteCarloPoT::Parameters::volatility,
                      "Volatility - standard deviation of returns (σ)")
        .def_readwrite("steps_per_day", &montestrike::MonteCarloPoT::Parameters::steps_per_day,
                      "Time resolution - steps per trading day")
        .def_readwrite("num_paths", &montestrike::MonteCarloPoT::Parameters::num_paths,
                      "Total paths to simulate (50K - 4M)")
        .def_readwrite("use_antithetic_variates", &montestrike::MonteCarloPoT::Parameters::use_antithetic_variates,
                      "Use antithetic variates for variance reduction")
        .def_readwrite("random_seed", &montestrike::MonteCarloPoT::Parameters::random_seed,
                      "Random seed for reproducible results (0 = system time)")
        .def_readwrite("backend", &montestrike::MonteCarloPoT::Parameters::backend,
                      "Compute backend to use (CUDA, AVX2, or CPU)")
        .def_readwrite("cpu_threads", &montestrike::MonteCarloPoT::Parameters::cpu_threads,
                      "Number of CPU threads to use (0 = auto-detect)")
        .def_readwrite("strict_backend_mode", &montestrike::MonteCarloPoT::Parameters::strict_backend_mode,
                      "If true, fail if requested backend unavailable")
        .def_readwrite("progress_report_interval_ms", &montestrike::MonteCarloPoT::Parameters::progress_report_interval_ms,
                      "Progress callback interval in milliseconds")
        .def_readwrite("device_id", &montestrike::MonteCarloPoT::Parameters::device_id,
                      "CUDA device ID (-1 for auto-select)")
        .def("__repr__", [](const montestrike::MonteCarloPoT::Parameters& p) {
            return "<Parameters S=" + std::to_string(p.current_price) + 
                   " K=" + std::to_string(p.strike_price) + 
                   " T=" + std::to_string(p.time_to_expiration) + 
                   " paths=" + std::to_string(p.num_paths) + ">";
        });
    
    // MonteCarloPoT Results
    py::class_<montestrike::MonteCarloPoT::Results>(m, "Results")
        .def(py::init<>())
        .def_readwrite("probability_of_touch", &montestrike::MonteCarloPoT::Results::probability_of_touch,
                      "Estimated probability of touch (0.0 - 1.0)")
        .def_readwrite("paths_processed", &montestrike::MonteCarloPoT::Results::paths_processed,
                      "Number of paths actually processed")
        .def_readwrite("paths_touched", &montestrike::MonteCarloPoT::Results::paths_touched,
                      "Number of paths that touched the strike")
        .def_readwrite("computation_successful", &montestrike::MonteCarloPoT::Results::computation_successful,
                      "Whether computation completed successfully")
        .def_readwrite("error_code", &montestrike::MonteCarloPoT::Results::error_code,
                      "Error code if computation failed")
        .def_readwrite("error_message", &montestrike::MonteCarloPoT::Results::error_message,
                      "Error message if computation failed")
        .def_readwrite("metrics", &montestrike::MonteCarloPoT::Results::metrics,
                      "Performance and timing metrics")
        .def_readwrite("device_used", &montestrike::MonteCarloPoT::Results::device_used,
                      "Information about the GPU device used")
        .def("__bool__", [](const montestrike::MonteCarloPoT::Results& r) { 
            return r.computation_successful; 
        })
        .def("__repr__", [](const montestrike::MonteCarloPoT::Results& r) {
            if (r.computation_successful) {
                return "<Results PoT=" + std::to_string(r.probability_of_touch) + 
                       " paths=" + std::to_string(r.paths_processed) + 
                       " time=" + std::to_string(r.metrics.computation_time_ms) + "ms>";
            } else {
                return "<Results FAILED: " + r.error_message + ">";
            }
        });
    
    // MonteCarloPoT main class
    py::class_<montestrike::MonteCarloPoT>(m, "MonteCarloPoT")
        .def(py::init<>())
        .def("estimate_pot", &montestrike::MonteCarloPoT::estimate_pot,
             "Estimate probability of touch for given parameters",
             py::arg("parameters"))
        .def("validate_parameters", &montestrike::MonteCarloPoT::validate_parameters,
             "Validate parameters before computation",
             py::arg("parameters"))
        .def("initialize", &montestrike::MonteCarloPoT::initialize,
             "Initialize the computation engine with specific device",
             py::arg("device_id") = -1)
        .def("shutdown", &montestrike::MonteCarloPoT::shutdown,
             "Shutdown and cleanup GPU resources")
        .def("is_initialized", &montestrike::MonteCarloPoT::is_initialized,
             "Check if the engine is initialized")
        .def("get_device_analyzer", &montestrike::MonteCarloPoT::get_device_analyzer,
             "Get the device analyzer instance",
             py::return_value_policy::reference_internal)
        .def("get_memory_info", &montestrike::MonteCarloPoT::get_memory_info,
             "Get current GPU memory information")
        .def("estimate_memory_requirements", &montestrike::MonteCarloPoT::estimate_memory_requirements,
             "Estimate GPU memory required for path count",
             py::arg("num_paths"))
        .def("estimate_computation_time", &montestrike::MonteCarloPoT::estimate_computation_time,
             "Estimate computation time for path count",
             py::arg("num_paths"));
    
    // Utility functions
    m.def("validate_monte_carlo_parameters", &montestrike::validate_monte_carlo_parameters,
          "Validate Monte Carlo parameters",
          py::arg("current_price"), py::arg("strike_price"), py::arg("time_to_expiration"),
          py::arg("drift"), py::arg("volatility"), py::arg("steps_per_day"), py::arg("num_paths"));
    
    m.def("error_code_to_string", &montestrike::error_code_to_string,
          "Convert error code to descriptive string",
          py::arg("error_code"));
    
    m.def("get_version_info", &montestrike::get_version_info,
          "Get version information");
    
    m.def("get_version_string", &montestrike::get_version_string,
          "Get version string");
    
    // Constants
    m.attr("VERSION_MAJOR") = montestrike::VERSION_MAJOR;
    m.attr("VERSION_MINOR") = montestrike::VERSION_MINOR;
    m.attr("VERSION_PATCH") = montestrike::VERSION_PATCH;
    
    // Add module-level documentation
    m.attr("__version__") = montestrike::get_version_string();
    m.attr("__author__") = "MonteStrike Contributors";
    m.attr("__license__") = "MIT";
}