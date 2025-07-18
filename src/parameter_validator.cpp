#include "montestrike/montestrike.hpp"
#include <cmath>
#include <sstream>

namespace montestrike {

ValidationResult validate_monte_carlo_parameters(
    float current_price,
    float strike_price, 
    float time_to_expiration,
    float drift,
    float volatility,
    uint32_t steps_per_day,
    uint32_t num_paths
) {
    ValidationResult result;
    result.is_valid = true;
    result.error_code = ErrorCode::SUCCESS;
    result.error_message = "";
    
    std::ostringstream error_stream;
    bool has_error = false;
    
    // Validate current_price
    if (current_price <= 0.0f || !std::isfinite(current_price)) {
        error_stream << "current_price must be positive and finite (got " << current_price << "); ";
        has_error = true;
    }
    
    // Validate strike_price  
    if (strike_price <= 0.0f || !std::isfinite(strike_price)) {
        error_stream << "strike_price must be positive and finite (got " << strike_price << "); ";
        has_error = true;
    }
    
    // Validate time_to_expiration
    if (time_to_expiration <= 0.0f || time_to_expiration > 10.0f || !std::isfinite(time_to_expiration)) {
        error_stream << "time_to_expiration must be between 0 and 10 years (got " << time_to_expiration << "); ";
        has_error = true;
    }
    
    // Validate drift (allow negative values, but check for reasonable range)
    if (!std::isfinite(drift) || drift < -1.0f || drift > 1.0f) {
        error_stream << "drift must be finite and between -100% and 100% (got " << drift << "); ";
        has_error = true;
    }
    
    // Validate volatility
    if (volatility <= 0.0f || volatility > 5.0f || !std::isfinite(volatility)) {
        error_stream << "volatility must be positive, finite and <= 500% (got " << volatility << "); ";
        has_error = true;
    }
    
    // Validate steps_per_day
    if (steps_per_day == 0 || steps_per_day > 100) {
        error_stream << "steps_per_day must be between 1 and 100 (got " << steps_per_day << "); ";
        has_error = true;
    }
    
    // Validate num_paths
    if (num_paths < 1000 || num_paths > 4000000) {
        error_stream << "num_paths must be between 1,000 and 4,000,000 (got " << num_paths << "); ";
        has_error = true;
    }
    
    // Check for reasonable price ratio
    if (std::isfinite(current_price) && std::isfinite(strike_price)) {
        float price_ratio = strike_price / current_price;
        if (price_ratio < 0.1f || price_ratio > 10.0f) {
            error_stream << "strike_price/current_price ratio is extreme (got " << price_ratio << "), consider values closer to current price; ";
            has_error = true;
        }
    }
    
    // Check for very short time to expiration
    if (std::isfinite(time_to_expiration) && time_to_expiration < (1.0f / 365.0f)) {
        error_stream << "time_to_expiration is very short (< 1 day), results may be unreliable; ";
        // This is a warning, not an error
    }
    
    if (has_error) {
        result.is_valid = false;
        result.error_code = ErrorCode::INVALID_PARAMETERS;
        result.error_message = error_stream.str();
        // Remove trailing "; "
        if (!result.error_message.empty()) {
            result.error_message = result.error_message.substr(0, result.error_message.length() - 2);
        }
    }
    
    return result;
}

std::string error_code_to_string(ErrorCode code) {
    switch (code) {
        case ErrorCode::SUCCESS:
            return "Success";
        case ErrorCode::INVALID_PARAMETERS:
            return "Invalid parameters";
        case ErrorCode::CUDA_ERROR:
            return "CUDA error";
        case ErrorCode::MEMORY_ALLOCATION_FAILED:
            return "Memory allocation failed";
        case ErrorCode::NO_CUDA_DEVICE:
            return "No CUDA device available";
        case ErrorCode::INCOMPATIBLE_DEVICE:
            return "Incompatible CUDA device";
        case ErrorCode::KERNEL_LAUNCH_FAILED:
            return "Kernel launch failed";
        case ErrorCode::MEMORY_COPY_FAILED:
            return "Memory copy failed";
        case ErrorCode::DEVICE_SELECTION_FAILED:
            return "Device selection failed";
        case ErrorCode::COMPUTATION_TIMEOUT:
            return "Computation timeout";
        case ErrorCode::UNKNOWN_ERROR:
        default:
            return "Unknown error";
    }
}

} // namespace montestrike