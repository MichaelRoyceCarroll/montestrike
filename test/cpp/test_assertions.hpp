#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <sstream>

namespace montestrike_test {

class TestResult {
public:
    bool passed;
    std::string test_name;
    std::string error_message;
    double execution_time_ms;
    
    TestResult(const std::string& name) 
        : passed(false), test_name(name), execution_time_ms(0.0) {}
};

extern int total_tests;
extern int passed_tests;
extern int failed_tests;

#define MONTESTRIKE_ASSERT(condition, message) \
    do { \
        total_tests++; \
        if (!(condition)) { \
            failed_tests++; \
            std::ostringstream oss; \
            oss << "ASSERTION FAILED: " << message \
                << " at " << __FILE__ << ":" << __LINE__; \
            std::cerr << "❌ " << oss.str() << std::endl; \
            return false; \
        } else { \
            passed_tests++; \
        } \
    } while(0)

#define MONTESTRIKE_ASSERT_NEAR(actual, expected, tolerance, message) \
    do { \
        total_tests++; \
        double diff = std::abs(static_cast<double>(actual) - static_cast<double>(expected)); \
        if (diff >= tolerance) { \
            failed_tests++; \
            std::ostringstream oss; \
            oss << "ASSERTION FAILED: " << message \
                << " (expected " << expected << ", got " << actual \
                << ", diff " << diff << " >= tolerance " << tolerance << ")" \
                << " at " << __FILE__ << ":" << __LINE__; \
            std::cerr << "❌ " << oss.str() << std::endl; \
            return false; \
        } else { \
            passed_tests++; \
        } \
    } while(0)

#define MONTESTRIKE_ASSERT_GT(actual, expected, message) \
    do { \
        total_tests++; \
        if (!((actual) > (expected))) { \
            failed_tests++; \
            std::ostringstream oss; \
            oss << "ASSERTION FAILED: " << message \
                << " (expected " << actual << " > " << expected << ")" \
                << " at " << __FILE__ << ":" << __LINE__; \
            std::cerr << "❌ " << oss.str() << std::endl; \
            return false; \
        } else { \
            passed_tests++; \
        } \
    } while(0)

#define MONTESTRIKE_ASSERT_LT(actual, expected, message) \
    do { \
        total_tests++; \
        if (!((actual) < (expected))) { \
            failed_tests++; \
            std::ostringstream oss; \
            oss << "ASSERTION FAILED: " << message \
                << " (expected " << actual << " < " << expected << ")" \
                << " at " << __FILE__ << ":" << __LINE__; \
            std::cerr << "❌ " << oss.str() << std::endl; \
            return false; \
        } else { \
            passed_tests++; \
        } \
    } while(0)

#define MONTESTRIKE_ASSERT_EQ(actual, expected, message) \
    do { \
        total_tests++; \
        if (!((actual) == (expected))) { \
            failed_tests++; \
            std::ostringstream oss; \
            oss << "ASSERTION FAILED: " << message \
                << " (expected " << expected << ", got " << actual << ")" \
                << " at " << __FILE__ << ":" << __LINE__; \
            std::cerr << "❌ " << oss.str() << std::endl; \
            return false; \
        } else { \
            passed_tests++; \
        } \
    } while(0)

#define MONTESTRIKE_ASSERT_TRUE(condition, message) \
    MONTESTRIKE_ASSERT(condition, message)

#define MONTESTRIKE_ASSERT_FALSE(condition, message) \
    MONTESTRIKE_ASSERT(!(condition), message)

inline void print_test_header(const std::string& test_name) {
    std::cout << "🧪 Running " << test_name << "..." << std::endl;
}

inline void print_test_result(const std::string& test_name, bool passed, double time_ms = 0.0) {
    if (passed) {
        std::cout << "✅ " << test_name << " PASSED";
        if (time_ms > 0) {
            std::cout << " (" << time_ms << " ms)";
        }
        std::cout << std::endl;
    } else {
        std::cout << "❌ " << test_name << " FAILED" << std::endl;
    }
}

inline void print_test_summary() {
    std::cout << "\n" << "=" * 50 << std::endl;
    std::cout << "📊 TEST SUMMARY" << std::endl;
    std::cout << "Total tests: " << total_tests << std::endl;
    std::cout << "Passed: " << passed_tests << std::endl;
    std::cout << "Failed: " << failed_tests << std::endl;
    
    if (failed_tests == 0) {
        std::cout << "🎉 All tests passed!" << std::endl;
    } else {
        std::cout << "💥 " << failed_tests << " test(s) failed!" << std::endl;
    }
    std::cout << "=" * 50 << std::endl;
}

} // namespace montestrike_test