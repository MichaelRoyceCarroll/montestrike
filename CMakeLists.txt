cmake_minimum_required(VERSION 3.18)
project(MonteStrike VERSION 0.1.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Check CUDA compute capability
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89)
endif()

# Find Python and pybind11 for Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 QUIET)
    
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, downloading...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# CUDA-specific options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(MONTESTRIKE_SOURCES
    src/montestrike.cpp
    src/device_analyzer.cpp
    src/parameter_validator.cpp
    src/memory_manager.cpp
    src/version.cpp
)

set(MONTESTRIKE_CUDA_SOURCES
    cuda/monte_carlo_kernel.cu
)

set(MONTESTRIKE_HEADERS
    include/montestrike/montestrike.hpp
    include/montestrike/device_analyzer.hpp
    include/montestrike/types.hpp
    include/montestrike/version.hpp
)

# Create main library
add_library(montestrike SHARED 
    ${MONTESTRIKE_SOURCES} 
    ${MONTESTRIKE_CUDA_SOURCES}
)

# Link CUDA libraries
target_link_libraries(montestrike 
    CUDA::cudart 
    CUDA::curand
)

# Add to export set
set_target_properties(montestrike PROPERTIES EXPORT_NAME montestrike)
add_library(MonteStrike::montestrike ALIAS montestrike)

# Set library properties
set_target_properties(montestrike PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Platform-specific library naming
if(WIN32)
    set_target_properties(montestrike PROPERTIES
        OUTPUT_NAME "libmontestrike"
        PREFIX ""
    )
endif()

# Installation rules
install(TARGETS montestrike
    EXPORT MonteStrikeTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/montestrike
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Generate pkg-config file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/montestrike.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/montestrike.pc"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/montestrike.pc"
    DESTINATION lib/pkgconfig
)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(montestrike_python python/montestrike_py.cpp)
    
    target_link_libraries(montestrike_python PRIVATE montestrike)
    
    set_target_properties(montestrike_python PROPERTIES
        OUTPUT_NAME "montestrike"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
    
    # Install Python module
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib'))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    install(TARGETS montestrike_python
        DESTINATION ${PYTHON_SITE_PACKAGES}
    )
    
    install(FILES python/__init__.py
        DESTINATION ${PYTHON_SITE_PACKAGES}/montestrike
    )
endif()

# Test framework
if(BUILD_TESTS)
    enable_testing()
    
    # C++ tests
    add_executable(test_montestrike_cpp
        test/cpp/test_runner.cpp
        test/cpp/test_pot_calculator.cpp
    )
    
    target_link_libraries(test_montestrike_cpp montestrike)
    
    add_test(NAME cpp_tests COMMAND test_montestrike_cpp)
    
    # Python tests (if Python bindings are built)
    if(BUILD_PYTHON_BINDINGS)
        add_test(NAME python_tests
            COMMAND ${Python3_EXECUTABLE} -m pytest test/python/
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endif()
endif()

# Examples
if(BUILD_EXAMPLES)
    # C++ examples
    add_executable(basic_usage_cpp examples/basic_usage.cpp)
    target_link_libraries(basic_usage_cpp montestrike)
    
    add_executable(performance_benchmark examples/performance_benchmark.cpp)
    target_link_libraries(performance_benchmark montestrike)
    
    # Install examples
    install(TARGETS basic_usage_cpp performance_benchmark
        DESTINATION bin/examples
    )
    
    install(FILES 
        examples/basic_usage.py
        DESTINATION share/montestrike/examples
    )
endif()

# Generate export header
include(GenerateExportHeader)
generate_export_header(montestrike
    EXPORT_FILE_NAME include/montestrike/export.hpp
)

# CMake package config
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MonteStrikeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MonteStrikeConfig.cmake"
    INSTALL_DESTINATION lib/cmake/MonteStrike
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MonteStrikeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MonteStrikeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MonteStrikeConfigVersion.cmake"
    DESTINATION lib/cmake/MonteStrike
)

# Export targets
install(EXPORT MonteStrikeTargets
    FILE MonteStrikeTargets.cmake
    NAMESPACE MonteStrike::
    DESTINATION lib/cmake/MonteStrike
)

export(EXPORT MonteStrikeTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/MonteStrikeTargets.cmake"
    NAMESPACE MonteStrike::
)

# Status information
message(STATUS "MonteStrike Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "  Examples: ${BUILD_EXAMPLES}")

if(BUILD_PYTHON_BINDINGS)
    message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
    message(STATUS "  Python site-packages: ${PYTHON_SITE_PACKAGES}")
endif()