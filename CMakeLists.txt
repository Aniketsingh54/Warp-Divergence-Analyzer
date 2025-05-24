cmake_minimum_required(VERSION 3.13)

project(WarpDivergence LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LLVM
find_package(LLVM REQUIRED CONFIG)

# Enable LLVM support and include directories
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Add LLVM includes and definitions
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Define your plugin as a proper pass plugin
set(LLVM_LINK_COMPONENTS
  Core
  Support
  Analysis
)

# Enable plugin support
include(AddLLVM)

add_llvm_pass_plugin(WarpDivergence
  lib/WarpDivergence.cpp
)
