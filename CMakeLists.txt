cmake_minimum_required(VERSION 3.13)
project(WarpAnalysis)

# Step 1: Locate llvm-config
find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config-19 llvm-config REQUIRED)

# Step 2: Extract LLVM configuration
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --includedir
  OUTPUT_VARIABLE LLVM_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir
  OUTPUT_VARIABLE LLVM_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --libs
  OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(LLVM_LIBS)

execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --system-libs
  OUTPUT_VARIABLE LLVM_SYS_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(LLVM_SYS_LIBS)

execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --cxxflags
  OUTPUT_VARIABLE LLVM_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(LLVM_CXX_FLAGS)

# Step 3: Set up build config
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include_directories(
  ${LLVM_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(${LLVM_LIB_DIR})

# Step 4: Add the WarpAnalysis plugin source files
add_library(WarpAnalysis MODULE
  src/Utils.cpp
  src/AnalysisLogic.cpp
  src/JsonEmitter.cpp
  src/WarpAnalysis.cpp
)

target_compile_options(WarpAnalysis PRIVATE ${LLVM_CXX_FLAGS})
target_link_libraries(WarpAnalysis PRIVATE ${LLVM_LIBS} ${LLVM_SYS_LIBS})

set_target_properties(WarpAnalysis PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)
