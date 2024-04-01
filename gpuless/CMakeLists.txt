cmake_minimum_required(VERSION 3.17)
project(gpuless)

enable_language(CUDA)
find_package(CUDAToolkit 11.4)

set(CUDNN_DIR "" CACHE STRING "")

option(BUILD_BENCHMARKS "Build the resnet50 benchmark" OFF)

option(SPDLOG_LEVEL_INFO "Set the spdlog level to info" OFF)
option(SPDLOG_LEVEL_DEBUG "Set the spdlog level to debug" OFF)
option(SPDLOG_LEVEL_TRACE "Set the spdlog level to trace" OFF)
option(SPDLOG_LEVEL_OFF "Set the spdlog level to off" OFF)

if(SPDLOG_LEVEL_INFO)
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
    message("Set SPDLOG_LEVEL_INFO")
elseif(SPDLOG_LEVEL_DEBUG)
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
    message("Set SPDLOG_LEVEL_DEBUG")
elseif(SPDLOG_LEVEL_OFF)
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF)
    message("Set SPDLOG_LEVEL_OFF")
elseif(SPDLOG_LEVEL_TRACE)
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
    message("Set SPDLOG_LEVEL_TRACE")
else()
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF)
    message("Set SPDLOG_LEVEL_OFF")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

# special options for compiling on the CSCS cluster
cmake_host_system_information(RESULT _hostname QUERY HOSTNAME)
if (${_hostname} MATCHES ".*\.cscs\.ch")
    add_compile_options(-DCSCS)
    #include_directories($ENV{HOME}/cudnn/include)
    #link_directories($ENV{HOME}/cudnn/lib64)
    include_directories(${CUDNN_DIR}/include)
    link_directories(${CUDNN_DIR}/lib64)
    link_directories(${CUDNN_DIR}/lib)
else()
  # FIXME: is there a better solution?
  # fix for missing link `-lcudnn`
  link_directories(gpuless PUBLIC ${CUDAToolkit_LIBRARY_DIR})
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -ggdb")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -Wall -Wextra -Wno-unused-parameter -ggdb -fno-operator-names")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# build flatbuffer schemas with external Makefile
#add_custom_target(
#    flatbuffer
#    COMMAND make
#    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/schemas)

add_executable(manager_trace
    trace/manager/manager.cpp
    trace/manager/manager_device.cpp
    trace/cuda_trace.cpp
    trace/cuda_trace_converter.cpp
    trace/cuda_api_calls.cpp
    trace/cudnn_api_calls.cpp
    trace/cublas_api_calls.cpp
    trace/cuda_virtual_device.cpp
    trace/cuda_virtual_device.cpp
    trace/trace_executor_local.cpp
    trace/trace_executor_tcp_client.cpp
    trace/dlsym_util.cpp
    utils.cpp)
target_include_directories(manager_trace SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>)
add_dependencies(manager_trace flatbuffers)
add_dependencies(manager_trace spdlog)
target_link_options(manager_trace PRIVATE "LINKER:-no-as-needed")
target_link_libraries(manager_trace
    iberty
    stdc++fs
    cudnn
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    pthread
    spdlog::spdlog
    flatbuffers
    iceoryx_posh::iceoryx_posh
)

add_library(analysis SHARED
    analysis/libcudaanalysis.cpp
    trace/cubin_analysis.cpp
    utils.cpp trace/trace_executor.hpp)
add_dependencies(analysis spdlog)
target_link_libraries(analysis
    CUDA::cudart
    CUDA::cuda_driver
    spdlog::spdlog
)

add_library(argument_analysis SHARED
        argument_analysis/libarganalysis.cpp
        trace/cubin_analysis.cpp
        utils.cpp trace/trace_executor.hpp)
target_link_libraries(argument_analysis
        CUDA::cudart
        CUDA::cuda_driver
        spdlog::spdlog
)

add_library(gpuless SHARED
    trace/libgpuless.cpp
    trace/hijack_cudnn.cpp
    trace/hijack_cublas.cpp
    trace/cuda_trace.cpp
    trace/cuda_trace_converter.cpp
    trace/cuda_api_calls.cpp
    trace/cudnn_api_calls.cpp
    trace/cublas_api_calls.cpp
    trace/cuda_virtual_device.cpp
    trace/cuda_virtual_device.cpp
    trace/trace_executor_local.cpp
    trace/trace_executor_tcp_client.cpp
    trace/trace_executor_shmem_client.cpp
    trace/dlsym_util.cpp
    utils.cpp
    trace/cubin_analysis.cpp)
target_include_directories(gpuless SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(gpuless
    iberty
    stdc++fs
    cudnn
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
    flatbuffers
    iceoryx_posh::iceoryx_posh
)

add_custom_command(
    TARGET gpuless POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR})
