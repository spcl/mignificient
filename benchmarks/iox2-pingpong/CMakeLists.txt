# Standalone (not part of the main build):
#   cmake -S benchmarks/iox2-pingpong -B <dir> -DCMAKE_PREFIX_PATH=<iceoryx2 install> && cmake --build <dir>
cmake_minimum_required(VERSION 3.22)
project(iox2_pingpong LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(iceoryx2-cxx 0.8.1 REQUIRED)
add_executable(pingpong pingpong.cpp)
target_link_libraries(pingpong iceoryx2-cxx::static-lib-cxx)
