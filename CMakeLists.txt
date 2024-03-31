cmake_minimum_required(VERSION 3.17)
include(FetchContent)
project(mignificient)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

find_package(spdlog 1.8...<1.10 QUIET)
if(NOT spdlog_FOUND)
  message(STATUS "Downloading and building spdlog dependency")
  FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    # default branch is v1.x - for some reason, cmake switches to master
    GIT_TAG v1.8.0
  )
  FetchContent_MakeAvailable(spdlog)
  set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
  message(STATUS "Using installed spdlog version")
  add_custom_target(spdlog)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/external/flatbuffers
                 ${EXTERNAL_INSTALL_LOCATION}/flatbuffers-build
                 EXCLUDE_FROM_ALL)

add_subdirectory(${CMAKE_SOURCE_DIR}/external/iceoryx/iceoryx_meta
                 ${EXTERNAL_INSTALL_LOCATION}/iceoryx-build)

add_subdirectory(gpuless)

add_subdirectory(executor)

