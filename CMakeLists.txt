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

###
# stduuid
###
find_package(stduuid QUIET)
if(NOT stduuid_FOUND)
  message(STATUS "Downloading and building stduuid dependency")
  FetchContent_Declare(stduuid
    GIT_REPOSITORY https://github.com/mariusbancila/stduuid.git
  )
  # disable installing gsl
  FetchContent_GetProperties(stduuid)
  if(NOT stduuid_POPULATED)
    FetchContent_Populate(stduuid)
    add_subdirectory(${stduuid_SOURCE_DIR} ${stduuid_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

###
# drogon
###
find_package(Drogon QUIET)
if(NOT Drogon_FOUND)
	set(JsonCpp_DIR ${jsoncpp_BINARY_DIR} CACHE PATH "JsonCpp directory" FORCE)
  message(STATUS "Downloading and building drogon dependency")
  FetchContent_Declare(Drogon
    GIT_REPOSITORY https://github.com/drogonframework/drogon
  )
  set(BUILD_TESTING OFF CACHE INTERNAL "Turn off tests")
  set(BUILD_SQLITE OFF CACHE INTERNAL "Turn off sqlite")
  set(BUILD_POSTGRESQL OFF CACHE INTERNAL "Turn off postgres")
  set(BUILD_MYSQL OFF CACHE INTERNAL "Turn off mysql")
  set(BUILD_ORM OFF CACHE INTERNAL "Turn off orm")
  set(BUILD_BROTLI OFF CACHE INTERNAL "Turn off brotli")
  set(BUILD_YAML_CONFIG OFF CACHE INTERNAL "Turn off brotli")
  FetchContent_GetProperties(Drogon)
  if(NOT drogon_POPULATED)
    FetchContent_Populate(Drogon)
    add_subdirectory(${drogon_SOURCE_DIR} ${drogon_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
  add_library(Drogon::Drogon ALIAS drogon)
else()
  message(STATUS "Found drogon dependency")
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/external/flatbuffers
                 ${EXTERNAL_INSTALL_LOCATION}/flatbuffers-build
                 EXCLUDE_FROM_ALL)

add_subdirectory(${CMAKE_SOURCE_DIR}/external/iceoryx/iceoryx_meta
                 ${EXTERNAL_INSTALL_LOCATION}/iceoryx-build)

add_subdirectory(gpuless)

add_subdirectory(executor)

add_subdirectory(orchestrator)
