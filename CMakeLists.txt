cmake_minimum_required(VERSION 3.17)
include(FetchContent)
project(mignificient)

# Single source of truth for the iceoryx2 version: external/iceoryx2-version (also read by
# external/iceoryx2-build.sh, which builds the Docker image's iceoryx2).
file(STRINGS ${CMAKE_SOURCE_DIR}/external/iceoryx2-version MIGNIFICIENT_ICEORYX2_VERSION LIMIT_COUNT 1)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

option(WITH_PROFILING "Additional profiling output" OFF)
if (${WITH_PROFILING})
  add_compile_options(-DMIGNIFICIENT_WITH_PROFILING)
endif()

option(WITH_MEMORY_PROFILING "Enable memory profiling on every API call" OFF)
if (${WITH_MEMORY_PROFILING})
  add_compile_options(-DMIGNIFICIENT_WITH_MEMORY_PROFILING)
endif()

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

  set(drogon_patch git apply ${CMAKE_CURRENT_SOURCE_DIR}/external/drogon.patch)

  message(STATUS "Downloading and building drogon dependency")
  FetchContent_Declare(Drogon
    GIT_REPOSITORY https://github.com/drogonframework/drogon
    PATCH_COMMAND ${drogon_patch}
    UPDATE_DISCONNECTED 1
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

###
# iceoryx2 (conditional)
###
if(BUILD_ICEORYX2)
  message(STATUS "Building with iceoryx2 support")

  # Add compile definition for all targets
  add_compile_definitions(MIGNIFICIENT_WITH_ICEORYX2)

  # Always the file-backed build (external/iceoryx2-file-backend.patch): per-client iceoryx2 directories
  # need it, and an unpatched process never finds the services of a patched one.
  set(MIGNIFICIENT_ICEORYX2_INSTALL "" CACHE PATH
    "Prefix of an iceoryx2 install from external/iceoryx2-build.sh (patched); empty: fetch and build it")
  if(MIGNIFICIENT_ICEORYX2_INSTALL)
    # Only there: an unpatched iceoryx2 found elsewhere would break the per-client directories.
    # iceoryx2-cxx's config pulls in its siblings with find_dependency; finding them here first pins them too.
    foreach(pkg iceoryx2-cmake-modules iceoryx2-bb-cxx iceoryx2-c iceoryx2-cxx)
      find_package(${pkg} REQUIRED PATHS ${MIGNIFICIENT_ICEORYX2_INSTALL} NO_DEFAULT_PATH)
    endforeach()
    message(STATUS "Using iceoryx2 from ${MIGNIFICIENT_ICEORYX2_INSTALL}")
  else()
    message(STATUS "Fetching iceoryx2 ${MIGNIFICIENT_ICEORYX2_VERSION} (file-backend patch)")
    set(BUILD_CXX ON CACHE INTERNAL "")
    set(BUILD_EXAMPLES OFF CACHE INTERNAL "")
    set(BUILD_TESTING OFF CACHE INTERNAL "")
    set(IOX2_PATCH ${CMAKE_SOURCE_DIR}/external/iceoryx2-file-backend.patch)
    FetchContent_Declare(iceoryx2
      GIT_REPOSITORY https://github.com/eclipse-iceoryx/iceoryx2.git
      GIT_TAG ${MIGNIFICIENT_ICEORYX2_VERSION}
      UPDATE_DISCONNECTED 1
      # Applied once; re-running on the patched tree is a no-op.
      PATCH_COMMAND sh -c "git apply --reverse --check ${IOX2_PATCH} 2>/dev/null || git apply ${IOX2_PATCH}"
    )
    FetchContent_MakeAvailable(iceoryx2)
  endif()
else()
  message(STATUS "Building with iceoryx1 only (BUILD_ICEORYX2=OFF)")
endif()

set(FLATBUFFERS_BUILD_TESTS OFF CACHE INTERNAL "")
add_subdirectory(${CMAKE_SOURCE_DIR}/external/flatbuffers
                 ${EXTERNAL_INSTALL_LOCATION}/flatbuffers-build
                 EXCLUDE_FROM_ALL)

add_subdirectory(${CMAKE_SOURCE_DIR}/external/iceoryx/iceoryx_meta
                 ${EXTERNAL_INSTALL_LOCATION}/iceoryx-build)

add_subdirectory(common)

add_subdirectory(gpuless)

add_subdirectory(executor)

add_subdirectory(orchestrator)

add_subdirectory(invoker)

add_subdirectory(examples)

configure_file(
  ${CMAKE_SOURCE_DIR}/config/orchestrator.json.in
  ${CMAKE_BINARY_DIR}/config/orchestrator.json
)
configure_file(
  ${CMAKE_SOURCE_DIR}/config/iceoryx2.toml
  ${CMAKE_BINARY_DIR}/config/iceoryx2.toml
  COPYONLY
)
configure_file(
  ${CMAKE_SOURCE_DIR}/config/invoker.json.in
  ${CMAKE_BINARY_DIR}/config/invoker.json
)

option(BUILD_INTEGRATION_TESTS "Register GPU integration tests (tests/integration)" ON)
if(BUILD_INTEGRATION_TESTS)
  enable_testing()
  add_subdirectory(tests/integration)
endif()
