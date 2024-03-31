
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(executor_lib SHARED
  executor.cpp
)
target_include_directories(executor_lib SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(executor_lib SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>)
add_dependencies(executor_lib spdlog)
target_link_libraries(executor_lib
  spdlog::spdlog
  iceoryx_posh::iceoryx_posh
)

add_executable(executor_cpp
  executor_cpp.cpp
)
target_include_directories(executor_cpp SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>)
add_dependencies(executor_cpp spdlog)
target_link_libraries(executor_cpp
  executor_lib
  spdlog::spdlog
)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

message(${Python_FOUND})
message(${pybind11_FOUND})

if(Python_FOUND AND pybind11_FOUND)

    message(STATUS "Building Python executor.")

    add_compile_definitions(WITH_PYTHON)
    pybind11_add_module(_mignificient python/executor.cpp)
    target_link_libraries(_mignificient PRIVATE executor_lib)

    set_target_properties(_mignificient PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mignificient)

    configure_file(python/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/mignificient/__init__.py COPYONLY)

    configure_file(python/cli.py ${CMAKE_CURRENT_BINARY_DIR}/bin/executor_python.py COPYONLY)
endif()
