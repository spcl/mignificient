
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

add_library(executor_lib SHARED
  src/executor.cpp
  src/function.cpp
)
target_include_directories(executor_lib SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(executor_lib SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(executor_lib SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>)
add_dependencies(executor_lib spdlog)
target_link_libraries(executor_lib
  spdlog::spdlog
  iceoryx_posh::iceoryx_posh
  mignificient_common
)

# Conditionally link iceoryx2
if(BUILD_ICEORYX2)
  target_link_libraries(executor_lib iceoryx2-cxx::static-lib-cxx)
endif()

set_target_properties(executor_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(executor_cpp
  src/executor_cpp.cpp
)
target_include_directories(executor_cpp SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>)
add_dependencies(executor_cpp spdlog)
target_link_libraries(executor_cpp
  dl
  executor_lib
  spdlog::spdlog
)
set_target_properties(executor_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/)

set(BARE_METAL_EXECUTOR_CPP ${CMAKE_BINARY_DIR}/executor/bin/executor_cpp PARENT_SCOPE)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

if(Python_FOUND AND pybind11_FOUND)

    message(STATUS "Building Python executor.")

    add_compile_definitions(WITH_PYTHON)
    pybind11_add_module(_mignificient python/executor.cpp)
    target_link_libraries(_mignificient PRIVATE executor_lib)

    set_target_properties(_mignificient PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mignificient)

    configure_file(python/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/mignificient/__init__.py COPYONLY)

    configure_file(python/cli.py ${CMAKE_CURRENT_BINARY_DIR}/bin/executor_python.py COPYONLY)

    set(BARE_METAL_EXECUTOR_PYTHON "[\"${Python_EXECUTABLE}\", \"${CMAKE_BINARY_DIR}/executor/bin/executor_python.py\"]" PARENT_SCOPE)
    set(BARE_METAL_EXECUTOR_PYTHONPATH "${CMAKE_BINARY_DIR}/executor/" PARENT_SCOPE)
else()
    message(STATUS "NOT building Python executor.")
    message("Python found? ${Python_FOUND}")
    message("Pybind11 found? ${pybind11_FOUND}")
    set(BARE_METAL_EXECUTOR_PYTHON "null" PARENT_SCOPE)
endif()
