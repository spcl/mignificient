
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
