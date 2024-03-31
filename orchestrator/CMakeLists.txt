
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

add_executable(orchestrator
  orchestrator.cpp
)
target_include_directories(orchestrator SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(orchestrator SYSTEM PUBLIC $<TARGET_PROPERTY:executor_lib,INTERFACE_INCLUDE_DIRECTORIES>)
add_dependencies(orchestrator spdlog)
target_link_libraries(orchestrator
  spdlog::spdlog
  iceoryx_posh::iceoryx_posh
)
