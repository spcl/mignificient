
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

add_executable(invoker
  cli.cpp
)
target_include_directories(invoker SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(invoker SYSTEM PUBLIC $<TARGET_PROPERTY:Drogon::Drogon,INTERFACE_INCLUDE_DIRECTORIES>)
add_dependencies(invoker spdlog)
add_dependencies(invoker Drogon::Drogon)
target_link_libraries(invoker
  Drogon::Drogon
  spdlog::spdlog
)
set_target_properties(invoker PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/)

