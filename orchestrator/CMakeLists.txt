
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

add_executable(orchestrator
  src/client.cpp
  src/orchestrator.cpp
  src/cli.cpp
  src/device.cpp
  src/executor.cpp
  src/http.cpp
)
target_include_directories(orchestrator SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(orchestrator SYSTEM PUBLIC common/include)
target_include_directories(orchestrator SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(orchestrator SYSTEM PUBLIC $<TARGET_PROPERTY:Drogon::Drogon,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(orchestrator SYSTEM PRIVATE $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(orchestrator SYSTEM PUBLIC $<TARGET_PROPERTY:executor_lib,INTERFACE_INCLUDE_DIRECTORIES>)
add_dependencies(orchestrator spdlog)
add_dependencies(orchestrator Drogon::Drogon)
target_link_libraries(orchestrator
  Drogon::Drogon
  spdlog::spdlog
  iceoryx_posh::iceoryx_posh
  mignificient_common
)
if(BUILD_ICEORYX2)
  target_link_libraries(orchestrator iceoryx2-cxx::static-lib-cxx)
endif()
