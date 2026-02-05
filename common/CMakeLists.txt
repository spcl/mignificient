set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

add_library(mignificient_common STATIC
  src/config.cpp
)

target_include_directories(mignificient_common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(Jsoncpp QUIET)
if(Jsoncpp_FOUND)
  target_include_directories(mignificient_common PUBLIC ${JSONCPP_INCLUDE_DIRS})
  target_link_libraries(mignificient_common PUBLIC ${JSONCPP_LIBRARIES})
else()
  target_include_directories(mignificient_common PUBLIC /usr/include/jsoncpp)
  target_link_libraries(mignificient_common PUBLIC jsoncpp)
endif()

if(TARGET spdlog::spdlog)
  target_include_directories(mignificient_common PRIVATE
    $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif()
