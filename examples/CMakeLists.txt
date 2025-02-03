
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

enable_language(CUDA)
find_package(CUDA REQUIRED)
find_package(jsoncpp REQUIRED)

set(EXAMPLES_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

#cuda_add_library(vector_add SHARED
#  vector_add.cu
#)
#target_include_directories(vector_add PRIVATE $<TARGET_PROPERTY:executor_lib,INTERFACE_INCLUDE_DIRECTORIES>)
#target_link_libraries(vector_add JsonCpp::JsonCpp)
#set_target_properties(vector_add PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
##target_link_options(vector_add PRIVATE --cudart shared)
#add_dependencies(vector_add executor_lib)
#set_target_properties(vector_add PROPERTIES POSITION_INDEPENDENT_CODE ON)


set(CMAKE_CUDA_ARCHITECTURES 80 86)
add_library(vector_add SHARED vector_add.cu)
target_include_directories(vector_add PRIVATE $<TARGET_PROPERTY:executor_lib,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(vector_add JsonCpp::JsonCpp)
target_compile_features(vector_add PRIVATE cuda_std_17)
set_target_properties(vector_add PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
add_dependencies(vector_add executor_lib)
set_target_properties(vector_add PROPERTIES POSITION_INDEPENDENT_CODE ON)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.json
  ${CMAKE_CURRENT_BINARY_DIR}/vector_add.json
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cubin.txt
  ${CMAKE_CURRENT_BINARY_DIR}/libvector_add.so.txt
  COPYONLY
)

