cmake_minimum_required(VERSION 3.14...3.99)

set(TARGET_NAME ReadMe)

add_executable(${TARGET_NAME} ReadMe.cpp)

set_target_properties(${TARGET_NAME}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_BINARY_DIR}
)

target_include_directories(${TARGET_NAME} PRIVATE 
  ${NUMCPP_INCLUDES}
)

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.68.0 REQUIRED 
  COMPONENTS 
  system
  filesystem
)

target_link_libraries(${TARGET_NAME} PRIVATE 
  Boost::headers
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:Boost::system>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:Boost::filesystem>
)

target_link_directories(${TARGET_NAME} PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:${Boost_LIBRARY_DIRS}>
)

target_link_libraries(${TARGET_NAME} PRIVATE 
  ${ALL_INTERFACE_TARGET}
)
