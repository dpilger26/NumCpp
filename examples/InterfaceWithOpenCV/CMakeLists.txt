cmake_minimum_required(VERSION 3.14...3.99)

set(TARGET_NAME InterfaceWithOpenCV)

add_executable(${TARGET_NAME} InterfaceWithOpenCV.cpp)

set_target_properties(${TARGET_NAME}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_BINARY_DIR}
)

target_include_directories(${TARGET_NAME} PRIVATE 
  ${NUMCPP_INCLUDES}
)

if (NOT NUMCPP_NO_USE_BOOST)
  target_link_libraries(${TARGET_NAME} PRIVATE 
    Boost::headers
  )
endif()

find_package(OpenCV REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE 
  ${OpenCV_LIBS}
)

target_link_libraries(${TARGET_NAME} PRIVATE 
  ${ALL_INTERFACE_TARGET}
)

target_compile_definitions(${TARGET_NAME} PRIVATE 
  $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING>
)
