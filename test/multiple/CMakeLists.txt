cmake_minimum_required(VERSION 3.14...3.99)

set(TARGET_NAME Multiple)

add_executable(${TARGET_NAME} 
  main.cpp
  function.cpp
)

set_target_properties(${TARGET_NAME}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin/$<0:>
)

target_include_directories(${TARGET_NAME} PRIVATE 
  ./
  ${NUMCPP_INCLUDES}
)

if (NOT NUMCPP_NO_USE_BOOST)
  target_link_libraries(${TARGET_NAME} PRIVATE 
    Boost::headers
  )
endif()

target_link_libraries(${TARGET_NAME} PRIVATE 
  ${ALL_INTERFACE_TARGET}
)
