cmake_minimum_required(VERSION 3.14...3.99)

set(TARGET_NAME CppCheck)

add_executable(${TARGET_NAME}
  CppCheck.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE 
  ${NUMCPP_INCLUDES}
)

target_link_libraries(${TARGET_NAME} PRIVATE 
  ${ALL_INTERFACE_TARGET}
)
