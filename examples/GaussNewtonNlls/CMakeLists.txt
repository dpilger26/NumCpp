cmake_minimum_required(VERSION 3.10)

project(GaussNewtonNlls)

set(CMAKE_CXX_STANDARD 17)

set(NUMCPP_INCLUDES ../../include)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(${PROJECT_NAME} GaussNewtonNlls.cpp)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/
)

find_package( Boost )

target_include_directories(${PROJECT_NAME} PUBLIC ${NUMCPP_INCLUDES})
target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS}) 
