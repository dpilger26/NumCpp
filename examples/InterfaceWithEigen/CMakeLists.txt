cmake_minimum_required(VERSION 3.10)

project(Example)

set(CMAKE_CXX_STANDARD 17)

set(NUMCPP_INCLUDES ../../include)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(${PROJECT_NAME} InterfaceWithEigen.cpp)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/
    COMPILE_FLAGS -DBOOST_NO_CXX11_SCOPED_ENUMS
)

target_include_directories(${PROJECT_NAME} PUBLIC ${NUMCPP_INCLUDES})

set(Boost_USE_STATIC_LIBS TRUE) 
find_package( Boost )

target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS}) 
