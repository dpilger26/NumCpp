cmake_minimum_required(VERSION 3.12)

set(PROJECT_NAME NumCpp)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(NUMCPP_INCLUDES ../../src)
set(BOOST_MIN_VERSION 1.63.0) # first version with numpy interface

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_library(${PROJECT_NAME} SHARED NumCppPy.cpp)

target_compile_definitions(${PROJECT_NAME} PUBLIC -DINCLUDE_PYTHON_INTERFACE)

target_include_directories(${PROJECT_NAME} PUBLIC ${NUMCPP_INCLUDES})

set(Boost_USE_STATIC_LIBS TRUE) 
find_package( Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS 
	system
	filesystem
	python37
	numpy37
)
target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS}) 
target_link_libraries(${PROJECT_NAME} PUBLIC ${Boost_LIBRARIES})

set(Python3_USE_STATIC_LIBS TRUE)
find_package(Python3 COMPONENTS Development)
target_include_directories(${PROJECT_NAME} PUBLIC ${Python3_INCLUDE_DIRS}) 

