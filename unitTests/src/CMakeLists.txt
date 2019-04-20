cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME NumCpp)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC
set(NUMCPP_INCLUDES ../../src)
set(BOOST_MIN_VERSION 1.63.0) # first version with numpy interface

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(${PROJECT_NAME} SHARED NumCppPy.cpp)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    PREFIX "" # don't prepend lib for python .so
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib/
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib/
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin/
)

target_compile_definitions(${PROJECT_NAME} PUBLIC 
	-DINCLUDE_PYTHON_INTERFACE
	-DINCLUDE_BOOST_PYTHON_INTERFACE
)

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

#set(Python3_USE_STATIC_LIBS TRUE)
#find_package(Python3 COMPONENTS Development)
#target_include_directories(${PROJECT_NAME} PUBLIC ${Python3_INCLUDE_DIRS}) 
target_include_directories(${PROJECT_NAME} PUBLIC $ENV{CONDA_PREFIX}/include/python3.7m)

