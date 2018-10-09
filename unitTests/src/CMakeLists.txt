cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(PROJECT_NAME NumCpp)
set(NUMCPP_INCLUDES ../../src)
set(PYTHON_INCLUDES /home/dpilger/anaconda3/include)
set(PYTHON_INCLUDES_SUBDIR /home/dpilger/anaconda3/include/python3.6m)
set(PYTHON_LIBS /home/dpilger/anaconda3/lib)

project(${PROJECT_NAME})

include_directories(${NUMCPP_INCLUDES})
include_directories(${PYTHON_INCLUDES})
include_directories(${PYTHON_INCLUDES_SUBDIR})

add_definitions(-DINCLUDE_PYTHON_INTERFACE)

# link_directories()

add_executable(${PROJECT_NAME} NumCppPy.cpp)