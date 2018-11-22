cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(PROJECT_NAME NumCpp)
set(NUMCPP_INCLUDES ../../src)
set(PYTHON_INCLUDES /home/dpilger/anaconda3/include)
set(PYTHON_INCLUDES_SUBDIR /home/dpilger/anaconda3/include/python3.7m)
set(PYTHON_LIBS /home/dpilger/anaconda3/lib)
set(BOOST_SYSTEM "libboost_system.a")
set(BOOST_FILESYSTEM "libboost_filesystem.a")
set(BOOST_PYTHON "libboost_python37.a")
set(BOOST_NUMPY "libboost_numpy37.a")

project(${PROJECT_NAME})

include_directories(${NUMCPP_INCLUDES})
include_directories(${PYTHON_INCLUDES})
include_directories(${PYTHON_INCLUDES_SUBDIR})

add_definitions(-DINCLUDE_PYTHON_INTERFACE)

link_directories(${PYTHON_LIBS})

add_library(${PROJECT_NAME} SHARED NumCppPy.cpp)

target_link_libraries(${PROJECT_NAME} ${BOOST_SYSTEM})
target_link_libraries(${PROJECT_NAME} ${BOOST_FILESYSTEM})
target_link_libraries(${PROJECT_NAME} ${BOOST_PYTHON})
target_link_libraries(${PROJECT_NAME} ${BOOST_NUMPY})
