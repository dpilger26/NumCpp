cmake_minimum_required(VERSION 3.16)

set(NUMCPP_INCLUDES ../../include)

project(NumCpp CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
message("Compiling with C++ standard: ${CMAKE_CXX_STANDARD}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC

file(
    GLOB_RECURSE _source_list 
    LIST_DIRECTORIES false
    "${NUMCPP_INCLUDES}/*.c*"
    "${NUMCPP_INCLUDES}/*.h*"
)

add_library(${PROJECT_NAME} SHARED NumCppPy.cpp ${_source_list})

foreach(_source IN ITEMS ${_source_list})
    get_filename_component(_source_path "${_source}" DIRECTORY)
    string(REPLACE "${CMAKE_SOURCE_DIR}" "" _group_path "${_source_path}")
    string(REPLACE "${NUMCPP_INCLUDES}" "" _group_path "${_group_path}")
    string(REPLACE "/" "\\" _group_path "${_group_path}")
    source_group("${_group_path}" FILES "${_source}")
endforeach()

target_compile_definitions(${PROJECT_NAME} PRIVATE 
  -DINCLUDE_PYTHON_INTERFACE
  # -DINCLUDE_PYBIND_PYTHON_INTERFACE
  -DINCLUDE_BOOST_PYTHON_INTERFACE
  -DBOOST_PYTHON_STATIC_LIB
  -DBOOST_NUMPY_STATIC_LIB
)

set(Boost_USE_STATIC_LIBS ON) 
set(Python3_USE_STATIC_LIBS True)

find_package(Boost 1.68.0 COMPONENTS python37 numpy37 REQUIRED)
find_package(Python COMPONENTS Development)

target_include_directories(${PROJECT_NAME} PRIVATE 
  ${NUMCPP_INCLUDES}
  ${Boost_INCLUDE_DIRS}
  ${Python_INCLUDE_DIRS}  
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:Boost::python37>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:Boost::numpy37>
  $<$<CXX_COMPILER_ID:MSVC>:${Python_LIBRARIES}>
)

target_link_directories(${PROJECT_NAME} PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:${Boost_LIBRARY_DIRS}>
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(OUTPUT_SUFFIX ".pyd")
else ()
  set(OUTPUT_SUFFIX ".so")
endif()

set_target_properties(${PROJECT_NAME}
  PROPERTIES
  PREFIX ""
  SUFFIX ${OUTPUT_SUFFIX}
  ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib/$<0:>
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib/$<0:>
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib/$<0:>
)

target_compile_options(${PROJECT_NAME} PRIVATE
$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-W>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wall>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wextra>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-pedantic>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wdouble-promotion>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wunused>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<CXX_COMPILER_ID:MSVC>:/Zi>
  $<$<CXX_COMPILER_ID:MSVC>:/sdl>
  $<$<CXX_COMPILER_ID:MSVC>:/MP>
  $<$<CXX_COMPILER_ID:MSVC>:/Gy>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/Oi>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/Ot>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/GL>
)

target_link_options(${PROJECT_NAME} PRIVATE
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/LTCG>
)
