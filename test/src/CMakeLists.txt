cmake_minimum_required(VERSION 3.14...3.99)

set(TARGET_NAME NumCppPy)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC

file(
  GLOB_RECURSE _source_list 
  LIST_DIRECTORIES false
  "${NUMCPP_INCLUDES}/*.c*"
  "${NUMCPP_INCLUDES}/*.h*"
)

foreach(_source IN ITEMS ${_source_list})
  get_filename_component(_source_path "${_source}" DIRECTORY)
  string(REPLACE "${CMAKE_SOURCE_DIR}" "" _group_path "${_source_path}")
  string(REPLACE "${NUMCPP_INCLUDES}" "" _group_path "${_group_path}")
  string(REPLACE "/" "\\" _group_path "${_group_path}")
  source_group("${_group_path}" FILES "${_source}")
endforeach()

add_library(${TARGET_NAME} SHARED 
  Constants.cpp 
  Coordinates.cpp 
  Core.cpp 
  DataCube.cpp 
  DateTime.cpp
  Filter.cpp 
  Functions.cpp 
  ImageProcessing.cpp 
  Integrate.cpp 
  Linalg.cpp 
  Misc.cpp 
  NdArray.cpp 
  Polynomial.cpp 
  Random.cpp 
  Roots.cpp 
  Rotations.cpp 
  Special.cpp 
  Utils.cpp 
  Vector.cpp
  NumCppPy.cpp
  ${_source_list}
)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
target_include_directories(${TARGET_NAME} PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${NUMCPP_INCLUDES}
)

set(Python_USE_STATIC_LIBS True)
find_package(Python 3.8 REQUIRED
  COMPONENTS
  Development.Module
)

target_link_libraries(${TARGET_NAME} PRIVATE 
  Python::Module
  ${ALL_INTERFACE_TARGET}
)

target_compile_definitions(${TARGET_NAME} PRIVATE 
  -DNUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE
)

target_compile_options(${TARGET_NAME} PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-W>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wall>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wextra>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Werror>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wdouble-promotion>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wunused>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wshadow>
  # $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wconversion>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wpedantic>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-pedantic-errors>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Waggregate-return>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wcast-align>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wcast-qual>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wfloat-equal>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wformat=2>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wmissing-include-dirs>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wpointer-arith>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wredundant-decls>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wsequence-point>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wswitch>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wundef>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wunreachable-code>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wunused-but-set-parameter>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wwrite-strings>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wunused-parameter>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

target_link_options(${TARGET_NAME} PRIVATE 
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-static-libstdc++>
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(OUTPUT_SUFFIX ".pyd")
else ()
  set(OUTPUT_SUFFIX ".so")
endif()

set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/test/pytest/$<0:>)
set_target_properties(${TARGET_NAME}
  PROPERTIES
  PREFIX ""
  SUFFIX ${OUTPUT_SUFFIX}
  ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)
