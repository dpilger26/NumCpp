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

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE .)
target_include_directories(${TARGET_NAME} PRIVATE 
  ${NUMCPP_INCLUDES}
)

set(Python_USE_STATIC_LIBS True)
find_package(Python 3.8
  COMPONENTS
  Development.Module
)

target_link_libraries(${TARGET_NAME} PRIVATE 
  Python::Module
)

if (NOT NUMCPP_NO_USE_BOOST)
  target_link_libraries(${TARGET_NAME} PRIVATE 
    Boost::headers
  )
endif()

if (NUMCPP_USE_MULTITHREAD)
  target_link_libraries(${TARGET_NAME} PRIVATE 
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:tbb>
  )
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE 
  -DNUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE
)

target_link_libraries(${TARGET_NAME} PRIVATE 
  ${ALL_INTERFACE_TARGET}
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(OUTPUT_SUFFIX ".pyd")
else ()
  set(OUTPUT_SUFFIX ".so")
endif()

set_target_properties(${TARGET_NAME}
  PROPERTIES
  PREFIX ""
  SUFFIX ${OUTPUT_SUFFIX}
  ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../pytest/$<0:>
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../pytest/$<0:>
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../pytest/$<0:>
)
