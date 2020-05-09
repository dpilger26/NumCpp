cmake_minimum_required(VERSION 3.17)

project(GaussNewtonNlls)

set(CMAKE_CXX_STANDARD 17)

set(NUMCPP_INCLUDES ../../include)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(${PROJECT_NAME} GaussNewtonNlls.cpp)

set_target_properties(${PROJECT_NAME}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/$<0:>
)

set(Boost_USE_STATIC_LIBS TRUE) 
find_package(Boost 1.68.0)

target_include_directories(${PROJECT_NAME} PRIVATE 
  ${NUMCPP_INCLUDES}
  ${Boost_INCLUDE_DIRS} 
)

target_compile_options(${PROJECT_NAME} PRIVATE
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
