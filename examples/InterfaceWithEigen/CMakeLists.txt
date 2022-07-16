cmake_minimum_required(VERSION 3.14...3.99)

set(TARGET_NAME InterfaceWithEigen)

add_executable(${TARGET_NAME} InterfaceWithEigen.cpp)

set_target_properties(${TARGET_NAME}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/$<0:>
)

target_include_directories(${TARGET_NAME} PRIVATE 
  ../../include
)

if (NOT NUMCPP_NO_USE_BOOST)
  target_link_libraries(${TARGET_NAME} PRIVATE 
    Boost::headers
  )
endif()

find_package (Eigen3 3.3 REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE 
  Eigen3::Eigen
)

target_link_libraries(${TARGET_NAME} PRIVATE 
  ${COMPILE_DEFINITIONS_TARGET}
)

target_compile_options(${TARGET_NAME} PRIVATE
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

target_link_options(${TARGET_NAME} PRIVATE
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/LTCG>
)

target_compile_definitions(${TARGET_NAME} PRIVATE 
  $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING>
)
