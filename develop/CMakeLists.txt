cmake_minimum_required(VERSION 3.17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

project(NdArray LANGUAGES CXX)

add_executable(${PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/NdArray/NdArrayCore.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NdArray/NdArrayIterators.hpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/NdArray/NdArrayOperators.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NdArray/Types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NdArray/TypeTraits.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NdArray/Utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    -DDEBUG
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/NdArray
)

find_package(NumCpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
    NumCpp::NumCpp
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:tbb>
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
