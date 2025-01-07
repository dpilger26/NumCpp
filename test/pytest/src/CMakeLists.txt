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
    Logger.cpp
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
find_package(Python 3.11 REQUIRED
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
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/bigobj>
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
