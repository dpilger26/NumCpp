set(TARGET_NAME InterfaceWithEigen)

add_executable(${TARGET_NAME} InterfaceWithEigen.cpp)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_BINARY_DIR}
)

target_include_directories(${TARGET_NAME} PRIVATE 
    ${NUMCPP_INCLUDES}
)

find_package (Eigen3 3.3 REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE 
    Eigen3::Eigen
)

target_link_libraries(${TARGET_NAME} PRIVATE 
    ${ALL_INTERFACE_TARGET}
)

target_compile_definitions(${TARGET_NAME} PRIVATE 
    $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING>
)
