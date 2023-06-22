set(TARGET_NAME GaussNewtonNlls)

add_executable(${TARGET_NAME} GaussNewtonNlls.cpp)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_BINARY_DIR}
)

target_include_directories(${TARGET_NAME} PRIVATE 
    ${NUMCPP_INCLUDES}
)

target_link_libraries(${TARGET_NAME} PRIVATE 
    ${ALL_INTERFACE_TARGET}
)
