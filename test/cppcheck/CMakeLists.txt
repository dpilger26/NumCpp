set(TARGET_NAME CppCheck)

add_executable(${TARGET_NAME}
    CppCheck.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE 
    ${NUMCPP_INCLUDES}
)

target_link_libraries(${TARGET_NAME} PRIVATE 
    ${ALL_INTERFACE_TARGET}
)
