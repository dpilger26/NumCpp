
set(TARGET_NAME NumCpp_tests)

add_executable(${TARGET_NAME}
    test_BinaryLogger.cpp
    test_Logger.cpp
)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_BINARY_DIR}
)

target_include_directories(${TARGET_NAME} PRIVATE 
    ${NUMCPP_INCLUDES}
)

find_package(GTest REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE 
    ${ALL_INTERFACE_TARGET}
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(${TARGET_NAME})

add_test(NAME ${TARGET_NAME}
    COMMAND ${TARGET_NAME}
)