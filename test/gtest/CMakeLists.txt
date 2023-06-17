
set(TARGET_NAME NumCpp_tests)

enable_testing()

add_executable(${TARGET_NAME}
    test_BinaryLogger.cpp
    test_Logger.cpp
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

set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/test/gtest/$<0:>)
set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

include(GoogleTest)
gtest_discover_tests(${TARGET_NAME})

