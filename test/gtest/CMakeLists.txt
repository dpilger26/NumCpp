
set(TARGET_NAME NumCpp_tests)

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

include(GoogleTest)
gtest_discover_tests(${TARGET_NAME})