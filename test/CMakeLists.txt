if(BUILD_TESTS)
    add_subdirectory(src)
endif()

if(BUILD_MULTIPLE_TEST)
    add_subdirectory(multiple)
endif()
