if(BUILD_TESTS)
    message(STATUS "Configuring Unit Tests")
    add_subdirectory(src)
endif()

if(BUILD_MULTIPLE_TEST)
    message(STATUS "Configuring Multiple Tests")
    add_subdirectory(multiple)
endif()
