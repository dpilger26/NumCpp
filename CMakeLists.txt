cmake_minimum_required(VERSION 3.14...3.99)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
message(STATUS "Compiling with C++ standard: ${CMAKE_CXX_STANDARD}")

option(BUILD_ALL "Build All targets" OFF)
option(BUILD_DOCS "Build the doxygen documentation" OFF)
option(BUILD_TESTS "Build the unit tests" OFF)
option(BUILD_MULTIPLE_TEST "Build the multiple translation unit test" OFF)
option(BUILD_EXAMPLE_ALL "Build all of the examples" OFF)
option(BUILD_EXAMPLE_GAUSS_NEWTON_NLLS "Build the Gauss-Newton NLLS example" OFF)
option(BUILD_EXAMPLE_INTERFACE_WITH_EIGEN "Build the Interface with Eigen example" OFF)
option(BUILD_EXAMPLE_INTERFACE_WITH_OPENCV "Build the Interface with OpenCV example" OFF)
option(BUILD_EXAMPLE_README "Build the README example" OFF)

option(NUMCPP_NO_USE_BOOST "Don't use the boost libraries" OFF)
option(NUMCPP_USE_MULTITHREAD "Enable multithreading" OFF)

if(BUILD_ALL)
    set(BUILD_DOCS ON)
    set(BUILD_TESTS ON)
    set(BUILD_MULTIPLE_TEST ON)
    set(BUILD_EXAMPLE_ALL ON)
endif()

if(BUILD_EXAMPLE_ALL)
    set(BUILD_EXAMPLE_GAUSS_NEWTON_NLLS ON)
    set(BUILD_EXAMPLE_INTERFACE_WITH_EIGEN ON)
    set(BUILD_EXAMPLE_INTERFACE_WITH_OPENCV ON)
    set(BUILD_EXAMPLE_README ON)
endif()

set(ALL_INTERFACE_TARGET compile_definitions)
add_library(${ALL_INTERFACE_TARGET} INTERFACE)

if(NUMCPP_NO_USE_BOOST)
    target_compile_definitions(${ALL_INTERFACE_TARGET} INTERFACE -DNUMCPP_NO_USE_BOOST)
else()
    find_package(Boost 1.68.0 REQUIRED)
    target_link_libraries(${ALL_INTERFACE_TARGET} INTERFACE Boost::boost)
endif()

if(NUMCPP_USE_MULTITHREAD)
    target_compile_definitions(${ALL_INTERFACE_TARGET} INTERFACE -DNUMCPP_USE_MULTITHREAD)
    target_link_libraries(${ALL_INTERFACE_TARGET} INTERFACE 
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:tbb>
    )
endif()

target_compile_options(${ALL_INTERFACE_TARGET} INTERFACE
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
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/bigobj>
)

target_link_options(${ALL_INTERFACE_TARGET} INTERFACE
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/LTCG>
)

get_filename_component(NUMCPP_INCLUDES ./include ABSOLUTE)

include(NumCppTools)
numcpp_find_version_file("" NUMCPP_VERSION_FILE)
message(STATUS "Found version file: ${NUMCPP_VERSION_FILE}")
numcpp_read_version("${NUMCPP_VERSION_FILE}" "" VERSION_STRING)

project("NumCpp"
    VERSION "${VERSION_STRING}"
    DESCRIPTION "A Templatized Header Only C++ Implementation of the Python NumPy Library"
    HOMEPAGE_URL "https://github.com/dpilger26/NumCpp"
    LANGUAGES CXX
)

message(STATUS "Building ${PROJECT_NAME} version ${VERSION_STRING}")

set(OUTPUT_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin/$<0:>)

if (BUILD_TESTS OR BUILD_MULTIPLE_TEST) 
    add_subdirectory(test)
endif()

add_subdirectory(examples)

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

include(GNUInstallDirs)
target_include_directories(${PROJECT_NAME} INTERFACE 
    $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(${PROJECT_NAME} INTERFACE $<INSTALL_INTERFACE:cxx_std_17>)

if (BUILD_DOCS)
    set(-Wno-dev)
    message(STATUS "Configuring Doxygen docs")
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        if (CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
            message("Doxygen build started")

            set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/Doxyfile.in)
            set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

            configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

            add_custom_target( docs ALL
                COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating API documentation with Doxygen"
                VERBATIM )
        endif()
    else(DOXYGEN_FOUND)
        message(WARNING "Doxygen needs to be installed to generate the doxygen documentation")
    endif(DOXYGEN_FOUND)
endif(BUILD_DOCS)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMajorVersion)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION
    ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(FILES 
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"   
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION 
    ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
