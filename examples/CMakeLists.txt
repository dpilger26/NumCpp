if (BUILD_EXAMPLE_GAUSS_NEWTON_NLLS)
    message(STATUS "Configuring GausNewtonNlls")
    add_subdirectory(GaussNewtonNlls)
endif()

if (BUILD_EXAMPLE_INTERFACE_WITH_EIGEN)
    message(STATUS "Configuring InterfaceWithEigen")
    add_subdirectory(InterfaceWithEigen)
endif()

if (BUILD_EXAMPLE_INTERFACE_WITH_OPENCV)
    message(STATUS "Configuring InterfaceWithOpenCV")
    add_subdirectory(InterfaceWithOpenCV)
endif()

if (BUILD_EXAMPLE_README)
    message(STATUS "Configuring ReadMe")
    add_subdirectory(ReadMe)
endif()
