if (BUILD_EXAMPLE_GAUSS_NEWTON_NLLS OR BUILD_EXAMPLE_ALL)
    add_subdirectory(GaussNewtonNlls)
endif()

if (BUILD_EXAMPLE_INTERFACE_WITH_EIGEN OR BUILD_EXAMPLE_ALL)
    add_subdirectory(InterfaceWithEigen)
endif()

if (BUILD_EXAMPLE_INTERFACE_WITH_OPENCV OR BUILD_EXAMPLE_ALL)
    add_subdirectory(InterfaceWithOpenCV)
endif()

if (BUILD_EXAMPLE_README OR BUILD_EXAMPLE_ALL)
    add_subdirectory(ReadMe)
endif()
