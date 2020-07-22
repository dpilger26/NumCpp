cmake_minimum_required(VERSION 3.14)

if (EXAMPLE_GAUSS_NEWTON_NLLS)
    add_subdirectory(GaussNewtonNlls)
endif()

if (EXAMPLE_INTERFACE_WITH_EIGEN)
    add_subdirectory(InterfaceWithEigen)
endif()

if (EXAMPLE_INTERFACE_WITH_OPENCV)
    add_subdirectory(InterfaceWithOpenCV)
endif()

if (EXAMPLE_README)
    add_subdirectory(ReadMe)
endif()
