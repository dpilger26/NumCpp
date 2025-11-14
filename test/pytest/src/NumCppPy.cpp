#if defined(NUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE)
#include "pybind11/pybind11.h"
namespace python_interface = pybind11;
using interface_module     = python_interface::module;
#elif defined(NUMCPP_INCLUDE_NANOBIND_PYTHON_INTERFACE)
#include "nanobind/nanobind.h"
namespace python_interface = nanobind;
using interface_module     = python_interface::module_;
#endif

//================================================================================

void initConstants(interface_module&);
// void initCoordinates(interface_module&);
// void initCore(interface_module&);
// void initDataCube(interface_module&);
// #ifndef NUMCPP_NO_USE_BOOST
// void initDateTime(interface_module&);
// #endif
// void initFilter(interface_module&);
// void initFunctions(interface_module&);
// void initImageProcessing(interface_module&);
// void initIntegrate(interface_module&);
// void initLinalg(interface_module&);
// void initLogger(interface_module&);
// void initNdArray(interface_module&);
// void initPolynomial(interface_module&);
// void initRandom(interface_module&);
// void initRoots(interface_module&);
// void initRotations(interface_module&);
// void initSpecial(interface_module&);
// void initUtils(interface_module&);
// void initVector(interface_module&);

//================================================================================

#if defined(NUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE)
PYBIND11_MODULE(NumCppPy, m)
#elif defined(NUMCPP_INCLUDE_NANOBIND_PYTHON_INTERFACE)
NB_MODULE(NumCppPy, m)
#endif
{
    m.doc() = "NumCpp unit test bindings";

    initConstants(m);
    //     initCoordinates(m);
    //     initCore(m);
    //     initDataCube(m);
    // #ifndef NUMCPP_NO_USE_BOOST
    //     initDateTime(m);
    // #endif
    //     initFilter(m);
    //     initFunctions(m);
    //     initImageProcessing(m);
    //     initIntegrate(m);
    //     initLinalg(m);
    //     initLogger(m);
    //     initNdArray(m);
    //     initPolynomial(m);
    //     initRandom(m);
    //     initRoots(m);
    //     initRotations(m);
    //     initSpecial(m);
    //     initUtils(m);
    //     initVector(m);
}
