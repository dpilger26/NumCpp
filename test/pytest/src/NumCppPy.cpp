#include "pybind11/pybind11.h"

namespace pb11 = pybind11;

//================================================================================

void initConstants(pb11::module&);
void initCoordinates(pb11::module&);
void initCore(pb11::module&);
void initDataCube(pb11::module&);
#ifndef NUMCPP_NO_USE_BOOST
void initDateTime(pb11::module&);
#endif
void initFFT(pb11::module&);
void initFilter(pb11::module&);
void initFunctions(pb11::module&);
void initImageProcessing(pb11::module&);
void initIntegrate(pb11::module&);
void initLinalg(pb11::module&);
void initLogger(pb11::module&);
void initNdArray(pb11::module&);
void initPolynomial(pb11::module&);
void initRandom(pb11::module&);
void initRoots(pb11::module&);
void initRotations(pb11::module&);
void initSpecial(pb11::module&);
void initUtils(pb11::module&);
void initVector(pb11::module&);

//================================================================================

PYBIND11_MODULE(NumCppPy, m)
{
    m.doc() = "NumCpp unit test bindings";

    initConstants(m);
    initCoordinates(m);
    initCore(m);
    initDataCube(m);
#ifndef NUMCPP_NO_USE_BOOST
    initDateTime(m);
#endif
    initFFT(m);
    initFilter(m);
    initFunctions(m);
    initImageProcessing(m);
    initIntegrate(m);
    initLinalg(m);
    initLogger(m);
    initNdArray(m);
    initPolynomial(m);
    initRandom(m);
    initRoots(m);
    initRotations(m);
    initSpecial(m);
    initUtils(m);
    initVector(m);
}
