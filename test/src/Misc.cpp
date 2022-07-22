#include "BindingsIncludes.hpp"

//================================================================================

void initMisc(pb11::module &m)
{
    typedef std::pair<NdArray<double>, NdArray<double>> doublePair;
    pb11::class_<doublePair>(m, "doublePair")
        .def(pb11::init<>())
        .def_readonly("first", &doublePair::first)
        .def_readonly("second", &doublePair::second);

    typedef std::pair<NdArray<uint32>, NdArray<uint32>> uint32Pair;
    pb11::class_<uint32Pair>(m, "uint32Pair")
        .def(pb11::init<>())
        .def_readonly("first", &uint32Pair::first)
        .def_readonly("second", &uint32Pair::second);
}
