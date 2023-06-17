#include "BindingsIncludes.hpp"

//================================================================================

void initMisc(pb11::module &m)
{
    using doublePair = std::pair<NdArray<double>, NdArray<double>>;
    pb11::class_<doublePair>(m, "doublePair")
        .def(pb11::init<>())
        .def_readonly("first", &doublePair::first)
        .def_readonly("second", &doublePair::second);

    using uint32Pair = std::pair<NdArray<uint32>, NdArray<uint32>>;
    pb11::class_<uint32Pair>(m, "uint32Pair")
        .def(pb11::init<>())
        .def_readonly("first", &uint32Pair::first)
        .def_readonly("second", &uint32Pair::second);
}
