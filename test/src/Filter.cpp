#include "NumCpp/Filter.hpp"

#include "BindingsIncludes.hpp"


//================================================================================

void initFilter(pb11::module &m)
{
    // Filters.hpp
    pb11::enum_<filter::Boundary>(m, "Mode")
        .value("REFLECT", filter::Boundary::REFLECT)
        .value("CONSTANT", filter::Boundary::CONSTANT)
        .value("NEAREST", filter::Boundary::NEAREST)
        .value("MIRROR", filter::Boundary::MIRROR)
        .value("WRAP", filter::Boundary::WRAP);

    m.def("complementaryMedianFilter", &filter::complementaryMedianFilter<double>);
    m.def("complementaryMedianFilter1d", &filter::complementaryMedianFilter1d<double>);
    m.def("convolve", &filter::convolve<double>);
    m.def("convolve1d", &filter::convolve1d<double>);
    m.def("gaussianFilter", &filter::gaussianFilter<double>);
    m.def("gaussianFilter1d", &filter::gaussianFilter1d<double>);
    m.def("laplaceFilter", &filter::laplace<double>);
    m.def("maximumFilter", &filter::maximumFilter<double>);
    m.def("maximumFilter1d", &filter::maximumFilter1d<double>);
    m.def("medianFilter", &filter::medianFilter<double>);
    m.def("medianFilter1d", &filter::medianFilter1d<double>);
    m.def("minimumFilter", &filter::minimumFilter<double>);
    m.def("minumumFilter1d", &filter::minumumFilter1d<double>);
    m.def("percentileFilter", &filter::percentileFilter<double>);
    m.def("percentileFilter1d", &filter::percentileFilter1d<double>);
    m.def("rankFilter", &filter::rankFilter<double>);
    m.def("rankFilter1d", &filter::rankFilter1d<double>);
    m.def("uniformFilter", &filter::uniformFilter<double>);
    m.def("uniformFilter1d", &filter::uniformFilter1d<double>);
}