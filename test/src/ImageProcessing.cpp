#include "NumCpp/ImageProcessing.hpp"

#include "BindingsIncludes.hpp"


//================================================================================

void initImageProcessing(pb11::module &m)
{
    // Image Processing
    using PixelDouble = imageProcessing::Pixel<double>;
    pb11::class_<PixelDouble>(m, "Pixel")
        .def(pb11::init<>())
        .def(pb11::init<uint32, uint32, double>())
        .def(pb11::init<PixelDouble>())
        .def("__eq__", &PixelDouble::operator==)
        .def("__ne__", &PixelDouble::operator!=)
        .def("__lt__", &PixelDouble::operator<)
        .def_readonly("clusterId", &PixelDouble::clusterId)
        .def_readonly("row", &PixelDouble::row)
        .def_readonly("col", &PixelDouble::col)
        .def_readonly("intensity", &PixelDouble::intensity)
        .def("__str__", &PixelDouble::str)
        .def("print", &PixelDouble::print);

    using ClusterDouble = imageProcessing::Cluster<double>;
    pb11::class_<ClusterDouble>(m, "Cluster")
        .def(pb11::init<>())
        .def(pb11::init<ClusterDouble>())
        .def("__eq__", &ClusterDouble::operator==)
        .def("__ne__", &ClusterDouble::operator!=)
        .def("__getitem__", &ClusterDouble::at, pb11::return_value_policy::reference)
        .def("size", &ClusterDouble::size)
        .def("clusterId", &ClusterDouble::clusterId)
        .def("rowMin", &ClusterDouble::rowMin)
        .def("rowMax", &ClusterDouble::rowMax)
        .def("colMin", &ClusterDouble::colMin)
        .def("colMax", &ClusterDouble::colMax)
        .def("height", &ClusterDouble::height)
        .def("width", &ClusterDouble::width)
        .def("intensity", &ClusterDouble::intensity)
        .def("peakPixelIntensity", &ClusterDouble::peakPixelIntensity)
        .def("eod", &ClusterDouble::eod)
        .def("__str__", &ClusterDouble::str)
        .def("print", &ClusterDouble::print);

    using CentroidDouble = imageProcessing::Centroid<double>;
    pb11::class_<CentroidDouble>(m, "Centroid")
        .def(pb11::init<>())
        .def(pb11::init<ClusterDouble>())
        .def(pb11::init<CentroidDouble>())
        .def("row", &CentroidDouble::row)
        .def("col", &CentroidDouble::col)
        .def("intensity", &CentroidDouble::intensity)
        .def("eod", &CentroidDouble::eod)
        .def("__str__", &CentroidDouble::str)
        .def("print", &CentroidDouble::print)
        .def("__eq__", &CentroidDouble::operator==)
        .def("__ne__", &CentroidDouble::operator!=)
        .def("__lt__", &CentroidDouble::operator<);

    m.def("applyThreshold", &imageProcessing::applyThreshold<double>);
    m.def("centroidClusters", &imageProcessing::centroidClusters<double>);
    m.def("clusterPixels", &imageProcessing::clusterPixels<double>);
    m.def("generateThreshold", &imageProcessing::generateThreshold<double>);
    m.def("generateCentroids", &imageProcessing::generateCentroids<double>);
    m.def("windowExceedances", &imageProcessing::windowExceedances);
}