#include "NumCpp/FFT.hpp"

#include "BindingsIncludes.hpp"

#include <algorithm>
#include <numeric>

//================================================================================

namespace FFTInterface
{
    template<typename dtype>
    pbArrayGeneric fft(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return nc2pybind(nc::fft::fft(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fftN(const NdArray<dtype>& inArray, uint32 inN, Axis inAxis)
    {
        return nc2pybind(nc::fft::fft(inArray, inN, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fftComplex(const NdArray<std::complex<dtype>>& inArray, Axis inAxis)
    {
        return nc2pybind(nc::fft::fft(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fftComplexN(const NdArray<std::complex<dtype>>& inArray, uint32 inN, Axis inAxis)
    {
        return nc2pybind(nc::fft::fft(inArray, inN, inAxis));
    }

} // namespace FFTInterface

//================================================================================

void initFFT(pb11::module& m)
{
    // FFT.hpp
    m.def("fft", &FFTInterface::fft<double>);
    m.def("fft", &FFTInterface::fftN<double>);
    m.def("fft", &FFTInterface::fftComplex<double>);
    m.def("fft", &FFTInterface::fftComplexN<double>);
}