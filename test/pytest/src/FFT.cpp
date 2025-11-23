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

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ifft(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return nc2pybind(nc::fft::ifft(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ifftN(const NdArray<dtype>& inArray, uint32 inN, Axis inAxis)
    {
        return nc2pybind(nc::fft::ifft(inArray, inN, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ifftComplex(const NdArray<std::complex<dtype>>& inArray, Axis inAxis)
    {
        return nc2pybind(nc::fft::ifft(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ifftComplexN(const NdArray<std::complex<dtype>>& inArray, uint32 inN, Axis inAxis)
    {
        return nc2pybind(nc::fft::ifft(inArray, inN, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric rfft(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return nc2pybind(nc::fft::rfft(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric rfftN(const NdArray<dtype>& inArray, uint32 inN, Axis inAxis)
    {
        return nc2pybind(nc::fft::rfft(inArray, inN, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric irfftComplex(const NdArray<std::complex<dtype>>& inArray, Axis inAxis)
    {
        return nc2pybind(nc::fft::irfft(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric irfftComplexN(const NdArray<std::complex<dtype>>& inArray, uint32 inN, Axis inAxis)
    {
        return nc2pybind(nc::fft::irfft(inArray, inN, inAxis));
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

    m.def("ifft", &FFTInterface::ifft<double>);
    m.def("ifft", &FFTInterface::ifftN<double>);
    m.def("ifft", &FFTInterface::ifftComplex<double>);
    m.def("ifft", &FFTInterface::ifftComplexN<double>);

    m.def("rfft", &FFTInterface::rfft<double>);
    m.def("rfft", &FFTInterface::rfftN<double>);

    m.def("irfft", &FFTInterface::irfftComplex<double>);
    m.def("irfft", &FFTInterface::irfftComplexN<double>);
}