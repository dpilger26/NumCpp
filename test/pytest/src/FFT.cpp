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
    pbArrayGeneric fftresample(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return nc2pybind(nc::fft::fftresample(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fftresampleN(const NdArray<dtype>& inArray, uint32 inN, Axis inAxis)
    {
        return nc2pybind(nc::fft::fftresample(inArray, inN, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fftresampleComplex(const NdArray<std::complex<dtype>>& inArray, Axis inAxis)
    {
        return nc2pybind(nc::fft::fftresample(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fftresampleComplexN(const NdArray<std::complex<dtype>>& inArray, uint32 inN, Axis inAxis)
    {
        return nc2pybind(nc::fft::fftresample(inArray, inN, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fftfreq(uint32 inN, double inD)
    {
        return nc2pybind(nc::fft::fftfreq(inN, inD));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fftshift(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return nc2pybind(nc::fft::fftshift(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ifftshift(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return nc2pybind(nc::fft::ifftshift(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fft2(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::fft::fft2(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fft2Shape(const NdArray<dtype>& inArray, const Shape& inShape)
    {
        return nc2pybind(nc::fft::fft2(inArray, inShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fft2Complex(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2pybind(nc::fft::fft2(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fft2ComplexShape(const NdArray<std::complex<dtype>>& inArray, const Shape& inShape)
    {
        return nc2pybind(nc::fft::fft2(inArray, inShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ifft2(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::fft::ifft2(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ifft2Shape(const NdArray<dtype>& inArray, const Shape& inShape)
    {
        return nc2pybind(nc::fft::ifft2(inArray, inShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ifft2Complex(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2pybind(nc::fft::ifft2(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ifft2ComplexShape(const NdArray<std::complex<dtype>>& inArray, const Shape& inShape)
    {
        return nc2pybind(nc::fft::ifft2(inArray, inShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fft2resample(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::fft::fft2resample(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fft2resampleShape(const NdArray<dtype>& inArray, const Shape& inShape)
    {
        return nc2pybind(nc::fft::fft2resample(inArray, inShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fft2resampleComplex(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2pybind(nc::fft::fft2resample(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fft2resampleComplexShape(const NdArray<std::complex<dtype>>& inArray, const Shape& inShape)
    {
        return nc2pybind(nc::fft::fft2resample(inArray, inShape));
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

    //================================================================================

    template<typename dtype>
    pbArrayGeneric rfftresample(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return nc2pybind(nc::fft::rfftresample(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric rfftresampleN(const NdArray<dtype>& inArray, uint32 inN, Axis inAxis)
    {
        return nc2pybind(nc::fft::rfftresample(inArray, inN, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric rfftfreq(uint32 inN, double inD)
    {
        return nc2pybind(nc::fft::rfftfreq(inN, inD));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric rfft2(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::fft::rfft2(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric rfft2Shape(const NdArray<dtype>& inArray, const Shape& inShape)
    {
        return nc2pybind(nc::fft::rfft2(inArray, inShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric irfft2Complex(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2pybind(nc::fft::irfft2(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric irfft2ComplexShape(const NdArray<std::complex<dtype>>& inArray, const Shape& inShape)
    {
        return nc2pybind(nc::fft::irfft2(inArray, inShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric rfft2resample(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::fft::rfft2resample(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric rfft2resampleShape(const NdArray<dtype>& inArray, const Shape& inShape)
    {
        return nc2pybind(nc::fft::rfft2resample(inArray, inShape));
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

    m.def("fftresample", &FFTInterface::fftresample<double>);
    m.def("fftresample", &FFTInterface::fftresampleN<double>);
    m.def("fftresample", &FFTInterface::fftresampleComplex<double>);
    m.def("fftresample", &FFTInterface::fftresampleComplexN<double>);

    m.def("fftfreq", &FFTInterface::fftfreq<double>);

    m.def("fftshift", &FFTInterface::fftshift<double>);
    m.def("ifftshift", &FFTInterface::ifftshift<double>);

    m.def("fft2", &FFTInterface::fft2<double>);
    m.def("fft2", &FFTInterface::fft2Shape<double>);
    m.def("fft2", &FFTInterface::fft2Complex<double>);
    m.def("fft2", &FFTInterface::fft2ComplexShape<double>);

    m.def("ifft2", &FFTInterface::ifft2<double>);
    m.def("ifft2", &FFTInterface::ifft2Shape<double>);
    m.def("ifft2", &FFTInterface::ifft2Complex<double>);
    m.def("ifft2", &FFTInterface::ifft2ComplexShape<double>);

    m.def("fft2resample", &FFTInterface::fft2resample<double>);
    m.def("fft2resample", &FFTInterface::fft2resampleShape<double>);
    m.def("fft2resample", &FFTInterface::fft2resampleComplex<double>);
    m.def("fft2resample", &FFTInterface::fft2resampleComplexShape<double>);

    m.def("rfft", &FFTInterface::rfft<double>);
    m.def("rfft", &FFTInterface::rfftN<double>);

    m.def("irfft", &FFTInterface::irfftComplex<double>);
    m.def("irfft", &FFTInterface::irfftComplexN<double>);

    m.def("rfftresample", &FFTInterface::rfftresample<double>);
    m.def("rfftresample", &FFTInterface::rfftresampleN<double>);

    m.def("rfftfreq", &FFTInterface::rfftfreq<double>);

    m.def("rfft2", &FFTInterface::rfft2<double>);
    m.def("rfft2", &FFTInterface::rfft2Shape<double>);

    m.def("irfft2", &FFTInterface::irfft2Complex<double>);
    m.def("irfft2", &FFTInterface::irfft2ComplexShape<double>);

    m.def("rfft2resample", &FFTInterface::rfft2resample<double>);
    m.def("rfft2resample", &FFTInterface::rfft2resampleShape<double>);
}