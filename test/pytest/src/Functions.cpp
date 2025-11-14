#include "NumCpp/Functions.hpp"

#include "BindingsIncludes.hpp"

#include <algorithm>
#include <numeric>

//================================================================================

namespace FunctionsInterface
{
    template<typename dtype>
    auto absScalar(dtype inValue) -> decltype(abs(inValue)) // trailing return type to help gcc
    {
        return abs(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric absArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(abs(inArray));
    }

    //================================================================================

    template<typename Type1, typename Type2>
    pbArrayGeneric add(const Type1& in1, const Type2& in2)
    {
        return nc2pybind(nc::add(in1, in2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric allArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(all(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric anyArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(any(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric argmaxArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(argmax(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric argminArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(argmin(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric argsortArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(argsort(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric argwhere(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::argwhere(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric amaxArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(amax(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric aminArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(amin(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    dtype angleScalar(const std::complex<dtype>& inValue)
    {
        return angle(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric angleArray(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2pybind(angle(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric arangeArray(dtype inStart, dtype inStop, dtype inStep)
    {
        return nc2pybind(arange(inStart, inStop, inStep));
    }

    //================================================================================

    template<typename dtype>
    auto arccosScalar(dtype inValue) -> decltype(arccos(inValue)) // trailing return type to help gcc
    {
        return arccos(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric arccosArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(arccos(inArray));
    }

    //================================================================================

    template<typename dtype>
    auto arccoshScalar(dtype inValue) -> decltype(arccosh(inValue)) // trailing return type to help gcc
    {
        return arccosh(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric arccoshArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(arccosh(inArray));
    }

    //================================================================================

    template<typename dtype>
    auto arcsinScalar(dtype inValue) -> decltype(arcsin(inValue)) // trailing return type to help gcc
    {
        return arcsin(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric arcsinArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(arcsin(inArray));
    }

    //================================================================================

    template<typename dtype>
    auto arcsinhScalar(dtype inValue) -> decltype(arcsinh(inValue)) // trailing return type to help gcc
    {
        return arcsinh(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric arcsinhArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(arcsinh(inArray));
    }

    //================================================================================

    template<typename dtype>
    auto arctanScalar(dtype inValue) -> decltype(arctan(inValue)) // trailing return type to help gcc
    {
        return arctan(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric arctanArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(arctan(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype arctan2Scalar(dtype inY, dtype inX)
    {
        return arctan2(inY, inX);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric arctan2Array(const NdArray<dtype>& inY, const NdArray<dtype>& inX)
    {
        return nc2pybind(arctan2(inY, inX));
    }

    //================================================================================

    template<typename dtype>
    auto arctanhScalar(dtype inValue) -> decltype(arctanh(inValue)) // trailing return type to help gcc
    {
        return arctanh(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric arctanhArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(arctanh(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype aroundScalar(dtype inValue, uint8 inNumDecimals)
    {
        return around(inValue, inNumDecimals);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric aroundArray(NdArray<dtype>& inArray, uint8 inNumDecimals)
    {
        return nc2pybind(around(inArray, inNumDecimals));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayInitializerList(dtype inValue1, dtype inValue2)
    {
        auto a = asarray({ inValue1, inValue2 });
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayInitializerList2D(dtype inValue1, dtype inValue2)
    {
        auto a = asarray({ { inValue1, inValue2 }, { inValue1, inValue2 } });
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayArray1D(dtype inValue1, dtype inValue2)
    {
        std::array<dtype, 2> arr = { inValue1, inValue2 };
        auto                 a   = asarray(arr, PointerPolicy::SHELL);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayArray1DCopy(dtype inValue1, dtype inValue2)
    {
        std::array<dtype, 2> arr = { inValue1, inValue2 };
        auto                 a   = asarray(arr, PointerPolicy::COPY);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayArray2D(dtype inValue1, dtype inValue2)
    {
        std::array<std::array<dtype, 2>, 2> arr{};
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr, PointerPolicy::SHELL);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayArray2DCopy(dtype inValue1, dtype inValue2)
    {
        std::array<std::array<dtype, 2>, 2> arr{};
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr, PointerPolicy::COPY);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayVector1D(dtype inValue1, dtype inValue2)
    {
        std::vector<dtype> arr = { inValue1, inValue2 };
        auto               a   = asarray(arr, PointerPolicy::SHELL);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayVector1DCopy(dtype inValue1, dtype inValue2)
    {
        std::vector<dtype> arr = { inValue1, inValue2 };
        auto               a   = asarray(arr, PointerPolicy::COPY);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayVector2D(dtype inValue1, dtype inValue2)
    {
        std::vector<std::vector<dtype>> arr(2, std::vector<dtype>(2));
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayVectorArray2D(dtype inValue1, dtype inValue2)
    {
        std::vector<std::array<dtype, 2>> arr(2);
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr, PointerPolicy::SHELL);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayVectorArray2DCopy(dtype inValue1, dtype inValue2)
    {
        std::vector<std::array<dtype, 2>> arr(2);
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr, PointerPolicy::COPY);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayDeque1D(dtype inValue1, dtype inValue2)
    {
        std::deque<dtype> arr = { inValue1, inValue2 };
        auto              a   = asarray(arr);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayDeque2D(dtype inValue1, dtype inValue2)
    {
        std::deque<std::deque<dtype>> arr(2, std::deque<dtype>(2));
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayList(dtype inValue1, dtype inValue2)
    {
        std::list<dtype> arr = { inValue1, inValue2 };
        auto             a   = asarray(arr);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayIterators(dtype inValue1, dtype inValue2)
    {
        std::vector<dtype> arr = { inValue1, inValue2 };
        auto               a   = asarray(arr.begin(), arr.end());
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointerIterators(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2); // NOLINT(modernize-avoid-c-arrays)
        ptr[0]   = inValue1;
        ptr[1]   = inValue2;
        auto a   = asarray(ptr.get(), ptr.get() + 2);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointer(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2); // NOLINT(modernize-avoid-c-arrays)
        ptr[0]   = inValue1;
        ptr[1]   = inValue2;
        auto a   = asarray(ptr.get(), uint32{ 2 });
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointer2D(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(4); // NOLINT(modernize-avoid-c-arrays)
        ptr[0]   = inValue1;
        ptr[1]   = inValue2;
        ptr[2]   = inValue1;
        ptr[3]   = inValue2;
        auto a   = asarray(ptr.get(), uint32{ 2 }, uint32{ 2 });
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointerShell(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2); // NOLINT(modernize-avoid-c-arrays)
        ptr[0]   = inValue1;
        ptr[1]   = inValue2;
        auto a   = asarray(ptr.get(), uint32{ 2 }, PointerPolicy::SHELL);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointerShell2D(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(4); // NOLINT(modernize-avoid-c-arrays)
        ptr[0]   = inValue1;
        ptr[1]   = inValue2;
        ptr[2]   = inValue1;
        ptr[3]   = inValue2;
        auto a   = asarray(ptr.get(), uint32{ 2 }, uint32{ 2 }, PointerPolicy::SHELL);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointerShellTakeOwnership(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2); // NOLINT(modernize-avoid-c-arrays)
        ptr[0]   = inValue1;
        ptr[1]   = inValue2;
        auto a   = asarray(ptr.get(), 2, PointerPolicy::COPY);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointerShell2DTakeOwnership(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(4); // NOLINT(modernize-avoid-c-arrays)
        ptr[0]   = inValue1;
        ptr[1]   = inValue2;
        ptr[2]   = inValue1;
        ptr[3]   = inValue2;
        auto a   = asarray(ptr.get(), 2, 2, PointerPolicy::COPY);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric average(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(nc::average(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        averageWeighted(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(average(inArray, inWeights, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric averageWeightedComplex(const NdArray<std::complex<dtype>>& inArray,
                                          const NdArray<dtype>&               inWeights,
                                          Axis                                inAxis = Axis::NONE)
    {
        return nc2pybind(average(inArray, inWeights, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric bincount(const NdArray<dtype>& inArray, uint16 inMinLength = 0)
    {
        return nc2pybind(nc::bincount(inArray, inMinLength));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        bincountWeighted(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0)
    {
        return nc2pybind(bincount(inArray, inWeights, inMinLength));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric bit_count(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::bit_count(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric bitwise_and(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(nc::bitwise_and(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric bitwise_not(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::bitwise_not(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric bitwise_or(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(nc::bitwise_or(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric bitwise_xor(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(nc::bitwise_xor(inArray1, inArray2));
    }

    //================================================================================

    pbArrayGeneric bartlett(nc::int32 m)
    {
        return nc2pybind(nc::bartlett(m));
    }

    //================================================================================

    pbArrayGeneric blackman(nc::int32 m)
    {
        return nc2pybind(nc::blackman(m));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric andOperatorArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 && inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric andOperatorScalar(const NdArray<dtype>& inArray, dtype inScalar)
    {
        return nc2pybind(inArray && inScalar);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric orOperatorArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 || inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric orOperatorScalar(const NdArray<dtype>& inArray, dtype inScalar)
    {
        return nc2pybind(inArray || inScalar);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric byteswap(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::byteswap(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype cbrtScalar(dtype inValue)
    {
        return cbrt(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric cbrtArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(cbrt(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype ceilScalar(dtype inValue)
    {
        return ceil(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ceilArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(ceil(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric centerOfMass(const NdArray<dtype>& inArray, const Axis inAxis = Axis::NONE)
    {
        return nc2pybind(nc::centerOfMass(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    dtype clipScalar(dtype inValue, dtype inMinValue, dtype inMaxValue)
    {
        return clip(inValue, inMinValue, inMaxValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric clipArray(const NdArray<dtype>& inArray, dtype inMinValue, dtype inMaxValue)
    {
        return nc2pybind(clip(inArray, inMinValue, inMaxValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric column_stack(const NdArray<dtype>& inArray1,
                                const NdArray<dtype>& inArray2,
                                const NdArray<dtype>& inArray3,
                                const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::column_stack({ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric column_stack_vec(const NdArray<dtype>& inArray1,
                                    const NdArray<dtype>& inArray2,
                                    const NdArray<dtype>& inArray3,
                                    const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::column_stack(std::vector{ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    std::complex<dtype> complexScalarSingle(dtype inReal)
    {
        return nc::complex(inReal);
    }

    //================================================================================

    template<typename dtype>
    std::complex<dtype> complexScalar(dtype inReal, dtype inImag)
    {
        return nc::complex(inReal, inImag);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric complexArraySingle(const NdArray<dtype>& inReal)
    {
        return nc2pybind(nc::complex(inReal));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric complexArray(const NdArray<dtype>& inReal, const NdArray<dtype>& inImag)
    {
        return nc2pybind(nc::complex(inReal, inImag));
    }

    //================================================================================

    template<typename dtype>
    std::complex<dtype> conjScalar(const std::complex<dtype>& inValue)
    {
        return nc::conj(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric conjArray(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2pybind(nc::conj(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric concatenate(const NdArray<dtype>& inArray1,
                               const NdArray<dtype>& inArray2,
                               const NdArray<dtype>& inArray3,
                               const NdArray<dtype>& inArray4,
                               Axis                  inAxis)
    {
        return nc2pybind(nc::concatenate({ inArray1, inArray2, inArray3, inArray4 }, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric concatenate_vec(const NdArray<dtype>& inArray1,
                                   const NdArray<dtype>& inArray2,
                                   const NdArray<dtype>& inArray3,
                                   const NdArray<dtype>& inArray4,
                                   Axis                  inAxis)
    {
        return nc2pybind(nc::concatenate(std::vector{ inArray1, inArray2, inArray3, inArray4 }, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric copy(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::copy(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric copySign(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(nc::copySign(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric copyto(NdArray<dtype>& inArrayDest, const NdArray<dtype>& inArraySrc)
    {
        return nc2pybind(nc::copyto(inArrayDest, inArraySrc));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric corrcoef(pbArray<dtype> x)
    {
        return nc2pybind(nc::corrcoef(pybind2nc(x)));
    }

    //================================================================================

    template<typename dtype>
    auto cosScalar(dtype inValue) -> decltype(cos(inValue)) // trailing return type to help gcc
    {
        return cos(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric cosArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(cos(inArray));
    }

    //================================================================================

    template<typename dtype>
    auto coshScalar(dtype inValue) -> decltype(cosh(inValue)) // trailing return type to help gcc
    {
        return cosh(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric coshArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(cosh(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric count_nonzero(const NdArray<dtype>& inArray, Axis inAxis = Axis::ROW)
    {
        return nc2pybind(nc::count_nonzero(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric cov(pbArray<dtype> x, Bias bias)
    {
        return nc2pybind(nc::cov(pybind2nc(x), bias));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric cov_inv(pbArray<dtype> x, Bias bias)
    {
        return nc2pybind(nc::cov_inv(pybind2nc(x), bias));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric cubeArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(cube(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric cumprodArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(cumprod(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric cumsumArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(cumsum(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    dtype deg2radScalar(dtype inValue)
    {
        return deg2rad(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric deg2radArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(deg2rad(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype degreesScalar(dtype inValue)
    {
        return degrees(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric degreesArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(degrees(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric deleteIndicesScalar(const NdArray<dtype>& inArray, uint32 inIndex, Axis inAxis)
    {
        return nc2pybind(deleteIndices(inArray, inIndex, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric deleteIndicesIndices(const NdArray<dtype>& inArray, const NdArray<int32>& inIndices, Axis inAxis)
    {
        return nc2pybind(deleteIndices(inArray, inIndices, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric deleteIndicesSlice(const NdArray<dtype>& inArray, const Slice& inIndices, Axis inAxis)
    {
        return nc2pybind(deleteIndices(inArray, inIndices, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric diag(const NdArray<dtype>& inArray, int32 k)
    {
        return nc2pybind(nc::diag(inArray, k));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric diagflat(const NdArray<dtype>& inArray, int32 k)
    {
        return nc2pybind(nc::diagflat(inArray, k));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric diagonal(const NdArray<dtype>& inArray, uint32 inOffset = 0, Axis inAxis = Axis::ROW)
    {
        return nc2pybind(nc::diagonal(inArray, inOffset, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric diff(const NdArray<dtype>& inArray, Axis inAxis = Axis::ROW)
    {
        return nc2pybind(nc::diff(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric digitize(const NdArray<dtype>& x, const NdArray<dtype>& bins)
    {
        return nc2pybind(nc::digitize(x, bins));
    }

    //================================================================================

    template<typename Type1, typename Type2>
    pbArrayGeneric divide(const Type1& in1, const Type2& in2)
    {
        return nc2pybind(nc::divide(in1, in2));
    }

    //================================================================================

    template<typename dtype1, typename dtype2>
    pbArrayGeneric dot(const NdArray<dtype1>& inArray1, const NdArray<dtype2>& inArray2)
    {
        return nc2pybind(nc::dot(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric emptyRowCol(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2pybind(nc::empty<dtype>(inNumRows, inNumCols));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric emptyShape(const Shape& inShape)
    {
        return nc2pybind(empty<dtype>(inShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(nc::equal(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric extract(pbArray<bool> condition, pbArray<dtype> arr)
    {
        return nc2pybind(nc::extract(pybind2nc(condition), pybind2nc(arr)));
    }

    //================================================================================

    template<typename dtype>
    auto expScalar(dtype inValue) -> decltype(exp(inValue)) // trailing return type to help gcc
    {
        return exp(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric expArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(exp(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype exp2Scalar(dtype inValue)
    {
        return exp2(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric exp2Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(exp2(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype expm1Scalar(dtype inValue)
    {
        return expm1(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric expm1Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(expm1(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric eye1D(uint32 inN, int32 inK)
    {
        return nc2pybind(eye<dtype>(inN, inK));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric eye2D(uint32 inN, uint32 inM, int32 inK)
    {
        return nc2pybind(eye<dtype>(inN, inM, inK));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric eyeShape(const Shape& inShape, int32 inK)
    {
        return nc2pybind(eye<dtype>(inShape, inK));
    }

    //================================================================================

    pbArrayGeneric find(const NdArray<bool>& inArray)
    {
        return nc2pybind(nc::find(inArray));
    }

    //================================================================================

    pbArrayGeneric findN(const NdArray<bool>& inArray, uint32 n)
    {
        return nc2pybind(nc::find(inArray, n));
    }

    //================================================================================

    template<typename dtype>
    dtype fixScalar(dtype inValue)
    {
        return fix(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fixArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(fix(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype floorScalar(dtype inValue)
    {
        return floor(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric floorArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(floor(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype floor_divideScalar(dtype inValue1, dtype inValue2)
    {
        return floor_divide(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric floor_divideArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(floor_divide(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    dtype fmaxScalar(dtype inValue1, dtype inValue2)
    {
        return fmax(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fmaxArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(fmax(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fmaxArrayScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(fmax(inArray, inValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fmaxScalarArray(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(fmax(inValue, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype fminScalar(dtype inValue1, dtype inValue2)
    {
        return fmin(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fminArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(fmin(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fminArrayScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(fmin(inArray, inValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fminScalarArray(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(fmin(inValue, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype fmodScalar(dtype inValue1, dtype inValue2)
    {
        return fmod(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fmodArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(fmod(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric frombuffer(const NdArray<dtype>& inArray)
    {
        auto buffer = reinterpret_cast<const char*>(inArray.data());
        return nc2pybind(nc::frombuffer<dtype>(buffer, static_cast<uint32>(inArray.nbytes())));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fromfileBinary(const std::string& inFilename)
    {
        return nc2pybind(nc::fromfile<dtype>(inFilename));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fromfileTxt(const std::string& inFilename, const char inSep)
    {
        return nc2pybind(nc::fromfile<dtype>(inFilename, inSep));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fromfunctionSize(typename NdArray<dtype>::size_type size)
    {
        const auto func = [](typename NdArray<dtype>::size_type idx) { return static_cast<double>(idx) / 10.; };
        return nc2pybind(nc::fromfunction<dtype>(func, size));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fromfunctionShape(Shape shape)
    {
        const auto func = [](typename NdArray<dtype>::size_type row, typename NdArray<dtype>::size_type col)
        { return static_cast<double>(row * col) / 10.; };
        return nc2pybind(nc::fromfunction<dtype>(func, shape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fromiter(const NdArray<dtype>& inArray)
    {
        std::vector<dtype> vec(inArray.begin(), inArray.end());
        return nc2pybind(nc::fromiter<dtype>(vec.begin(), vec.end()));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fromstring(const std::string& str, const char sep)
    {
        return nc2pybind(nc::fromstring<dtype>(str, sep));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fullSquare(uint32 inSquareSize, dtype inValue)
    {
        return nc2pybind(full(inSquareSize, inValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fullRowCol(uint32 inNumRows, uint32 inNumCols, dtype inValue)
    {
        return nc2pybind(full(inNumRows, inNumCols, inValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fullShape(const Shape& inShape, dtype inValue)
    {
        return nc2pybind(full(inShape, inValue));
    }

    //================================================================================

#if defined(__cpp_lib_gcd_lcm) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    dtype gcdScalar(dtype inValue1, dtype inValue2)
    {
        return gcd(inValue1, inValue2);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype gcdArray(const NdArray<dtype>& inArray)
    {
        return gcd(inArray);
    }
#endif

    //================================================================================

    template<typename dtype>
    pbArrayGeneric geomspace(dtype start, dtype stop, uint32 num, EndPoint endPoint)
    {
        return nc2pybind(nc::geomspace(start, stop, num, endPoint));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric gradient(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(nc::gradient(inArray, inAxis));
    }

    //================================================================================

    pbArrayGeneric hamming(nc::int32 m)
    {
        return nc2pybind(nc::hamming(m));
    }

    //================================================================================

    pbArrayGeneric hanning(nc::int32 m)
    {
        return nc2pybind(nc::hanning(m));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric histogramWithEdges(const NdArray<dtype>& inArray, const NdArray<dtype>& inBinEdges)
    {
        auto histo = nc::histogram(inArray, inBinEdges);
        return nc2pybind(histo);
    }

    //================================================================================

    template<typename dtype>
    python_interface::tuple histogram(const NdArray<dtype>& inArray, uint32 inNumBins = 10)
    {
        std::pair<NdArray<uint32>, NdArray<double>> output = nc::histogram(inArray, inNumBins);
        return python_interface::make_tuple(output.first, output.second);
    }

    //================================================================================

    template<typename dtype>
    std::vector<pbArrayGeneric> hsplit(const NdArray<dtype>& inArray, const NdArray<int32>& indices)
    {
        const auto                  splits = nc::hsplit(inArray, indices);
        std::vector<pbArrayGeneric> result{};
        for (const auto& split : splits)
        {
            result.push_back(nc2pybind(split));
        }
        return result;
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric hstack(const NdArray<dtype>& inArray1,
                          const NdArray<dtype>& inArray2,
                          const NdArray<dtype>& inArray3,
                          const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::hstack({ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric hstack_vec(const NdArray<dtype>& inArray1,
                              const NdArray<dtype>& inArray2,
                              const NdArray<dtype>& inArray3,
                              const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::hstack(std::vector{ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    dtype hypotScalar(dtype inValue1, dtype inValue2)
    {
        return hypot(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    dtype hypotScalarTriple(dtype inValue1, dtype inValue2, dtype inValue3)
    {
        return hypot(inValue1, inValue2, inValue3);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric hypotArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(hypot(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        hypotArrayTriple(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, const NdArray<dtype>& inArray3)
    {
        return nc2pybind(hypot(inArray1, inArray2, inArray3));
    }

    //================================================================================

    template<typename dtype>
    dtype imagScalar(const std::complex<dtype>& inValue)
    {
        return nc::imag(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric imagArray(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2pybind(imag(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype inner(pbArray<dtype> a, pbArray<dtype> b)
    {
        return nc::inner(pybind2nc(a), pybind2nc(b));
    }

    //============================================================================

    template<typename dtype>
    pbArrayGeneric insertIndexScalar(const NdArray<dtype>& arr, int32 index, const dtype& value)
    {
        return nc2pybind(insert(arr, index, value));
    }

    //============================================================================

    template<typename dtype>
    pbArrayGeneric insertIndexArray(const NdArray<dtype>& arr, int32 index, const NdArray<dtype>& values)
    {
        return nc2pybind(insert(arr, index, values));
    }

    //============================================================================

    template<typename dtype>
    pbArrayGeneric insertIndexScalarAxis(const NdArray<dtype>& arr, int32 index, const dtype& value, Axis axis)
    {
        return nc2pybind(insert(arr, index, value, axis));
    }

    //============================================================================

    template<typename dtype>
    pbArrayGeneric insertIndexArrayAxis(const NdArray<dtype>& arr, int32 index, const NdArray<dtype>& values, Axis axis)
    {
        return nc2pybind(insert(arr, index, values, axis));
    }

    //============================================================================

    template<typename dtype>
    pbArrayGeneric
        insertIndicesScalar(const NdArray<dtype>& arr, const NdArray<int32>& indices, const dtype& value, Axis axis)
    {
        return nc2pybind(insert(arr, indices, value, axis));
    }

    //============================================================================

    template<typename dtype>
    pbArrayGeneric insertSliceScalar(const NdArray<dtype>& arr, const Slice& slice, const dtype& value, Axis axis)
    {
        return nc2pybind(insert(arr, slice, value, axis));
    }

    //============================================================================

    template<typename dtype>
    pbArrayGeneric insertIndicesArray(const NdArray<dtype>& arr,
                                      const NdArray<int32>& indices,
                                      const NdArray<dtype>& values,
                                      Axis                  axis)
    {
        return nc2pybind(insert(arr, indices, values, axis));
    }

    //============================================================================

    template<typename dtype>
    pbArrayGeneric
        insertSliceArray(const NdArray<dtype>& arr, const Slice& slice, const NdArray<dtype>& values, Axis axis)
    {
        return nc2pybind(insert(arr, slice, values, axis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric interp(const NdArray<dtype>& inX, const NdArray<dtype>& inXp, const NdArray<dtype>& inFp)
    {
        return nc2pybind(nc::interp(inX, inXp, inFp));
    }

    //================================================================================

    template<typename dtype>
    bool isinfScalar(dtype inValue)
    {
        return nc::isinf(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric isinfArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::isinf(inArray));
    }

    //================================================================================

    template<typename dtype>
    bool isposinfScalar(dtype inValue)
    {
        return nc::isposinf(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric isposinfArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::isposinf(inArray));
    }

    //================================================================================

    template<typename dtype>
    bool isneginfScalar(dtype inValue)
    {
        return nc::isneginf(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric isneginfArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::isneginf(inArray));
    }

    //================================================================================

    template<typename dtype>
    bool isnanScalar(dtype inValue)
    {
        return nc::isnan(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric isnanArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::isnan(inArray));
    }

    //================================================================================

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    pbArrayGeneric kaiser(nc::int32 m, double beta)
    {
        return nc2pybind(nc::kaiser(m, beta));
    }
#endif

    //================================================================================

    template<typename dtype>
    dtype ldexpScalar(dtype inValue1, uint8 inValue2)
    {
        return ldexp(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ldexpArray(const NdArray<dtype>& inArray1, const NdArray<uint8>& inArray2)
    {
        return nc2pybind(ldexp(inArray1, inArray2));
    }

    //================================================================================

    pbArray<double> nansSquare(uint32 inSquareSize)
    {
        return nc2pybind(nans(inSquareSize));
    }

    //================================================================================

    pbArray<double> nansRowCol(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2pybind(nans(inNumRows, inNumCols));
    }

    //================================================================================

    pbArray<double> nansShape(const Shape& inShape)
    {
        return nc2pybind(nans(inShape));
    }

    //================================================================================

    pbArray<double> nansList(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2pybind(nans({ inNumRows, inNumCols }));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric negative(const NdArray<dtype> inArray)
    {
        return nc2pybind(nc::negative(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric noneArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(none(inArray, inAxis));
    }

    //================================================================================

#if defined(__cpp_lib_gcd_lcm) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    dtype lcmScalar(dtype inValue1, dtype inValue2)
    {
        return lcm(inValue1, inValue2);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype lcmArray(const NdArray<dtype>& inArray)
    {
        return lcm(inArray);
    }
#endif

    //================================================================================

    template<typename dtype>
    auto logScalar(dtype inValue) -> decltype(log(inValue)) // trailing return type to help gcc
    {
        return log(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric logArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(log(inArray));
    }

    //================================================================================

    template<typename dtype>
    auto logbScalar(dtype inValue, dtype base) -> decltype(logb(inValue)) // trailing return type to help gcc
    {
        return logb(inValue, base);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric logbArray(const NdArray<dtype>& inArray, dtype base)
    {
        return nc2pybind(logb(inArray, base));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric logspace(dtype start, dtype stop, uint32 num, EndPoint endPoint, double base)
    {
        return nc2pybind(nc::logspace(start, stop, num, endPoint, base));
    }

    //================================================================================

    template<typename dtype>
    auto log10Scalar(dtype inValue) -> decltype(log10(inValue)) // trailing return type to help gcc
    {
        return log10(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric log10Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(log10(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype log1pScalar(dtype inValue)
    {
        return log1p(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric log1pArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(log1p(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype log2Scalar(dtype inValue)
    {
        return log2(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric log2Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(log2(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype logaddexpScalar(dtype x1, dtype x2)
    {
        return logaddexp(x1, x2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric logaddexpArray(const NdArray<dtype>& x1, const NdArray<dtype>& x2)
    {
        return nc2pybind(logaddexp(x1, x2));
    }

    //================================================================================

    template<typename dtype>
    dtype logaddexp2Scalar(dtype x1, dtype x2)
    {
        return logaddexp2(x1, x2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric logaddexp2Array(const NdArray<dtype>& x1, const NdArray<dtype>& x2)
    {
        return nc2pybind(logaddexp2(x1, x2));
    }

    //================================================================================

    template<typename dtype1, typename dtype2>
    pbArrayGeneric matmul(const NdArray<dtype1>& inArray1, const NdArray<dtype2>& inArray2)
    {
        return nc2pybind(nc::matmul(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric max(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(nc::max(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric maximumArrays(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(nc::maximum(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric maximumArrayScalar(const NdArray<dtype>& inArray, const dtype& inValue)
    {
        return nc2pybind(nc::maximum(inArray, inValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric maximumScalarArray(const dtype& inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::maximum(inValue, inArray));
    }

    //================================================================================

    template<typename dtype>
    std::pair<NdArray<dtype>, NdArray<dtype>> meshgrid(const Slice& inISlice, const Slice& inJSlice)
    {
        return nc::meshgrid<dtype>(inISlice, inJSlice);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric min(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(nc::min(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric minimumArrays(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(nc::minimum(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric minimumArrayScalar(const NdArray<dtype>& inArray, const dtype& inValue)
    {
        return nc2pybind(nc::minimum(inArray, inValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric minimumScalarArray(const dtype& inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::minimum(inValue, inArray));
    }

    //================================================================================

    template<typename Type1, typename Type2>
    pbArrayGeneric multiply(const Type1& in1, const Type2& in2)
    {
        return nc2pybind(nc::multiply(in1, in2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric nan_to_num(const NdArray<dtype>& inArray, dtype nan, dtype posInf, dtype negInf)
    {
        return nc2pybind(nc::nan_to_num(inArray, nan, posInf, negInf));
    }

    //================================================================================

    template<typename dtype>
    dtype newbyteorderScalar(dtype inValue, Endian inEndianess)
    {
        return newbyteorder(inValue, inEndianess);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric newbyteorderArray(const NdArray<dtype>& inArray, Endian inEndianess)
    {
        return nc2pybind(newbyteorder(inArray, inEndianess));
    }

    //================================================================================

    template<typename dtype1, typename dtype2>
    double nth_rootScalar(dtype1 inValue, dtype2 inRoot)
    {
        return nth_root(inValue, inRoot);
    }

    //================================================================================

    template<typename dtype1, typename dtype2>
    pbArrayGeneric nth_rootArray(const NdArray<dtype1>& inArray, dtype2 inRoot)
    {
        return nc2pybind(nth_root(inArray, inRoot));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric onesSquare(uint32 inSquareSize)
    {
        return nc2pybind(ones<dtype>(inSquareSize));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric onesRowCol(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2pybind(ones<dtype>(inNumRows, inNumCols));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric onesShape(const Shape& inShape)
    {
        return nc2pybind(ones<dtype>(inShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric outer(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(nc::outer(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric packbitsLittleEndian(const NdArray<dtype>& x, Axis axis)
    {
        return nc2pybind(nc::packbitsLittleEndian(x, axis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric packbitsBigEndian(const NdArray<dtype>& x, Axis axis)
    {
        return nc2pybind(nc::packbitsBigEndian(x, axis));
    }

    //================================================================================

    template<typename dtype>
    typename NdArray<dtype>::index_type searchsortedScalar(const NdArray<dtype>& array, dtype value, Side side)
    {
        return nc::searchsorted(array, value, side);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric searchsorted(const NdArray<dtype>& array, const NdArray<dtype>& values, Side side)
    {
        return nc2pybind(nc::searchsorted(array, values, side));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        select(std::vector<pbArray<bool>> condlist, std::vector<pbArray<dtype>> choicelist, dtype defaultValue)
    {
        std::vector<NdArray<bool>>        condVec{};
        std::vector<const NdArray<bool>*> condVecPtr{};
        condVec.reserve(condlist.size());
        condVecPtr.reserve(condlist.size());
        for (auto& cond : condlist)
        {
            condVec.push_back(pybind2nc(cond));
            condVecPtr.push_back(&condVec.back());
        }

        std::vector<NdArray<dtype>>        choiceVec{};
        std::vector<const NdArray<dtype>*> choiceVecPtr{};
        choiceVec.reserve(choicelist.size());
        choiceVecPtr.reserve(choicelist.size());
        for (auto& choice : choicelist)
        {
            choiceVec.push_back(pybind2nc(choice));
            choiceVecPtr.push_back(&choiceVec.back());
        }

        return nc2pybind(nc::select(condVecPtr, choiceVecPtr, defaultValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        selectVector(std::vector<pbArray<bool>> condlist, std::vector<pbArray<dtype>> choicelist, dtype defaultValue)
    {
        std::vector<NdArray<bool>> condVec{};
        condVec.reserve(condlist.size());
        for (auto& cond : condlist)
        {
            condVec.push_back(pybind2nc(cond));
        }

        std::vector<NdArray<dtype>> choiceVec{};
        choiceVec.reserve(choicelist.size());
        for (auto& choice : choicelist)
        {
            choiceVec.push_back(pybind2nc(choice));
        }

        return nc2pybind(nc::select(condVec, choiceVec, defaultValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric selectInitializerList(pbArray<bool>  cond1,
                                         pbArray<bool>  cond2,
                                         pbArray<bool>  cond3,
                                         pbArray<dtype> choice1,
                                         pbArray<dtype> choice2,
                                         pbArray<dtype> choice3,
                                         dtype          defaultValue)
    {
        return nc2pybind(nc::select({ pybind2nc(cond1), pybind2nc(cond2), pybind2nc(cond3) },
                                    { pybind2nc(choice1), pybind2nc(choice2), pybind2nc(choice3) },
                                    defaultValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric sqrArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(sqr(inArray));
    }

    //================================================================================

    template<typename dtype>
    std::complex<dtype> polarScalar(dtype mag, dtype angle)
    {
        return polar(mag, angle);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric polarArray(const NdArray<dtype>& mag, const NdArray<dtype>& angle)
    {
        return nc2pybind(polar(mag, angle));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric powerArrayScalar(const NdArray<dtype>& inArray, uint8 inExponent)
    {
        return nc2pybind(nc::power(inArray, inExponent));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric powerArrayArray(const NdArray<dtype>& inArray, const NdArray<uint8>& inExponents)
    {
        return nc2pybind(nc::power(inArray, inExponents));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric powerfArrayScalar(const NdArray<dtype>& inArray, dtype inExponent)
    {
        return nc2pybind(nc::powerf(inArray, inExponent));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric powerfArrayArray(const NdArray<dtype>& inArray, const NdArray<dtype>& inExponents)
    {
        return nc2pybind(nc::powerf(inArray, inExponents));
    }

    //================================================================================

    template<typename dtype>
    std::complex<dtype> projScalar(const std::complex<dtype>& inValue)
    {
        return nc::proj(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric projArray(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2pybind(proj(inArray));
    }

    template<typename dtype>
    pbArrayGeneric putFlat(NdArray<dtype>& inArray, int32 inIndex, dtype inValue)
    {
        put(inArray, inIndex, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putRowCol(NdArray<dtype>& inArray, int32 inRow, int32 inCol, dtype inValue)
    {
        put(inArray, inRow, inCol, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putIndices1DValue(NdArray<dtype>& inArray, const NdArray<int32>& inIndices, dtype inValue)
    {
        put(inArray, inIndices, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        putIndices1DValues(NdArray<dtype>& inArray, const NdArray<int32>& inIndices, pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        put(inArray, inIndices, inValues);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice1DValue(NdArray<dtype>& inArray, const Slice& inSlice, dtype inValue)
    {
        put(inArray, inSlice, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice1DValues(NdArray<dtype>& inArray, const Slice& inSlice, pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        put(inArray, inSlice, inValues);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putIndices2DValue(NdArray<dtype>&       inArray,
                                     const NdArray<int32>& inRowIndices,
                                     const NdArray<int32>& inColIndices,
                                     dtype                 inValue)
    {
        put(inArray, inRowIndices, inColIndices, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putRowIndicesColSliceValues(NdArray<dtype>&       inArray,
                                               const NdArray<int32>& inRowIndices,
                                               const Slice&          inColSlice,
                                               pbArray<dtype>&       inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        put(inArray, inRowIndices, inColSlice, inValues);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putRowSliceColIndicesValues(NdArray<dtype>&       inArray,
                                               const Slice&          inRowSlice,
                                               const NdArray<int32>& inColIndices,
                                               pbArray<dtype>&       inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        put(inArray, inRowSlice, inColIndices, inValues);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        putSlice2DValue(NdArray<dtype>& inArray, const Slice& inSliceRow, const Slice& inSliceCol, dtype inValue)
    {
        put(inArray, inSliceRow, inSliceCol, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putIndices2DValueRow(NdArray<dtype>&       inArray,
                                        int32                 inRowIndex,
                                        const NdArray<int32>& inColIndices,
                                        dtype                 inValue)
    {
        put(inArray, inRowIndex, inColIndices, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValueRow(NdArray<dtype>& inArray, int32 inRowIndex, const Slice& inSliceCol, dtype inValue)
    {
        put(inArray, inRowIndex, inSliceCol, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putIndices2DValueCol(NdArray<dtype>&       inArray,
                                        const NdArray<int32>& inRowIndices,
                                        int32                 inColIndex,
                                        dtype                 inValue)
    {
        put(inArray, inRowIndices, inColIndex, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValueCol(NdArray<dtype>& inArray, const Slice& inSliceRow, int32 inColIndex, dtype inValue)
    {
        put(inArray, inSliceRow, inColIndex, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putIndices2DValues(NdArray<dtype>&       inArray,
                                      const NdArray<int32>& inRowIndices,
                                      const NdArray<int32>& inColIndices,
                                      pbArray<dtype>&       inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        put(inArray, inRowIndices, inColIndices, inValues);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putRowIndicesColSliceValue(NdArray<dtype>&       inArray,
                                              const NdArray<int32>& inRowIndices,
                                              const Slice&          inColSlice,
                                              dtype                 inValue)
    {
        put(inArray, inRowIndices, inColSlice, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putRowSliceColIndicesValue(NdArray<dtype>&       inArray,
                                              const Slice&          inRowSlice,
                                              const NdArray<int32>& inColIndices,
                                              dtype                 inValue)
    {
        put(inArray, inRowSlice, inColIndices, inValue);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValues(NdArray<dtype>& inArray,
                                    const Slice&    inSliceRow,
                                    const Slice&    inSliceCol,
                                    pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        put(inArray, inSliceRow, inSliceCol, inValues);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putIndices2DValuesRow(NdArray<dtype>&       inArray,
                                         int32                 inRowIndex,
                                         const NdArray<int32>& inColIndices,
                                         pbArray<dtype>&       inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        put(inArray, inRowIndex, inColIndices, inValues);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValuesRow(NdArray<dtype>& inArray,
                                       int32           inRowIndex,
                                       const Slice&    inSliceCol,
                                       pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        put(inArray, inRowIndex, inSliceCol, inValues);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putIndices2DValuesCol(NdArray<dtype>&       inArray,
                                         const NdArray<int32>& inRowIndices,
                                         int32                 inColIndex,
                                         pbArray<dtype>&       inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        put(inArray, inRowIndices, inColIndex, inValues);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValuesCol(NdArray<dtype>& inArray,
                                       const Slice&    inSliceRow,
                                       int32           inColIndex,
                                       pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        put(inArray, inSliceRow, inColIndex, inValues);
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putmask(NdArray<dtype>& inArray, const NdArray<bool>& inMask, const NdArray<dtype>& inValues)
    {
        return nc2pybind(nc::putmask(inArray, inMask, inValues));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putmaskScalar(NdArray<dtype>& inArray, const NdArray<bool>& inMask, dtype inValue)
    {
        return nc2pybind(putmask(inArray, inMask, inValue));
    }

    //================================================================================

    template<typename dtype>
    dtype rad2degScalar(dtype inValue)
    {
        return rad2deg(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric rad2degArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(rad2deg(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype radiansScalar(dtype inValue)
    {
        return radians(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric radiansArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(radians(inArray));
    }

    //================================================================================

    template<typename dtype>
    NdArray<dtype>& ravel(NdArray<dtype>& inArray)
    {
        return nc::ravel(inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric reciprocal(NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::reciprocal(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype realScalar(const std::complex<dtype>& inValue)
    {
        return nc::real(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric realArray(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2pybind(real(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype remainderScalar(dtype inValue1, dtype inValue2)
    {
        return remainder(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric remainderArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(remainder(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric replace(NdArray<dtype>& inArray, dtype oldValue, dtype newValue)
    {
        return nc2pybind(nc::replace(inArray, oldValue, newValue));
    }

    //================================================================================

    template<typename dtype>
    NdArray<dtype>& reshapeInt(NdArray<dtype>& inArray, uint32 inSize)
    {
        return nc::reshape(inArray, inSize);
    }

    //================================================================================

    template<typename dtype>
    NdArray<dtype>& reshapeValues(NdArray<dtype>& inArray, int32 inNumRows, int32 inNumCols)
    {
        return nc::reshape(inArray, inNumRows, inNumCols);
    }

    //================================================================================

    template<typename dtype>
    NdArray<dtype>& reshapeShape(NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        return nc::reshape(inArray, inNewShape);
    }

    //================================================================================

    template<typename dtype>
    NdArray<dtype>& reshapeList(NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        return reshape(inArray, inNewShape.rows, inNewShape.cols);
    }

    //================================================================================

    template<typename dtype>
    NdArray<dtype>& resizeFast(NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        return nc::resizeFast(inArray, inNewShape);
    }

    //================================================================================

    template<typename dtype>
    NdArray<dtype>& resizeFastList(NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        return resizeFast(inArray, inNewShape.rows, inNewShape.cols);
    }

    //================================================================================

    template<typename dtype>
    NdArray<dtype>& resizeSlow(NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        return nc::resizeSlow(inArray, inNewShape);
    }

    //================================================================================

    template<typename dtype>
    NdArray<dtype>& resizeSlowList(NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        return resizeSlow(inArray, inNewShape.rows, inNewShape.cols);
    }

    //================================================================================

    template<typename dtype>
    dtype rintScalar(dtype inValue)
    {
        return rint(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric rintArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(rint(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype roundScalar(dtype inValue, uint8 inDecimals)
    {
        return round(inValue, inDecimals);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric roundArray(const NdArray<dtype>& inArray, uint8 inDecimals)
    {
        return nc2pybind(round(inArray, inDecimals));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric row_stack(const NdArray<dtype>& inArray1,
                             const NdArray<dtype>& inArray2,
                             const NdArray<dtype>& inArray3,
                             const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::row_stack({ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric row_stack_vec(const NdArray<dtype>& inArray1,
                                 const NdArray<dtype>& inArray2,
                                 const NdArray<dtype>& inArray3,
                                 const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::row_stack(std::vector{ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    int8 signScalar(dtype inValue)
    {
        return sign(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric signArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(sign(inArray));
    }

    //================================================================================

    template<typename dtype>
    bool signbitScalar(dtype inValue)
    {
        return signbit(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric signbitArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(signbit(inArray));
    }

    //================================================================================

    template<typename dtype>
    auto sinScalar(dtype inValue) -> decltype(sin(inValue)) // trailing return type to help gcc
    {
        return sin(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric sinArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(sin(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype sincScalar(dtype inValue)
    {
        return sinc(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric sincArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(sinc(inArray));
    }

    //================================================================================

    template<typename dtype>
    auto sinhScalar(dtype inValue) -> decltype(sinh(inValue)) // trailing return type to help gcc
    {
        return sinh(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric sinhArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(sinh(inArray));
    }

    //================================================================================

    template<typename dtype>
    std::vector<pbArrayGeneric> split(const NdArray<dtype>& inArray, const NdArray<int32>& indices, nc::Axis inAxis)
    {
        const auto                  splits = nc::split(inArray, indices, inAxis);
        std::vector<pbArrayGeneric> result{};
        for (const auto& split : splits)
        {
            result.push_back(nc2pybind(split));
        }
        return result;
    }

    //================================================================================

    template<typename dtype>
    auto sqrtScalar(dtype inValue) -> decltype(sqrt(inValue)) // trailing return type to help gcc
    {
        return sqrt(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric sqrtArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(sqrt(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype squareScalar(dtype inValue)
    {
        return square(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric squareArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(square(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric stack(const NdArray<dtype>& inArray1,
                         const NdArray<dtype>& inArray2,
                         const NdArray<dtype>& inArray3,
                         const NdArray<dtype>& inArray4,
                         nc::Axis              inAxis)
    {
        return nc2pybind(nc::stack({ inArray1, inArray2, inArray3, inArray4 }, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric stack_vec(const NdArray<dtype>& inArray1,
                             const NdArray<dtype>& inArray2,
                             const NdArray<dtype>& inArray3,
                             const NdArray<dtype>& inArray4,
                             nc::Axis              inAxis)
    {
        return nc2pybind(nc::stack(std::vector{ inArray1, inArray2, inArray3, inArray4 }, inAxis));
    }

    //================================================================================

    template<typename Type1, typename Type2>
    pbArrayGeneric subtract(const Type1& in1, const Type2& in2)
    {
        return nc2pybind(nc::subtract(in1, in2));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric take(const NdArray<dtype>& inArray, const NdArray<uint32>& inIndices, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(nc::take(inArray, inIndices, inAxis));
    }

    //================================================================================

    template<typename dtype>
    auto tanScalar(dtype inValue) -> decltype(tan(inValue)) // trailing return type to help gcc
    {
        return tan(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric tanArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(tan(inArray));
    }

    //================================================================================

    template<typename dtype>
    auto tanhScalar(dtype inValue) -> decltype(tanh(inValue)) // trailing return type to help gcc
    {
        return tanh(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric tanhArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(tanh(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric tileRectangle(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        return nc2pybind(tile(inArray, inNumRows, inNumCols));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric tileShape(const NdArray<dtype>& inArray, const Shape& inRepShape)
    {
        return nc2pybind(tile(inArray, inRepShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric tileList(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        return nc2pybind(tile(inArray, { inNumRows, inNumCols }));
    }

    //================================================================================

    template<typename dtype>
    void tofileBinary(const NdArray<dtype>& inArray, const std::string& filename)
    {
        tofile(inArray, filename);
    }

    //================================================================================

    template<typename dtype>
    void tofileTxt(const NdArray<dtype>& inArray, const std::string& filename, const char sep)
    {
        tofile(inArray, filename, sep);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric trapzDx(const NdArray<dtype>& inY, double dx = 1.0, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(trapz(inY, dx, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric trapz(const NdArray<dtype>& inY, const NdArray<dtype>& inX, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(nc::trapz(inY, inX, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric triuSquare(uint32 inSquareSize, int32 inOffset)
    {
        return nc2pybind(triu<dtype>(inSquareSize, inOffset));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric triuRect(uint32 inNumRows, uint32 inNumCols, int32 inOffset)
    {
        return nc2pybind(triu<dtype>(inNumRows, inNumCols, inOffset));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric triuArray(const NdArray<dtype>& inArray, int32 inOffset)
    {
        return nc2pybind(triu(inArray, inOffset));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric trilSquare(uint32 inSquareSize, int32 inOffset)
    {
        return nc2pybind(tril<dtype>(inSquareSize, inOffset));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric trilRect(uint32 inNumRows, uint32 inNumCols, int32 inOffset)
    {
        return nc2pybind(tril<dtype>(inNumRows, inNumCols, inOffset));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric trilArray(const NdArray<dtype>& inArray, int32 inOffset)
    {
        return nc2pybind(tril(inArray, inOffset));
    }

    //================================================================================

    template<typename dtype>
    double truncScalar(dtype inValue)
    {
        return trunc(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric truncArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(trunc(inArray));
    }

    //================================================================================

    pbArrayGeneric unpackbitsLittleEndian(const NdArray<uint8>& x, Axis axis)
    {
        return nc2pybind(nc::unpackbitsLittleEndian(x, axis));
    }

    //================================================================================

    pbArrayGeneric unpackbitsBigEndian(const NdArray<uint8>& x, Axis axis)
    {
        return nc2pybind(nc::unpackbitsBigEndian(x, axis));
    }

    //================================================================================

    template<typename dtype>
    dtype unwrapScalar(dtype inValue)
    {
        return unwrap(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric unwrapArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(unwrap(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric vander(const NdArray<dtype>& x, Increasing increasing)
    {
        return nc2pybind(nc::vander(x, increasing));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric vander(const NdArray<dtype>& x, uint32 n, Increasing increasing)
    {
        return nc2pybind(nc::vander(x, n, increasing));
    }

    //================================================================================

    template<typename dtype>
    std::vector<pbArrayGeneric> vsplit(const NdArray<dtype>& inArray, const NdArray<int32>& indices)
    {
        const auto                  splits = nc::vsplit(inArray, indices);
        std::vector<pbArrayGeneric> result{};
        for (const auto& split : splits)
        {
            result.push_back(nc2pybind(split));
        }
        return result;
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric vstack(const NdArray<dtype>& inArray1,
                          const NdArray<dtype>& inArray2,
                          const NdArray<dtype>& inArray3,
                          const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::vstack({ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric vstack_vec(const NdArray<dtype>& inArray1,
                              const NdArray<dtype>& inArray2,
                              const NdArray<dtype>& inArray3,
                              const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::vstack(std::vector{ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric whereArrayArray(const NdArray<bool>& inMask, const NdArray<dtype>& inA, const NdArray<dtype>& inB)
    {
        return nc2pybind(nc::where(inMask, inA, inB));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric whereScalarArray(const NdArray<bool>& inMask, const NdArray<dtype>& inA, dtype inB)
    {
        return nc2pybind(nc::where(inMask, inA, inB));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric whereArrayScalar(const NdArray<bool>& inMask, dtype inA, const NdArray<dtype>& inB)
    {
        return nc2pybind(nc::where(inMask, inA, inB));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric whereScalarScalar(const NdArray<bool>& inMask, dtype inA, dtype inB)
    {
        return nc2pybind(nc::where(inMask, inA, inB));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric zerosSquare(uint32 inSquareSize)
    {
        return nc2pybind(zeros<dtype>(inSquareSize));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric zerosRowCol(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2pybind(zeros<dtype>(inNumRows, inNumCols));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric zerosShape(const Shape& inShape)
    {
        return nc2pybind(zeros<dtype>(inShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric zerosList(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2pybind(zeros<dtype>({ inNumRows, inNumCols }));
    }
} // namespace FunctionsInterface

//================================================================================

void initFunctions(python_interface::module& m)
{
    // Functions.hpp
    m.def("absScalar", &FunctionsInterface::absScalar<double>);
    m.def("absArray", &FunctionsInterface::absArray<double>);
    m.def("absScalar", &FunctionsInterface::absScalar<ComplexDouble>);
    m.def("absArray", &FunctionsInterface::absArray<ComplexDouble>);
    m.def("add", &FunctionsInterface::add<NdArray<double>, NdArray<double>>);
    m.def("add", &FunctionsInterface::add<NdArray<double>, double>);
    m.def("add", &FunctionsInterface::add<double, NdArray<double>>);
    m.def("add", &FunctionsInterface::add<NdArray<ComplexDouble>, NdArray<ComplexDouble>>);
    m.def("add", &FunctionsInterface::add<NdArray<ComplexDouble>, ComplexDouble>);
    m.def("add", &FunctionsInterface::add<ComplexDouble, NdArray<ComplexDouble>>);
    m.def("add", &FunctionsInterface::add<NdArray<double>, NdArray<ComplexDouble>>);
    m.def("add", &FunctionsInterface::add<NdArray<ComplexDouble>, NdArray<double>>);
    m.def("add", &FunctionsInterface::add<NdArray<double>, ComplexDouble>);
    m.def("add", &FunctionsInterface::add<ComplexDouble, NdArray<double>>);
    m.def("add", &FunctionsInterface::add<NdArray<ComplexDouble>, double>);
    m.def("add", &FunctionsInterface::add<double, NdArray<ComplexDouble>>);
    m.def("alen", &alen<double>);
    m.def("all", &FunctionsInterface::allArray<double>);
    m.def("all", &FunctionsInterface::allArray<ComplexDouble>);
    m.def("allclose", &allclose<double>);
    m.def("amax", &FunctionsInterface::amaxArray<double>);
    m.def("amax", &FunctionsInterface::amaxArray<ComplexDouble>);
    m.def("amin", &FunctionsInterface::aminArray<double>);
    m.def("amin", &FunctionsInterface::aminArray<ComplexDouble>);
    m.def("angleScalar", &FunctionsInterface::angleScalar<double>);
    m.def("angleArray", &FunctionsInterface::angleArray<double>);
    m.def("any", &FunctionsInterface::anyArray<double>);
    m.def("any", &FunctionsInterface::anyArray<ComplexDouble>);
    m.def("append", &append<double>);
    m.def("applyPoly1d", &applyPoly1d<double>);
    m.def("arange", &FunctionsInterface::arangeArray<double>);
    m.def("arccosScalar", &FunctionsInterface::arccosScalar<double>);
    m.def("arccosArray", &FunctionsInterface::arccosArray<double>);
    m.def("arccosScalar", &FunctionsInterface::arccosScalar<ComplexDouble>);
    m.def("arccosArray", &FunctionsInterface::arccosArray<ComplexDouble>);
    m.def("arccoshScalar", &FunctionsInterface::arccoshScalar<double>);
    m.def("arccoshArray", &FunctionsInterface::arccoshArray<double>);
    m.def("arccoshScalar", &FunctionsInterface::arccoshScalar<ComplexDouble>);
    m.def("arccoshArray", &FunctionsInterface::arccoshArray<ComplexDouble>);
    m.def("arcsinScalar", &FunctionsInterface::arcsinScalar<double>);
    m.def("arcsinArray", &FunctionsInterface::arcsinArray<double>);
    m.def("arcsinScalar", &FunctionsInterface::arcsinScalar<ComplexDouble>);
    m.def("arcsinArray", &FunctionsInterface::arcsinArray<ComplexDouble>);
    m.def("arcsinhScalar", &FunctionsInterface::arcsinhScalar<double>);
    m.def("arcsinhArray", &FunctionsInterface::arcsinhArray<double>);
    m.def("arcsinhScalar", &FunctionsInterface::arcsinhScalar<ComplexDouble>);
    m.def("arcsinhArray", &FunctionsInterface::arcsinhArray<ComplexDouble>);
    m.def("arctanScalar", &FunctionsInterface::arctanScalar<double>);
    m.def("arctanArray", &FunctionsInterface::arctanArray<double>);
    m.def("arctanScalar", &FunctionsInterface::arctanScalar<ComplexDouble>);
    m.def("arctanArray", &FunctionsInterface::arctanArray<ComplexDouble>);
    m.def("arctan2Scalar", &FunctionsInterface::arctan2Scalar<double>);
    m.def("arctan2Array", &FunctionsInterface::arctan2Array<double>);
    m.def("arctanhScalar", &FunctionsInterface::arctanhScalar<double>);
    m.def("arctanhArray", &FunctionsInterface::arctanhArray<double>);
    m.def("arctanhScalar", &FunctionsInterface::arctanhScalar<ComplexDouble>);
    m.def("arctanhArray", &FunctionsInterface::arctanhArray<ComplexDouble>);
    m.def("argmax", &FunctionsInterface::argmaxArray<double>);
    m.def("argmax", &FunctionsInterface::argmaxArray<ComplexDouble>);
    m.def("argmin", &FunctionsInterface::argminArray<double>);
    m.def("argmin", &FunctionsInterface::argminArray<ComplexDouble>);
    m.def("argpartition", &argpartition<double>);
    m.def("argpartition", &argpartition<ComplexDouble>);
    m.def("argsort", &FunctionsInterface::argsortArray<double>);
    m.def("argsort", &FunctionsInterface::argsortArray<ComplexDouble>);
    m.def("argwhere", &FunctionsInterface::argwhere<double>);
    m.def("argwhere", &FunctionsInterface::argwhere<ComplexDouble>);
    m.def("aroundScalar", &FunctionsInterface::aroundScalar<double>);
    m.def("aroundArray", &FunctionsInterface::aroundArray<double>);
    m.def("array_equal", &array_equal<double>);
    m.def("array_equal", &array_equal<ComplexDouble>);
    m.def("array_equiv", &array_equiv<double>);
    m.def("array_equiv", &array_equiv<ComplexDouble>);
    m.def("asarrayInitializerList", &FunctionsInterface::asarrayInitializerList<double>);
    m.def("asarrayInitializerList", &FunctionsInterface::asarrayInitializerList<ComplexDouble>);
    m.def("asarrayInitializerList2D", &FunctionsInterface::asarrayInitializerList2D<double>);
    m.def("asarrayInitializerList2D", &FunctionsInterface::asarrayInitializerList2D<ComplexDouble>);
    m.def("asarrayArray1D", &FunctionsInterface::asarrayArray1D<double>);
    m.def("asarrayArray1D", &FunctionsInterface::asarrayArray1D<ComplexDouble>);
    m.def("asarrayArray1DCopy", &FunctionsInterface::asarrayArray1DCopy<double>);
    m.def("asarrayArray1DCopy", &FunctionsInterface::asarrayArray1DCopy<ComplexDouble>);
    m.def("asarrayArray2D", &FunctionsInterface::asarrayArray2D<double>);
    m.def("asarrayArray2D", &FunctionsInterface::asarrayArray2D<ComplexDouble>);
    m.def("asarrayArray2DCopy", &FunctionsInterface::asarrayArray2DCopy<double>);
    m.def("asarrayArray2DCopy", &FunctionsInterface::asarrayArray2DCopy<ComplexDouble>);
    m.def("asarrayVector1D", &FunctionsInterface::asarrayVector1D<double>);
    m.def("asarrayVector1D", &FunctionsInterface::asarrayVector1D<ComplexDouble>);
    m.def("asarrayVector1DCopy", &FunctionsInterface::asarrayVector1DCopy<double>);
    m.def("asarrayVector1DCopy", &FunctionsInterface::asarrayVector1DCopy<ComplexDouble>);
    m.def("asarrayVector2D", &FunctionsInterface::asarrayVector2D<double>);
    m.def("asarrayVector2D", &FunctionsInterface::asarrayVector2D<ComplexDouble>);
    m.def("asarrayVectorArray2D", &FunctionsInterface::asarrayVectorArray2D<double>);
    m.def("asarrayVectorArray2D", &FunctionsInterface::asarrayVectorArray2D<ComplexDouble>);
    m.def("asarrayVectorArray2DCopy", &FunctionsInterface::asarrayVectorArray2DCopy<double>);
    m.def("asarrayVectorArray2DCopy", &FunctionsInterface::asarrayVectorArray2DCopy<ComplexDouble>);
    m.def("asarrayDeque1D", &FunctionsInterface::asarrayDeque1D<double>);
    m.def("asarrayDeque1D", &FunctionsInterface::asarrayDeque1D<ComplexDouble>);
    m.def("asarrayDeque2D", &FunctionsInterface::asarrayDeque2D<double>);
    m.def("asarrayDeque2D", &FunctionsInterface::asarrayDeque2D<ComplexDouble>);
    m.def("asarrayList", &FunctionsInterface::asarrayList<double>);
    m.def("asarrayList", &FunctionsInterface::asarrayList<ComplexDouble>);
    m.def("asarrayIterators", &FunctionsInterface::asarrayIterators<double>);
    m.def("asarrayIterators", &FunctionsInterface::asarrayIterators<ComplexDouble>);
    m.def("asarrayPointerIterators", &FunctionsInterface::asarrayPointerIterators<double>);
    m.def("asarrayPointerIterators", &FunctionsInterface::asarrayPointerIterators<ComplexDouble>);
    m.def("asarrayPointer", &FunctionsInterface::asarrayPointer<double>);
    m.def("asarrayPointer", &FunctionsInterface::asarrayPointer<ComplexDouble>);
    m.def("asarrayPointer2D", &FunctionsInterface::asarrayPointer2D<double>);
    m.def("asarrayPointer2D", &FunctionsInterface::asarrayPointer2D<ComplexDouble>);
    m.def("asarrayPointerShell", &FunctionsInterface::asarrayPointerShell<double>);
    m.def("asarrayPointerShell", &FunctionsInterface::asarrayPointerShell<ComplexDouble>);
    m.def("asarrayPointerShell2D", &FunctionsInterface::asarrayPointerShell2D<double>);
    m.def("asarrayPointerShell2D", &FunctionsInterface::asarrayPointerShell2D<ComplexDouble>);
    m.def("asarrayPointerShellTakeOwnership", &FunctionsInterface::asarrayPointerShellTakeOwnership<double>);
    m.def("asarrayPointerShellTakeOwnership", &FunctionsInterface::asarrayPointerShellTakeOwnership<ComplexDouble>);
    m.def("asarrayPointerShell2DTakeOwnership", &FunctionsInterface::asarrayPointerShell2DTakeOwnership<double>);
    m.def("asarrayPointerShell2DTakeOwnership", &FunctionsInterface::asarrayPointerShell2DTakeOwnership<ComplexDouble>);
    m.def("astypeDoubleToUint32", &astype<uint32, double>);
    m.def("astypeDoubleToComplex", &astype<ComplexDouble, double>);
    m.def("astypeComplexToComplex", &astype<std::complex<float>, ComplexDouble>);
    m.def("astypeComplexToDouble", &astype<double, ComplexDouble>);
    m.def("average", &FunctionsInterface::average<double>);
    m.def("average", &FunctionsInterface::average<ComplexDouble>);
    m.def("averageWeighted", &FunctionsInterface::averageWeighted<double>);
    m.def("averageWeighted", &FunctionsInterface::averageWeightedComplex<double>);

    m.def("bartlett", &FunctionsInterface::bartlett);
    m.def("binaryRepr", &binaryRepr<uint64>);
    m.def("bincount", &FunctionsInterface::bincount<uint32>);
    m.def("bincountWeighted", &FunctionsInterface::bincountWeighted<uint32>);
    m.def("bit_count", &FunctionsInterface::bit_count<uint64>);
    m.def("bitwise_and", &FunctionsInterface::bitwise_and<uint64>);
    m.def("bitwise_not", &FunctionsInterface::bitwise_not<uint64>);
    m.def("bitwise_or", &FunctionsInterface::bitwise_or<uint64>);
    m.def("bitwise_xor", &FunctionsInterface::bitwise_xor<uint64>);
    m.def("blackman", &FunctionsInterface::blackman);
    m.def("andOperatorArray", &FunctionsInterface::andOperatorArray<uint64>);
    m.def("andOperatorScalar", &FunctionsInterface::andOperatorScalar<uint64>);
    m.def("orOperatorArray", &FunctionsInterface::orOperatorArray<uint64>);
    m.def("orOperatorScalar", &FunctionsInterface::orOperatorScalar<uint64>);
    m.def("byteswap", &FunctionsInterface::byteswap<uint64>);

    m.def("cbrtScalar", &FunctionsInterface::cbrtScalar<double>);
    m.def("cbrtArray", &FunctionsInterface::cbrtArray<double>);
    m.def("ceilScalar", &FunctionsInterface::ceilScalar<double>);
    m.def("centerOfMass", &FunctionsInterface::centerOfMass<double>);
    m.def("ceilArray", &FunctionsInterface::ceilArray<double>);
    m.def("clipScalar", &FunctionsInterface::clipScalar<double>);
    m.def("clipScalar", &FunctionsInterface::clipScalar<ComplexDouble>);
    m.def("clipArray", &FunctionsInterface::clipArray<double>);
    m.def("clipArray", &FunctionsInterface::clipArray<ComplexDouble>);
    m.def("column_stack", &FunctionsInterface::column_stack<double>);
    m.def("column_stack_vec", &FunctionsInterface::column_stack_vec<double>);
    m.def("complexScalar", &FunctionsInterface::complexScalarSingle<double>);
    m.def("complexScalar", &FunctionsInterface::complexScalar<double>);
    m.def("complexArray", &FunctionsInterface::complexArraySingle<double>);
    m.def("complexArray", &FunctionsInterface::complexArray<double>);
    m.def("conjScalar", &FunctionsInterface::conjScalar<double>);
    m.def("conjArray", &FunctionsInterface::conjArray<double>);
    m.def("concatenate", &FunctionsInterface::concatenate<double>);
    m.def("concatenate_vec", &FunctionsInterface::concatenate_vec<double>);
    m.def("contains", &contains<double>);
    m.def("contains", &contains<ComplexDouble>);
    m.def("copy", &FunctionsInterface::copy<double>);
    m.def("copysign", &FunctionsInterface::copySign<double>);
    m.def("copyto", &FunctionsInterface::copyto<double>);
    m.def("corrcoef", &FunctionsInterface::corrcoef<double>);
    m.def("cosScalar", &FunctionsInterface::cosScalar<double>);
    m.def("cosScalar", &FunctionsInterface::cosScalar<ComplexDouble>);
    m.def("cosArray", &FunctionsInterface::cosArray<double>);
    m.def("cosArray", &FunctionsInterface::cosArray<ComplexDouble>);
    m.def("coshScalar", &FunctionsInterface::coshScalar<double>);
    m.def("coshScalar", &FunctionsInterface::coshScalar<ComplexDouble>);
    m.def("coshArray", &FunctionsInterface::coshArray<double>);
    m.def("coshArray", &FunctionsInterface::coshArray<ComplexDouble>);
    m.def("count_nonzero", &FunctionsInterface::count_nonzero<double>);
    m.def("count_nonzero", &FunctionsInterface::count_nonzero<ComplexDouble>);
    m.def("cov", &FunctionsInterface::cov<double>);
    m.def("cov_inv", &FunctionsInterface::cov_inv<double>);
    m.def("cross", &cross<double>);
    m.def("cross", &cross<ComplexDouble>);
    m.def("cube", &FunctionsInterface::cubeArray<double>);
    m.def("cube", &FunctionsInterface::cubeArray<ComplexDouble>);
    m.def("cumprod", &FunctionsInterface::cumprodArray<double>);
    m.def("cumprod", &FunctionsInterface::cumprodArray<ComplexDouble>);
    m.def("cumsum", &FunctionsInterface::cumsumArray<double>);
    m.def("cumsum", &FunctionsInterface::cumsumArray<ComplexDouble>);

    m.def("deg2radScalar", &FunctionsInterface::deg2radScalar<double>);
    m.def("deg2radArray", &FunctionsInterface::deg2radArray<double>);
    m.def("degreesScalar", &FunctionsInterface::degreesScalar<double>);
    m.def("degreesArray", &FunctionsInterface::degreesArray<double>);
    m.def("deleteIndicesScalar", &FunctionsInterface::deleteIndicesScalar<double>);
    m.def("deleteIndicesIndices", &FunctionsInterface::deleteIndicesIndices<double>);
    m.def("deleteIndicesSlice", &FunctionsInterface::deleteIndicesSlice<double>);
    m.def("diag", &FunctionsInterface::diag<double>);
    m.def("diag", &FunctionsInterface::diag<ComplexDouble>);
    m.def("diagflat", &FunctionsInterface::diagflat<double>);
    m.def("diagflat", &FunctionsInterface::diagflat<ComplexDouble>);
    m.def("diagonal", &FunctionsInterface::diagonal<double>);
    m.def("diagonal", &FunctionsInterface::diagonal<ComplexDouble>);
    m.def("diff", &FunctionsInterface::diff<double>);
    m.def("diff", &FunctionsInterface::diff<ComplexDouble>);
    m.def("digitize", &FunctionsInterface::digitize<double>);
    m.def("divide", &FunctionsInterface::divide<NdArray<double>, NdArray<double>>);
    m.def("divide", &FunctionsInterface::divide<NdArray<double>, double>);
    m.def("divide", &FunctionsInterface::divide<double, NdArray<double>>);
    m.def("divide", &FunctionsInterface::divide<NdArray<ComplexDouble>, NdArray<ComplexDouble>>);
    m.def("divide", &FunctionsInterface::divide<NdArray<ComplexDouble>, ComplexDouble>);
    m.def("divide", &FunctionsInterface::divide<ComplexDouble, NdArray<ComplexDouble>>);
    m.def("divide", &FunctionsInterface::divide<NdArray<double>, NdArray<ComplexDouble>>);
    m.def("divide", &FunctionsInterface::divide<NdArray<ComplexDouble>, NdArray<double>>);
    m.def("divide", &FunctionsInterface::divide<NdArray<double>, ComplexDouble>);
    m.def("divide", &FunctionsInterface::divide<ComplexDouble, NdArray<double>>);
    m.def("divide", &FunctionsInterface::divide<NdArray<ComplexDouble>, double>);
    m.def("divide", &FunctionsInterface::divide<double, NdArray<ComplexDouble>>);
    m.def("dot", &FunctionsInterface::dot<double, double>);
    m.def("dot", &FunctionsInterface::dot<ComplexDouble, ComplexDouble>);
    m.def("dot", &FunctionsInterface::dot<double, ComplexDouble>);
    m.def("dot", &FunctionsInterface::dot<ComplexDouble, double>);
    m.def("dump", &dump<double>);
    m.def("dump", &dump<ComplexDouble>);

    m.def("emptyRowCol", &FunctionsInterface::emptyRowCol<double>);
    m.def("emptyShape", &FunctionsInterface::emptyShape<double>);
    m.def("empty_like", &empty_like<double>);
    m.def("endianess", &endianess<double>);
    m.def("equal", &FunctionsInterface::equal<double>);
    m.def("equal", &FunctionsInterface::equal<ComplexDouble>);
    m.def("extract", &FunctionsInterface::extract<double>);
    m.def("expScalar", &FunctionsInterface::expScalar<double>);
    m.def("expScalar", &FunctionsInterface::expScalar<ComplexDouble>);
    m.def("expArray", &FunctionsInterface::expArray<double>);
    m.def("expArray", &FunctionsInterface::expArray<ComplexDouble>);
    m.def("exp2Scalar", &FunctionsInterface::exp2Scalar<double>);
    m.def("exp2Array", &FunctionsInterface::exp2Array<double>);
    m.def("expm1Scalar", &FunctionsInterface::expm1Scalar<double>);
    m.def("expm1Scalar", &FunctionsInterface::expm1Scalar<ComplexDouble>);
    m.def("expm1Array", &FunctionsInterface::expm1Array<double>);
    m.def("expm1Array", &FunctionsInterface::expm1Array<ComplexDouble>);
    m.def("eye1D", &FunctionsInterface::eye1D<double>);
    m.def("eye1DComplex", &FunctionsInterface::eye1D<ComplexDouble>);
    m.def("eye2D", &FunctionsInterface::eye2D<double>);
    m.def("eye2DComplex", &FunctionsInterface::eye2D<ComplexDouble>);
    m.def("eyeShape", &FunctionsInterface::eyeShape<double>);
    m.def("eyeShapeComplex", &FunctionsInterface::eyeShape<ComplexDouble>);

    m.def("fillDiagonal", &fillDiagonal<double>);
    m.def("find", &FunctionsInterface::find);
    m.def("findN", &FunctionsInterface::findN);
    m.def("fixScalar", &FunctionsInterface::fixScalar<double>);
    m.def("fixArray", &FunctionsInterface::fixArray<double>);
    m.def("flatten", &flatten<double>);
    m.def("flatnonzero", &flatnonzero<double>);
    m.def("flatnonzero", &flatnonzero<ComplexDouble>);
    m.def("flip", &flip<double>);
    m.def("fliplr", &fliplr<double>);
    m.def("flipud", &flipud<double>);
    m.def("floorScalar", &FunctionsInterface::floorScalar<double>);
    m.def("floorArray", &FunctionsInterface::floorArray<double>);
    m.def("floor_divideScalar", &FunctionsInterface::floor_divideScalar<double>);
    m.def("floor_divideArray", &FunctionsInterface::floor_divideArray<double>);
    m.def("fmaxScalar", &FunctionsInterface::fmaxScalar<double>);
    m.def("fmaxScalar", &FunctionsInterface::fmaxScalar<ComplexDouble>);
    m.def("fmaxArray", &FunctionsInterface::fmaxArray<double>);
    m.def("fmaxArray", &FunctionsInterface::fmaxArray<ComplexDouble>);
    m.def("fmaxArrayScalar", &FunctionsInterface::fmaxArrayScalar<double>);
    m.def("fmaxArrayScalar", &FunctionsInterface::fmaxArrayScalar<ComplexDouble>);
    m.def("fmaxArrayScalar", &FunctionsInterface::fmaxScalarArray<double>);
    m.def("fmaxArrayScalar", &FunctionsInterface::fmaxScalarArray<ComplexDouble>);
    m.def("fminScalar", &FunctionsInterface::fminScalar<double>);
    m.def("fminScalar", &FunctionsInterface::fminScalar<ComplexDouble>);
    m.def("fminArray", &FunctionsInterface::fminArray<double>);
    m.def("fminArray", &FunctionsInterface::fminArray<ComplexDouble>);
    m.def("fminArrayScalar", &FunctionsInterface::fminArrayScalar<double>);
    m.def("fminArrayScalar", &FunctionsInterface::fminArrayScalar<ComplexDouble>);
    m.def("fminArrayScalar", &FunctionsInterface::fminScalarArray<double>);
    m.def("fminArrayScalar", &FunctionsInterface::fminScalarArray<ComplexDouble>);
    m.def("fmodScalarInt", &FunctionsInterface::fmodScalar<uint32>);
    m.def("fmodArrayInt", &FunctionsInterface::fmodArray<uint32>);
    m.def("fmodScalarFloat", &FunctionsInterface::fmodScalar<double>);
    m.def("fmodArrayFloat", &FunctionsInterface::fmodArray<double>);
    m.def("frombuffer", &FunctionsInterface::frombuffer<double>);
    m.def("frombuffer", &FunctionsInterface::frombuffer<ComplexDouble>);
    m.def("fromfile", &FunctionsInterface::fromfileBinary<double>);
    m.def("fromfile", &FunctionsInterface::fromfileTxt<double>);
    m.def("fromfunction", &FunctionsInterface::fromfunctionSize<double>);
    m.def("fromfunction", &FunctionsInterface::fromfunctionShape<double>);
    m.def("fromiter", &FunctionsInterface::fromiter<double>);
    m.def("fromiter", &FunctionsInterface::fromiter<ComplexDouble>);
    m.def("fromstringInt", &FunctionsInterface::fromstring<int>);
    m.def("fromstringDouble", &FunctionsInterface::fromstring<double>);
    m.def("fullSquare", &FunctionsInterface::fullSquare<double>);
    m.def("fullSquareComplex", &FunctionsInterface::fullSquare<ComplexDouble>);
    m.def("fullRowCol", &FunctionsInterface::fullRowCol<double>);
    m.def("fullRowColComplex", &FunctionsInterface::fullRowCol<ComplexDouble>);
    m.def("fullShape", &FunctionsInterface::fullShape<double>);
    m.def("fullShapeComplex", &FunctionsInterface::fullShape<ComplexDouble>);
    m.def("full_like", &full_like<double>);
    m.def("full_likeComplex", &full_like<ComplexDouble>);

#if defined(__cpp_lib_gcd_lcm) || !defined(NUMCPP_NO_USE_BOOST)
    m.def("gcdScalar", &FunctionsInterface::gcdScalar<uint32>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("gcdArray", &FunctionsInterface::gcdArray<uint32>);
#endif
    m.def("geomspace", &FunctionsInterface::geomspace<double>);
    m.def("greater", &greater<double>);
    m.def("greater", &greater<ComplexDouble>);
    m.def("greater_equal", &greater_equal<double>);
    m.def("greater_equal", &greater_equal<ComplexDouble>);
    m.def("gradient", &FunctionsInterface::gradient<double>);
    m.def("gradient", &FunctionsInterface::gradient<ComplexDouble>);

    m.def("hamming", &FunctionsInterface::hamming);
    m.def("hanning", &FunctionsInterface::hanning);
    m.def("histogram", &FunctionsInterface::histogram<double>);
    m.def("histogram", &FunctionsInterface::histogramWithEdges<double>);
    m.def("hsplit", &FunctionsInterface::hsplit<double>);
    m.def("hstack", &FunctionsInterface::hstack<double>);
    m.def("hstack_vec", &FunctionsInterface::hstack_vec<double>);
    m.def("hypotScalar", &FunctionsInterface::hypotScalar<double>);
    m.def("hypotScalarTriple", &FunctionsInterface::hypotScalarTriple<double>);
    m.def("hypotArray", &FunctionsInterface::hypotArray<double>);
    m.def("hypotArray", &FunctionsInterface::hypotArrayTriple<double>);

    m.def("identity", &identity<double>);
    m.def("identityComplex", &identity<ComplexDouble>);
    m.def("imagScalar", &FunctionsInterface::imagScalar<double>);
    m.def("imagArray", &FunctionsInterface::imagArray<double>);
    m.def("inner", &FunctionsInterface::inner<double>);
    m.def("insert", &FunctionsInterface::insertIndexScalar<double>);
    m.def("insert", &FunctionsInterface::insertIndexArray<double>);
    m.def("insert", &FunctionsInterface::insertIndexScalarAxis<double>);
    m.def("insert", &FunctionsInterface::insertIndexArrayAxis<double>);
    m.def("insert", &FunctionsInterface::insertIndicesScalar<double>);
    m.def("insert", &FunctionsInterface::insertSliceScalar<double>);
    m.def("insert", &FunctionsInterface::insertIndicesArray<double>);
    m.def("insert", &FunctionsInterface::insertSliceArray<double>);
    m.def("interp", &FunctionsInterface::interp<double>);
    m.def("intersect1d", &intersect1d<uint32>);
    m.def("invert", &invert<uint32>);
    m.def("isclose", &isclose<double>);
    m.def("isinfScalar", &FunctionsInterface::isinfScalar<double>);
    m.def("isinfArray", &FunctionsInterface::isinfArray<double>);
    m.def("isposinfScalar", &FunctionsInterface::isposinfScalar<double>);
    m.def("isposinfArray", &FunctionsInterface::isposinfArray<double>);
    m.def("isneginfScalar", &FunctionsInterface::isneginfScalar<double>);
    m.def("isneginfArray", &FunctionsInterface::isneginfArray<double>);
    m.def("isnanScalar", &FunctionsInterface::isnanScalar<double>);
    m.def("isnanArray", &FunctionsInterface::isnanArray<double>);

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    m.def("kaiser", &FunctionsInterface::kaiser);
#endif

#if defined(__cpp_lib_gcd_lcm) || !defined(NUMCPP_NO_USE_BOOST)
    m.def("lcmScalar", &FunctionsInterface::lcmScalar<uint32>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("lcmArray", &FunctionsInterface::lcmArray<uint32>);
#endif
    m.def("ldexpScalar", &FunctionsInterface::ldexpScalar<double>);
    m.def("ldexpArray", &FunctionsInterface::ldexpArray<double>);
    m.def("left_shift", &left_shift<uint32>);
    m.def("less", &less<double>);
    m.def("less", &less<ComplexDouble>);
    m.def("less_equal", &less_equal<double>);
    m.def("less_equal", &less_equal<ComplexDouble>);
    m.def("linspace", &linspace<double>);
    m.def("load", &load<double>);
    m.def("logScalar", &FunctionsInterface::logScalar<double>);
    m.def("logArray", &FunctionsInterface::logArray<double>);
    m.def("logScalar", &FunctionsInterface::logScalar<ComplexDouble>);
    m.def("logArray", &FunctionsInterface::logArray<ComplexDouble>);
    m.def("logbScalar", &FunctionsInterface::logbScalar<double>);
    m.def("logbArray", &FunctionsInterface::logbArray<double>);
    m.def("logspace", &FunctionsInterface::logspace<double>);
    m.def("log10Scalar", &FunctionsInterface::log10Scalar<double>);
    m.def("log10Array", &FunctionsInterface::log10Array<ComplexDouble>);
    m.def("log10Scalar", &FunctionsInterface::log10Scalar<ComplexDouble>);
    m.def("log10Array", &FunctionsInterface::log10Array<double>);
    m.def("log1pScalar", &FunctionsInterface::log1pScalar<double>);
    m.def("log1pArray", &FunctionsInterface::log1pArray<double>);
    m.def("log2Scalar", &FunctionsInterface::log2Scalar<double>);
    m.def("log2Array", &FunctionsInterface::log2Array<double>);
    m.def("logaddexpScalar", &FunctionsInterface::logaddexpScalar<double>);
    m.def("logaddexpArray", &FunctionsInterface::logaddexpArray<double>);
    m.def("logaddexpScalar", &FunctionsInterface::logaddexpScalar<ComplexDouble>);
    m.def("logaddexpArray", &FunctionsInterface::logaddexpArray<ComplexDouble>);
    m.def("logaddexp2Scalar", &FunctionsInterface::logaddexp2Scalar<double>);
    m.def("logaddexp2Array", &FunctionsInterface::logaddexp2Array<double>);
    m.def("logical_and", &logical_and<double>);
    m.def("logical_not", &logical_not<double>);
    m.def("logical_or", &logical_or<double>);
    m.def("logical_xor", &logical_xor<double>);

    m.def("matmul", &FunctionsInterface::matmul<double, double>);
    m.def("matmul", &FunctionsInterface::matmul<ComplexDouble, ComplexDouble>);
    m.def("matmul", &FunctionsInterface::matmul<double, ComplexDouble>);
    m.def("matmul", &FunctionsInterface::matmul<ComplexDouble, double>);
    m.def("max", &FunctionsInterface::max<double>);
    m.def("max", &FunctionsInterface::max<ComplexDouble>);
    m.def("maximum", &FunctionsInterface::maximumArrays<double>);
    m.def("maximum", &FunctionsInterface::maximumArrays<ComplexDouble>);
    m.def("maximum", &FunctionsInterface::maximumArrayScalar<double>);
    m.def("maximum", &FunctionsInterface::maximumArrayScalar<ComplexDouble>);
    m.def("maximum", &FunctionsInterface::maximumScalarArray<double>);
    m.def("maximum", &FunctionsInterface::maximumScalarArray<ComplexDouble>);
    NdArray<double> (*meanDouble)(const NdArray<double>&, Axis) = &mean<double>;
    m.def("mean", meanDouble);
    NdArray<ComplexDouble> (*meanComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &mean<double>;
    m.def("mean", meanComplexDouble);
    m.def("median", &median<double>);
    m.def("meshgrid", &FunctionsInterface::meshgrid<double>);
    m.def("min", &FunctionsInterface::min<double>);
    m.def("min", &FunctionsInterface::min<ComplexDouble>);
    m.def("minimum", &FunctionsInterface::minimumArrays<double>);
    m.def("minimum", &FunctionsInterface::minimumArrays<ComplexDouble>);
    m.def("minimum", &FunctionsInterface::minimumArrayScalar<double>);
    m.def("minimum", &FunctionsInterface::minimumArrayScalar<ComplexDouble>);
    m.def("minimum", &FunctionsInterface::minimumScalarArray<double>);
    m.def("minimum", &FunctionsInterface::minimumScalarArray<ComplexDouble>);
    m.def("mod", &mod<uint32>);
    m.def("multiply", &FunctionsInterface::multiply<NdArray<double>, NdArray<double>>);
    m.def("multiply", &FunctionsInterface::multiply<NdArray<double>, double>);
    m.def("multiply", &FunctionsInterface::multiply<double, NdArray<double>>);
    m.def("multiply", &FunctionsInterface::multiply<NdArray<ComplexDouble>, NdArray<ComplexDouble>>);
    m.def("multiply", &FunctionsInterface::multiply<NdArray<ComplexDouble>, ComplexDouble>);
    m.def("multiply", &FunctionsInterface::multiply<ComplexDouble, NdArray<ComplexDouble>>);
    m.def("multiply", &FunctionsInterface::multiply<NdArray<double>, NdArray<ComplexDouble>>);
    m.def("multiply", &FunctionsInterface::multiply<NdArray<ComplexDouble>, NdArray<double>>);
    m.def("multiply", &FunctionsInterface::multiply<NdArray<double>, ComplexDouble>);
    m.def("multiply", &FunctionsInterface::multiply<ComplexDouble, NdArray<double>>);
    m.def("multiply", &FunctionsInterface::multiply<NdArray<ComplexDouble>, double>);
    m.def("multiply", &FunctionsInterface::multiply<double, NdArray<ComplexDouble>>);

    m.def("nan_to_num", &FunctionsInterface::nan_to_num<double>);
    m.def("nanargmax", &nanargmax<double>);
    m.def("nanargmin", &nanargmin<double>);
    m.def("nancumprod", &nancumprod<double>);
    m.def("nancumsum", &nancumsum<double>);
    m.def("nanmax", &nanmax<double>);
    m.def("nanmean", &nanmean<double>);
    m.def("nanmedian", &nanmedian<double>);
    m.def("nanmin", &nanmin<double>);
    m.def("nanpercentile", &nanpercentile<double>);
    m.def("nanprod", &nanprod<double>);
    m.def("nansSquare", &FunctionsInterface::nansSquare);
    m.def("nansRowCol", &FunctionsInterface::nansRowCol);
    m.def("nansShape", &FunctionsInterface::nansShape);
    m.def("nansList", &FunctionsInterface::nansList);
    m.def("nans_like", &nans_like<double>);
    m.def("nanstdev", &nanstdev<double>);
    m.def("nansum", &nansum<double>);
    m.def("nanvar", &nanvar<double>);
    m.def("nbytes", &nbytes<double>);
    m.def("nbytes", &nbytes<ComplexDouble>);
    m.def("negative", &negative<double>);
    m.def("negative", &negative<ComplexDouble>);
    m.def("newbyteorderScalar", &FunctionsInterface::newbyteorderScalar<uint32>);
    m.def("newbyteorderArray", &FunctionsInterface::newbyteorderArray<uint32>);
    m.def("nth_rootScalar", &FunctionsInterface::nth_rootScalar<double, double>);
    m.def("nth_rootArray", &FunctionsInterface::nth_rootArray<double, double>);
    m.def("none", &FunctionsInterface::noneArray<double>);
    m.def("none", &FunctionsInterface::noneArray<ComplexDouble>);
    m.def("nonzero", &nonzero<double>);
    m.def("nonzero", &nonzero<ComplexDouble>);
    NdArray<double> (*normDouble)(const NdArray<double>&, Axis) = &norm<double>;
    m.def("norm", normDouble);
    NdArray<ComplexDouble> (*normComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &norm<double>;
    m.def("norm", normComplexDouble);
    NdArray<double> (*normalizeDouble)(const NdArray<double>&, Axis) = &normalize<double>;
    m.def("normalize", normalizeDouble);
    NdArray<ComplexDouble> (*normalizeComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &normalize<double>;
    m.def("normalize", normalizeComplexDouble);
    m.def("not_equal", &not_equal<double>);
    m.def("not_equal", &not_equal<ComplexDouble>);

    m.def("onesSquare", &FunctionsInterface::onesSquare<double>);
    m.def("onesSquareComplex", &FunctionsInterface::onesSquare<ComplexDouble>);
    m.def("onesRowCol", &FunctionsInterface::onesRowCol<double>);
    m.def("onesRowColComplex", &FunctionsInterface::onesRowCol<ComplexDouble>);
    m.def("onesShape", &FunctionsInterface::onesShape<double>);
    m.def("onesShapeComplex", &FunctionsInterface::onesShape<ComplexDouble>);
    m.def("ones_like", &ones_like<double, double>);
    m.def("ones_likeComplex", &ones_like<ComplexDouble, double>);
    m.def("outer", &FunctionsInterface::outer<double>);
    m.def("outer", &FunctionsInterface::outer<ComplexDouble>);

    m.def("packbitsLittleEndian", &FunctionsInterface::packbitsLittleEndian<bool>);
    m.def("packbitsLittleEndian", &FunctionsInterface::packbitsLittleEndian<uint8>);
    m.def("packbitsBigEndian", &FunctionsInterface::packbitsBigEndian<bool>);
    m.def("packbitsBigEndian", &FunctionsInterface::packbitsBigEndian<uint8>);
    m.def("pad", &pad<double>);
    m.def("pad", &pad<ComplexDouble>);
    m.def("partition", &partition<double>);
    m.def("partition", &partition<ComplexDouble>);
    m.def("percentile", &percentile<double>);
    m.def("place", &place<double>);
    m.def("polarScalar", &FunctionsInterface::polarScalar<double>);
    m.def("polarArray", &FunctionsInterface::polarArray<double>);
    m.def("powerArrayScalar", &FunctionsInterface::powerArrayScalar<double>);
    m.def("powerArrayArray", &FunctionsInterface::powerArrayArray<double>);
    m.def("powerArrayScalar", &FunctionsInterface::powerArrayScalar<ComplexDouble>);
    m.def("powerArrayArray", &FunctionsInterface::powerArrayArray<ComplexDouble>);
    m.def("powerfArrayScalar", &FunctionsInterface::powerfArrayScalar<double>);
    m.def("powerfArrayArray", &FunctionsInterface::powerfArrayArray<double>);
    m.def("powerfArrayScalar", &FunctionsInterface::powerfArrayScalar<ComplexDouble>);
    m.def("powerfArrayArray", &FunctionsInterface::powerfArrayArray<ComplexDouble>);
    m.def("prod", &prod<double>);
    m.def("prod", &prod<ComplexDouble>);
    m.def("projScalar", &FunctionsInterface::projScalar<double>);
    m.def("projArray", &FunctionsInterface::projArray<double>);
    m.def("ptp", &ptp<double>);
    m.def("ptp", &ptp<ComplexDouble>);
    m.def("put", &FunctionsInterface::putFlat<double>);
    m.def("put", &FunctionsInterface::putFlat<ComplexDouble>);
    m.def("put", &FunctionsInterface::putRowCol<double>);
    m.def("put", &FunctionsInterface::putRowCol<ComplexDouble>);
    m.def("put", &FunctionsInterface::putIndices1DValue<double>);
    m.def("put", &FunctionsInterface::putIndices1DValue<ComplexDouble>);
    m.def("put", &FunctionsInterface::putIndices1DValues<double>);
    m.def("put", &FunctionsInterface::putIndices1DValues<ComplexDouble>);
    m.def("put", &FunctionsInterface::putSlice1DValue<double>);
    m.def("put", &FunctionsInterface::putSlice1DValue<ComplexDouble>);
    m.def("put", &FunctionsInterface::putSlice1DValues<double>);
    m.def("put", &FunctionsInterface::putSlice1DValues<ComplexDouble>);
    m.def("put", &FunctionsInterface::putIndices2DValue<double>);
    m.def("put", &FunctionsInterface::putIndices2DValue<ComplexDouble>);
    m.def("put", &FunctionsInterface::putRowIndicesColSliceValue<double>);
    m.def("put", &FunctionsInterface::putRowIndicesColSliceValue<ComplexDouble>);
    m.def("put", &FunctionsInterface::putRowSliceColIndicesValue<double>);
    m.def("put", &FunctionsInterface::putRowSliceColIndicesValue<ComplexDouble>);
    m.def("put", &FunctionsInterface::putSlice2DValue<double>);
    m.def("put", &FunctionsInterface::putSlice2DValue<ComplexDouble>);
    m.def("put", &FunctionsInterface::putIndices2DValueRow<double>);
    m.def("put", &FunctionsInterface::putIndices2DValueRow<ComplexDouble>);
    m.def("put", &FunctionsInterface::putSlice2DValueRow<double>);
    m.def("put", &FunctionsInterface::putSlice2DValueRow<ComplexDouble>);
    m.def("put", &FunctionsInterface::putIndices2DValueCol<double>);
    m.def("put", &FunctionsInterface::putIndices2DValueCol<ComplexDouble>);
    m.def("put", &FunctionsInterface::putSlice2DValueCol<double>);
    m.def("put", &FunctionsInterface::putSlice2DValueCol<ComplexDouble>);
    m.def("put", &FunctionsInterface::putIndices2DValues<double>);
    m.def("put", &FunctionsInterface::putIndices2DValues<ComplexDouble>);
    m.def("put", &FunctionsInterface::putRowIndicesColSliceValues<double>);
    m.def("put", &FunctionsInterface::putRowIndicesColSliceValues<ComplexDouble>);
    m.def("put", &FunctionsInterface::putRowSliceColIndicesValues<double>);
    m.def("put", &FunctionsInterface::putRowSliceColIndicesValues<ComplexDouble>);
    m.def("put", &FunctionsInterface::putSlice2DValues<double>);
    m.def("put", &FunctionsInterface::putSlice2DValues<ComplexDouble>);
    m.def("put", &FunctionsInterface::putIndices2DValuesRow<double>);
    m.def("put", &FunctionsInterface::putIndices2DValuesRow<ComplexDouble>);
    m.def("put", &FunctionsInterface::putSlice2DValuesRow<double>);
    m.def("put", &FunctionsInterface::putSlice2DValuesRow<ComplexDouble>);
    m.def("put", &FunctionsInterface::putIndices2DValuesCol<double>);
    m.def("put", &FunctionsInterface::putIndices2DValuesCol<ComplexDouble>);
    m.def("put", &FunctionsInterface::putSlice2DValuesCol<double>);
    m.def("put", &FunctionsInterface::putSlice2DValuesCol<ComplexDouble>);
    m.def("putmask", &FunctionsInterface::putmask<double>);
    m.def("putmaskScalar", &FunctionsInterface::putmaskScalar<double>);

    m.def("rad2degScalar", &FunctionsInterface::rad2degScalar<double>);
    m.def("rad2degArray", &FunctionsInterface::rad2degArray<double>);
    m.def("radiansScalar", &FunctionsInterface::radiansScalar<double>);
    m.def("radiansArray", &FunctionsInterface::radiansArray<double>);
    m.def("ravel", &FunctionsInterface::ravel<double>, python_interface::return_value_policy::reference);
    m.def("reciprocal", &FunctionsInterface::reciprocal<double>);
    m.def("reciprocal", &FunctionsInterface::reciprocal<ComplexDouble>);
    m.def("realScalar", &FunctionsInterface::realScalar<double>);
    m.def("realArray", &FunctionsInterface::realArray<double>);
    m.def("remainderScalar", &FunctionsInterface::remainderScalar<double>);
    m.def("remainderArray", &FunctionsInterface::remainderArray<double>);
    m.def("replace", &FunctionsInterface::replace<double>);
    m.def("replace", &FunctionsInterface::replace<ComplexDouble>);
    m.def("reshape", &FunctionsInterface::reshapeInt<double>, python_interface::return_value_policy::reference);
    m.def("reshape", &FunctionsInterface::reshapeShape<double>, python_interface::return_value_policy::reference);
    m.def("reshape", &FunctionsInterface::reshapeValues<double>, python_interface::return_value_policy::reference);
    m.def("reshapeList", &FunctionsInterface::reshapeList<double>, python_interface::return_value_policy::reference);
    m.def("resizeFast", &FunctionsInterface::resizeFast<double>, python_interface::return_value_policy::reference);
    m.def("resizeFastList",
          &FunctionsInterface::resizeFastList<double>,
          python_interface::return_value_policy::reference);
    m.def("resizeSlow", &FunctionsInterface::resizeSlow<double>, python_interface::return_value_policy::reference);
    m.def("resizeSlowList",
          &FunctionsInterface::resizeSlowList<double>,
          python_interface::return_value_policy::reference);
    m.def("right_shift", &right_shift<uint32>);
    m.def("rintScalar", &FunctionsInterface::rintScalar<double>);
    m.def("rintArray", &FunctionsInterface::rintArray<double>);
    NdArray<double> (*rmsDouble)(const NdArray<double>&, Axis) = &rms<double>;
    m.def("rms", rmsDouble);
    NdArray<ComplexDouble> (*rmsComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &rms<double>;
    m.def("rms", rmsComplexDouble);
    m.def("roll", &roll<double>);
    m.def("rot90", &rot90<double>);
    m.def("roundScalar", &FunctionsInterface::roundScalar<double>);
    m.def("roundArray", &FunctionsInterface::roundArray<double>);
    m.def("row_stack", &FunctionsInterface::row_stack<double>);
    m.def("row_stack_vec", &FunctionsInterface::row_stack_vec<double>);

    m.def("searchsorted", &FunctionsInterface::searchsortedScalar<uint32>);
    m.def("searchsorted", &FunctionsInterface::searchsorted<uint32>);
    m.def("select", &FunctionsInterface::select<double>);
    m.def("selectVector", &FunctionsInterface::selectVector<double>);
    m.def("select", &FunctionsInterface::selectInitializerList<double>);
    m.def("setdiff1d", &setdiff1d<uint32>);
    m.def("setdiff1d", &setdiff1d<std::complex<double>>);
    m.def("signScalar", &FunctionsInterface::signScalar<double>);
    m.def("signScalar", &FunctionsInterface::signScalar<ComplexDouble>);
    m.def("signArray", &FunctionsInterface::signArray<double>);
    m.def("signArray", &FunctionsInterface::signArray<ComplexDouble>);
    m.def("signbitScalar", &FunctionsInterface::signbitScalar<double>);
    m.def("signbitArray", &FunctionsInterface::signbitArray<double>);
    m.def("sinScalar", &FunctionsInterface::sinScalar<double>);
    m.def("sinScalar", &FunctionsInterface::sinScalar<ComplexDouble>);
    m.def("sinArray", &FunctionsInterface::sinArray<double>);
    m.def("sinArray", &FunctionsInterface::sinArray<ComplexDouble>);
    m.def("sincScalar", &FunctionsInterface::sincScalar<double>);
    m.def("sincArray", &FunctionsInterface::sincArray<double>);
    m.def("sinhScalar", &FunctionsInterface::sinhScalar<ComplexDouble>);
    m.def("sinhScalar", &FunctionsInterface::sinhScalar<double>);
    m.def("sinhArray", &FunctionsInterface::sinhArray<double>);
    m.def("sinhArray", &FunctionsInterface::sinhArray<ComplexDouble>);
    m.def("size", &size<double>);
    m.def("sort", &sort<double>);
    m.def("sort", &sort<ComplexDouble>);
    m.def("split", &FunctionsInterface::split<double>);
    m.def("sqrtScalar", &FunctionsInterface::sqrtScalar<double>);
    m.def("sqrtScalar", &FunctionsInterface::sqrtScalar<ComplexDouble>);
    m.def("sqrtArray", &FunctionsInterface::sqrtArray<double>);
    m.def("sqrtArray", &FunctionsInterface::sqrtArray<ComplexDouble>);
    m.def("squareScalar", &FunctionsInterface::squareScalar<double>);
    m.def("squareScalar", &FunctionsInterface::squareScalar<ComplexDouble>);
    m.def("squareArray", &FunctionsInterface::squareArray<double>);
    m.def("squareArray", &FunctionsInterface::squareArray<ComplexDouble>);
    m.def("stack", &FunctionsInterface::stack<double>);
    m.def("stack_vec", &FunctionsInterface::stack_vec<double>);
    NdArray<double> (*stdevDouble)(const NdArray<double>&, Axis) = &stdev<double>;
    m.def("stdev", stdevDouble);
    NdArray<ComplexDouble> (*stdevComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &stdev<double>;
    m.def("stdev", stdevComplexDouble);
    m.def("subtract", &FunctionsInterface::subtract<NdArray<double>, NdArray<double>>);
    m.def("subtract", &FunctionsInterface::subtract<NdArray<double>, double>);
    m.def("subtract", &FunctionsInterface::subtract<double, NdArray<double>>);
    m.def("subtract", &FunctionsInterface::subtract<NdArray<ComplexDouble>, NdArray<ComplexDouble>>);
    m.def("subtract", &FunctionsInterface::subtract<NdArray<ComplexDouble>, ComplexDouble>);
    m.def("subtract", &FunctionsInterface::subtract<ComplexDouble, NdArray<ComplexDouble>>);
    m.def("subtract", &FunctionsInterface::subtract<NdArray<double>, NdArray<ComplexDouble>>);
    m.def("subtract", &FunctionsInterface::subtract<NdArray<ComplexDouble>, NdArray<double>>);
    m.def("subtract", &FunctionsInterface::subtract<NdArray<double>, ComplexDouble>);
    m.def("subtract", &FunctionsInterface::subtract<ComplexDouble, NdArray<double>>);
    m.def("subtract", &FunctionsInterface::subtract<NdArray<ComplexDouble>, double>);
    m.def("subtract", &FunctionsInterface::subtract<double, NdArray<ComplexDouble>>);
    m.def("sum", &sum<double>);
    m.def("sum", &sum<ComplexDouble>);
    m.def("swap", &nc::swap<double>);
    m.def("swapaxes", &swapaxes<double>);
    m.def("swapRows", &nc::swapRows<double>);
    m.def("swapCols", &nc::swapCols<double>);

    m.def("take", &FunctionsInterface::take<double>);
    m.def("tanScalar", &FunctionsInterface::tanScalar<double>);
    m.def("tanScalar", &FunctionsInterface::tanScalar<ComplexDouble>);
    m.def("tanArray", &FunctionsInterface::tanArray<double>);
    m.def("tanArray", &FunctionsInterface::tanArray<ComplexDouble>);
    m.def("tanhScalar", &FunctionsInterface::tanhScalar<double>);
    m.def("tanhScalar", &FunctionsInterface::tanhScalar<ComplexDouble>);
    m.def("tanhArray", &FunctionsInterface::tanhArray<double>);
    m.def("tanhArray", &FunctionsInterface::tanhArray<ComplexDouble>);
    m.def("tileRectangle", &FunctionsInterface::tileRectangle<double>);
    m.def("tileShape", &FunctionsInterface::tileShape<double>);
    m.def("tileList", &FunctionsInterface::tileList<double>);
    m.def("tofile", &FunctionsInterface::tofileBinary<double>);
    m.def("tofile", &FunctionsInterface::tofileTxt<double>);
    m.def("toStlVector", &toStlVector<double>);
    m.def("trace", &trace<double>);
    m.def("trace", &trace<ComplexDouble>);
    m.def("transpose", &transpose<double>);
    m.def("trapzDx", &FunctionsInterface::trapzDx<double>);
    m.def("trapz", &FunctionsInterface::trapz<double>);
    m.def("trilSquare", &FunctionsInterface::trilSquare<double>);
    m.def("trilSquareComplex", &FunctionsInterface::trilSquare<ComplexDouble>);
    m.def("trilRect", &FunctionsInterface::trilRect<double>);
    m.def("trilRectComplex", &FunctionsInterface::trilRect<ComplexDouble>);
    m.def("trilArray", &FunctionsInterface::trilArray<double>);
    m.def("trilArray", &FunctionsInterface::trilArray<ComplexDouble>);
    m.def("triuSquare", &FunctionsInterface::triuSquare<double>);
    m.def("triuSquareComplex", &FunctionsInterface::triuSquare<ComplexDouble>);
    m.def("triuRect", &FunctionsInterface::triuRect<double>);
    m.def("triuRectComplex", &FunctionsInterface::triuRect<ComplexDouble>);
    m.def("triuArray", &FunctionsInterface::triuArray<double>);
    m.def("triuArray", &FunctionsInterface::triuArray<ComplexDouble>);
    m.def("trim_zeros", &trim_zeros<double>);
    m.def("trim_zeros", &trim_zeros<ComplexDouble>);
    m.def("truncScalar", &FunctionsInterface::truncScalar<double>);
    m.def("truncArray", &FunctionsInterface::truncArray<double>);

    m.def("union1d", &union1d<uint32>);
    m.def("union1d", &union1d<std::complex<double>>);
    m.def("unique", &unique<uint32>);
    m.def("unique", &unique<std::complex<double>>);
    m.def("unpackbitsBigEndian", &FunctionsInterface::unpackbitsBigEndian);
    m.def("unpackbitsLittleEndian", &FunctionsInterface::unpackbitsLittleEndian);
    m.def("unwrapScalar", &FunctionsInterface::unwrapScalar<double>);
    m.def("unwrapArray", &FunctionsInterface::unwrapArray<double>);

    pbArrayGeneric (*vanderDouble)(const NdArray<double>&, Increasing increasing) = &FunctionsInterface::vander<double>;
    m.def("vander", vanderDouble);
    pbArrayGeneric (*vanderComplexDouble)(const NdArray<ComplexDouble>&, Increasing increasing) =
        &FunctionsInterface::vander<ComplexDouble>;
    m.def("vander", vanderComplexDouble);
    pbArrayGeneric (*vanderNDouble)(const NdArray<double>&, uint32 n, Increasing increasing) =
        &FunctionsInterface::vander<double>;
    m.def("vander", vanderNDouble);
    pbArrayGeneric (*vanderNComplexDouble)(const NdArray<ComplexDouble>&, uint32 n, Increasing increasing) =
        &FunctionsInterface::vander<ComplexDouble>;
    m.def("vander", vanderNComplexDouble);
    NdArray<double> (*varDouble)(const NdArray<double>&, Axis) = &var<double>;
    m.def("var", varDouble);
    NdArray<ComplexDouble> (*varComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &var<double>;
    m.def("var", varComplexDouble);
    m.def("vsplit", &FunctionsInterface::vsplit<double>);
    m.def("vstack", &FunctionsInterface::vstack<double>);
    m.def("vstack_vec", &FunctionsInterface::vstack_vec<double>);

    m.def("where", &FunctionsInterface::whereArrayArray<double>);
    m.def("where", &FunctionsInterface::whereArrayArray<ComplexDouble>);
    m.def("where", &FunctionsInterface::whereArrayScalar<double>);
    m.def("where", &FunctionsInterface::whereArrayScalar<ComplexDouble>);
    m.def("where", &FunctionsInterface::whereScalarArray<double>);
    m.def("where", &FunctionsInterface::whereScalarArray<ComplexDouble>);
    m.def("where", &FunctionsInterface::whereScalarScalar<double>);
    m.def("where", &FunctionsInterface::whereScalarScalar<ComplexDouble>);
    m.def("wrap", [](double value) { return nc::wrap(value); });
    m.def("wrap", [](const NdArray<double>& values) { return nc::wrap(values); });
    m.def("wrap2Pi", [](double value) { return nc::wrap2Pi(value); });
    m.def("wrap2Pi", [](const NdArray<double>& values) { return nc::wrap2Pi(values); });

    m.def("zerosSquare", &FunctionsInterface::zerosSquare<double>);
    m.def("zerosSquareComplex", &FunctionsInterface::zerosSquare<ComplexDouble>);
    m.def("zerosRowCol", &FunctionsInterface::zerosRowCol<double>);
    m.def("zerosRowColComplex", &FunctionsInterface::zerosRowCol<ComplexDouble>);
    m.def("zerosShape", &FunctionsInterface::zerosShape<double>);
    m.def("zerosShapeComplex", &FunctionsInterface::zerosShape<ComplexDouble>);
    m.def("zerosList", &FunctionsInterface::zerosList<double>);
    m.def("zerosListComplex", &FunctionsInterface::zerosList<ComplexDouble>);
    m.def("zeros_like", &zeros_like<double, double>);
    m.def("zeros_likeComplex", &zeros_like<ComplexDouble, double>);
}