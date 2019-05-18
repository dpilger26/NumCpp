/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
/// Copyright 2019 David Pilger
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy of this
/// software and associated documentation files(the "Software"), to deal in the Software
/// without restriction, including without limitation the rights to use, copy, modify,
/// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
/// permit persons to whom the Software is furnished to do so, subject to the following
/// conditions :
///
/// The above copyright notice and this permission notice shall be included in all copies
/// or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
/// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
/// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
/// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
/// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
/// DEALINGS IN THE SOFTWARE.
///
/// @section Description
/// Methods for working with NdArrays
///
#pragma once

#include"NumCpp/Constants.hpp"
#include"NumCpp/DtypeInfo.hpp"
#include"NumCpp/NdArray.hpp"
#include"NumCpp/Polynomial.hpp"
#include"NumCpp/Types.hpp"

#include"boost/algorithm/clamp.hpp"
#include"boost/filesystem.hpp"
#include"boost/integer/common_factor_rt.hpp"
#include"boost/math/special_functions/erf.hpp"

#include<algorithm>
#include<bitset>
#include<cmath>
#include<fstream>
#include<functional>
#include<initializer_list>
#include<iostream>
#include<set>
#include<sstream>
#include<stdexcept>
#include<string>
#include<utility>
#include<vector>

namespace nc
{
    // forward declare all functions
    template<typename dtype>
    dtype abs(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<dtype> abs(const NdArray<dtype>& inArray);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> add(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    uint32 alen(const NdArray<dtype>& inArray) noexcept;

    template<typename dtype>
    NdArray<bool> all(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    bool allclose(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, double inTolerance = 1e-5);

    template<typename dtype>
    NdArray<dtype> amax(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> amin(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<bool> any(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> applyFunction(const NdArray<dtype>& inArray, const std::function<dtype(dtype)>& inFunc);

    template<typename dtype>
    NdArray<dtype> append(const NdArray<dtype>& inArray, const NdArray<dtype>& inAppendValues, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> arange(dtype inStart, dtype inStop, dtype inStep = 1);

    template<typename dtype>
    NdArray<dtype> arange(dtype inStop);

    template<typename dtype>
    NdArray<dtype> arange(const Slice& inSlice);

    template<typename dtype>
    double arccos(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> arccos(const NdArray<dtype>& inArray);

    template<typename dtype>
    double arccosh(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> arccosh(const NdArray<dtype>& inArray);

    template<typename dtype>
    double arcsin(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> arcsin(const NdArray<dtype>& inArray);

    template<typename dtype>
    double arcsinh(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> arcsinh(const NdArray<dtype>& inArray);

    template<typename dtype>
    double arctan(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> arctan(const NdArray<dtype>& inArray);

    template<typename dtype>
    double arctan2(dtype inY, dtype inX) noexcept;

    template<typename dtype>
    NdArray<double> arctan2(const NdArray<dtype>& inY, const NdArray<dtype>& inX);

    template<typename dtype>
    double arctanh(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> arctanh(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<uint32> argmax(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<uint32> argmin(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<uint32> argsort(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<uint32> argwhere(const NdArray<dtype>& inArray);

    template<typename dtype>
    dtype around(dtype inValue, uint8 inNumDecimals = 0);

    template<typename dtype>
    NdArray<dtype> around(const NdArray<dtype>& inArray, uint8 inNumDecimals = 0);

    template<typename dtype>
    bool array_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    bool array_equiv(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> asarray(const std::vector<dtype>& inVector);

    template<typename dtype>
    NdArray<dtype> asarray(std::initializer_list<dtype>& inList);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> astype(const NdArray<dtype> inArray);

    template<typename dtype>
    NdArray<double> average(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<double> average(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, Axis inAxis = Axis::NONE);

    template<typename dtype>
    std::string binaryRepr(dtype inValue);

    template<typename dtype>
    NdArray<dtype> bincount(const NdArray<dtype>& inArray, uint16 inMinLength = 0);

    template<typename dtype>
    NdArray<dtype> bincount(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0);

    template<typename dtype>
    NdArray<dtype> bitwise_and(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> bitwise_not(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<dtype> bitwise_or(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> bitwise_xor(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> byteswap(const NdArray<dtype>& inArray);

    template<typename dtype>
    double cbrt(dtype inValue);

    template<typename dtype>
    NdArray<double> cbrt(const NdArray<dtype>& inArray);

    template<typename dtype>
    dtype ceil(dtype inValue);

    template<typename dtype>
    NdArray<dtype> ceil(const NdArray<dtype>& inArray);

    template<typename dtype>
    dtype clip(dtype inValue, dtype inMinValue, dtype inMaxValue);

    template<typename dtype>
    NdArray<dtype> clip(const NdArray<dtype>& inArray, dtype inMinValue, dtype inMaxValue);

    template<typename dtype>
    NdArray<dtype> column_stack(const std::initializer_list<NdArray<dtype> >& inArrayList);

    template<typename dtype>
    NdArray<dtype> concatenate(const std::initializer_list<NdArray<dtype> >& inArrayList, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<bool> contains(const NdArray<dtype>& inArray, dtype inValue, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> copy(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<dtype> copySign(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype>& copyto(NdArray<dtype>& inDestArray, const NdArray<dtype>& inSrcArray);

    template<typename dtype>
    double cos(dtype inValue);

    template<typename dtype>
    NdArray<double> cos(const NdArray<dtype>& inArray);

    template<typename dtype>
    double cosh(dtype inValue);

    template<typename dtype>
    NdArray<double> cosh(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<uint32> count_nonzero(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> cross(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, Axis inAxis = Axis::NONE);

    template<typename dtypeOut, typename dtype>
    dtypeOut cube(dtype inValue);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> cube(const NdArray<dtype>& inArray);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> cumprod(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> cumsum(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    double deg2rad(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> deg2rad(const NdArray<dtype>& inArray = Axis::NONE);

    template<typename dtype>
    double degrees(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> degrees(const NdArray<dtype>& inArray = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, const NdArray<uint32>& inArrayIdxs, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, const Slice& inIndicesSlice, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, uint32 inIndex, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> diagflat(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<dtype> diagonal(const NdArray<dtype>& inArray, int32 inOffset = 0, Axis inAxis = Axis::ROW);

    template<typename dtype>
    NdArray<dtype> diff(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> divide(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> dot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    void dump(const NdArray<dtype>& inArray, const std::string& inFilename);

    template<typename dtype>
    NdArray<dtype> empty(uint32 inNumRows, uint32 inNumCols);

    template<typename dtype>
    NdArray<dtype> empty(const Shape& inShape);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> empty_like(const NdArray<dtype>& inArray);

    template<typename dtype>
    Endian endianess(const NdArray<dtype>& inArray) noexcept;

    template<typename dtype>
    double erf(dtype inValue);

    template<typename dtype>
    NdArray<double> erf(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<bool> equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    double exp(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> exp(const NdArray<dtype>& inArray);

    template<typename dtype>
    double exp2(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> exp2(const NdArray<dtype>& inArray);

    template<typename dtype>
    double expm1(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> expm1(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<dtype> eye(uint32 inN, int32 inK = 0);

    template<typename dtype>
    NdArray<dtype> eye(uint32 inN, uint32 inM, int32 inK = 0);

    template<typename dtype>
    NdArray<dtype> eye(const Shape& inShape, int32 inK = 0);

    template<typename dtype>
    void fillDiagonal(NdArray<dtype>& inArray, dtype inValue) noexcept;

    template<typename dtype>
    dtype fix(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<dtype> fix(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<dtype> flatten(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<uint32> flatnonzero(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<dtype> flip(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> fliplr(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<dtype> flipud(const NdArray<dtype>& inArray);

    template<typename dtype>
    dtype floor(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<dtype> floor(const NdArray<dtype>& inArray);

    template<typename dtype>
    dtype floor_divide(dtype inValue1, dtype inValue2) noexcept;

    template<typename dtype>
    NdArray<dtype> floor_divide(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    dtype fmax(dtype inValue1, dtype inValue2);

    template<typename dtype>
    NdArray<dtype> fmax(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    dtype fmin(dtype inValue1, dtype inValue2);

    template<typename dtype>
    NdArray<dtype> fmin(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    dtype fmod(dtype inValue1, dtype inValue2) noexcept;

    template<typename dtype>
    NdArray<dtype> fmod(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> frombuffer(char* inBufferPtr, uint32 inNumBytes);

    template<typename dtype>
    NdArray<dtype> fromfile(const std::string& inFilename, const std::string& inSep = "");

    template<typename dtype, typename Iter>
    NdArray<dtype> fromiter(Iter inBegin, Iter inEnd);

    template<typename dtype>
    NdArray<dtype> full(uint32 inSquareSize, dtype inFillValue);

    template<typename dtype>
    NdArray<dtype> full(uint32 inNumRows, uint32 inNumCols, dtype inFillValue);

    template<typename dtype>
    NdArray<dtype> full(const Shape& inShape, dtype inFillValue);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> full_like(const NdArray<dtype>& inArray, dtype inFillValue);

    template<typename dtype>
    dtype gcd(dtype inValue1, dtype inValue2) noexcept;

    template<typename dtype>
    dtype gcd(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<double> gradient(const NdArray<dtype>& inArray, Axis inAxis = Axis::ROW);

    template<typename dtype>
    NdArray<bool> greater(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<bool> greater_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    std::pair<NdArray<uint32>, NdArray<double> > histogram(const NdArray<dtype>& inArray, uint32 inNumBins = 10);

    template<typename dtype>
    NdArray<dtype> hstack(const std::initializer_list<NdArray<dtype> >& inArrayList);

    template<typename dtypeOut, typename dtype>
    dtypeOut hypot(dtype inValue1, dtype inValue2) noexcept;

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> hypot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> identity(uint32 inSquareSize);

    template<typename dtype>
    double interp(dtype inValue1, dtype inValue2, double inPercent);

    template<typename dtype>
    NdArray<dtype> interp(const NdArray<dtype>& inX, const NdArray<dtype>& inXp, const NdArray<dtype>& inFp);

    template<typename dtype>
    NdArray<dtype> intersect1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> invert(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<bool> isclose(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        double inRtol = 1e-05, double inAtol = 1e-08);

    template<typename dtype>
    bool isinf(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<bool> isinf(const NdArray<dtype>& inArray);

    template<typename dtype>
    bool isnan(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<bool> isnan(const NdArray<dtype>& inArray);

    template<typename dtype>
    dtype lcm(dtype inValue1, dtype inValue2) noexcept;

    template<typename dtype>
    NdArray<dtype> lcm(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    dtype ldexp(dtype inValue1, uint8 inValue2) noexcept;

    template<typename dtype>
    NdArray<dtype> ldexp(const NdArray<dtype>& inArray1, const NdArray<uint8>& inArray2);

    template<typename dtype>
    NdArray<dtype> left_shift(const NdArray<dtype>& inArray, uint8 inNumBits);

    template<typename dtype>
    NdArray<bool> less(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<bool> less_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> linspace(dtype inStart, dtype inStop, uint32 inNum = 50, bool endPoint = true);

    template<typename dtype>
    NdArray<dtype> load(const std::string& inFilename);

    template<typename dtype>
    double log(dtype inValue);

    template<typename dtype>
    NdArray<double> log(const NdArray<dtype>& inArray);

    template<typename dtype>
    double log10(dtype inValue);

    template<typename dtype>
    NdArray<double> log10(const NdArray<dtype>& inArray);

    template<typename dtype>
    double log1p(dtype inValue);

    template<typename dtype>
    NdArray<double> log1p(const NdArray<dtype>& inArray);

    template<typename dtype>
    double log2(dtype inValue);

    template<typename dtype>
    NdArray<double> log2(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<bool> logical_and(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<bool> logical_not(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<bool> logical_or(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<bool> logical_xor(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> matmul(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> max(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> maximum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<double> mean(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> median(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    std::pair<NdArray<dtype>, NdArray<dtype> > meshgrid(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    std::pair<NdArray<dtype>, NdArray<dtype> > meshgrid(const Slice& inSlice1, const Slice& inSlice2);

    template<typename dtype>
    NdArray<dtype> min(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> minimum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> mod(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> multiply(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<uint32> nanargmax(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<uint32> nanargmin(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> nancumprod(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> nancumsum(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> nanmax(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<double> nanmean(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> nanmedian(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> nanmin(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtypeOut, typename dtype>
    NdArray<double> nanpercentile(const NdArray<dtype>& inArray, double inPercentile,
        Axis inAxis = Axis::NONE, const std::string& inInterpMethod = "linear");

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> nanprod(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> nans(uint32 inSquareSize);

    template<typename dtype>
    NdArray<dtype> nans(uint32 inNumRows, uint32 inNumCols);

    template<typename dtype>
    NdArray<dtype> nans(const Shape& inShape);

    template<typename dtype>
    NdArray<double> nans_like(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<double> nanstdev(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> nansum(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<double> nanvar(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    uint64 nbytes(const NdArray<dtype>& inArray) noexcept;

    template<typename dtype>
    dtype newbyteorder(dtype inValue, Endian inEndianess);

    template<typename dtype>
    NdArray<dtype> newbyteorder(const NdArray<dtype>& inArray, Endian inEndianess);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> negative(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<uint32> nonzero(const NdArray<dtype>& inArray);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> norm(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<bool> not_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> ones(uint32 inSquareSize);

    template<typename dtype>
    NdArray<dtype> ones(uint32 inNumRows, uint32 inNumCols);

    template<typename dtype>
    NdArray<dtype> ones(const Shape& inShape);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> ones_like(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<dtype> pad(const NdArray<dtype>& inArray, uint16 inPadWidth, dtype inPadValue);

    template<typename dtype>
    NdArray<dtype> partition(const NdArray<dtype>& inArray, uint32 inKth, Axis inAxis = Axis::NONE);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> percentile(const NdArray<dtype>& inArray, double inPercentile,
        Axis inAxis = Axis::NONE, const std::string& inInterpMethod = "linear");

    template<typename dtypeOut, typename dtype>
    dtypeOut power(dtype inValue, uint8 inExponent);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> power(const NdArray<dtype>& inArray, uint8 inExponent);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> power(const NdArray<dtype>& inArray, const NdArray<uint8>& inExponents);

    template<typename dtype>
    void print(const NdArray<dtype>& inArray);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> prod(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> ptp(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype>& put(NdArray<dtype>& inArray, const NdArray<uint32>& inIndices, const NdArray<dtype>& inValues);

    template<typename dtype>
    NdArray<dtype>& putmask(NdArray<dtype>& inArray, const NdArray<bool>& inMask, dtype inValue);

    template<typename dtype>
    NdArray<dtype>& putmask(NdArray<dtype>& inArray, const NdArray<bool>& inMask, const NdArray<dtype>& inValues);

    template<typename dtype>
    double rad2deg(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> rad2deg(const NdArray<dtype>& inArray);

    template<typename dtype>
    double radians(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> radians(const NdArray<dtype>& inArray);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> reciprocal(const NdArray<dtype>& inArray);

    template<typename dtypeOut, typename dtype>
    dtypeOut remainder(dtype inValue1, dtype inValue2) noexcept;

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> remainder(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> repeat(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols);

    template<typename dtype>
    NdArray<dtype> repeat(const NdArray<dtype>& inArray, const Shape& inRepeatShape);

    template<typename dtype>
    NdArray<dtype>& reshape(NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols);

    template<typename dtype>
    NdArray<dtype>& reshape(NdArray<dtype>& inArray, const Shape& inNewShape);

    template<typename dtype>
    NdArray<dtype>& resizeFast(NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols);

    template<typename dtype>
    NdArray<dtype>& resizeFast(NdArray<dtype>& inArray, const Shape& inNewShape);

    template<typename dtype>
    NdArray<dtype>& resizeSlow(NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols);

    template<typename dtype>
    NdArray<dtype>& resizeSlow(NdArray<dtype>& inArray, const Shape& inNewShape);

    template<typename dtype>
    NdArray<dtype> right_shift(const NdArray<dtype>& inArray, uint8 inNumBits);

    template<typename dtype>
    dtype rint(dtype inValue);

    template<typename dtype>
    NdArray<dtype> rint(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<double> rms(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> roll(const NdArray<dtype>& inArray, int32 inShift, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> rot90(const NdArray<dtype>& inArray, uint8 inK = 1);

    template<typename dtype>
    dtype round(dtype inValue, uint8 inDecimals = 0);

    template<typename dtype>
    NdArray<dtype> round(const NdArray<dtype>& inArray, uint8 inDecimals = 0);

    template<typename dtype>
    NdArray<dtype> row_stack(const std::initializer_list<NdArray<dtype> >& inArrayList);

    template<typename dtype>
    NdArray<dtype> setdiff1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    Shape shape(const NdArray<dtype>& inArray);

    template<typename dtype>
    int8 sign(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<int8> sign(const NdArray<dtype>& inArray);

    template<typename dtype>
    bool signbit(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<bool> signbit(const NdArray<dtype>& inArray);

    template<typename dtype>
    double sin(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> sin(const NdArray<dtype>& inArray);

    template<typename dtype>
    double sinc(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> sinc(const NdArray<dtype>& inArray);

    template<typename dtype>
    double sinh(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> sinh(const NdArray<dtype>& inArray);

    template<typename dtype>
    uint32 size(const NdArray<dtype>& inArray) noexcept;

    template<typename dtype>
    NdArray<dtype> sort(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    double sqrt(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> sqrt(const NdArray<dtype>& inArray);

    template<typename dtype>
    dtype square(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<dtype> square(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<dtype> stack(const std::initializer_list<NdArray<dtype> >& inArrayList, Axis inAxis = Axis::ROW);

    template<typename dtype>
    NdArray<double> stdev(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> sum(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> swapaxes(const NdArray<dtype>& inArray);

    template<typename dtype>
    double tan(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> tan(const NdArray<dtype>& inArray);

    template<typename dtype>
    double tanh(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<double> tanh(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<dtype> tile(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols);

    template<typename dtype>
    NdArray<dtype> tile(const NdArray<dtype>& inArray, const Shape& inReps);

    template<typename dtype>
    void tofile(const NdArray<dtype>& inArray, const std::string& inFilename, const std::string& inSep = "");

    template<typename dtype>
    std::vector<dtype> toStlVector(const NdArray<dtype>& inArray);

    template<typename dtypeOut, typename dtype>
    dtypeOut trace(const NdArray<dtype>& inArray, int16 inOffset = 0, Axis inAxis = Axis::ROW) noexcept;

    template<typename dtype>
    NdArray<dtype> transpose(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<double> trapz(const NdArray<dtype>& inArray, double dx = 1.0, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<double> trapz(const NdArray<dtype>& inArrayY, const NdArray<dtype>& inArrayX, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> tril(uint32 inN, int32 inOffset = 0);

    template<typename dtype>
    NdArray<dtype> tril(uint32 inN, uint32 inM, int32 inOffset = 0);

    template<typename dtype>
    NdArray<dtype> tril(const NdArray<dtype>& inArray, int32 inOffset = 0);

    template<typename dtype>
    NdArray<dtype> triu(uint32 inN, int32 inOffset = 0);

    template<typename dtype>
    NdArray<dtype> triu(uint32 inN, uint32 inM, int32 inOffset = 0);

    template<typename dtype>
    NdArray<dtype> triu(const NdArray<dtype>& inArray, int32 inOffset = 0);

    template<typename dtype>
    NdArray<dtype> trim_zeros(const NdArray<dtype>& inArray, const std::string inTrim = "fb");

    template<typename dtype>
    dtype trunc(dtype inValue);

    template<typename dtype>
    NdArray<dtype> trunc(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<dtype> union1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2);

    template<typename dtype>
    NdArray<dtype> unique(const NdArray<dtype>& inArray);

    template<typename dtype>
    dtype unwrap(dtype inValue) noexcept;

    template<typename dtype>
    NdArray<dtype> unwrap(const NdArray<dtype>& inArray);

    template<typename dtype>
    NdArray<double> var(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE);

    template<typename dtype>
    NdArray<dtype> vstack(const std::initializer_list<NdArray<dtype> >& inArrayList);

    template<typename dtype>
    NdArray<dtype> where(const NdArray<bool>& inMask, const NdArray<dtype>& inA, const NdArray<dtype>& inB);

    template<typename dtype>
    NdArray<dtype> zeros(uint32 inSquareSize);

    template<typename dtype>
    NdArray<dtype> zeros(uint32 inNumRows, uint32 inNumCols);

    template<typename dtype>
    NdArray<dtype> zeros(const Shape& inShape);

    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> zeros_like(const NdArray<dtype>& inArray);

    //============================================================================
    // Method Description:
    ///						Calculate the absolute value.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.absolute.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype abs(dtype inValue) noexcept
    {
        return std::abs(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Calculate the absolute value element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.absolute.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> abs(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> dtype { return std::abs(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Add arguments element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.add.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> add(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1.template astype<dtypeOut>() + inArray2.template astype<dtypeOut>();
    }

    //============================================================================
    // Method Description:
    ///						Return the length of the first dimension of the input array.
    ///
    /// @param
    ///				inArray
    /// @return
    ///				length uint16
    ///
    template<typename dtype>
    uint32 alen(const NdArray<dtype>& inArray) noexcept
    {
        return inArray.shape().rows;
    }

    //============================================================================
    // Method Description:
    ///						Test whether all array elements along a given axis evaluate to True.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.all.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				bool
    ///
    template<typename dtype>
    NdArray<bool> all(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.all(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Returns True if two arrays are element-wise equal within a tolerance.
    ///						inTolerance must be a positive number
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.allclose.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @param				inTolerance: (Optional, default 1e-5)
    /// @return
    ///				bool
    ///
    template<typename dtype>
    bool allclose(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, double inTolerance)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: allclose: input array dimensions are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        for (uint32 i = 0; i < inArray1.size(); ++i)
        {
            if (std::abs(inArray1[i] - inArray2[i]) > inTolerance)
            {
                return false;
            }
        }

        return true;
    }

    //============================================================================
    // Method Description:
    ///						Return the maximum of an array or maximum along an axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.amax.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				max value
    ///
    template<typename dtype>
    NdArray<dtype> amax(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.max(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Return the minimum of an array or minimum along an axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.amin.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				min value
    ///
    template<typename dtype>
    NdArray<dtype> amin(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.min(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Test whether any array element along a given axis evaluates to True.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.any.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> any(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.any(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Apply the input function element wise to the input
    ///                     array in place.
    ///
    /// @param				inArray
    /// @param				inFunc
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    void applyFunction(NdArray<dtype>& inArray, const std::function<dtype(dtype)>& inFunc)
    {
        std::transform(inArray.begin(), inArray.end(), inArray.begin(), inFunc);
    }

    //============================================================================
    // Method Description:
    ///						Apply polynomial elemnt wise to the input values.
    ///
    /// @param				inArray
    /// @param				inPoly
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    void applyPoly1d(NdArray<dtype>& inArray, const Poly1d<dtype>& inPoly)
    {
        applyFunction<dtype>(inArray, inPoly);
    }

    //============================================================================
    // Method Description:
    ///						Append values to the end of an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.append.html
    ///
    /// @param				inArray
    /// @param				inAppendValues
    /// @param				inAxis (Optional, default NONE): The axis along which values are appended.
    ///									If axis is not given, both inArray and inAppendValues
    ///									are flattened before use.
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> append(const NdArray<dtype>& inArray, const NdArray<dtype>& inAppendValues, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                NdArray<dtype> returnArray(1, inArray.size() + inAppendValues.size());
                std::copy(inArray.cbegin(), inArray.cend(), returnArray.begin());
                std::copy(inAppendValues.cbegin(), inAppendValues.cend(), returnArray.begin() + inArray.size());

                return returnArray;
            }
            case Axis::ROW:
            {
                const Shape inShape = inArray.shape();
                const Shape appendShape = inAppendValues.shape();
                if (inShape.cols != appendShape.cols)
                {
                    std::string errStr = "ERROR: append: all the input array dimensions except for the concatenation axis must match exactly";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                NdArray<dtype> returnArray(inShape.rows + appendShape.rows, inShape.cols);
                std::copy(inArray.cbegin(), inArray.cend(), returnArray.begin());
                std::copy(inAppendValues.cbegin(), inAppendValues.cend(), returnArray.begin() + inArray.size());

                return returnArray;
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();
                const Shape appendShape = inAppendValues.shape();
                if (inShape.rows != appendShape.rows)
                {
                    std::string errStr = "ERROR: append: all the input array dimensions except for the concatenation axis must match exactly";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                NdArray<dtype> returnArray(inShape.rows, inShape.cols + appendShape.cols);
                for (uint32 row = 0; row < returnArray.shape().rows; ++row)
                {
                    std::copy(inArray.cbegin(row), inArray.cend(row), returnArray.begin(row));
                    std::copy(inAppendValues.cbegin(row), inAppendValues.cend(row), returnArray.begin(row) + inShape.cols);
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtype>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Return evenly spaced values within a given interval.
    ///
    ///						Values are generated within the half - open interval[start, stop)
    ///						(in other words, the interval including start but excluding stop).
    ///						For integer arguments the function is equivalent to the Python built - in
    ///						range function, but returns an ndarray rather than a list.
    ///
    ///						When using a non - integer step, such as 0.1, the results will often
    ///						not be consistent.It is better to use linspace for these cases.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arange.html
    ///
    /// @param				inStart
    /// @param				inStop
    /// @param				inStep: (Optional, defaults to 1)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> arange(dtype inStart, dtype inStop, dtype inStep)
    {
        if (inStep > 0 && inStop < inStart)
        {
            std::string errStr = "ERROR: arange: stop value must be larger than the start value for positive step.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        if (inStep < 0 && inStop > inStart)
        {
            std::string errStr = "ERROR: arange: start value must be larger than the stop value for negative step.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        std::vector<dtype> values;

        dtype theValue = inStart;

        if (inStep > 0)
        {
            while (theValue < inStop)
            {
                values.push_back(theValue);
                theValue += inStep;
            }
        }
        else
        {
            while (theValue > inStop)
            {
                values.push_back(theValue);
                theValue += inStep;
            }
        }

        return NdArray<dtype>(values);
    }

    //============================================================================
    // Method Description:
    ///						Return evenly spaced values within a given interval.
    ///
    ///						Values are generated within the half - open interval[start, stop)
    ///						(in other words, the interval including start but excluding stop).
    ///						For integer arguments the function is equivalent to the Python built - in
    ///						range function, but returns an ndarray rather than a list.
    ///
    ///						When using a non - integer step, such as 0.1, the results will often
    ///						not be consistent.It is better to use linspace for these cases.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arange.html
    ///
    /// @param
    ///				inStop: start is 0 and step is 1
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> arange(dtype inStop)
    {
        if (inStop <= 0)
        {
            std::string errStr = "ERROR: arange: stop value must ge greater than 0.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        return arange<dtype>(0, inStop, 1);
    }

    //============================================================================
// Method Description:
///						Return evenly spaced values within a given interval.
///
///						Values are generated within the half - open interval[start, stop)
///						(in other words, the interval including start but excluding stop).
///						For integer arguments the function is equivalent to the Python built - in
///						range function, but returns an ndarray rather than a list.
///
///						When using a non - integer step, such as 0.1, the results will often
///						not be consistent.It is better to use linspace for these cases.
///
///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arange.html
///
/// @param
///				inSlice
/// @return
///				NdArray
///
    template<typename dtype>
    NdArray<dtype> arange(const Slice& inSlice)
    {
        return arange<dtype>(inSlice.start, inSlice.stop, inSlice.step);
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse cosine
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arccos.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double arccos(dtype inValue) noexcept
    {
        return std::acos(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse cosine, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arccos.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> arccos(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue)  noexcept -> double { return std::acos(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse hyperbolic cosine.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arccosh.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double arccosh(dtype inValue) noexcept
    {
        return std::acosh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse hyperbolic cosine, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arccosh.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> arccosh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return std::acosh(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse sine.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arcsin.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double arcsin(dtype inValue) noexcept
    {
        return std::asin(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse sine, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arcsin.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> arcsin(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue)  noexcept -> double { return std::asin(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse hyperbolic sine.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arcsinh.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double arcsinh(dtype inValue) noexcept
    {
        return std::asinh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse hyperbolic sine, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arcsinh.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> arcsinh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue)  noexcept-> double { return std::asinh(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse tangent.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arctan.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double arctan(dtype inValue) noexcept
    {
        return std::atan(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse tangent, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arctan.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> arctan(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue)  noexcept-> double { return std::atan(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse tangent.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arctan2.html
    ///
    /// @param				inY
    /// @param				inX
    /// @return
    ///				value
    ///
    template<typename dtype>
    double arctan2(dtype inY, dtype inX) noexcept
    {
        return std::atan2(static_cast<double>(inY), static_cast<double>(inX));
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse tangent, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arctan2.html
    ///
    /// @param				inY
    /// @param				inX
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> arctan2(const NdArray<dtype>& inY, const NdArray<dtype>& inX)
    {
        if (inX.shape() != inY.shape())
        {
            std::string errStr = "Error: arctan2: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<double> returnArray(inY.shape());
        std::transform(inY.cbegin(), inY.cend(), inX.cbegin(), returnArray.begin(),
            [](dtype y, dtype x) noexcept -> double { return std::atan2(static_cast<double>(y), static_cast<double>(x)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse hyperbolic tangent.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arctanh.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double arctanh(dtype inValue) noexcept
    {
        return std::atanh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric inverse hyperbolic tangent, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arctanh.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> arctanh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return std::atanh(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns the indices of the maximum values along an axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.argmax.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<uint32> argmax(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.argmax(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Returns the indices of the minimum values along an axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.argmin.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<uint32> argmin(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.argmin(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Returns the indices that would sort an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.argsort.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<uint32> argsort(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.argsort(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Find the indices of array elements that are non-zero, grouped by element.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.argwhere.html
    ///
    /// @param				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<uint32> argwhere(const NdArray<dtype>& inArray)
    {
        return inArray.nonzero();
    }

    //============================================================================
    // Method Description:
    ///						Evenly round to the given number of decimals.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.around.html
    ///
    /// @param			inValue
    /// @param			inNumDecimals: (Optional, default = 0)
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype around(dtype inValue, uint8 inNumDecimals)
    {
        NdArray<dtype> value = { inValue };
        return value.round(inNumDecimals).item();
    }

    //============================================================================
    // Method Description:
    ///						Evenly round to the given number of decimals.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.around.html
    ///
    /// @param			inArray
    /// @param			inNumDecimals: (Optional, default = 0)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> around(const NdArray<dtype>& inArray, uint8 inNumDecimals)
    {
        return inArray.round(inNumDecimals);
    }

    //============================================================================
    // Method Description:
    ///						True if two arrays have the same shape and elements, False otherwise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.array_equal.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				bool
    ///
    template<typename dtype>
    bool array_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            return false;
        }

        return std::equal(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin());
    }

    //============================================================================
    // Method Description:
    ///						Returns True if input arrays are shape consistent and all elements equal.
    ///
    ///						Shape consistent means they are either the same shape, or one input array
    ///						can be broadcasted to create the same shape as the other one.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.array_equiv.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				bool
    ///
    template<typename dtype>
    bool array_equiv(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.size() != inArray2.size())
        {
            return false;
        }

        return std::equal(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin());
    }

    //============================================================================
    // Method Description:
    ///						Convert the vector to an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param
    ///				inVector
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(const std::vector<dtype>& inVector)
    {
        return NdArray<dtype>(inVector);
    }

    //============================================================================
    // Method Description:
    ///						Convert the list initializer to an array.
    ///						eg: NdArray<int> myArray = NC::asarray<int>({1,2,3});
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param
    ///				inList
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(std::initializer_list<dtype>& inList)
    {
        return NdArray<dtype>(inList);
    }

    //============================================================================
    // Method Description:
    ///						Returns a copy of the array, cast to a specified type.
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> astype(const NdArray<dtype> inArray)
    {
        return inArray.template astype<dtypeOut>();
    }

    //============================================================================
    // Method Description:
    ///						Compute the average along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.average.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> average(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.mean(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Compute the weighted average along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.average.html
    ///
    /// @param				inArray
    /// @param				inWeights
    /// @param  			inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> average(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inWeights.shape() != inArray.shape())
                {
                    std::string errStr = "ERROR: average: input array and weight values are not consistant.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                NdArray<double> weightedArray(inArray.shape());
                std::transform(inArray.cbegin(), inArray.cend(), inWeights.cbegin(),
                    weightedArray.begin(), std::multiplies<double>());

                double sum = static_cast<double>(std::accumulate(weightedArray.begin(), weightedArray.end(), 0.0));
                NdArray<double> returnArray = { sum /= inWeights.template sum<double>().item() };

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape arrayShape = inArray.shape();
                if (inWeights.size() != arrayShape.cols)
                {
                    std::string errStr = "ERROR: average: input array and weights value are not consistant.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                double weightSum = inWeights.template sum<double>().item();
                NdArray<double> returnArray(1, arrayShape.rows);
                for (uint32 row = 0; row < arrayShape.rows; ++row)
                {
                    NdArray<double> weightedArray(1, arrayShape.cols);
                    std::transform(inArray.cbegin(row), inArray.cend(row), inWeights.cbegin(),
                        weightedArray.begin(), std::multiplies<double>());

                    double sum = static_cast<double>(std::accumulate(weightedArray.begin(), weightedArray.end(), 0.0));
                    returnArray(0, row) = sum / weightSum;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                if (inWeights.size() != inArray.shape().rows)
                {
                    std::string errStr = "ERROR: average: input array and weight values are not consistant.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                NdArray<dtype> transposedArray = inArray.transpose();

                const Shape transShape = transposedArray.shape();
                double weightSum = inWeights.template sum<double>().item();
                NdArray<double> returnArray(1, transShape.rows);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    NdArray<double> weightedArray(1, transShape.cols);
                    std::transform(transposedArray.cbegin(row), transposedArray.cend(row), inWeights.cbegin(),
                        weightedArray.begin(), std::multiplies<double>());

                    double sum = static_cast<double>(std::accumulate(weightedArray.begin(), weightedArray.end(), 0.0));
                    returnArray(0, row) = sum / weightSum;
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<double>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Return the binary representation of the input number as a string.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.binary_repr.html
    ///
    /// @param				inValue
    /// @return
    ///				std::string
    ///
    template<typename dtype>
    std::string binaryRepr(dtype inValue)
    {
        return std::bitset<DtypeInfo<dtype>::bits()>(inValue).to_string();
    }

    //============================================================================
    // Method Description:
    ///						Count number of occurrences of each value in array of non-negative ints.
    ///						Negative values will be counted in the zero bin.
    ///
    ///						The number of bins(of size 1) is one larger than the largest value in x.
    ///						If minlength is specified, there will be at least this number of bins in
    ///						the output array(though it will be longer if necessary, depending on the
    ///						contents of x).Each bin gives the number of occurrences of its index value
    ///						in x.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.bincount.html
    ///
    /// @param				inArray
    /// @param				inMinLength
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> bincount(const NdArray<dtype>& inArray, uint16 inMinLength)
    {
        // only works with integer input types
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: bincount: can only use with integer types.");

        dtype maxValue = inArray.max().item();
        if (maxValue < 0)
        {
            // no positive values so just return an empty array
            return NdArray<dtype>(0);
        }

        if (maxValue + 1 > DtypeInfo<dtype>::max())
        {
            std::string errStr = "Error: bincount: array values too large, will result in gigantic array that will take up alot of memory...";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        const uint16 outArraySize = std::max(static_cast<uint16>(maxValue + 1), inMinLength);
        NdArray<dtype> clippedArray = inArray.clip(0, maxValue);

        NdArray<dtype> outArray(1, outArraySize);
        outArray.zeros();
        for (auto value : clippedArray)
        {
            ++outArray[value];
        }

        return outArray;
    }

    //============================================================================
    // Method Description:
    ///						Count number of occurrences of each value in array of non-negative ints.
    ///						Negative values will be counted in the zero bin.
    ///
    ///						The number of bins(of size 1) is one larger than the largest value in x.
    ///						If minlength is specified, there will be at least this number of bins in
    ///						the output array(though it will be longer if necessary, depending on the
    ///						contents of x).Each bin gives the number of occurrences of its index value
    ///						in x.If weights is specified the input array is weighted by it, i.e. if a
    ///						value n is found at position i, out[n] += weight[i] instead of out[n] += 1.
    ///						Weights array shall be of the same shape as inArray.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.bincount.html
    ///
    /// @param				inArray
    /// @param				inWeights
    /// @param				inMinLength
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> bincount(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength)
    {
        // only works with integer input types
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: bincount: can only use with integer types.");

        if (inArray.shape() != inWeights.shape())
        {
            std::string errStr = "ERROR: bincount: weights array must be the same shape as the input array.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        dtype maxValue = inArray.max().item();
        if (maxValue < 0)
        {
            // no positive values so just return an empty array
            return NdArray<dtype>(0);
        }

        if (maxValue + 1 > DtypeInfo<dtype>::max())
        {
            std::string errStr = "Error: bincount: array values too large, will result in gigantic array that will take up alot of memory...";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        const uint16 outArraySize = std::max(static_cast<uint16>(maxValue + 1), inMinLength);
        NdArray<dtype> clippedArray = inArray.clip(0, maxValue);

        NdArray<dtype> outArray(1, outArraySize);
        outArray.zeros();
        for (uint32 i = 0; i < inArray.size(); ++i)
        {
            outArray[clippedArray[i]] += inWeights[i];
        }

        return outArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the bit-wise AND of two arrays element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.bitwise_and.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> bitwise_and(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 & inArray2;
    }

    //============================================================================
    // Method Description:
    ///						Compute the bit-wise NOT the input array element-wise.
    ///
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> bitwise_not(const NdArray<dtype>& inArray)
    {
        return ~inArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the bit-wise OR of two arrays element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.bitwise_or.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> bitwise_or(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 | inArray2;
    }

    //============================================================================
    // Method Description:
    ///						Compute the bit-wise XOR of two arrays element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.bitwise_xor.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> bitwise_xor(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 ^ inArray2;
    }

    //============================================================================
    // Method Description:
    ///						Return a new array with the bytes of the array elements
    ///						swapped.
    ///
    /// @param				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> byteswap(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray);
        returnArray.byteswap();
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the cube-root of an array. Not super usefull
    ///						if not using a floating point type
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.cbrt.html
    ///
    /// @param				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double cbrt(dtype inValue)
    {
        return std::cbrt(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Return the cube-root of an array, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.cbrt.html
    ///
    /// @param				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> cbrt(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return std::cbrt(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the ceiling of the input.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ceil.html
    ///
    /// @param				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype ceil(dtype inValue)
    {
        return std::ceil(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Return the ceiling of the input, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ceil.html
    ///
    /// @param				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> ceil(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> dtype { return std::ceil(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Clip (limit) the value.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.clip.html
    ///
    /// @param				inValue
    /// @param				inMinValue
    /// @param				inMaxValue
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    dtype clip(dtype inValue, dtype inMinValue, dtype inMaxValue)
    {
        return boost::algorithm::clamp(inValue, inMinValue, inMaxValue);
    }

    //============================================================================
    // Method Description:
    ///						Clip (limit) the values in an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.clip.html
    ///
    /// @param				inArray
    /// @param				inMinValue
    /// @param				inMaxValue
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> clip(const NdArray<dtype>& inArray, dtype inMinValue, dtype inMaxValue)
    {
        return inArray.clip(inMinValue, inMaxValue);
    }

    //============================================================================
    // Method Description:
    ///						Stack 1-D arrays as columns into a 2-D array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.column_stack.html
    ///
    /// @param
    ///				inArrayList: {list} of arrays to stack
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> column_stack(const std::initializer_list<NdArray<dtype> >& inArrayList)
    {
        // first loop through to calculate the final size of the array
        Shape finalShape;
        for (auto& ndarray : inArrayList)
        {
            if (finalShape.isnull())
            {
                finalShape = ndarray.shape();
            }
            else if (ndarray.shape().rows != finalShape.rows)
            {
                std::string errStr = "ERROR: column_stack: input arrays must have the same number of rows.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }
            else
            {
                finalShape.cols += ndarray.shape().cols;
            }
        }

        // now that we know the final size, contruct the output array
        NdArray<dtype> returnArray(finalShape);
        uint32 colStart = 0;
        for (auto& ndarray : inArrayList)
        {
            const Shape theShape = ndarray.shape();
            for (uint32 row = 0; row < theShape.rows; ++row)
            {
                for (uint32 col = 0; col < theShape.cols; ++col)
                {
                    returnArray(row, colStart + col) = ndarray(row, col);
                }
            }
            colStart += theShape.cols;
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Join a sequence of arrays along an existing axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.concatenate.html
    ///
    /// @param				inArrayList
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> concatenate(const std::initializer_list<NdArray<dtype> >& inArrayList, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                uint32 finalSize = 0;
                for (auto& ndarray : inArrayList)
                {
                    finalSize += ndarray.size();
                }

                NdArray<dtype> returnArray(1, finalSize);
                uint32 offset = 0;
                for (auto& ndarray : inArrayList)
                {
                    std::copy(ndarray.cbegin(), ndarray.cend(), returnArray.begin() + offset);
                    offset += ndarray.size();
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return row_stack(inArrayList);
            }
            case Axis::COL:
            {
                return column_stack(inArrayList);
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtype>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						returns whether or not a value is included the array
    ///
    /// @param				inArray
    /// @param				inValue
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				bool
    ///
    template<typename dtype>
    NdArray<bool> contains(const NdArray<dtype>& inArray, dtype inValue, Axis inAxis)
    {
        return inArray.contains(inValue, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Return an array copy of the given object.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.copy.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> copy(const NdArray<dtype>& inArray)
    {
        return NdArray<dtype>(inArray);
    }

    //============================================================================
    // Method Description:
    ///						Change the sign of x1 to that of x2, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.copysign.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> copySign(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: copysign: input arrays are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<dtype> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> dtype { return inValue2 < 0 ? std::abs(inValue1) * -1 : std::abs(inValue1); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Copies values from one array to another
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.copyto.html
    ///
    /// @param				inDestArray
    /// @param				inSrcArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& copyto(NdArray<dtype>& inDestArray, const NdArray<dtype>& inSrcArray)
    {
        inDestArray = inSrcArray;
        return inDestArray;
    }

    //============================================================================
    // Method Description:
    ///						Cosine
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.cos.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double cos(dtype inValue)
    {
        return std::cos(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Cosine element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.cos.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> cos(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return std::cos(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Hyperbolic Cosine.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.cosh.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double cosh(dtype inValue)
    {
        return std::cosh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Hyperbolic Cosine element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.cosh.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> cosh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return std::cosh(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Counts the number of non-zero values in the array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.count_nonzero.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<uint32> count_nonzero(const NdArray<dtype>& inArray, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                NdArray<uint32> count = { inArray.size() - static_cast<uint32>(std::count(inArray.cbegin(), inArray.cend(), static_cast<dtype>(0))) };
                return count;
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();

                NdArray<uint32> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray(0, row) = inShape.cols -
                        static_cast<uint32>(std::count(inArray.cbegin(row), inArray.cend(row), static_cast<dtype>(0)));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> inArrayTranspose = inArray.transpose();
                Shape inShapeTransposed = inArrayTranspose.shape();
                NdArray<uint32> returnArray(1, inShapeTransposed.rows);
                for (uint32 row = 0; row < inShapeTransposed.rows; ++row)
                {
                    returnArray(0, row) = inShapeTransposed.cols -
                        static_cast<uint32>(std::count(inArrayTranspose.cbegin(row), inArrayTranspose.cend(row), static_cast<dtype>(0)));
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<uint32>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Return the cross product of two (arrays of) vectors.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.cross.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @param  			inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> cross(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, Axis inAxis)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: cross: the input array dimensions are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        switch (inAxis)
        {
            case Axis::NONE:
            {
                const uint32 arraySize = inArray1.size();
                if (arraySize != inArray2.size() || arraySize < 2 || arraySize > 3)
                {
                    std::string errStr = "ERROR: cross: incompatible dimensions for cross product (dimension must be 2 or 3)";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                NdArray<dtype> in1 = inArray1.flatten();
                NdArray<dtype> in2 = inArray2.flatten();

                switch (arraySize)
                {
                    case 2:
                    {
                        NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(in1[0]) * static_cast<dtypeOut>(in2[1])
                            - static_cast<dtypeOut>(in1[1]) * static_cast<dtypeOut>(in2[0]) };
                        return returnArray;
                    }
                    case 3:
                    {
                        dtypeOut i = static_cast<dtypeOut>(in1[1]) * static_cast<dtypeOut>(in2[2])
                            - static_cast<dtypeOut>(in1[2]) * static_cast<dtypeOut>(in2[1]);
                        dtypeOut j = -(static_cast<dtypeOut>(in1[0]) * static_cast<dtypeOut>(in2[2])
                            - static_cast<dtypeOut>(in1[2]) * static_cast<dtypeOut>(in2[0]));
                        dtypeOut k = (static_cast<dtypeOut>(in1[0]) * static_cast<dtypeOut>(in2[1])
                            - static_cast<dtypeOut>(in1[1]) * static_cast<dtypeOut>(in2[0]));

                        NdArray<dtypeOut> returnArray = { i, j, k };
                        return returnArray;
                    }
                    default:
                    {
                        // this isn't actually possible, just putting this here to get rid
                        // of the compiler warning.
                        return NdArray<dtypeOut>(0);
                    }
                }
            }
            case Axis::ROW:
            {
                const Shape arrayShape = inArray1.shape();
                if (arrayShape != inArray2.shape() || arrayShape.rows < 2 || arrayShape.rows > 3)
                {
                    std::string errStr = "ERROR: cross: incompatible dimensions for cross product (dimension must be 2 or 3)";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                Shape returnArrayShape;
                if (arrayShape.rows == 2)
                {
                    returnArrayShape = Shape(1, arrayShape.cols);
                }
                else
                {
                    returnArrayShape = Shape(3, arrayShape.cols);
                }

                NdArray<dtypeOut> returnArray(returnArrayShape);
                for (uint32 col = 0; col < arrayShape.cols; ++col)
                {
                    const int32 theCol = static_cast<int32>(col);
                    NdArray<dtype> vec1 = inArray1({ 0, static_cast<int32>(arrayShape.rows) }, { theCol, theCol + 1 });
                    NdArray<dtype> vec2 = inArray2({ 0, static_cast<int32>(arrayShape.rows) }, { theCol, theCol + 1 });
                    NdArray<dtypeOut> vecCross = cross<dtypeOut>(vec1, vec2, Axis::NONE);

                    returnArray.put({ 0, static_cast<int32>(returnArrayShape.rows) }, { theCol, theCol + 1 }, vecCross);
                }

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape arrayShape = inArray1.shape();
                if (arrayShape != inArray2.shape() || arrayShape.cols < 2 || arrayShape.cols > 3)
                {
                    std::string errStr = "ERROR: cross: incompatible dimensions for cross product (dimension must be 2 or 3)";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                Shape returnArrayShape;
                if (arrayShape.cols == 2)
                {
                    returnArrayShape = Shape(arrayShape.rows, 1);
                }
                else
                {
                    returnArrayShape = Shape(arrayShape.rows, 3);
                }

                NdArray<dtypeOut> returnArray(returnArrayShape);
                for (uint32 row = 0; row < arrayShape.rows; ++row)
                {
                    const int32 theRow = static_cast<int32>(row);
                    NdArray<dtype> vec1 = inArray1({ theRow, theRow + 1 }, { 0, static_cast<int32>(arrayShape.cols) });
                    NdArray<dtype> vec2 = inArray2({ theRow, theRow + 1 }, { 0, static_cast<int32>(arrayShape.cols) });
                    NdArray<dtypeOut> vecCross = cross<dtypeOut>(vec1, vec2, Axis::NONE);

                    returnArray.put({ theRow, theRow + 1 }, { 0, static_cast<int32>(returnArrayShape.cols) }, vecCross);
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtypeOut>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Cubes the input
    ///
    /// @param
    ///				inValue
    /// @return
    ///				cubed value
    ///
    template<typename dtypeOut = double, typename dtype>
    dtypeOut cube(dtype inValue)
    {
        return utils::cube(static_cast<dtypeOut>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Cubes the elements of the array
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> cube(const NdArray<dtype>& inArray)
    {
        NdArray<dtypeOut> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> dtypeOut { return utils::cube(static_cast<dtypeOut>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the cumulative product of elements along a given axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.cumprod.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> cumprod(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.template cumprod<dtypeOut>(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Return the cumulative sum of the elements along a given axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.cumsum.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> cumsum(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.template cumsum<dtypeOut>(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Convert angles from degrees to radians.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.deg2rad.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double deg2rad(dtype inValue) noexcept
    {
        return inValue * constants::pi / 180.0;
    }

    //============================================================================
    // Method Description:
    ///						Convert angles from degrees to radians.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.deg2rad.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> deg2rad(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return deg2rad(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Convert angles from degrees to radians.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.degrees.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double degrees(dtype inValue) noexcept
    {
        return rad2deg(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Convert angles from degrees to radians.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.degrees.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> degrees(const NdArray<dtype>& inArray)
    {
        return rad2deg(inArray);
    }

    //============================================================================
    // Method Description:
    ///						Return a new array with sub-arrays along an axis deleted.
    ///
    /// @param				inArray
    /// @param				inArrayIdxs
    /// @param				inAxis (Optional, default NONE) if none the indices will be applied to the flattened array
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, const NdArray<uint32>& inArrayIdxs, Axis inAxis)
    {
        // make sure that the indices are unique first
        NdArray<uint32> indices = unique(inArrayIdxs);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::vector<dtype> values;
                for (uint32 i = 0; i < inArray.size(); ++i)
                {
                    if (indices.contains(i).item())
                    {
                        continue;
                    }

                    values.push_back(inArray[i]);
                }

                return NdArray<dtype>(values);
            }
            case Axis::ROW:
            {
                const Shape inShape = inArray.shape();
                if (indices.max().item() >= inShape.rows)
                {
                    std::string errStr = "ERROR: deleteIndices: input index value is greater than the number of rows in the array.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                const uint32 numNewRows = inShape.rows - indices.size();
                NdArray<dtype> returnArray(numNewRows, inShape.cols);

                uint32 rowCounter = 0;
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    if (indices.contains(row).item())
                    {
                        continue;
                    }

                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        returnArray(rowCounter, col) = inArray(row, col);
                    }
                    ++rowCounter;
                }

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();
                if (indices.max().item() >= inShape.cols)
                {
                    std::string errStr = "ERROR: deleteIndices: input index value is greater than the number of cols in the array.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                const uint32 numNewCols = inShape.cols - indices.size();
                NdArray<dtype> returnArray(inShape.rows, numNewCols);

                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    uint32 colCounter = 0;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (indices.contains(col).item())
                        {
                            continue;
                        }

                        returnArray(row, colCounter++) = inArray(row, col);
                    }
                }

                return returnArray;


            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtype>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Return a new array with sub-arrays along an axis deleted.
    ///
    /// @param				inArray
    /// @param				inIndicesSlice
    /// @param  			inAxis (Optional, default NONE) if none the indices will be applied to the flattened array
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, const Slice& inIndicesSlice, Axis inAxis)
    {
        Slice sliceCopy(inIndicesSlice);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                sliceCopy.makePositiveAndValidate(inArray.size());
                break;
            }
            case Axis::ROW:
            {
                sliceCopy.makePositiveAndValidate(inArray.shape().cols);
                break;
            }
            case Axis::COL:
            {
                sliceCopy.makePositiveAndValidate(inArray.shape().rows);
                break;
            }
        }

        std::vector<uint32> indices;
        for (uint32 i = static_cast<uint32>(sliceCopy.start); i < static_cast<uint32>(sliceCopy.stop); i += sliceCopy.step)
        {
            indices.push_back(i);
        }

        return deleteIndices(inArray, NdArray<uint32>(indices), inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Return a new array with sub-arrays along an axis deleted.
    ///
    /// @param				inArray
    /// @param				inIndex
    /// @param				inAxis (Optional, default NONE) if none the indices will be applied to the flattened array
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, uint32 inIndex, Axis inAxis)
    {
        NdArray<uint32> inIndices = { inIndex };
        return deleteIndices(inArray, inIndices, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Create a two-dimensional array with the flattened input as a diagonal.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.diagflat.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> diagflat(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray.size());
        returnArray.zeros();
        for (uint32 i = 0; i < inArray.size(); ++i)
        {
            returnArray(i, i) = inArray[i];
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return specified diagonals.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.diagonal.html
    ///
    /// @param				inArray
    /// @param				inOffset (Defaults to 0)
    /// @param				inAxis (Optional, default NONE) axis the offset is applied to
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> diagonal(const NdArray<dtype>& inArray, int32 inOffset, Axis inAxis)
    {
        return inArray.diagonal(inOffset, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Calculate the n-th discrete difference along given axis.
    ///						Unsigned dtypes will give you weird results...obviously.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.diff.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> diff(const NdArray<dtype>& inArray, Axis inAxis)
    {
        const Shape inShape = inArray.shape();

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inArray.size() < 2)
                {
                    return NdArray<dtype>(0);
                }

                NdArray<dtype> returnArray(1, inArray.size() - 1);
                std::transform(inArray.cbegin(), inArray.cend() - 1, inArray.cbegin() + 1, returnArray.begin(),
                    [](dtype inValue1, dtype inValue2) noexcept -> dtype { return inValue2 - inValue1; });

                return returnArray;
            }
            case Axis::COL:
            {
                if (inShape.cols < 2)
                {
                    return NdArray<dtype>(0);
                }

                NdArray<dtype> returnArray(inShape.rows, inShape.cols - 1);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    std::transform(inArray.cbegin(row), inArray.cend(row) - 1, inArray.cbegin(row) + 1, returnArray.begin(row),
                        [](dtype inValue1, dtype inValue2) noexcept -> dtype { return inValue2 - inValue1; });
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                if (inShape.rows < 2)
                {
                    return NdArray<dtype>(0);
                }

                NdArray<dtype> transArray = inArray.transpose();
                const Shape transShape = transArray.shape();
                NdArray<dtype> returnArray(transShape.rows, transShape.cols - 1);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    std::transform(transArray.cbegin(row), transArray.cend(row) - 1, transArray.cbegin(row) + 1, returnArray.begin(row),
                        [](dtype inValue1, dtype inValue2) noexcept -> dtype { return inValue2 - inValue1; });
                }

                return returnArray.transpose();
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtype>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Returns a true division of the inputs, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.divide.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> divide(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1.template astype<dtypeOut>() / inArray2.template astype<dtypeOut>();
    }

    //============================================================================
    // Method Description:
    ///						Dot product of two arrays.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.dot.html
    ///
    /// @param			inArray1
    /// @param			inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> dot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1.template dot<dtypeOut>(inArray2);
    }

    //============================================================================
    // Method Description:
    ///						Dump a binary file of the array to the specified file.
    ///						The array can be read back with or NC::load.
    ///
    /// @param				inArray
    /// @param				inFilename
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    void dump(const NdArray<dtype>& inArray, const std::string& inFilename)
    {
        inArray.dump(inFilename);
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, without initializing entries.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.empty.html
    ///
    /// @param				inNumRows
    /// @param				inNumCols
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> empty(uint32 inNumRows, uint32 inNumCols)
    {
        return NdArray<dtype>(inNumRows, inNumCols);
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, without initializing entries.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.empty.html
    ///
    /// @param
    ///				inShape
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> empty(const Shape& inShape)
    {
        return NdArray<dtype>(inShape);
    }

    //============================================================================
    // Method Description:
    ///						Return a new array with the same shape as a given array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.empty_like.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> empty_like(const NdArray<dtype>& inArray)
    {
        return NdArray<dtypeOut>(inArray.shape());
    }

    //============================================================================
    // Method Description:
    ///						Return the endianess of the array values.
    ///
    /// @param
    ///				inArray
    /// @return
    ///				Endian
    ///
    template<typename dtype>
    Endian endianess(const NdArray<dtype>& inArray) noexcept
    {
        return inArray.endianess();
    }

    //============================================================================
    // Method Description:
    ///						Calculate the error function of all elements in the input array.
    ///                     Integral (from [-x, x]) of np.exp(np.power(-t, 2)) dt, multiplied by 1/np.pi.
    ///
    /// @param
    ///				inValue
    /// @return
    ///				double
    ///
    template<typename dtype>
    double erf(dtype inValue)
    {
        return boost::math::erf(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Calculate the error function of all elements in the input array.
    ///                     Integral (from [-x, x]) of np.exp(np.power(-t, 2)) dt, multiplied by 1/np.pi.
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray<double>
    ///
    template<typename dtype>
    NdArray<double> erf(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) -> double { return erf(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns the complement of the error function of inValue.
    ///
    /// @param
    ///				inValue
    /// @return
    ///				double
    ///
    template<typename dtype>
    double erfc(dtype inValue)
    {
        return boost::math::erfc(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Returns the element-wise complement of the error
    ///                     function of inValue.
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray<double>
    ///
    template<typename dtype>
    NdArray<double> erfc(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue)  -> double { return erfc(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return (x1 == x2) element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.equal.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 == inArray2;
    }

    //============================================================================
    // Method Description:
    ///						Calculate the exponential of the input value.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.exp.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double exp(dtype inValue) noexcept
    {
        return std::exp(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Calculate the exponential of all elements in the input array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.exp.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> exp(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return exp(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Calculate 2**p for all p in the input value.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.exp2.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double exp2(dtype inValue) noexcept
    {
        return std::exp2(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Calculate 2**p for all p in the input array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.exp2.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> exp2(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return exp2(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Calculate exp(x) - 1 for the input value.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.expm1.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double expm1(dtype inValue) noexcept
    {
        return std::exp(static_cast<double>(inValue)) - 1.0;
    }

    //============================================================================
    // Method Description:
    ///						Calculate exp(x) - 1 for all elements in the array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.expm1.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> expm1(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return expm1(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return a 2-D array with ones on the diagonal and zeros elsewhere.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.eye.html
    ///
    /// @param				inN: number of rows and columns (N)
    /// @param				inK: Index of the diagonal: 0 (the default) refers to the main diagonal,
    ///				a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> eye(uint32 inN, int32 inK)
    {
        return eye<dtype>(inN, inN, inK);
    }

    //============================================================================
    // Method Description:
    ///						Return a 2-D array with ones on the diagonal and zeros elsewhere.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.eye.html
    ///
    /// @param				inN: number of rows (N)
    /// @param				inM: number of columns (M)
    /// @param				inK: Index of the diagonal: 0 (the default) refers to the main diagonal,
    ///				a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> eye(uint32 inN, uint32 inM, int32 inK)
    {
        NdArray<dtype> returnArray(inN, inM);
        returnArray.zeros();

        if (inK < 0)
        {
            uint32 col = 0;
            for (uint32 row = inK; row < inN; ++row)
            {
                if (col >= inM)
                {
                    break;
                }

                returnArray(row, col++) = 1;
            }
        }
        else
        {
            uint32 row = 0;
            for (uint32 col = inK; col < inM; ++col)
            {
                if (row >= inN)
                {
                    break;
                }

                returnArray(row++, col) = 1;
            }
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return a 2-D array with ones on the diagonal and zeros elsewhere.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.eye.html
    ///
    /// @param				inShape
    /// @param				inK: Index of the diagonal: 0 (the default) refers to the main diagonal,
    ///				a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> eye(const Shape& inShape, int32 inK)
    {
        return eye<dtype>(inShape.rows, inShape.cols, inK);
    }

    //============================================================================
    // Method Description:
    ///						Fill the main diagonal of the given array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fill_diagonal.html
    ///
    /// @param      inArray
    /// @param      inValue
    ///
    template<typename dtype>
    void fillDiagonal(NdArray<dtype>& inArray, dtype inValue) noexcept
    {
        const auto inShape = inArray.shape();
        for (uint32 row = 0; row < inShape.rows; ++row)
        {
            if (row < inShape.cols)
            {
                inArray(row, row) = inValue;
            }
        }
    }


    //============================================================================
    // Method Description:
    ///						Find flat indices of nonzero elements.
    ///
    /// @param      mask: the mask to apply to the array
    /// @param      n: the first n indices to return (optional, default all)
    ///
    /// @return
    ///				NdArray
    ///
    NdArray<uint32> find(const NdArray<bool>& mask, uint32 n = std::numeric_limits<uint32>::max())
    {
        NdArray<uint32> indices = mask.nonzero();

        if (indices.size() <= n)
        {
            return indices;
        }
        else
        {
            return indices[Slice(0, n)];
        }
    }

    //============================================================================
    // Method Description:
    ///						Round to nearest integer towards zero.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fix.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype fix(dtype inValue) noexcept
    {
        return inValue > 0 ? std::floor(inValue) : std::ceil(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Round to nearest integer towards zero.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fix.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fix(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return fix(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return a copy of the array collapsed into one dimension.
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> flatten(const NdArray<dtype>& inArray)
    {
        return inArray.flatten();
    }

    //============================================================================
    // Method Description:
    ///						Return indices that are non-zero in the flattened version of a.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.flatnonzero.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<uint32> flatnonzero(const NdArray<dtype>& inArray)
    {
        return inArray.flatten().nonzero();
    }

    //============================================================================
    // Method Description:
    ///						Reverse the order of elements in an array along the given axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.flip.html
    ///
    /// @param				inArray
    /// @param				inAxis
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> flip(const NdArray<dtype>& inArray, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                NdArray<dtype> returnArray(inArray);
                std::reverse(returnArray.begin(), returnArray.end());
                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<dtype> returnArray(inArray);
                for (uint32 row = 0; row < inArray.shape().rows; ++row)
                {
                    std::reverse(returnArray.begin(row), returnArray.end(row));
                }
                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> returnArray = inArray.transpose();
                for (uint32 row = 0; row < returnArray.shape().rows; ++row)
                {
                    std::reverse(returnArray.begin(row), returnArray.end(row));
                }
                return returnArray.transpose();
            }
            default:
            {
                return NdArray<dtype>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Flip array in the left/right direction.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fliplr.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fliplr(const NdArray<dtype>& inArray)
    {
        return flip(inArray, Axis::COL);
    }

    //============================================================================
    // Method Description:
    ///						Flip array in the up/down direction.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.flipud.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> flipud(const NdArray<dtype>& inArray)
    {
        return flip(inArray, Axis::ROW);
    }

    //============================================================================
    // Method Description:
    ///						Return the floor of the input.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.floor.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype floor(dtype inValue) noexcept
    {
        return std::floor(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Return the floor of the input, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.floor.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> floor(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return floor(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the largest integer smaller or equal to the division of the inputs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.floor_divide.html
    ///
    /// @param				inValue1
    /// @param				inValue2
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype floor_divide(dtype inValue1, dtype inValue2) noexcept
    {
        return std::floor(inValue1 / inValue2);
    }

    //============================================================================
    // Method Description:
    ///						Return the largest integer smaller or equal to the division of the inputs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.floor_divide.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> floor_divide(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return floor(inArray1 / inArray2);
    }

    //============================================================================
    // Method Description:
    ///						maximum of inputs.
    ///
    ///						Compare two value and returns a value containing the
    ///						maxima
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmax.html
    ///
    /// @param				inValue1
    /// @param				inValue2
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype fmax(dtype inValue1, dtype inValue2)
    {
        return std::max(inValue1, inValue2);
    }

    //============================================================================
    // Method Description:
    ///						Element-wise maximum of array elements.
    ///
    ///						Compare two arrays and returns a new array containing the
    ///						element - wise maxima
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmax.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fmax(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: fmax: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<double> returnArray(inArray1.shape());

        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> double { return std::max(inValue1, inValue2); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						minimum of inputs.
    ///
    ///						Compare two value and returns a value containing the
    ///						minima
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmin.html
    ///
    /// @param				inValue1
    /// @param				inValue2
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype fmin(dtype inValue1, dtype inValue2)
    {
        return std::min(inValue1, inValue2);
    }

    //============================================================================
    // Method Description:
    ///						Element-wise minimum of array elements.
    ///
    ///						Compare two arrays and returns a new array containing the
    ///						element - wise minima
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmin.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fmin(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: fmin: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<double> returnArray(inArray1.shape());

        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> double { return std::min(inValue1, inValue2); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the remainder of division.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmod.html
    ///
    ///
    /// @param				inValue1
    /// @param				inValue2
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype fmod(dtype inValue1, dtype inValue2) noexcept
    {
        // can only be called on integer types
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: fmod can only be compiled with integer types.");

        return inValue1 % inValue2;
    }

    //============================================================================
    // Method Description:
    ///						Return the element-wise remainder of division.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmod.html
    ///
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fmod(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        // can only be called on integer types
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: fmod can only be compiled with integer types.");

        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: fmod: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<dtype> returnArray(inArray1.shape());

        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> dtype { return inValue1 % inValue2; });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Interpret a buffer as a 1-dimensional array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.frombuffer.html
    ///
    /// @param				inBufferPtr
    /// @param				inNumBytes
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> frombuffer(char* inBufferPtr, uint32 inNumBytes)
    {
        return NdArray<dtype>(reinterpret_cast<dtype*>(inBufferPtr), inNumBytes);
    }

    //============================================================================
    // Method Description:
    ///						Construct an array from data in a text or binary file.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fromfile.html
    ///
    /// @param				inFilename
    /// @param				inSep: Separator between items if file is a text file. Empty ("")
    ///							separator means the file should be treated as binary.
    ///							Right now the only supported seperators are " ", "\t", "\n"
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fromfile(const std::string& inFilename, const std::string& inSep)
    {
        boost::filesystem::path p(inFilename);
        if (!boost::filesystem::exists(inFilename))
        {
            std::string errStr = "ERROR: fromfile: input filename does not exist.\n\t" + inFilename;
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        if (inSep.compare("") == 0)
        {
            // read in as binary file
            std::ifstream in(inFilename.c_str(), std::ios::in | std::ios::binary);
            in.seekg(0, in.end);
            const uint32 fileSize = static_cast<uint32>(in.tellg());

            FILE* filePtr = fopen(inFilename.c_str(), "rb");
            if (filePtr == nullptr)
            {
                std::string errStr = "ERROR: fromfile: unable to open the file.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            char* fileBuffer = new char[fileSize];
            const size_t bytesRead = fread(fileBuffer, sizeof(char), fileSize, filePtr);
            fclose(filePtr);

            NdArray<dtype> returnArray(reinterpret_cast<dtype*>(fileBuffer), static_cast<uint32>(bytesRead));
            delete[] fileBuffer;

            return returnArray;
        }
        else
        {
            // read in as txt file
            if (!(inSep.compare(" ") == 0 || inSep.compare("\t") == 0 || inSep.compare("\n") == 0))
            {
                std::string errStr = "ERROR: fromfile: only [' ', '\\t', '\\n'] seperators are supported";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            std::vector<dtype> values;

            std::ifstream file(inFilename.c_str());
            if (file.is_open())
            {
                while (!file.eof())
                {
                    std::string line;
                    std::getline(file, line);

                    std::istringstream iss(line);
                    try
                    {
                        values.push_back(static_cast<dtype>(std::stod(iss.str())));
                    }
                    catch (const std::invalid_argument& ia)
                    {
                        std::cout << "Warning: fromfile: " << ia.what() << std::endl;
                        ///throw;
                    }
                }
                file.close();
            }
            else
            {
                std::string errStr = "ERROR: fromfile: unable to open file.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            return NdArray<dtype>(values);
        }
    }

    //============================================================================
    // Method Description:
    ///						Create a new 1-dimensional array from an iterable object.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fromiter.html
    ///
    /// @param				inBegin
    /// @param				inEnd
    /// @return
    ///				NdArray
    ///
    template<typename dtype, typename Iter>
    NdArray<dtype> fromiter(Iter inBegin, Iter inEnd)
    {
        std::vector<dtype> values;
        for (Iter iter = inBegin; iter != inEnd; ++iter)
        {
            values.push_back(*iter);
        }
        return NdArray<dtype>(values);
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with inFillValue
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.full.html
    ///
    /// @param				inSquareSize
    /// @param				inFillValue
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> full(uint32 inSquareSize, dtype inFillValue)
    {
        NdArray<dtype> returnArray(inSquareSize, inSquareSize);
        returnArray.fill(inFillValue);
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with inFillValue
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.full.html
    ///
    /// @param				inNumRows
    /// @param				inNumCols
    /// @param				inFillValue
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> full(uint32 inNumRows, uint32 inNumCols, dtype inFillValue)
    {
        NdArray<dtype> returnArray(inNumRows, inNumCols);
        returnArray.fill(inFillValue);
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with inFillValue
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.full.html
    ///
    /// @param				inShape
    /// @param				inFillValue
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> full(const Shape& inShape, dtype inFillValue)
    {
        return full(inShape.rows, inShape.cols, inFillValue);
    }

    //============================================================================
    // Method Description:
    ///						Return a full array with the same shape and type as a given array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.full_like.html
    ///
    /// @param				inArray
    /// @param				inFillValue
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> full_like(const NdArray<dtype>& inArray, dtype inFillValue)
    {
        return full(inArray.shape(), static_cast<dtypeOut>(inFillValue));
    }

    //============================================================================
    // Method Description:
    ///						Returns the greatest common divisor of |x1| and |x2|
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.gcd.html
    ///
    /// @param      inValue1
    /// @param      inValue2
    /// @return
    ///				dtype
    ///
    template<typename dtype>
    dtype gcd(dtype inValue1, dtype inValue2) noexcept
    {
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: gcd can only be called with integer types.");
        return boost::integer::gcd(inValue1, inValue2);
    }

    //============================================================================
    // Method Description:
    ///						Returns the greatest common divisor of the values in the
    ///                     input array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.gcd.html
    ///
    /// @param      inArray
    /// @return
    ///				NdArray<double>
    ///
    template<typename dtype>
    dtype gcd(const NdArray<dtype>& inArray)
    {
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: gcd can only be called with integer types.");
        return boost::integer::gcd_range(inArray.cbegin(), inArray.cend()).first;
    }

    //============================================================================
    // Method Description:
    ///						Return the gradient of the array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.gradient.html
    ///
    ///
    /// @param				inArray
    /// @param				inAxis (default ROW)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> gradient(const NdArray<dtype>& inArray, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::ROW:
            {
                const auto inShape = inArray.shape();
                if (inShape.rows < 2)
                {
                    std::string errStr = "ERROR: gradient: input array must have more than 1 row.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                // first do the first and last rows
                auto returnArray = NdArray<double>(inShape);
                for (uint32 col = 0; col < inShape.cols; ++col)
                {
                    returnArray(0, col) = static_cast<double>(inArray(1, col)) - static_cast<double>(inArray(0, col));
                    returnArray(-1, col) = static_cast<double>(inArray(-1, col)) - static_cast<double>(inArray(-2, col));
                }

                // then rip through the rest of the array
                for (uint32 col = 0; col < inShape.cols; ++col)
                {
                    for (uint32 row = 1; row < inShape.rows - 1; ++row)
                    {
                        returnArray(row, col) = (static_cast<double>(inArray(row + 1, col)) - static_cast<double>(inArray(row - 1, col))) / 2.0;
                    }
                }

                return returnArray;
            }
            case Axis::COL:
            {
                const auto inShape = inArray.shape();
                if (inShape.cols < 2)
                {
                    std::string errStr = "ERROR: gradient: input array must have more than 1 columns.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                // first do the first and last columns
                auto returnArray = NdArray<double>(inShape);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray(row, 0) = static_cast<double>(inArray(row, 1)) - static_cast<double>(inArray(row, 0));
                    returnArray(row, -1) = static_cast<double>(inArray(row, -1)) - static_cast<double>(inArray(row, -2));
                }

                // then rip through the rest of the array
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    for (uint32 col = 1; col < inShape.cols - 1; ++col)
                    {
                        returnArray(row, col) = (static_cast<double>(inArray(row, col + 1)) - static_cast<double>(inArray(row, col - 1))) / 2.0;
                    }
                }

                return returnArray;
            }
            default:
            {
                // will return the gradient of the flattened array
                if (inArray.size() < 2)
                {
                    std::string errStr = "ERROR: gradient: input array must have more than 1 element.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                auto returnArray = NdArray<double>(1, inArray.size());
                returnArray[0] = static_cast<double>(inArray[1]) - static_cast<double>(inArray[0]);
                returnArray[-1] = static_cast<double>(inArray[-1]) - static_cast<double>(inArray[-2]);

                std::transform(inArray.cbegin() + 2, inArray.cend(), inArray.cbegin(), returnArray.begin() + 1,
                    [](dtype value1, dtype value2) noexcept -> double { return (static_cast<double>(value1) - static_cast<double>(value2)) / 2.0; });

                return returnArray;
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Return the truth value of (x1 > x2) element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.greater.html
    ///
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> greater(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 > inArray2;
    }

    //============================================================================
    // Method Description:
    ///						Return the truth value of (x1 >= x2) element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.greater_equal.html
    ///
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> greater_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 >= inArray2;
    }

    //============================================================================
    // Method Description:
    ///						Compute the histogram of a set of data.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.histogram.html
    ///
    ///
    /// @param				inArray
    /// @param				inNumBins( default 10)
    ///
    /// @return
    ///				std::pair of NdArrays; first is histogram counts, seconds is the bin edges
    ///
    template<typename dtype>
    std::pair<NdArray<uint32>, NdArray<double> > histogram(const NdArray<dtype>& inArray, uint32 inNumBins)
    {
        if (inNumBins == 0)
        {
            std::string errStr = "ERROR: histogram: number of bins must be positive.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<uint32> histo = zeros<uint32>(1, inNumBins);

        const bool useEndPoint = true;
        NdArray<double> binEdges = linspace(static_cast<double>(inArray.min().item()),
            static_cast<double>(inArray.max().item()), inNumBins + 1, useEndPoint);

        for (uint32 i = 0; i < inArray.size(); ++i)
        {
            // binary search to find the bin idx
            const bool keepSearching = true;
            uint32 lowIdx = 0;
            uint32 highIdx = binEdges.size() - 1;
            while (keepSearching)
            {
                const uint32 idx = (lowIdx + highIdx) / 2; // integer division
                if (lowIdx == highIdx || lowIdx == highIdx - 1)
                {
                    // we found the bin
                    ++histo[lowIdx];
                    break;
                }

                if (inArray[i] > binEdges[idx])
                {
                    lowIdx = idx;
                }
                else if (inArray[i] < binEdges[idx])
                {
                    highIdx = idx;
                }
                else
                {
                    // we found the bin
                    ++histo[idx];
                    break;
                }
            }
        }

        return std::make_pair(histo, binEdges);
    }

    //============================================================================
    // Method Description:
    ///						Stack arrays in sequence horizontally (column wise).
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.hstack.html
    ///
    ///
    /// @param
    ///				inArrayList: {list} of arrays to stack
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> hstack(const std::initializer_list<NdArray<dtype> >& inArrayList)
    {
        return column_stack(inArrayList);
    }

    //============================================================================
    // Method Description:
    ///						Given the "legs" of a right triangle, return its hypotenuse.
    ///
    ///						Equivalent to sqrt(x1**2 + x2 * *2), element - wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.hypot.html
    ///
    ///
    /// @param				inValue1
    /// @param				inValue2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    dtypeOut hypot(dtype inValue1, dtype inValue2) noexcept
    {
        return std::hypot(static_cast<dtypeOut>(inValue1), static_cast<dtypeOut>(inValue2));
    }

    //============================================================================
    // Method Description:
    ///						Given the "legs" of a right triangle, return its hypotenuse.
    ///
    ///						Equivalent to sqrt(x1**2 + x2 * *2), element - wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.hypot.html
    ///
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> hypot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: hypot: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<dtypeOut> returnArray(inArray1.shape());

        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> dtypeOut
        { return std::hypot(static_cast<dtypeOut>(inValue1), static_cast<dtypeOut>(inValue2)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the identity array.
    ///
    ///						The identity array is a square array with ones on the main diagonal.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.identity.html
    ///
    /// @param
    ///				inSquareSize
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> identity(uint32 inSquareSize)
    {
        NdArray<dtype> returnArray(inSquareSize);
        returnArray.zeros();
        for (uint32 i = 0; i < inSquareSize; ++i)
        {
            returnArray(i, i) = 1;
        }

        return returnArray;
    }

    //============================================================================
    ///						Returns the linear interpolation between two points
    ///
    /// @param      inValue1
    /// @param      inValue2
    /// @param      inPercent
    ///
    /// @return     linear interpolated point
    ///
    template<typename dtype>
    double interp(dtype inValue1, dtype inValue2, double inPercent)
    {
        return utils::interp(inValue1, inValue2, inPercent);
    }

    //============================================================================
    // Method Description:
    ///						One-dimensional linear interpolation.
    ///
    ///                     Returns the one - dimensional piecewise linear interpolant
    ///                     to a function with given values at discrete data - points.
    ///                     If input arrays are not one dimensional they will be
    ///                     internally flattened.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.interp.html
    ///
    /// @param				inX: The x-coordinates at which to evaluate the interpolated values.
    /// @param              inXp: The x-coordinates of the data points, must be increasing. Otherwise, xp is internally sorted.
    /// @param				inFp: The y-coordinates of the data points, same length as inXp.
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> interp(const NdArray<dtype>& inX, const NdArray<dtype>& inXp, const NdArray<dtype>& inFp)
    {
        // do some error checking first
        if (inXp.size() != inFp.size())
        {
            std::string errStr = "ERROR: interp: inXp and inFp need to be the same size().";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        if (inX.min().item() < inXp.min().item() || inX.max().item() > inXp.max().item())
        {
            std::string errStr = "ERROR: interp: endpoints of inX should be contained within inXp.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        // sort the input inXp and inFp data
        NdArray<uint32> sortedXpIdxs = argsort(inXp);
        NdArray<dtype> sortedXp(1, inFp.size());
        NdArray<dtype> sortedFp(1, inFp.size());
        uint32 counter = 0;
        for (auto sortedXpIdx : sortedXpIdxs)
        {
            sortedXp[counter] = inXp[sortedXpIdx];
            sortedFp[counter++] = inFp[sortedXpIdx];
        }

        // sort the input inX array
        NdArray<dtype> sortedX = sort(inX);

        NdArray<dtype> returnArray(1, inX.size());

        uint32 currXpIdx = 0;
        uint32 currXidx = 0;
        while (currXidx < sortedX.size())
        {
            if (sortedXp[currXpIdx] <= sortedX[currXidx] && sortedX[currXidx] <= sortedXp[currXpIdx + 1])
            {
                const double percent = static_cast<double>(sortedX[currXidx] - sortedXp[currXpIdx]) /
                    static_cast<double>(sortedXp[currXpIdx + 1] - sortedXp[currXpIdx]);
                returnArray[currXidx++] = utils::interp(sortedFp[currXpIdx], sortedFp[currXpIdx + 1], percent);
            }
            else
            {
                ++currXpIdx;
            }
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Find the intersection of two arrays.
    ///
    ///						Return the sorted, unique values that are in both of the input arrays.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.intersect1d.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> intersect1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        std::vector<dtype> res(inArray1.size() + inArray2.size());
        std::set<dtype> in1(inArray1.cbegin(), inArray1.cend());
        std::set<dtype> in2(inArray2.cbegin(), inArray2.cend());

        const typename std::vector<dtype>::iterator iter = std::set_intersection(in1.begin(), in1.end(),
            in2.begin(), in2.end(), res.begin());
        res.resize(iter - res.begin());
        return NdArray<dtype>(res);
    }

    //============================================================================
    // Method Description:
    ///						Compute bit-wise inversion, or bit-wise NOT, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.invert.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> invert(const NdArray<dtype>& inArray)
    {
        return ~inArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns a boolean array where two arrays are element-wise
    ///						equal within a tolerance.
    ///
    ///						For finite values, isclose uses the following equation to test whether two floating point values are equivalent.
    ///						absolute(a - b) <= (atol + rtol * absolute(b))
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.isclose.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @param				inRtol: relative tolerance (default 1e-5)
    /// @param				inAtol: absolute tolerance (default 1e-9)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> isclose(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, double inRtol, double inAtol)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: isclose: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<bool> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [inRtol, inAtol](dtype inValueA, dtype inValueB) noexcept -> bool
        { return std::abs(inValueA - inValueB) <= (inAtol + inRtol * std::abs(inValueB)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Test for inf and return result as a boolean.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.isinf.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				bool
    ///
    template<typename dtype>
    bool isinf(dtype inValue) noexcept
    {
        return std::isinf(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Test element-wise for inf and return result as a boolean array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.isinf.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> isinf(const NdArray<dtype>& inArray)
    {
        NdArray<bool> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> bool { return std::isinf(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Test for NaN and return result as a boolean.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.isnan.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				bool
    ///
    template<typename dtype>
    bool isnan(dtype inValue) noexcept
    {
        static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: isnan: can only be used with floating point types.");
        return std::isnan(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Test element-wise for NaN and return result as a boolean array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.isnan.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> isnan(const NdArray<dtype>& inArray)
    {
        static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: isnan: can only be used with floating point types.");

        NdArray<bool> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> bool { return std::isnan(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns the least common multiple of |x1| and |x2|
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.lcm.html
    ///
    /// @param      inValue1
    /// @param      inValue2
    /// @return
    ///				dtype
    ///
    template<typename dtype>
    dtype lcm(dtype inValue1, dtype inValue2) noexcept
    {
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: lcm: Can only be called with integer types.");
        return boost::integer::lcm(inValue1, inValue2);
    }

    //============================================================================
    // Method Description:
    ///						Returns the least common multiple of the values of the input array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.lcm.html
    ///
    /// @param      inArray
    /// @return
    ///				NdArray<double>
    ///
    template<typename dtype>
    dtype lcm(const NdArray<dtype>& inArray)
    {
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: lcm: Can only be called with integer types.");
        return boost::integer::lcm_range(inArray.cbegin(), inArray.cend()).first;
    }

    //============================================================================
    // Method Description:
    ///						Returns x1 * 2^x2.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ldexp.html
    ///
    /// @param				inValue1
    /// @param				inValue2
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype ldexp(dtype inValue1, uint8 inValue2) noexcept
    {
        return static_cast<dtype>(std::ldexp(static_cast<double>(inValue1), inValue2));
    }

    //============================================================================
    // Method Description:
    ///						Returns x1 * 2^x2, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ldexp.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> ldexp(const NdArray<dtype>& inArray1, const NdArray<uint8>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: ldexp: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<dtype> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, uint8 inValue2) noexcept -> dtype
        { return static_cast<dtype>(std::ldexp(static_cast<double>(inValue1), inValue2)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Shift the bits of an integer to the left.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.left_shift.html
    ///
    /// @param				inArray
    /// @param				inNumBits
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> left_shift(const NdArray<dtype>& inArray, uint8 inNumBits)
    {
        return inArray << inNumBits;
    }

    //============================================================================
    // Method Description:
    ///						Return the truth value of (x1 < x2) element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.less.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> less(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 < inArray2;
    }

    //============================================================================
    // Method Description:
    ///						Return the truth value of (x1 <= x2) element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.less_equal.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> less_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 <= inArray2;
    }

    //============================================================================
    // Method Description:
    ///						Return evenly spaced numbers over a specified interval.
    ///
    ///						Returns num evenly spaced samples, calculated over the
    ///						interval[start, stop].
    ///
    ///						The endpoint of the interval can optionally be excluded.
    ///
    ///						Mostly only usefull if called with a floating point type
    ///						for the template argument.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.linspace.html
    ///
    /// @param				inStart
    /// @param				inStop
    /// @param				inNum: number of points (default = 50)
    /// @param				endPoint: include endPoint (default = true)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> linspace(dtype inStart, dtype inStop, uint32 inNum, bool endPoint)
    {
        if (inNum == 0)
        {
            return NdArray<dtype>(0);
        }
        else if (inNum == 1)
        {
            NdArray<dtype> returnArray = { inStart };
            return returnArray;
        }

        if (inStop <= inStart)
        {
            std::string errStr = "ERROR: linspace: stop value must be greater than the start value.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        if (endPoint)
        {
            if (inNum == 2)
            {
                NdArray<dtype> returnArray = { inStart, inStop };
                return returnArray;
            }
            else
            {
                NdArray<dtype> returnArray(1, inNum);
                returnArray[0] = inStart;
                returnArray[inNum - 1] = inStop;

                dtype step = (inStop - inStart) / (inNum - 1);
                for (uint32 i = 1; i < inNum - 1; ++i)
                {
                    returnArray[i] = returnArray[i - 1] + step;
                }

                return returnArray;
            }
        }
        else
        {
            if (inNum == 2)
            {
                dtype step = (inStop - inStart) / (inNum);
                NdArray<dtype> returnArray = { inStart, inStart + step };
                return returnArray;
            }
            else
            {
                NdArray<dtype> returnArray(1, inNum);
                returnArray[0] = inStart;

                dtype step = (inStop - inStart) / inNum;
                for (uint32 i = 1; i < inNum; ++i)
                {
                    returnArray[i] = returnArray[i - 1] + step;
                }

                return returnArray;
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						loads a .bin file from the dump() method into an NdArray
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.load.html
    ///
    /// @param
    ///				inFilename
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> load(const std::string& inFilename)
    {
        return fromfile<dtype>(inFilename, "");
    }

    //============================================================================
    // Method Description:
    ///						Natural logarithm.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.log.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    double log(dtype inValue)
    {
        return std::log(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Natural logarithm, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.log.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> log(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return std::log(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the base 10 logarithm of the input array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.log10.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    double log10(dtype inValue)
    {
        return std::log10(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Return the base 10 logarithm of the input array, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.log10.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> log10(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return std::log10(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the natural logarithm of one plus the input array.
    ///
    ///						Calculates log(1 + x).
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.log1p.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    double log1p(dtype inValue)
    {
        return std::log1p(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Return the natural logarithm of one plus the input array, element-wise.
    ///
    ///						Calculates log(1 + x).
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.log1p.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> log1p(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return std::log1p(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Base-2 logarithm of x.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.log2.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    double log2(dtype inValue)
    {
        return std::log2(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Base-2 logarithm of x.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.log2.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> log2(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return std::log2(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the truth value of x1 AND x2 element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.logical_and.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> logical_and(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: logical_and: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<bool> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> bool { return (inValue1 != 0) && (inValue2 != 0); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the truth value of NOT x element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.logical_not.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> logical_not(const NdArray<dtype>& inArray)
    {
        NdArray<bool> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> bool { return inValue == 0; });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the truth value of x1 OR x2 element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.logical_or.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> logical_or(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: logical_or: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<bool> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> bool { return (inValue1 != 0) || (inValue2 != 0); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the truth value of x1 XOR x2 element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.logical_xor.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> logical_xor(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: logical_xor: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<bool> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> bool { return (inValue1 != 0) != (inValue2 != 0); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Matrix product of two arrays.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matmul.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> matmul(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1.template dot<dtypeOut>(inArray2);
    }

    //============================================================================
    // Method Description:
    ///						Return the maximum of an array or maximum along an axis.
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> max(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.max(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Element-wise maximum of array elements.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.maximum.html
    ///
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> maximum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: maximum: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<dtype> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> dtype { return std::max(inValue1, inValue2); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the mean along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.mean.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> mean(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.mean(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Compute the median along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.median.html
    ///
    /// @param				inArray
    /// @param  			inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> median(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.median(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Return coordinate matrices from coordinate vectors.
    ///                     Make 2D coordinate arrays for vectorized evaluations of 2D scalar
    ///                     vector fields over 2D grids, given one - dimensional coordinate arrays x1, x2, ..., xn.
    ///                     If input arrays are not one dimensional they will be flattened.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.meshgrid.html
    ///
    /// @param				inICoords
    /// @param  			inJCoords
    ///
    /// @return
    ///				std::pair<NdArray<dtype>, NdArray<dtype> >, i and j matrices
    ///
    template<typename dtype>
    std::pair<NdArray<dtype>, NdArray<dtype> > meshgrid(const NdArray<dtype>& inICoords, const NdArray<dtype>& inJCoords)
    {
        const uint32 numRows = inJCoords.size();
        const uint32 numCols = inICoords.size();
        auto returnArrayI = NdArray<dtype>(numRows, numCols);
        auto returnArrayJ = NdArray<dtype>(numRows, numCols);

        // first the I array
        for (uint32 row = 0; row < numRows; ++row)
        {
            for (uint32 col = 0; col < numCols; ++col)
            {
                returnArrayI(row, col) = inICoords[col];
            }
        }

        // then the I array
        for (uint32 col = 0; col < numCols; ++col)
        {
            for (uint32 row = 0; row < numRows; ++row)
            {
                returnArrayJ(row, col) = inJCoords[row];
            }
        }

        return std::make_pair(returnArrayI, returnArrayJ);
    }

    //============================================================================
    // Method Description:
    ///						Return coordinate matrices from coordinate vectors.
    ///                     Make 2D coordinate arrays for vectorized evaluations of 2D scalar
    ///                     vector fields over 2D grids, given one - dimensional coordinate arrays x1, x2, ..., xn.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.meshgrid.html
    ///
    /// @param				inSlice1
    /// @param  			inSlice2
    ///
    /// @return
    ///				std::pair<NdArray<dtype>, NdArray<dtype> >, i and j matrices
    ///
    template<typename dtype>
    std::pair<NdArray<dtype>, NdArray<dtype> > meshgrid(const Slice& inSlice1, const Slice& inSlice2)
    {
        return meshgrid(arange<dtype>(inSlice1), arange<dtype>(inSlice2));
    }

    //============================================================================
    // Method Description:
    ///						Return the minimum of an array or maximum along an axis.
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> min(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.min(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Element-wise minimum of array elements.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.minimum.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> minimum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: minimum: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<dtype> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> dtype { return std::min(inValue1, inValue2); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return element-wise remainder of division.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.mod.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> mod(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 % inArray2;
    }

    //============================================================================
    // Method Description:
    ///						Multiply arguments element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.multiply.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> multiply(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 * inArray2;
    }

    //============================================================================
    // Method Description:
    ///						Returns the indices of the maximum values along an axis ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanargmax.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<uint32> nanargmax(const NdArray<dtype>& inArray, Axis inAxis)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (auto& value : arrayCopy)
        {
            if (std::isnan(value))
            {
                value = DtypeInfo<dtype>::min();
            }
        }

        return argmax(arrayCopy, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Returns the indices of the minimum values along an axis ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanargmin.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<uint32> nanargmin(const NdArray<dtype>& inArray, Axis inAxis)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (auto& value : arrayCopy)
        {
            if (std::isnan(value))
            {
                value = DtypeInfo<dtype>::max();
            }
        }

        return argmin(arrayCopy, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Return the cumulative product of elements along a given axis ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nancumprod.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> nancumprod(const NdArray<dtype>& inArray, Axis inAxis)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (auto& value : arrayCopy)
        {
            if (std::isnan(value))
            {
                value = 1;
            }
        }

        return cumprod<dtypeOut>(arrayCopy, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Return the cumulative sum of the elements along a given axis ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nancumsum.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> nancumsum(const NdArray<dtype>& inArray, Axis inAxis)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (auto& value : arrayCopy)
        {
            if (std::isnan(value))
            {
                value = 0;
            }
        }

        return cumsum<dtypeOut>(arrayCopy, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Return the maximum of an array or maximum along an axis ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanmax.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> nanmax(const NdArray<dtype>& inArray, Axis inAxis)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (auto& value : arrayCopy)
        {
            if (std::isnan(value))
            {
                value = DtypeInfo<dtype>::min();
            }
        }

        return max(arrayCopy, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Compute the mean along the specified axis ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanmean.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> nanmean(const NdArray<dtype>& inArray, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                double sum = static_cast<double>(std::accumulate(inArray.cbegin(), inArray.cend(), 0.0,
                    [](dtype inValue1, dtype inValue2) noexcept -> dtype { return std::isnan(inValue2) ? inValue1 : inValue1 + inValue2; }));

                const double numberNonNan = static_cast<double>(std::accumulate(inArray.cbegin(), inArray.cend(), 0.0,
                    [](dtype inValue1, dtype inValue2) noexcept -> dtype { return std::isnan(inValue2) ? inValue1 : inValue1 + 1; }));

                NdArray<double> returnArray = { sum /= numberNonNan };

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();
                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    double sum = static_cast<double>(std::accumulate(inArray.cbegin(row), inArray.cend(row), 0.0,
                        [](dtype inValue1, dtype inValue2) noexcept -> dtype { return std::isnan(inValue2) ? inValue1 : inValue1 + inValue2; }));

                    double numberNonNan = static_cast<double>(std::accumulate(inArray.cbegin(row), inArray.cend(row), 0.0,
                        [](dtype inValue1, dtype inValue2) noexcept -> dtype { return std::isnan(inValue2) ? inValue1 : inValue1 + 1; }));

                    returnArray(0, row) = sum / numberNonNan;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> transposedArray = inArray.transpose();
                const Shape transShape = transposedArray.shape();
                NdArray<double> returnArray(1, transShape.rows);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    double sum = static_cast<double>(std::accumulate(transposedArray.cbegin(row), transposedArray.cend(row), 0.0,
                        [](dtype inValue1, dtype inValue2) noexcept -> dtype { return std::isnan(inValue2) ? inValue1 : inValue1 + inValue2; }));

                    double numberNonNan = static_cast<double>(std::accumulate(transposedArray.cbegin(row), transposedArray.cend(row), 0.0,
                        [](dtype inValue1, dtype inValue2) noexcept -> dtype { return std::isnan(inValue2) ? inValue1 : inValue1 + 1; }));

                    returnArray(0, row) = sum / numberNonNan;
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<double>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Compute the median along the specified axis ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanmedian.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> nanmedian(const NdArray<dtype>& inArray, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::vector<dtype> values;
                for (auto value : inArray)
                {
                    if (!std::isnan(value))
                    {
                        values.push_back(value);
                    }
                }

                const uint32 middle = static_cast<uint32>(values.size()) / 2;
                std::nth_element(values.begin(), values.begin() + middle, values.end());
                NdArray<dtype> returnArray = { values[middle] };

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();
                NdArray<dtype> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    std::vector<dtype> values;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (!std::isnan(inArray(row, col)))
                        {
                            values.push_back(inArray(row, col));
                        }
                    }

                    const uint32 middle = static_cast<uint32>(values.size()) / 2;
                    std::nth_element(values.begin(), values.begin() + middle, values.end());
                    returnArray(0, row) = values[middle];
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> transposedArray = inArray.transpose();
                const Shape inShape = transposedArray.shape();
                NdArray<dtype> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    std::vector<dtype> values;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (!std::isnan(transposedArray(row, col)))
                        {
                            values.push_back(transposedArray(row, col));
                        }
                    }

                    const uint32 middle = static_cast<uint32>(values.size()) / 2;
                    std::nth_element(values.begin(), values.begin() + middle, values.end());
                    returnArray(0, row) = values[middle];
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtype>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Return the minimum of an array or maximum along an axis ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanmin.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> nanmin(const NdArray<dtype>& inArray, Axis inAxis)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (auto& value : arrayCopy)
        {
            if (std::isnan(value))
            {
                value = DtypeInfo<dtype>::max();
            }
        }

        return min(arrayCopy, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Compute the qth percentile of the data along the specified axis, while ignoring nan values.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanpercentile.html
    ///
    /// @param				inArray
    /// @param              inPercentile
    /// @param				inAxis (Optional, default NONE)
    /// @param              inInterpMethod (default linear) choices = ['linear','lower','higher','nearest','midpoint']
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<double> nanpercentile(const NdArray<dtype>& inArray, double inPercentile,
        Axis inAxis, const std::string& inInterpMethod)
    {
        if (inPercentile < 0.0 || inPercentile > 100.0)
        {
            std::string errStr = "ERROR: percentile: input percentile value must be of the range [0, 100].";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        if (inInterpMethod.compare("linear") != 0 &&
            inInterpMethod.compare("lower") != 0 &&
            inInterpMethod.compare("higher") != 0 &&
            inInterpMethod.compare("nearest") != 0 &&
            inInterpMethod.compare("midpoint") != 0)
        {
            std::string errStr = "ERROR: percentile: input interpolation method is not a vaid option.\n";
            errStr += "\tValid options are 'linear', 'lower', 'higher', 'nearest', 'midpoint'.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (utils::essentiallyEqual(inPercentile, 0.0))
                {
                    for (auto value : inArray)
                    {
                        if (!isnan(value))
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(value) };
                            return returnArray;
                        }
                    }
                    return NdArray<dtypeOut>(0);
                }
                else if (utils::essentiallyEqual(inPercentile, 100.0))
                {
                    for (int32 i = static_cast<int32>(inArray.size()) - 1; i > -1; --i)
                    {
                        if (!isnan(inArray[i]))
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(inArray[i]) };
                            return returnArray;
                        }
                    }
                    return NdArray<dtypeOut>(0);
                }

                std::vector<double> arrayCopy;
                uint32 numNonNan = 0;
                for (auto value : inArray)
                {
                    if (!isnan(value))
                    {
                        arrayCopy.push_back(value);
                        ++numNonNan;
                    }
                }

                if (arrayCopy.size() < 2)
                {
                    return NdArray<dtypeOut>(0);
                }

                const int32 i = static_cast<int32>(std::floor(static_cast<double>(numNonNan - 1) * inPercentile / 100.0));
                const uint32 indexLower = static_cast<uint32>(clip<uint32>(i, 0, numNonNan - 2));

                std::sort(arrayCopy.begin(), arrayCopy.end());

                if (inInterpMethod.compare("linear") == 0)
                {
                    const double percentI = static_cast<double>(indexLower) / static_cast<double>(numNonNan - 1);
                    const double fraction = (inPercentile / 100.0 - percentI) /
                        (static_cast<double>(indexLower + 1) / static_cast<double>(numNonNan - 1) - percentI);

                    const double returnValue = arrayCopy[indexLower] + (arrayCopy[indexLower + 1] - arrayCopy[indexLower]) * fraction;
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(returnValue) };
                    return returnArray;
                }
                else if (inInterpMethod.compare("lower") == 0)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower]) };
                    return returnArray;
                }
                else if (inInterpMethod.compare("higher") == 0)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower + 1]) };
                    return returnArray;
                }
                else if (inInterpMethod.compare("nearest") == 0)
                {
                    const double percent = inPercentile / 100.0;
                    const double percent1 = static_cast<double>(indexLower) / static_cast<double>(numNonNan - 1);
                    const double percent2 = static_cast<double>(indexLower + 1) / static_cast<double>(numNonNan - 1);
                    const double diff1 = percent - percent1;
                    const double diff2 = percent2 - percent;

                    switch (argmin<double>({ diff1, diff2 }).item())
                    {
                        case 0:
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower]) };
                            return returnArray;
                        }
                        case 1:
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower + 1]) };
                            return returnArray;
                        }
                    }
                }
                else if (inInterpMethod.compare("midpoint") == 0)
                {
                    NdArray<dtypeOut> returnArray = { (arrayCopy[indexLower] + arrayCopy[indexLower + 1]) / 2.0 };
                    return returnArray;
                }
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();

                NdArray<dtypeOut> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    NdArray<dtypeOut> outValue = nanpercentile<dtypeOut>(NdArray<dtype>(inArray.cbegin(row), inArray.cend(row)),
                        inPercentile, Axis::NONE, inInterpMethod);

                    if (outValue.size() == 1)
                    {
                        returnArray[row] = outValue.item();
                    }
                    else
                    {
                        returnArray[row] = constants::nan;
                    }
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> arrayTrans = inArray.transpose();
                const Shape inShape = arrayTrans.shape();

                NdArray<dtypeOut> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    NdArray<dtypeOut> outValue = nanpercentile<dtypeOut>(NdArray<dtype>(arrayTrans.cbegin(row), arrayTrans.cend(row)),
                        inPercentile, Axis::NONE, inInterpMethod);

                    if (outValue.size() == 1)
                    {
                        returnArray[row] = outValue.item();
                    }
                    else
                    {
                        returnArray[row] = constants::nan;
                    }
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtypeOut>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanprod.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> nanprod(const NdArray<dtype>& inArray, Axis inAxis)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (auto& value : arrayCopy)
        {
            if (std::isnan(value))
            {
                value = 1;
            }
        }

        return prod<dtypeOut>(arrayCopy, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with nans.
    ///                     Only really works for dtype = float/double
    ///
    /// @param
    ///				inSquareSize
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> nans(uint32 inSquareSize)
    {
        return full(inSquareSize, static_cast<dtype>(constants::nan));
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with nans.
    ///                     Only really works for dtype = float/double
    ///
    /// @param				inNumRows
    /// @param				inNumCols
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> nans(uint32 inNumRows, uint32 inNumCols)
    {
        return full(inNumRows, inNumCols, static_cast<dtype>(constants::nan));
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with nans.
    ///                     Only really works for dtype = float/double
    ///
    /// @param
    ///				inShape
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> nans(const Shape& inShape)
    {
        return full(inShape, static_cast<dtype>(constants::nan));
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with nans.
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> nans_like(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        returnArray.nans();
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the standard deviation along the specified axis, while ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanstd.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> nanstdev(const NdArray<dtype>& inArray, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                double meanValue = nanmean(inArray, inAxis).item();
                double sum = 0;
                double counter = 0;
                for (auto value : inArray)
                {
                    if (std::isnan(value))
                    {
                        continue;
                    }

                    sum += utils::sqr(static_cast<double>(value) - meanValue);
                    ++counter;
                }
                NdArray<double> returnArray = { std::sqrt(sum / counter) };
                return returnArray;
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();
                NdArray<double> meanValue = nanmean(inArray, inAxis);
                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    double sum = 0;
                    double counter = 0;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (std::isnan(inArray(row, col)))
                        {
                            continue;
                        }

                        sum += utils::sqr(static_cast<double>(inArray(row, col)) - meanValue[row]);
                        ++counter;
                    }
                    returnArray(0, row) = std::sqrt(sum / counter);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<double> meanValue = nanmean(inArray, inAxis);
                NdArray<dtype> transposedArray = inArray.transpose();
                const Shape inShape = transposedArray.shape();
                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    double sum = 0;
                    double counter = 0;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (std::isnan(transposedArray(row, col)))
                        {
                            continue;
                        }

                        sum += utils::sqr(static_cast<double>(transposedArray(row, col)) - meanValue[row]);
                        ++counter;
                    }
                    returnArray(0, row) = std::sqrt(sum / counter);
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<double>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nansum.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> nansum(const NdArray<dtype>& inArray, Axis inAxis)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (auto& value : arrayCopy)
        {
            if (std::isnan(value))
            {
                value = 0;
            }
        }

        return sum<dtypeOut>(arrayCopy, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Compute the variance along the specified axis, while ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanvar.html
    ///
    /// @param			inArray
    /// @param			inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> nanvar(const NdArray<dtype>& inArray, Axis inAxis)
    {
        NdArray<double> stdValues = nanstdev(inArray, inAxis);
        for (auto& value : stdValues)
        {
            value *= value;
        }
        return stdValues;
    }

    //============================================================================
    // Method Description:
    ///						Returns the number of bytes held by the array
    ///
    /// @param
    ///				inArray
    /// @return
    ///				number of bytes
    ///
    template<typename dtype>
    uint64 nbytes(const NdArray<dtype>& inArray) noexcept
    {
        return inArray.nbytes();
    }

    //============================================================================
    // Method Description:
    ///						Return the array with the same data viewed with a
    ///						different byte order. only works for integer types,
    ///						floating point types will not compile and you will
    ///						be confused as to why...
    ///
    ///
    /// @param				inValue
    /// @param				inEndianess
    ///
    /// @return
    ///				inValue
    ///
    template<typename dtype>
    dtype newbyteorder(dtype inValue, Endian inEndianess)
    {
        NdArray<dtype> valueArray = { inValue };
        return valueArray.newbyteorder(inEndianess).item();
    }

    //============================================================================
    // Method Description:
    ///						Return the array with the same data viewed with a
    ///						different byte order. only works for integer types,
    ///						floating point types will not compile and you will
    ///						be confused as to why...
    ///
    ///
    /// @param				inArray
    /// @param				inEndianess
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> newbyteorder(const NdArray<dtype>& inArray, Endian inEndianess)
    {
        return inArray.newbyteorder(inEndianess);
    }

    //============================================================================
    // Method Description:
    ///						Numerical negative, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.negative.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> negative(const NdArray<dtype>& inArray)
    {
        NdArray<dtypeOut> returnArray = inArray.template astype<dtypeOut>();
        return returnArray *= -1;
    }

    //============================================================================
    // Method Description:
    ///						Return the indices of the flattened array of the
    ///						elements that are non-zero.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nonzero.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<uint32> nonzero(const NdArray<dtype>& inArray)
    {
        return inArray.nonzero();
    }

    //============================================================================
    // Method Description:
    ///						Matrix or vector norm.
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> norm(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.template norm<dtypeOut>(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Return (x1 != x2) element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.not_equal.html
    ///
    /// @param			inArray1
    /// @param			inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> not_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 != inArray2;
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with ones.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ones.html
    ///
    /// @param
    ///				inSquareSize
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> ones(uint32 inSquareSize)
    {
        return full(inSquareSize, static_cast<dtype>(1));
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with ones.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ones.html
    ///
    /// @param			inNumRows
    /// @param			inNumCols
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> ones(uint32 inNumRows, uint32 inNumCols)
    {
        return full(inNumRows, inNumCols, static_cast<dtype>(1));
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with ones.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ones.html
    ///
    /// @param
    ///				inShape
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> ones(const Shape& inShape)
    {
        return full(inShape, static_cast<dtype>(1));
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with ones.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ones_like.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> ones_like(const NdArray<dtype>& inArray)
    {
        NdArray<dtypeOut> returnArray(inArray.shape());
        returnArray.ones();
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Pads an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.pad.html
    ///
    /// @param				inArray
    /// @param				inPadWidth
    /// @param				inPadValue
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> pad(const NdArray<dtype>& inArray, uint16 inPadWidth, dtype inPadValue)
    {
        const Shape inShape = inArray.shape();
        Shape outShape(inShape);
        outShape.rows += 2 * inPadWidth;
        outShape.cols += 2 * inPadWidth;

        NdArray<dtype> returnArray(outShape);
        returnArray.fill(inPadValue);
        returnArray.put(Slice(inPadWidth, inPadWidth + inShape.rows), Slice(inPadWidth, inPadWidth + inShape.cols), inArray);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Rearranges the elements in the array in such a way that
    ///						value of the element in kth position is in the position it
    ///						would be in a sorted array. All elements smaller than the kth
    ///						element are moved before this element and all equal or greater
    ///						are moved behind it. The ordering of the elements in the two
    ///						partitions is undefined.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.partition.html
    ///
    /// @param              inArray
    /// @param				inKth: kth element
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> partition(const NdArray<dtype>& inArray, uint32 inKth, Axis inAxis)
    {
        NdArray<dtype> returnArray(inArray);
        returnArray.partition(inKth, inAxis);
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the qth percentile of the data along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.percentile.html
    ///
    /// @param				inArray
    /// @param				inPercentile: percentile must be in the range [0, 100]
    /// @param				inAxis (Optional, default NONE)
    /// @param				inInterpMethod (Optional) interpolation method
    ///					linear: i + (j - i) * fraction, where fraction is the fractional part of the index surrounded by i and j.
    ///					lower : i.
    ///					higher : j.
    ///					nearest : i or j, whichever is nearest.
    ///					midpoint : (i + j) / 2.
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> percentile(const NdArray<dtype>& inArray, double inPercentile,
        Axis inAxis, const std::string& inInterpMethod)
    {
        if (inPercentile < 0.0 || inPercentile > 100.0)
        {
            std::string errStr = "ERROR: percentile: input percentile value must be of the range [0, 100].";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        if (inInterpMethod.compare("linear") != 0 &&
            inInterpMethod.compare("lower") != 0 &&
            inInterpMethod.compare("higher") != 0 &&
            inInterpMethod.compare("nearest") != 0 &&
            inInterpMethod.compare("midpoint") != 0)
        {
            std::string errStr = "ERROR: percentile: input interpolation method is not a vaid option.\n";
            errStr += "\tValid options are 'linear', 'lower', 'higher', 'nearest', 'midpoint'.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (utils::essentiallyEqual(inPercentile, 0.0))
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(*inArray.cbegin()) };
                    return returnArray;
                }
                else if (utils::essentiallyEqual(inPercentile, 100.0))
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(*inArray.cend()) };
                    return returnArray;
                }

                const int32 i = static_cast<int32>(std::floor(static_cast<double>(inArray.size() - 1) * inPercentile / 100.0));
                const uint32 indexLower = static_cast<uint32>(clip<uint32>(i, 0, inArray.size() - 2));

                NdArray<double> arrayCopy = inArray.template astype<double>();
                std::sort(arrayCopy.begin(), arrayCopy.end());

                if (inInterpMethod.compare("linear") == 0)
                {
                    const double percentI = static_cast<double>(indexLower) / static_cast<double>(inArray.size() - 1);
                    const double fraction = (inPercentile / 100.0 - percentI) /
                        (static_cast<double>(indexLower + 1) / static_cast<double>(inArray.size() - 1) - percentI);

                    const double returnValue = arrayCopy[indexLower] + (arrayCopy[indexLower + 1] - arrayCopy[indexLower]) * fraction;
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(returnValue) };
                    return returnArray;
                }
                else if (inInterpMethod.compare("lower") == 0)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower]) };
                    return returnArray;
                }
                else if (inInterpMethod.compare("higher") == 0)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower + 1]) };
                    return returnArray;
                }
                else if (inInterpMethod.compare("nearest") == 0)
                {
                    const double percent = inPercentile / 100.0;
                    const double percent1 = static_cast<double>(indexLower) / static_cast<double>(inArray.size() - 1);
                    const double percent2 = static_cast<double>(indexLower + 1) / static_cast<double>(inArray.size() - 1);
                    const double diff1 = percent - percent1;
                    const double diff2 = percent2 - percent;

                    switch (argmin<double>({ diff1, diff2 }).item())
                    {
                        case 0:
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower]) };
                            return returnArray;
                        }
                        case 1:
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower + 1]) };
                            return returnArray;
                        }
                    }
                }
                else if (inInterpMethod.compare("midpoint") == 0)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>((arrayCopy[indexLower] + arrayCopy[indexLower + 1]) / 2.0) };
                    return returnArray;
                }
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();

                NdArray<dtypeOut> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray[row] = percentile<dtypeOut>(NdArray<dtype>(inArray.cbegin(row), inArray.cend(row)),
                        inPercentile, Axis::NONE, inInterpMethod).item();
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> arrayTrans = inArray.transpose();
                const Shape inShape = arrayTrans.shape();

                NdArray<dtypeOut> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray[row] = percentile<dtypeOut>(NdArray<dtype>(arrayTrans.cbegin(row), arrayTrans.cend(row)),
                        inPercentile, Axis::NONE, inInterpMethod).item();
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtypeOut>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Raises the elements of the array to the input integer power
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param				inValue
    /// @param				inExponent
    /// @return
    ///				value raised to the power
    ///
    template<typename dtypeOut = double, typename dtype>
    dtypeOut power(dtype inValue, uint8 inExponent)
    {
        return utils::power(static_cast<dtypeOut>(inValue), inExponent);
    }

    //============================================================================
    // Method Description:
    ///						Raises the elements of the array to the input integer power
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param				inArray
    /// @param				inExponent
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> power(const NdArray<dtype>& inArray, uint8 inExponent)
    {
        NdArray<dtypeOut> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [inExponent](dtype inValue) noexcept -> dtypeOut
        { return utils::power(static_cast<dtypeOut>(inValue), inExponent); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Raises the elements of the array to the input integer powers
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param				inArray
    /// @param				inExponents
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> power(const NdArray<dtype>& inArray, const NdArray<uint8>& inExponents)
    {
        if (inArray.shape() != inExponents.shape())
        {
            std::string errStr = "ERROR: power: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<dtypeOut> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), inExponents.cbegin(), returnArray.begin(),
            [](dtype inValue, uint8 inExponent) noexcept -> dtypeOut 
        { return utils::power(static_cast<dtypeOut>(inValue), inExponent); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Raises the elements of the array to the input floating point power
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param				inValue
    /// @param				inExponent
    /// @return
    ///				value raised to the power
    ///
    template<typename dtype>
    double powerf(dtype inValue, double inExponent)
    {
        return utils::powerf(inValue, inExponent);
    }

    //============================================================================
    // Method Description:
    ///						Raises the elements of the array to the input floating point power
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param				inArray
    /// @param				inExponent
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> powerf(const NdArray<dtype>& inArray, double inExponent)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [inExponent](dtype inValue) noexcept -> double
        { return utils::powerf(inValue, inExponent); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Raises the elements of the array to the input floating point powers
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param				inArray
    /// @param				inExponents
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> powerf(const NdArray<dtype>& inArray, const NdArray<double>& inExponents)
    {
        if (inArray.shape() != inExponents.shape())
        {
            std::string errStr = "ERROR: powerf: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), inExponents.cbegin(), returnArray.begin(),
            [](dtype inValue, double inExponent) noexcept -> double
        { return utils::powerf(inValue, inExponent); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Prints the array to the console.
    ///
    /// @param
    ///				inArray
    /// @return
    ///				None
    ///
    template<typename dtype>
    void print(const NdArray<dtype>& inArray)
    {
        std::cout << inArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the product of array elements over a given axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.prod.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> prod(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.template prod<dtypeOut>(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Range of values (maximum - minimum) along an axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ptp.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> ptp(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.ptp(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Replaces specified elements of an array with given values.
    ///						The indexing works on the flattened target array
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.put.html
    ///
    /// @param				inArray
    /// @param				inIndices
    /// @param				inValues
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& put(NdArray<dtype>& inArray, const NdArray<uint32>& inIndices, const NdArray<dtype>& inValues)
    {
        inArray.put(inIndices, inValues);
        return inArray;
    }

    //============================================================================
    // Method Description:
    ///						Changes elements of an array based on conditional and input values.
    ///
    ///						Sets a.flat[n] = values[n] for each n where mask.flat[n] == True.
    ///
    ///						If values is not the same size as a and mask then it will repeat.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.putmask.html
    ///
    /// @param				inArray
    /// @param				inMask
    /// @param				inValue
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& putmask(NdArray<dtype>& inArray, const NdArray<bool>& inMask, dtype inValue)
    {
        inArray.putMask(inMask, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    ///						Changes elements of an array based on conditional and input values.
    ///
    ///						Sets a.flat[n] = values[n] for each n where mask.flat[n] == True.
    ///
    ///						If values is not the same size as a and mask then it will repeat.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.putmask.html
    ///
    /// @param				inArray
    /// @param				inMask
    /// @param				inValues
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& putmask(NdArray<dtype>& inArray, const NdArray<bool>& inMask, const NdArray<dtype>& inValues)
    {
        inArray.putMask(inMask, inValues);
        return inArray;
    }

    //============================================================================
    // Method Description:
    ///						Convert angles from radians to degrees.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.rad2deg.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    double rad2deg(dtype inValue) noexcept
    {
        return inValue * 180.0 / constants::pi;
    }

    //============================================================================
    // Method Description:
    ///						Convert angles from radians to degrees.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.rad2deg.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> rad2deg(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return rad2deg(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Convert angles from degrees to radians.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.radians.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double radians(dtype inValue) noexcept
    {
        return deg2rad(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Convert angles from degrees to radians.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.radians.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> radians(const NdArray<dtype>& inArray)
    {
        return deg2rad(inArray);
    }

    //============================================================================
    // Method Description:
    ///						Return the reciprocal of the argument, element-wise.
    ///
    ///						Calculates 1 / x.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.reciprocal.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> reciprocal(const NdArray<dtype>& inArray)
    {
        NdArray<dtypeOut> returnArray(inArray.shape());
        uint32 counter = 0;
        for (auto value : inArray)
        {
            returnArray[counter++] = static_cast<dtypeOut>(1.0 / static_cast<double>(value));
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return remainder of division.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.remainder.html
    ///
    /// @param				inValue1
    /// @param				inValue2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    dtypeOut remainder(dtype inValue1, dtype inValue2) noexcept
    {
        return std::remainder(static_cast<dtypeOut>(inValue1), static_cast<dtypeOut>(inValue2));
    }

    //============================================================================
    // Method Description:
    ///						Return element-wise remainder of division.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.remainder.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> remainder(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: remainder: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<dtypeOut> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> dtypeOut
        { return std::remainder(static_cast<dtypeOut>(inValue1), static_cast<dtypeOut>(inValue2)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Repeat elements of an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.repeat.html
    ///
    /// @param				inArray
    /// @param				inNumRows
    /// @param				inNumCols
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> repeat(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        return inArray.repeat(inNumRows, inNumCols);
    }

    //============================================================================
    // Method Description:
    ///						Repeat elements of an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.repeat.html
    ///
    /// @param				inArray
    /// @param				inRepeatShape
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> repeat(const NdArray<dtype>& inArray, const Shape& inRepeatShape)
    {
        return inArray.repeat(inRepeatShape);
    }

    //============================================================================
    // Method Description:
    ///						Gives a new shape to an array without changing its data.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.reshape.html
    ///
    /// @param				inArray
    /// @param				inNumRows
    /// @param				inNumCols
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& reshape(NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        inArray.reshape(inNumRows, inNumCols);
        return inArray;
    }

    //============================================================================
    // Method Description:
    ///						Gives a new shape to an array without changing its data.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.reshape.html
    ///
    /// @param				inArray
    /// @param				inNewShape
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& reshape(NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        inArray.reshape(inNewShape);
        return inArray;
    }

    //============================================================================
    // Method Description:
    ///						Change shape and size of array in-place. All previous
    ///						data of the array is lost.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.resize.html
    ///
    /// @param				inArray
    /// @param				inNumRows
    /// @param				inNumCols
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& resizeFast(NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        inArray.resizeFast(inNumRows, inNumCols);
        return inArray;
    }

    //============================================================================
    // Method Description:
    ///						Change shape and size of array in-place. All previous
    ///						data of the array is lost.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.resize.html
    ///
    /// @param				inArray
    /// @param				inNewShape
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& resizeFast(NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        inArray.resizeFast(inNewShape);
        return inArray;
    }

    //============================================================================
    // Method Description:
    ///						Return a new array with the specified shape. If new shape
    ///						is larger than old shape then array will be padded with zeros.
    ///						If new shape is smaller than the old shape then the data will
    ///						be discarded.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.resize.html
    ///
    /// @param				inArray
    /// @param				inNumRows
    /// @param				inNumCols
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& resizeSlow(NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        inArray.resizeSlow(inNumRows, inNumCols);
        return inArray;
    }

    //============================================================================
    // Method Description:
    ///						Return a new array with the specified shape. If new shape
    ///						is larger than old shape then array will be padded with zeros.
    ///						If new shape is smaller than the old shape then the data will
    ///						be discarded.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.resize.html
    ///
    /// @param				inArray
    /// @param				inNewShape
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& resizeSlow(NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        inArray.resizeSlow(inNewShape);
        return inArray;
    }

    //============================================================================
    // Method Description:
    ///						Shift the bits of an integer to the right.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.right_shift.html
    ///
    /// @param				inArray
    /// @param				inNumBits
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> right_shift(const NdArray<dtype>& inArray, uint8 inNumBits)
    {
        return inArray >> inNumBits;
    }

    //============================================================================
    // Method Description:
    ///						Round value to the nearest integer.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.rint.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype rint(dtype inValue)
    {
        return std::rint(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Round elements of the array to the nearest integer.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.rint.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> rint(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return std::rint(static_cast<double>(inValue)); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the root mean square (RMS) along the specified axis.
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> rms(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.rms(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Roll array elements along a given axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.roll.html
    ///
    /// @param				inArray
    /// @param				inShift: (elements to shift, positive means forward, negative means backwards)
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> roll(const NdArray<dtype>& inArray, int32 inShift, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                uint32 shift = std::abs(inShift) % inArray.size();
                if (inShift > 0)
                {
                    shift = inArray.size() - shift;
                }

                NdArray<dtype> returnArray(inArray);
                std::rotate(returnArray.begin(), returnArray.begin() + shift, returnArray.end());

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();

                uint32 shift = std::abs(inShift) % inShape.cols;
                if (inShift > 0)
                {
                    shift = inShape.cols - shift;
                }

                NdArray<dtype> returnArray(inArray);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    std::rotate(returnArray.begin(row), returnArray.begin(row) + shift, returnArray.end(row));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                const Shape inShape = inArray.shape();

                uint32 shift = std::abs(inShift) % inShape.rows;
                if (inShift > 0)
                {
                    shift = inShape.rows - shift;
                }

                NdArray<dtype> returnArray = inArray.transpose();
                for (uint32 row = 0; row < inShape.cols; ++row)
                {
                    std::rotate(returnArray.begin(row), returnArray.begin(row) + shift, returnArray.end(row));
                }

                return returnArray.transpose();
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtype>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Rotate an array by 90 degrees counter clockwise in the plane.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.rot90.html
    ///
    /// @param				inArray
    /// @param				inK: the number of times to rotate 90 degrees
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> rot90(const NdArray<dtype>& inArray, uint8 inK)
    {
        inK %= 4;
        switch (inK)
        {
            case 1:
            {
                return flipud(inArray.transpose());
            }
            case 2:
            {
                return flip(inArray, Axis::NONE);
            }
            case 3:
            {
                return fliplr(inArray.transpose());
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtype>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Round value to the given number of decimals.
    ///
    /// @param				inValue
    /// @param				inDecimals
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype round(dtype inValue, uint8 inDecimals)
    {
        NdArray<dtype> input = { inValue };
        return input.round(inDecimals).item();
    }

    //============================================================================
    // Method Description:
    ///						Round an array to the given number of decimals.
    ///
    /// @param				inArray
    /// @param				inDecimals
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> round(const NdArray<dtype>& inArray, uint8 inDecimals)
    {
        return inArray.copy().round(inDecimals);
    }

    //============================================================================
    // Method Description:
    ///						Stack arrays in sequence vertically (row wise).
    ///
    /// @param
    ///				inArrayList: {list} of arrays to stack
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> row_stack(const std::initializer_list<NdArray<dtype> >& inArrayList)
    {
        // first loop through to calculate the final size of the array
        Shape finalShape;
        for (auto& ndarray : inArrayList)
        {
            if (finalShape.isnull())
            {
                finalShape = ndarray.shape();
            }
            else if (ndarray.shape().cols != finalShape.cols)
            {
                std::string errStr = "ERROR: row_stack: input arrays must have the same number of columns.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }
            else
            {
                finalShape.rows += ndarray.shape().rows;
            }
        }

        // now that we know the final size, contruct the output array
        NdArray<dtype> returnArray(finalShape);
        uint32 rowStart = 0;
        for (auto& ndarray : inArrayList)
        {
            const Shape theShape = ndarray.shape();
            for (uint32 row = 0; row < theShape.rows; ++row)
            {
                for (uint32 col = 0; col < theShape.cols; ++col)
                {
                    returnArray(rowStart + row, col) = ndarray(row, col);
                }
            }
            rowStart += theShape.rows;
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Find the set difference of two arrays.
    ///
    ///						Return the sorted, unique values in ar1 that are not in ar2.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.setdiff1d.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> setdiff1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        std::vector<dtype> res(inArray1.size() + inArray2.size());
        std::set<dtype> in1(inArray1.cbegin(), inArray1.cend());
        std::set<dtype> in2(inArray2.cbegin(), inArray2.cend());

        const typename std::vector<dtype>::iterator iter = std::set_difference(in1.begin(), in1.end(),
            in2.begin(), in2.end(), res.begin());
        res.resize(iter - res.begin());
        return NdArray<dtype>(res);
    }

    //============================================================================
    // Method Description:
    ///						Return the shape of the array
    ///
    /// @param
    ///				inArray
    /// @return
    ///				Shape
    ///
    template<typename dtype>
    Shape shape(const NdArray<dtype>& inArray)
    {
        return inArray.shape();
    }

    //============================================================================
    // Method Description:
    ///						Returns an element-wise indication of the sign of a number.
    ///
    ///						The sign function returns - 1 if x < 0, 0 if x == 0, 1 if x > 0.
    ///						nan is returned for nan inputs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sign.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    int8 sign(dtype inValue) noexcept
    {
        if (inValue < 0)
        {
            return -1;
        }
        else if (inValue > 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    //============================================================================
    // Method Description:
    ///						Returns an element-wise indication of the sign of a number.
    ///
    ///						The sign function returns - 1 if x < 0, 0 if x == 0, 1 if x > 0.
    ///						nan is returned for nan inputs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sign.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<int8> sign(const NdArray<dtype>& inArray)
    {
        NdArray<int8> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> int8 { return sign(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Returns element-wise True where signbit is set (less than zero).
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.signbit.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    bool signbit(dtype inValue) noexcept
    {
        return inValue < 0 ? true : false;
    }

    //============================================================================
    // Method Description:
    ///						Returns element-wise True where signbit is set (less than zero).
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.signbit.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> signbit(const NdArray<dtype>& inArray)
    {
        NdArray<bool> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> bool { return signbit(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric sine.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sin.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double sin(dtype inValue) noexcept
    {
        return std::sin(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Trigonometric sine, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sin.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> sin(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return sin(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the sinc function.
    ///
    ///						The sinc function is sin(pi*x) / (pi*x).
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sinc.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double sinc(dtype inValue) noexcept
    {
        const double input = static_cast<double>(inValue);
        return std::sin(constants::pi * input) / (constants::pi * input);
    }

    //============================================================================
    // Method Description:
    ///						Return the sinc function.
    ///
    ///						The sinc function is sin(pi*x) / (pi*x).
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sinc.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> sinc(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return sinc(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Hyperbolic sine.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sinh.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double sinh(dtype inValue) noexcept
    {
        return std::sinh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Hyperbolic sine, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sinh.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> sinh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return sinh(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the number of elements.
    ///
    /// @param
    ///				inArray
    /// @return
    ///				uint32 size
    ///
    template<typename dtype>
    uint32 size(const NdArray<dtype>& inArray) noexcept
    {
        return inArray.size();
    }

    //============================================================================
    // Method Description:
    ///						Return a sorted copy of an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sort.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> sort(const NdArray<dtype>& inArray, Axis inAxis)
    {
        NdArray<dtype> returnArray(inArray);
        returnArray.sort(inAxis);
        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the positive square-root of a value.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sqrt.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double sqrt(dtype inValue) noexcept
    {
        return std::sqrt(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Return the positive square-root of an array, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sqrt.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> sqrt(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return sqrt(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the square of an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.square.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype square(dtype inValue) noexcept
    {
        return utils::sqr(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Return the square of an array, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.square.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> square(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> dtype { return square(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the variance along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.stack.html
    ///
    /// @param      inArrayList: {list} of arrays to stack
    /// @param      inAxis: axis to stack the input NdArrays
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> stack(const std::initializer_list<NdArray<dtype> >& inArrayList, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::ROW:
            {
                return row_stack(inArrayList);
            }
            case Axis::COL:
            {
                return column_stack(inArrayList);
            }
            default:
            {
                std::string errStr = "ERROR: stack: inAxis must be either ROW or COL.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Compute the standard deviation along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.std.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> stdev(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.stdev(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Sum of array elements over a given axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sum.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> sum(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.template sum<dtypeOut>(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Interchange two axes of an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.swapaxes.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> swapaxes(const NdArray<dtype>& inArray)
    {
        return inArray.swapaxes();
    }

    //============================================================================
    // Method Description:
    ///						Compute tangent.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.tan.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double tan(dtype inValue) noexcept
    {
        return std::tan(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Compute tangent element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.tan.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> tan(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return tan(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute hyperbolic tangent.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.tanh.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				value
    ///
    template<typename dtype>
    double tanh(dtype inValue) noexcept
    {
        return std::tanh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Compute hyperbolic tangent element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.tanh.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> tanh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double { return tanh(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Construct an array by repeating A the number of times given by reps.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.tile.html
    ///
    /// @param				inArray
    /// @param				inNumRows
    /// @param				inNumCols
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> tile(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        return inArray.repeat(inNumRows, inNumCols);
    }

    //============================================================================
    // Method Description:
    ///						Construct an array by repeating A the number of times given by reps.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.tile.html
    ///
    /// @param				inArray
    /// @param				inReps
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> tile(const NdArray<dtype>& inArray, const Shape& inReps)
    {
        return inArray.repeat(inReps);
    }

    //============================================================================
    // Method Description:
    ///						Write array to a file as text or binary (default)..
    ///						The data produced by this method can be recovered
    ///						using the function fromfile().
    ///
    /// @param				inArray
    /// @param				inFilename
    /// @param				inSep: (Separator between array items for text output. If �� (empty), a binary file is written)
    /// @return
    ///				None
    ///
    template<typename dtype>
    void tofile(const NdArray<dtype>& inArray, const std::string& inFilename, const std::string& inSep)
    {
        return inArray.tofile(inFilename, inSep);
    }

    //============================================================================
    // Method Description:
    ///						Write flattened array to an STL vector
    ///
    /// @param
    ///				inArray
    /// @return
    ///				std::vector
    ///
    template<typename dtype>
    std::vector<dtype> toStlVector(const NdArray<dtype>& inArray)
    {
        return inArray.toStlVector();
    }

    //============================================================================
    // Method Description:
    ///						Return the sum along diagonals of the array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.trace.html
    ///
    /// @param				inArray
    /// @param				inOffset: (Offset from main diaganol, default = 0, negative=above, positve=below)
    /// @param				inAxis (Optional, default ROW)
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    dtypeOut trace(const NdArray<dtype>& inArray, int16 inOffset, Axis inAxis) noexcept
    {
        return inArray.template trace<dtypeOut>(inOffset, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Permute the dimensions of an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.transpose.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> transpose(const NdArray<dtype>& inArray)
    {
        return inArray.transpose();
    }

    //============================================================================
    // Method Description:
    ///						Integrate along the given axis using the composite trapezoidal rule.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.trapz.html
    ///
    /// @param				inArray
    /// @param              dx: (Optional defaults to 1.0)
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> trapz(const NdArray<dtype>& inArray, double dx, Axis inAxis)
    {
        const Shape inShape = inArray.shape();
        switch (inAxis)
        {
            case Axis::COL:
            {
                NdArray<double> returnArray(inShape.rows, 1);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    double sum = 0;
                    for (uint32 col = 0; col < inShape.cols - 1; ++col)
                    {
                        sum += static_cast<double>(inArray(row, col + 1) - inArray(row, col)) / 2.0 +
                            static_cast<double>(inArray(row, col));
                    }

                    returnArray[row] = sum * dx;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> arrayTranspose = inArray.transpose();
                const Shape transShape = arrayTranspose.shape();
                NdArray<double> returnArray(transShape.rows, 1);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    double sum = 0;
                    for (uint32 col = 0; col < transShape.cols - 1; ++col)
                    {
                        sum += static_cast<double>(arrayTranspose(row, col + 1) - arrayTranspose(row, col)) / 2.0 +
                            static_cast<double>(arrayTranspose(row, col));
                    }

                    returnArray[row] = sum * dx;
                }

                return returnArray;
            }
            case Axis::NONE:
            {
                double sum = 0.0;
                for (uint32 i = 0; i < inArray.size() - 1; ++i)
                {
                    sum += static_cast<double>(inArray[i + 1] - inArray[i]) / 2.0 + static_cast<double>(inArray[i]);
                }

                NdArray<double> returnArray = { sum * dx };
                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<double>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Integrate along the given axis using the composite trapezoidal rule.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.trapz.html
    ///
    /// @param				inArrayY
    /// @param				inArrayX
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> trapz(const NdArray<dtype>& inArrayY, const NdArray<dtype>& inArrayX, Axis inAxis)
    {
        const Shape inShapeY = inArrayY.shape();
        const Shape inShapeX = inArrayX.shape();

        if (inShapeY != inShapeX)
        {
            std::string errStr = "ERROR: trapz: input x and y arrays should be the same shape.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        switch (inAxis)
        {
            case Axis::COL:
            {
                NdArray<double> returnArray(inShapeY.rows, 1);
                for (uint32 row = 0; row < inShapeY.rows; ++row)
                {
                    double sum = 0;
                    for (uint32 col = 0; col < inShapeY.cols - 1; ++col)
                    {
                        const double dx = static_cast<double>(inArrayX(row, col + 1) - inArrayX(row, col));
                        sum += dx * (static_cast<double>(inArrayY(row, col + 1) - inArrayY(row, col)) / 2.0 +
                            static_cast<double>(inArrayY(row, col)));
                    }

                    returnArray[row] = sum;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> arrayYTranspose = inArrayY.transpose();
                NdArray<dtype> arrayXTranspose = inArrayX.transpose();
                const Shape transShape = arrayYTranspose.shape();
                NdArray<double> returnArray(transShape.rows, 1);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    double sum = 0;
                    for (uint32 col = 0; col < transShape.cols - 1; ++col)
                    {
                        const double dx = static_cast<double>(arrayXTranspose(row, col + 1) - arrayXTranspose(row, col));
                        sum += dx * (static_cast<double>(arrayYTranspose(row, col + 1) - arrayYTranspose(row, col)) / 2.0 +
                            static_cast<double>(arrayYTranspose(row, col)));
                    }

                    returnArray[row] = sum;
                }

                return returnArray;
            }
            case Axis::NONE:
            {
                double sum = 0.0;
                for (uint32 i = 0; i < inArrayY.size() - 1; ++i)
                {
                    const double dx = static_cast<double>(inArrayX[i + 1] - inArrayX[i]);
                    sum += dx * (static_cast<double>(inArrayY[i + 1] - inArrayY[i]) / 2.0 + static_cast<double>(inArrayY[i]));
                }

                NdArray<double> returnArray = { sum };
                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<double>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						An array with ones at and below the given diagonal and zeros elsewhere.
    ///
    /// @param				inN: number of rows and cols
    /// @param				inOffset: (the sub-diagonal at and below which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> tril(uint32 inN, int32 inOffset)
    {
        uint32 rowStart = 0;
        uint32 colStart = 0;
        if (inOffset > 0)
        {
            colStart = inOffset;
        }
        else
        {
            rowStart = inOffset * -1;
        }

        NdArray<dtype> returnArray(inN);
        returnArray.zeros();
        for (uint32 row = rowStart; row < inN; ++row)
        {
            for (uint32 col = 0; col < row + colStart + 1 - rowStart; ++col)
            {
                if (col == inN)
                {
                    break;
                }

                returnArray(row, col) = static_cast<dtype>(1);
            }
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						An array with ones at and below the given diagonal and zeros elsewhere.
    ///
    /// @param				inN: number of rows
    /// @param				inM: number of columns
    /// @param				inOffset: (the sub-diagonal at and below which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> tril(uint32 inN, uint32 inM, int32 inOffset)
    {
        uint32 rowStart = 0;
        uint32 colStart = 0;
        if (inOffset > 0)
        {
            colStart = inOffset;
        }
        else if (inOffset < 0)
        {
            rowStart = inOffset * -1;
        }

        NdArray<dtype> returnArray(inN, inM);
        returnArray.zeros();
        for (uint32 row = rowStart; row < inN; ++row)
        {
            for (uint32 col = 0; col < row + colStart + 1 - rowStart; ++col)
            {
                if (col == inM)
                {
                    break;
                }

                returnArray(row, col) = static_cast<dtype>(1);
            }
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///                     Lower triangle of an array.
    ///
    ///                     Return a copy of an array with elements above the k - th diagonal zeroed.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.tril.html
    ///
    /// @param				inArray: number of rows and cols
    /// @param				inOffset: (the sub-diagonal at and below which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> tril(const NdArray<dtype>& inArray, int32 inOffset)
    {
        const Shape inShape = inArray.shape();
        auto outArray = inArray.copy();
        outArray.putMask(triu<bool>(inShape.rows, inShape.cols, inOffset + 1), 0);
        return outArray;
    }

    //============================================================================
    // Method Description:
    ///						An array with ones at and above the given diagonal and zeros elsewhere.
    ///
    /// @param				inN: number of rows and cols
    /// @param				inOffset: (the sub-diagonal at and above which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> triu(uint32 inN, int32 inOffset)
    {
        return tril<dtype>(inN, -inOffset).transpose();
    }

    //============================================================================
    // Method Description:
    ///						An array with ones at and above the given diagonal and zeros elsewhere.
    ///
    /// @param				inN: number of rows
    /// @param				inM: number of columns
    /// @param				inOffset: (the sub-diagonal at and above which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> triu(uint32 inN, uint32 inM, int32 inOffset)
    {
        // because i'm stealing the lines of code from tril and reversing it, this is necessary
        inOffset -= 1;

        uint32 rowStart = 0;
        uint32 colStart = 0;
        if (inOffset > 0)
        {
            colStart = inOffset;
        }
        else if (inOffset < 0)
        {
            rowStart = inOffset * -1;
        }

        NdArray<dtype> returnArray(inN, inM);
        returnArray.ones();
        for (uint32 row = rowStart; row < inN; ++row)
        {
            for (uint32 col = 0; col < row + colStart + 1 - rowStart; ++col)
            {
                if (col == inM)
                {
                    break;
                }

                returnArray(row, col) = static_cast<dtype>(0);
            }
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///                     Upper triangle of an array.
    ///
    ///                     Return a copy of an array with elements below the k - th diagonal zeroed.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.triu.html
    ///
    /// @param				inArray: number of rows and cols
    /// @param				inOffset: (the sub-diagonal at and below which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> triu(const NdArray<dtype>& inArray, int32 inOffset)
    {
        const Shape inShape = inArray.shape();
        auto outArray = inArray.copy();
        outArray.putMask(tril<bool>(inShape.rows, inShape.cols, inOffset - 1), 0);
        return outArray;
    }

    //============================================================================
    // Method Description:
    ///						Trim the leading and/or trailing zeros from a 1-D array or sequence.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.trim_zeros.html
    ///
    /// @param				inArray
    /// @param				inTrim: ("f" = front, "b" = back, "fb" = front and back)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> trim_zeros(const NdArray<dtype>& inArray, const std::string inTrim)
    {
        if (inTrim == "f")
        {
            uint32 place = 0;
            for (auto value : inArray)
            {
                if (value != static_cast<dtype>(0))
                {
                    break;
                }
                else
                {
                    ++place;
                }
            }

            if (place == inArray.size())
            {
                return NdArray<dtype>(0);
            }

            NdArray<dtype> returnArray(1, inArray.size() - place);
            std::copy(inArray.cbegin() + place, inArray.cend(), returnArray.begin());

            return returnArray;
        }
        else if (inTrim == "b")
        {
            uint32 place = inArray.size();
            for (uint32 i = inArray.size() - 1; i > 0; --i)
            {
                if (inArray[i] != static_cast<dtype>(0))
                {
                    break;
                }
                else
                {
                    --place;
                }
            }

            if (place == 0 || (place == 1 && inArray[0] == 0))
            {
                return NdArray<dtype>(0);
            }

            NdArray<dtype> returnArray(1, place);
            std::copy(inArray.cbegin(), inArray.cbegin() + place, returnArray.begin());

            return returnArray;
        }
        else if (inTrim == "fb")
        {
            uint32 placeBegin = 0;
            for (auto value : inArray)
            {
                if (value != static_cast<dtype>(0))
                {
                    break;
                }
                else
                {
                    ++placeBegin;
                }
            }

            if (placeBegin == inArray.size())
            {
                return NdArray<dtype>(0);
            }

            uint32 placeEnd = inArray.size();
            for (uint32 i = inArray.size() - 1; i > 0; --i)
            {
                if (inArray[i] != static_cast<dtype>(0))
                {
                    break;
                }
                else
                {
                    --placeEnd;
                }
            }

            if (placeEnd == 0 || (placeEnd == 1 && inArray[0] == 0))
            {
                return NdArray<dtype>(0);
            }

            NdArray<dtype> returnArray(1, placeEnd - placeBegin);
            std::copy(inArray.cbegin() + placeBegin, inArray.cbegin() + placeEnd, returnArray.begin());

            return returnArray;
        }
        else
        {
            std::string errStr = "ERROR: trim_zeros: trim options are 'f' = front, 'b' = back, 'fb' = front and back.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }
    }

    //============================================================================
    // Method Description:
    ///						Return the truncated value of the input.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.trunc.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype trunc(dtype inValue)
    {
        return std::trunc(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Return the truncated value of the input, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.trunc.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> trunc(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> dtype { return std::trunc(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Find the union of two arrays.
    ///
    ///						Return the unique, sorted array of values that are in
    ///						either of the two input arrays.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.union1d.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> union1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: union1d: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        std::set<dtype> theSet(inArray1.cbegin(), inArray1.cend());
        theSet.insert(inArray2.cbegin(), inArray2.cend());
        return NdArray<dtype>(theSet);
    }

    //============================================================================
    // Method Description:
    ///						Find the unique elements of an array.
    ///
    ///						Returns the sorted unique elements of an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.unique.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> unique(const NdArray<dtype>& inArray)
    {
        std::set<dtype> theSet(inArray.cbegin(), inArray.cend());
        return NdArray<dtype>(theSet);
    }

    //============================================================================
    // Method Description:
    ///						Unwrap by changing deltas between values to 2*pi complement.
    ///                     Unwraps to [-pi, pi].
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.unwrap.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype unwrap(dtype inValue) noexcept
    {
        return std::atan2(std::sin(inValue), std::cos(inValue));
    }

    //============================================================================
    // Method Description:
    ///						Unwrap by changing deltas between values to 2*pi complement.
    ///                     Unwraps to [-pi, pi].
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.unwrap.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> unwrap(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> dtype { return unwrap(inValue); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Compute the variance along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.var.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> var(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return inArray.var(inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Compute the variance along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.vstack.html
    ///
    /// @param
    ///				inArrayList: {list} of arrays to stack
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> vstack(const std::initializer_list<NdArray<dtype> >& inArrayList)
    {
        return row_stack(inArrayList);
    }

    //============================================================================
    // Method Description:
    ///						Return elements, either from x or y, depending on the input mask.
    ///                     The output array contains elements of x where mask is True, and
    ///                     elements from y elsewhere.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.where.html
    ///
    /// @param      inMask
    /// @param      inA
    /// @param      inB
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> where(const NdArray<bool>& inMask, const NdArray<dtype>& inA, const NdArray<dtype>& inB)
    {
        const auto shapeMask = inMask.shape();
        const auto shapeA = inA.shape();
        if (shapeA != inB.shape())
        {
            std::string errStr = "ERROR: where: input inA and inB must be the same shapes.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        if (shapeMask != shapeA)
        {
            std::string errStr = "ERROR: where: input inMask must be the same shape as the input arrays.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        auto outArray = NdArray<dtype>(shapeMask);

        uint32 idx = 0;
        for (auto maskValue : inMask)
        {
            if (maskValue)
            {
                outArray[idx] = inA[idx];
            }
            else
            {
                outArray[idx] = inB[idx];
            }
            ++idx;
        }

        return outArray;
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with zeros.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.zeros.html
    ///
    /// @param
    ///				inSquareSize
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> zeros(uint32 inSquareSize)
    {
        return full(inSquareSize, static_cast<dtype>(0));
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with zeros.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.zeros.html
    ///
    /// @param				inNumRows
    /// @param				inNumCols
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> zeros(uint32 inNumRows, uint32 inNumCols)
    {
        return full(inNumRows, inNumCols, static_cast<dtype>(0));
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with zeros.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.zeros.html
    ///
    /// @param
    ///				inShape
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> zeros(const Shape& inShape)
    {
        return full(inShape, static_cast<dtype>(0));
    }

    //============================================================================
    // Method Description:
    ///						Return a new array of given shape and type, filled with zeros.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.zeros_like.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtypeOut = double, typename dtype>
    NdArray<dtypeOut> zeros_like(const NdArray<dtype>& inArray)
    {
        NdArray<dtypeOut> returnArray(inArray.shape());
        returnArray.zeros();
        return returnArray;
    }
}
