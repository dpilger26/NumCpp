#include "NumCpp.hpp"

#include <complex>
#include <functional>
#include <iostream>
#include <string>
#include <utility>

#ifndef BOOST_PYTHON_STATIC_LIB
#define BOOST_PYTHON_STATIC_LIB
#endif

#ifndef BOOST_NUMPY_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#endif

#include "boost/python.hpp"
#include "boost/python/suite/indexing/vector_indexing_suite.hpp" // needed for returning a std::vector directly
#include "boost/python/return_internal_reference.hpp" // needed for returning references and pointers
#include "boost/python/numpy.hpp" // needed for working with numpy
#define BOOST_LIB_NAME "boost_numpy37"
#include "boost/config/auto_link.hpp"

namespace bp = boost::python;
namespace np = boost::python::numpy;

using namespace nc;
using namespace nc::boostPythonInterface;

//================================================================================

namespace ShapeInterface
{
    bool testListContructor() noexcept
    {
        const Shape test = { 357, 666 };
        if (test.rows == 357 && test.cols == 666)
        {
            return true;
        }

        return false;
    }
}

//================================================================================

namespace NdArrayInterface
{
    template<typename dtype>
    bool test1DListContructor()
    {
        NdArray<dtype> test = { 1,2,3,4,666,357,314159 };
        if (test.size() != 7)
        {
            return false;
        }

        if (test.shape().rows != 1 || test.shape().cols != test.size())
        {
            return false;
        }

        return test[0] == 1 && test[1] == 2 && test[2] == 3 && test[3] == 4 && test[4] == 666 && test[5] == 357 && test[6] == 314159;
    }

    //================================================================================

    template<typename dtype>
    bool test2DListContructor()
    {
        NdArray<dtype> test = { {1,2}, {4,666}, {314159, 9}, {0, 8} };
        if (test.size() != 8)
        {
            return false;
        }

        if (test.shape().rows != 4 || test.shape().cols != 2)
        {
            return false;
        }

        return test[0] == 1 && test[1] == 2 && test[2] == 4 && test[3] == 666 && test[4] == 314159 && test[5] == 9 && test[6] == 0 && test[7] == 8;
    }

    //================================================================================

    template<typename dtype>
    np::ndarray getNumpyArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(inArray);
    }

    //================================================================================

    template<typename dtype>
    void setArray(NdArray<dtype>& self, const np::ndarray& inBoostArray)
    {
        BoostNdarrayHelper newNdArrayHelper(inBoostArray);
        const uint8 numDims = newNdArrayHelper.numDimensions();
        if (numDims > 2)
        {
            std::string errorString = "ERROR: Input array can only have up to 2 dimensions!";
            PyErr_SetString(PyExc_RuntimeError, errorString.c_str());
        }

        self = boost2Nc<dtype>(inBoostArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray all(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.all(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray any(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.any(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray argmax(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.argmax(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray argmin(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.argmin(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray argsort(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.argsort(inAxis));
    }

    //================================================================================

    template<typename dtype>
    dtype back(const NdArray<dtype>& self)
    {
        return self.back();
    }

    //================================================================================

    template<typename dtype>
    dtype backReference(NdArray<dtype>& self)
    {
        return self.back();
    }

    //================================================================================

    template<typename dtype>
    np::ndarray clip(const NdArray<dtype>& self, dtype inMin, dtype inMax)
    {
        return nc2Boost(self.clip(inMin, inMax));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray copy(const NdArray<dtype>& self)
    {
        return nc2Boost(self.copy());
    }

    //================================================================================

    template<typename dtype>
    np::ndarray contains(const NdArray<dtype>& self, dtype inValue, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.contains(inValue, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray cumprod(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.cumprod(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray cumsum(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.cumsum(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray diagonal(const NdArray<dtype>& self, int32 inOffset = 0, Axis inAxis = Axis::ROW)
    {
        return nc2Boost(self.diagonal(inOffset, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray dot(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self.dot(inOtherArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray fill(NdArray<dtype>& self, dtype inFillValue)
    {
        self.fill(inFillValue);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray flatnonzero(const NdArray<dtype>& self)
    {
        return nc2Boost(self.flatnonzero());
    }

    //================================================================================

    template<typename dtype>
    np::ndarray flatten(const NdArray<dtype>& self)
    {
        return nc2Boost(self.flatten());
    }

    //================================================================================

    template<typename dtype>
    dtype front(const NdArray<dtype>& self)
    {
        return self.front();
    }

    //================================================================================

    template<typename dtype>
    dtype frontReference(NdArray<dtype>& self)
    {
        return self.front();
    }

    //================================================================================

    template<typename dtype>
    dtype getValueFlat(NdArray<dtype>& self, int32 inIndex)
    {
        return self.at(inIndex);
    }

    //================================================================================

    template<typename dtype>
    dtype getValueRowCol(NdArray<dtype>& self, int32 inRow, int32 inCol)
    {
        return self.at(inRow, inCol);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray getSlice1D(const NdArray<dtype>& self, const Slice& inSlice)
    {
        return nc2Boost(self.at(inSlice));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray getSlice2D(const NdArray<dtype>& self, const Slice& inRowSlice, const Slice& inColSlice)
    {
        return nc2Boost(self.at(inRowSlice, inColSlice));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray getSlice2DCol(const NdArray<dtype>& self, const Slice& inRowSlice, int32 inColIndex)
    {
        return nc2Boost(self.at(inRowSlice, inColIndex));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray getSlice2DRow(const NdArray<dtype>& self, int32 inRowIndex, const Slice& inColSlice)
    {
        return nc2Boost(self.at(inRowIndex, inColSlice));
    }


    //================================================================================

    template<typename dtype>
    np::ndarray getByIndices(const NdArray<dtype>& self, const NdArray<uint32>& inIndices)
    {
        return nc2Boost(self.getByIndices(inIndices));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray getByMask(const NdArray<dtype>& self, const NdArray<bool>& inMask)
    {
        return nc2Boost(self.getByMask(inMask));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray issorted(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.issorted(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray max(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.max(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray min(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.min(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray mean(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.mean(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray median(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.median(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray newbyteorder(const NdArray<dtype>& self, Endian inEndiness = Endian::NATIVE)
    {
        return nc2Boost(self.newbyteorder(inEndiness));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray none(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.none(inAxis));
    }

    //================================================================================

    template<typename dtype>
    bp::tuple nonzero(const NdArray<dtype>& self)
    {
        auto rowCol = self.nonzero();
        return bp::make_tuple(nc2Boost(rowCol.first), nc2Boost(rowCol.second));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray norm(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.norm(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray ones(NdArray<dtype>& self)
    {
        self.ones();
        return nc2Boost<dtype>(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray partition(NdArray<dtype>& self, uint32 inKth, Axis inAxis = Axis::NONE)
    {
        self.partition(inKth, inAxis);
        return nc2Boost<dtype>(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray prod(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.prod(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray ptp(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.ptp(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putFlat(NdArray<dtype>& self, int32 inIndex, dtype inValue)
    {
        self.put(inIndex, inValue);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putRowCol(NdArray<dtype>& self, int32 inRow, int32 inCol, dtype inValue)
    {
        self.put(inRow, inCol, inValue);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putSlice1DValue(NdArray<dtype>& self, const Slice& inSlice, dtype inValue)
    {
        self.put(inSlice, inValue);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putSlice1DValues(NdArray<dtype>& self, const Slice& inSlice, const np::ndarray& inArrayValues)
    {
        NdArray<dtype> inValues = boost2Nc<dtype>(inArrayValues);
        self.put(inSlice, inValues);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putSlice2DValue(NdArray<dtype>& self, const Slice& inSliceRow, const Slice& inSliceCol, dtype inValue)
    {
        self.put(inSliceRow, inSliceCol, inValue);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putSlice2DValueRow(NdArray<dtype>& self, int32 inRowIndex, const Slice& inSliceCol, dtype inValue)
    {
        self.put(inRowIndex, inSliceCol, inValue);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putSlice2DValueCol(NdArray<dtype>& self, const Slice& inSliceRow, int32 inColIndex, dtype inValue)
    {
        self.put(inSliceRow, inColIndex, inValue);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putSlice2DValues(NdArray<dtype>& self, const Slice& inSliceRow, const Slice& inSliceCol, const np::ndarray& inArrayValues)
    {
        NdArray<dtype> inValues = boost2Nc<dtype>(inArrayValues);
        self.put(inSliceRow, inSliceCol, inValues);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putSlice2DValuesRow(NdArray<dtype>& self, int32 inRowIndex, const Slice& inSliceCol, const np::ndarray& inArrayValues)
    {
        NdArray<dtype> inValues = boost2Nc<dtype>(inArrayValues);
        self.put(inRowIndex, inSliceCol, inValues);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putSlice2DValuesCol(NdArray<dtype>& self, const Slice& inSliceRow, int32 inColIndex, const np::ndarray& inArrayValues)
    {
        NdArray<dtype> inValues = boost2Nc<dtype>(inArrayValues);
        self.put(inSliceRow, inColIndex, inValues);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putMaskSingle(NdArray<dtype>& self, const np::ndarray& inMask, dtype inValue)
    {
        auto mask = boost2Nc<bool>(inMask);
        self.putMask(mask, inValue);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putMaskMultiple(NdArray<dtype>& self, const np::ndarray& inMask, const np::ndarray& inArrayValues)
    {
        auto mask = boost2Nc<bool>(inMask);
        auto inValues = boost2Nc<dtype>(inArrayValues);
        self.putMask(mask, inValues);
        return nc2Boost(self);
    }

    template<typename dtype>
    np::ndarray ravel(NdArray<dtype>& self)
    {
        self.ravel();
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray repeat(const NdArray<dtype>& self, const Shape& inRepeatShape)
    {
        return nc2Boost(self.repeat(inRepeatShape));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray reshapeInt(NdArray<dtype>& self, uint32 size)
    {
        self.reshape(size);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray reshapeValues(NdArray<dtype>& self, int32 inNumRows, int32 inNumCols)
    {
        self.reshape(inNumRows, inNumCols);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray reshapeShape(NdArray<dtype>& self, const Shape& inShape)
    {
        self.reshape(inShape);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray reshapeList(NdArray<dtype>& self, const Shape& inShape)
    {
        self.reshape({ inShape.rows, inShape.cols });
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray replace(NdArray<dtype>& self, dtype oldValue, dtype newValue)
    {
        self.replace(oldValue, newValue);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray resizeFast(NdArray<dtype>& self, const Shape& inShape)
    {
        self.resizeFast(inShape);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray resizeFastList(NdArray<dtype>& self, const Shape& inShape)
    {
        self.resizeFast({ inShape.rows, inShape.cols });
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray resizeSlow(NdArray<dtype>& self, const Shape& inShape)
    {
        self.resizeSlow(inShape);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray resizeSlowList(NdArray<dtype>& self, const Shape& inShape)
    {
        self.resizeSlow({ inShape.rows, inShape.cols });
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray rms(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.rms(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray round(const NdArray<dtype>& self, uint8 inNumDecimals)
    {
        return nc2Boost(self.round(inNumDecimals));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray sort(NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        self.sort(inAxis);
        return nc2Boost(self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray stdev(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.stdev(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray sum(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.sum(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray swapaxes(const NdArray<dtype>& self)
    {
        return nc2Boost(self.swapaxes());
    }

    //================================================================================

    template<typename dtype>
    np::ndarray transpose(const NdArray<dtype>& self)
    {
        return nc2Boost(self.transpose());
    }

    //================================================================================

    template<typename dtype>
    np::ndarray var(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(self.var(inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorPlusScaler(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(self + inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorPlusScalerReversed(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(inScaler + self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorPlusArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self + inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorNegative(const NdArray<dtype>& self)
    {
        return nc2Boost(-self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorMinusScaler(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(self - inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorMinusScalerReversed(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(inScaler - self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorMinusArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self - inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorMultiplyScaler(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(self * inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorMultiplyScalerReversed(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(inScaler * self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorMultiplyArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self * inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorDivideScaler(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(self / inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorDivideScalerReversed(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(inScaler / self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorDivideArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self / inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorModulusScaler(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(self % inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorModulusScalerReversed(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(inScaler % self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorModulusArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self % inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseOrScaler(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(self | inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseOrScalerReversed(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(inScaler | self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseOrArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self | inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseAndScaler(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(self & inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseAndScalerReversed(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(inScaler & self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseAndArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self & inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseXorScaler(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(self ^ inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseXorScalerReversed(const NdArray<dtype>& self, dtype inScaler)
    {
        return nc2Boost(inScaler ^ self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseXorArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self ^ inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseNot(const NdArray<dtype>& self)
    {
        return nc2Boost(~self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalAndArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self && inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalAndScalar(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(self && inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalAndScalarReversed(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(inValue && self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalOrArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self || inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalOrScalar(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(self || inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalOrScalarReversed(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(inValue || self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorNot(const NdArray<dtype>& self)
    {
        return nc2Boost(!self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorEqualityScaler(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(self == inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorEqualityScalerReversed(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(inValue == self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorEqualityArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self == inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorNotEqualityScaler(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(self != inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorNotEqualityScalerReversed(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(inValue != self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorNotEqualityArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self != inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessScaler(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(self < inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessScalerReversed(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(inValue < self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self < inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterScaler(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(self > inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterScalerReversed(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(inValue > self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self > inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessEqualScaler(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(self <= inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessEqualScalerReversed(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(inValue <= self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessEqualArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self <= inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterEqualScaler(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(self >= inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterEqualScalerReversed(const NdArray<dtype>& self, dtype inValue)
    {
        return nc2Boost(inValue >= self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterEqualArray(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2Boost(self >= inOtherArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitshiftLeft(const NdArray<dtype>& self, uint8 inNumBits)
    {
        return nc2Boost(self << inNumBits);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitshiftRight(const NdArray<dtype>& self, uint8 inNumBits)
    {
        return nc2Boost(self >> inNumBits);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorPrePlusPlus(NdArray<dtype>& self)
    {
        return nc2Boost(++self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorPostPlusPlus(NdArray<dtype>& self)
    {
        return nc2Boost(self++);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorPreMinusMinus(NdArray<dtype>& self)
    {
        return nc2Boost(--self);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorPostMinusMinus(NdArray<dtype>& self)
    {
        return nc2Boost(self--);
    }
}

//================================================================================

namespace MethodsInterface
{
    template<typename dtype>
    dtype absScaler(dtype inValue) noexcept
    {
        return abs(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray absArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(abs(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray addArrays(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(add(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray allArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(all(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray anyArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(any(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray argmaxArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(argmax(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray argminArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(argmin(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray argsortArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(argsort(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray argwhere(const NdArray<dtype>& inArray)
    {
        return nc2Boost(nc::argwhere(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray amaxArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(amax(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray aminArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(amin(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray arangeArray(dtype inStart, dtype inStop, dtype inStep)
    {
        return nc2Boost(arange(inStart, inStop, inStep));
    }

    //================================================================================

    template<typename dtype>
    dtype arccosScaler(dtype inValue) noexcept
    {
        return arccos(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray arccosArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(arccos(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype arccoshScaler(dtype inValue) noexcept
    {
        return arccosh(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray arccoshArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(arccosh(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype arcsinScaler(dtype inValue) noexcept
    {
        return arcsin(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray arcsinArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(arcsin(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype arcsinhScaler(dtype inValue) noexcept
    {
        return arcsinh(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray arcsinhArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(arcsinh(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype arctanScaler(dtype inValue) noexcept
    {
        return arctan(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray arctanArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(arctan(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype arctan2Scaler(dtype inY, dtype inX) noexcept
    {
        return arctan2(inY, inX);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray arctan2Array(const NdArray<dtype>& inY, const NdArray<dtype>& inX)
    {
        return nc2Boost(arctan2(inY, inX));
    }

    //================================================================================

    template<typename dtype>
    dtype arctanhScaler(dtype inValue) noexcept
    {
        return arctanh(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray arctanhArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(arctanh(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype aroundScaler(dtype inValue, uint8 inNumDecimals)
    {
        return around(inValue, inNumDecimals);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray aroundArray(const NdArray<dtype>& inArray, uint8 inNumDecimals)
    {
        return nc2Boost(around(inArray, inNumDecimals));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayVector(const std::vector<double>& inVec)
    {
        return nc2Boost(asarray(inVec));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayList(dtype inValue1, dtype inValue2)
    {
        return nc2Boost<dtype>({ {inValue1, inValue2}, {inValue1, inValue2} });
    }

    //================================================================================

    template<typename dtype>
    np::ndarray average(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(nc::average(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray averageWeighted(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(average(inArray, inWeights, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bincount(const NdArray<dtype>& inArray, uint16 inMinLength = 0)
    {
        return nc2Boost(nc::bincount(inArray, inMinLength));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bincountWeighted(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0)
    {
        return nc2Boost(bincount(inArray, inWeights, inMinLength));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bitwise_and(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(nc::bitwise_and(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bitwise_not(const NdArray<dtype>& inArray)
    {
        return nc2Boost(nc::bitwise_not(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bitwise_or(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(nc::bitwise_or(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bitwise_xor(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(nc::bitwise_xor(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray andOperatorArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 && inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray andOperatorScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2Boost(inArray && inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray orOperatorArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 || inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray orOperatorScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2Boost(inArray || inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray byteswap(const NdArray<dtype>& inArray)
    {
        return nc2Boost(nc::byteswap(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype cbrtScaler(dtype inValue) noexcept
    {
        return cbrt(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray cbrtArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(cbrt(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype ceilScaler(dtype inValue) noexcept
    {
        return ceil(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray ceilArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(ceil(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray centerOfMass(const NdArray<dtype>& inArray, const Axis inAxis = Axis::NONE) noexcept
    {
        return nc2Boost(nc::centerOfMass(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    dtype clipScaler(dtype inValue, dtype inMinValue, dtype inMaxValue)
    {
        return clip(inValue, inMinValue, inMaxValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray clipArray(const NdArray<dtype>& inArray, dtype inMinValue, dtype inMaxValue)
    {
        return nc2Boost(clip(inArray, inMinValue, inMaxValue));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray column_stack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4)
    {
        return nc2Boost(nc::column_stack({ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray concatenate(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4, Axis inAxis)
    {
        return nc2Boost(nc::concatenate({ inArray1, inArray2, inArray3, inArray4 }, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray copy(const NdArray<dtype>& inArray)
    {
        return nc2Boost(nc::copy(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray copySign(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(nc::copySign(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray copyto(NdArray<dtype>& inArrayDest, const NdArray<dtype>& inArraySrc)
    {
        return nc2Boost(nc::copyto(inArrayDest, inArraySrc));
    }

    //================================================================================

    template<typename dtype>
    dtype cosScaler(dtype inValue) noexcept
    {
        return cos(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray cosArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(cos(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype coshScaler(dtype inValue) noexcept
    {
        return cosh(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray coshArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(cosh(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray count_nonzero(const NdArray<dtype>& inArray, Axis inAxis = Axis::ROW)
    {
        return nc2Boost(nc::count_nonzero(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray cubeArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(cube(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray cumprodArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(cumprod(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray cumsumArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(cumsum(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    dtype deg2radScaler(dtype inValue) noexcept
    {
        return deg2rad(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray deg2radArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(deg2rad(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype degreesScaler(dtype inValue) noexcept
    {
        return degrees(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray degreesArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(degrees(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray deleteIndicesScaler(const NdArray<dtype>& inArray, uint32 inIndex, Axis inAxis)
    {
        return nc2Boost(deleteIndices(inArray, inIndex, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray deleteIndicesSlice(const NdArray<dtype>& inArray, const Slice& inIndices, Axis inAxis)
    {
        return nc2Boost(deleteIndices(inArray, inIndices, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray diag(const NdArray<dtype>& inArray, int32 k)
    {
        return nc2Boost(nc::diag(inArray, k));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray diagflat(const NdArray<dtype>& inArray, int32 k)
    {
        return nc2Boost(nc::diagflat(inArray, k));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray diagonal(const NdArray<dtype>& inArray, uint32 inOffset = 0, Axis inAxis = Axis::ROW)
    {
        return nc2Boost(nc::diagonal(inArray, inOffset, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray diff(const NdArray<dtype>& inArray, Axis inAxis = Axis::ROW)
    {
        return nc2Boost(nc::diff(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray divide(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(nc::divide(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray dot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(nc::dot(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray emptyRowCol(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2Boost(nc::empty<dtype>(inNumRows, inNumCols));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray emptyShape(const Shape& inShape)
    {
        return nc2Boost(empty<dtype>(inShape));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(nc::equal(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    dtype expScaler(dtype inValue) noexcept
    {
        return exp(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray expArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(exp(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype exp2Scaler(dtype inValue) noexcept
    {
        return exp2(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray exp2Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(exp2(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype expm1Scaler(dtype inValue) noexcept
    {
        return expm1(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray expm1Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(expm1(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray eye1D(uint32 inN, int32 inK)
    {
        return nc2Boost(eye<dtype>(inN, inK));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray eye2D(uint32 inN, uint32 inM, int32 inK)
    {
        return nc2Boost(eye<dtype>(inN, inM, inK));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray eyeShape(const Shape& inShape, int32 inK)
    {
        return nc2Boost(eye<dtype>(inShape, inK));
    }

    //================================================================================

    np::ndarray find(const NdArray<bool>& inArray) noexcept
    {
        return nc2Boost(nc::find(inArray));
    }

    //================================================================================

    np::ndarray findN(const NdArray<bool>& inArray, uint32 n) noexcept
    {
        return nc2Boost(nc::find(inArray, n));
    }

    //================================================================================

    template<typename dtype>
    dtype fixScaler(dtype inValue) noexcept
    {
        return fix(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray fixArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(fix(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype floorScaler(dtype inValue) noexcept
    {
        return floor(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray floorArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(floor(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype floor_divideScaler(dtype inValue1, dtype inValue2) noexcept
    {
        return floor_divide(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray floor_divideArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(floor_divide(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    dtype fmaxScaler(dtype inValue1, dtype inValue2) noexcept
    {
        return fmax(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray fmaxArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(fmax(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    dtype fminScaler(dtype inValue1, dtype inValue2) noexcept
    {
        return fmin(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray fminArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(fmin(inArray1, inArray2));
    }

    template<typename dtype>
    dtype fmodScaler(dtype inValue1, dtype inValue2) noexcept
    {
        return fmod(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray fmodArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(fmod(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray fullSquare(uint32 inSquareSize, dtype inValue)
    {
        return nc2Boost(full(inSquareSize, inValue));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray fullRowCol(uint32 inNumRows, uint32 inNumCols, dtype inValue)
    {
        return nc2Boost(full(inNumRows, inNumCols, inValue));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray fullShape(const Shape& inShape, dtype inValue)
    {
        return nc2Boost(full(inShape, inValue));
    }

    //================================================================================

    template<typename dtype>
    dtype gcdScaler(dtype inValue1, dtype inValue2) noexcept
    {
        return gcd(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    dtype gcdArray(const NdArray<dtype>& inArray)
    {
        return gcd(inArray);
    }

    //================================================================================

    template<typename dtype>
    bp::tuple histogram(const NdArray<dtype>& inArray, uint32 inNumBins = 10)
    {
        std::pair<NdArray<uint32>, NdArray<double> > output = nc::histogram(inArray, inNumBins);
        return bp::make_tuple(output.first, output.second);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray hstack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4)
    {
        return nc2Boost(nc::hstack({ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    dtype hypotScaler(dtype inValue1, dtype inValue2) noexcept
    {
        return hypot(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    dtype hypotScalerTriple(dtype inValue1, dtype inValue2, dtype inValue3) noexcept
    {
        return hypot(inValue1, inValue2, inValue3);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray hypotArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(hypot(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray interp(const NdArray<dtype>& inX, const NdArray<dtype>& inXp, const NdArray<dtype>& inFp)
    {
        return nc2Boost(nc::interp(inX, inXp, inFp));
    }

    //================================================================================

    template<typename dtype>
    bool isinfScaler(dtype inValue) noexcept
    {
        return nc::isinf(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray isinfArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(nc::isinf(inArray));
    }

    //================================================================================

    template<typename dtype>
    bool isnanScaler(dtype inValue) noexcept
    {
        return nc::isnan(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray isnanArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(nc::isnan(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype ldexpScaler(dtype inValue1, uint8 inValue2) noexcept
    {
        return ldexp(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray ldexpArray(const NdArray<dtype>& inArray1, const NdArray<uint8>& inArray2)
    {
        return nc2Boost(ldexp(inArray1, inArray2));
    }

    //================================================================================

    np::ndarray nansSquare(uint32 inSquareSize)
    {
        return nc2Boost(nans(inSquareSize));
    }

    //================================================================================

    np::ndarray nansRowCol(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2Boost(nans(inNumRows, inNumCols));
    }

    //================================================================================

    np::ndarray nansShape(const Shape& inShape)
    {
        return nc2Boost(nans(inShape));
    }

    //================================================================================

    np::ndarray nansList(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2Boost(nans({ inNumRows, inNumCols }));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray negative(const NdArray<dtype> inArray)
    {
        return nc2Boost(nc::negative(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray noneArray(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(none(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    dtype lcmScaler(dtype inValue1, dtype inValue2) noexcept
    {
        return lcm(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    dtype lcmArray(const NdArray<dtype>& inArray)
    {
        return lcm(inArray);
    }

    //================================================================================

    template<typename dtype>
    dtype logScaler(dtype inValue) noexcept
    {
        return log(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray logArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(log(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype log10Scaler(dtype inValue) noexcept
    {
        return log10(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray log10Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(log10(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype log1pScaler(dtype inValue) noexcept
    {
        return log1p(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray log1pArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(log1p(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype log2Scaler(dtype inValue) noexcept
    {
        return log2(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray log2Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(log2(inArray));
    }

    //================================================================================

    template<typename dtype>
    std::pair<NdArray<dtype>, NdArray<dtype> > meshgrid(const Slice& inISlice, const Slice& inJSlice)
    {
        return nc::meshgrid<dtype>(inISlice, inJSlice);
    }

    //================================================================================

    template<typename dtype>
    dtype newbyteorderScaler(dtype inValue, Endian inEndianess)
    {
        return newbyteorder(inValue, inEndianess);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray newbyteorderArray(const NdArray<dtype>& inArray, Endian inEndianess)
    {
        return nc2Boost(newbyteorder(inArray, inEndianess));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray onesSquare(uint32 inSquareSize)
    {
        return nc2Boost(ones<dtype>(inSquareSize));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray onesRowCol(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2Boost(ones<dtype>(inNumRows, inNumCols));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray onesShape(const Shape& inShape)
    {
        return nc2Boost(ones<dtype>(inShape));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray outer(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(nc::outer(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray sqrArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(sqr(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray powerArrayScaler(const NdArray<dtype>& inArray, uint8 inExponent)
    {
        return nc2Boost(power(inArray, inExponent));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray powerArrayArray(const NdArray<dtype>& inArray, const NdArray<uint8>& inExponents)
    {
        return nc2Boost(power(inArray, inExponents));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray powerfArrayScaler(const NdArray<dtype>& inArray, double inExponent)
    {
        return nc2Boost(powerf(inArray, inExponent));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray powerfArrayArray(const NdArray<dtype>& inArray, const NdArray<double>& inExponents)
    {
        return nc2Boost(powerf(inArray, inExponents));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putmask(NdArray<dtype>& inArray, const NdArray<bool>& inMask, const NdArray<dtype>& inValues)
    {
        return nc2Boost(nc::putmask(inArray, inMask, inValues));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray putmaskScaler(NdArray<dtype>& inArray, const NdArray<bool>& inMask, dtype inValue)
    {
        return nc2Boost(putmask(inArray, inMask, inValue));
    }

    //================================================================================

    template<typename dtype>
    dtype rad2degScaler(dtype inValue) noexcept
    {
        return rad2deg(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray rad2degArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(rad2deg(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype radiansScaler(dtype inValue) noexcept
    {
        return radians(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray radiansArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(radians(inArray));
    }

    //================================================================================

    template<typename dtype>
    NdArray<dtype>& ravel(NdArray<dtype>& inArray)
    {
        return nc::ravel(inArray);
    }

    //================================================================================

    template<typename dtype>
    dtype remainderScaler(dtype inValue1, dtype inValue2) noexcept
    {
        return remainder(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray remainderArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(remainder(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray replace(NdArray<dtype>& inArray, dtype oldValue, dtype newValue)
    {
        return nc2Boost(nc::replace(inArray, oldValue, newValue));
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
    dtype rintScaler(dtype inValue) noexcept
    {
        return rint(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray rintArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(rint(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype roundScaler(dtype inValue, uint8 inDecimals)
    {
        return round(inValue, inDecimals);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray roundArray(const NdArray<dtype>& inArray, uint8 inDecimals)
    {
        return nc2Boost(round(inArray, inDecimals));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray row_stack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4)
    {
        return nc2Boost(nc::row_stack({ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    int8 signScaler(dtype inValue) noexcept
    {
        return sign(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray signArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(sign(inArray));
    }

    //================================================================================

    template<typename dtype>
    bool signbitScaler(dtype inValue) noexcept
    {
        return signbit(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray signbitArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(signbit(inArray));
    }

    //================================================================================

    template<typename dtype>
    double sinScaler(dtype inValue) noexcept
    {
        return sin(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray sinArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(sin(inArray));
    }

    //================================================================================

    template<typename dtype>
    double sincScaler(dtype inValue) noexcept
    {
        return sinc(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray sincArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(sinc(inArray));
    }

    //================================================================================

    template<typename dtype>
    double sinhScaler(dtype inValue) noexcept
    {
        return sinh(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray sinhArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(sinh(inArray));
    }

    //================================================================================

    template<typename dtype>
    double sqrtScaler(dtype inValue) noexcept
    {
        return sqrt(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray sqrtArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(sqrt(inArray));
    }

    //================================================================================

    template<typename dtype>
    double squareScaler(dtype inValue) noexcept
    {
        return square(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray squareArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(square(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray subtractArrays(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(subtract(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    double tanScaler(dtype inValue) noexcept
    {
        return tan(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray tanArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(tan(inArray));
    }

    //================================================================================

    template<typename dtype>
    double tanhScaler(dtype inValue) noexcept
    {
        return tanh(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray tanhArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(tanh(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray tileRectangle(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        return nc2Boost(tile(inArray, inNumRows, inNumCols));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray tileShape(const NdArray<dtype>& inArray, const Shape& inRepShape)
    {
        return nc2Boost(tile(inArray, inRepShape));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray tileList(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        return nc2Boost(tile(inArray, { inNumRows, inNumCols }));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray trapzDx(const NdArray<dtype>& inY, double dx = 1.0, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(trapz(inY, dx, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray trapz(const NdArray<dtype>& inY, const NdArray<dtype>& inX, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(nc::trapz(inY, inX, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray triuSquare(uint32 inSquareSize, int32 inOffset)
    {
        return nc2Boost(triu<dtype>(inSquareSize, inOffset));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray triuRect(uint32 inNumRows, uint32 inNumCols, int32 inOffset)
    {
        return nc2Boost(triu<dtype>(inNumRows, inNumCols, inOffset));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray triuArray(const NdArray<dtype>& inArray, int32 inOffset)
    {
        return nc2Boost(triu(inArray, inOffset));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray trilSquare(uint32 inSquareSize, int32 inOffset)
    {
        return nc2Boost(tril<dtype>(inSquareSize, inOffset));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray trilRect(uint32 inNumRows, uint32 inNumCols, int32 inOffset)
    {
        return nc2Boost(tril<dtype>(inNumRows, inNumCols, inOffset));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray trilArray(const NdArray<dtype>& inArray, int32 inOffset)
    {
        return nc2Boost(tril(inArray, inOffset));
    }

    //================================================================================

    template<typename dtype>
    dtype unwrapScaler(dtype inValue) noexcept
    {
        return unwrap(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray unwrapArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(unwrap(inArray));
    }

    //================================================================================

    template<typename dtype>
    double truncScaler(dtype inValue) noexcept
    {
        return trunc(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray truncArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(trunc(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray stack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4, nc::Axis inAxis)
    {
        return nc2Boost(nc::stack({ inArray1, inArray2, inArray3, inArray4 }, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray vstack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4)
    {
        return nc2Boost(nc::vstack({ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray where(const NdArray<bool>& inMask, const NdArray<dtype>& inA, const NdArray<dtype>& inB)
    {
        return nc2Boost(nc::where(inMask, inA, inB));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray zerosSquare(uint32 inSquareSize)
    {
        return nc2Boost(zeros<dtype>(inSquareSize));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray zerosRowCol(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2Boost(zeros<dtype>(inNumRows, inNumCols));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray zerosShape(const Shape& inShape)
    {
        return nc2Boost(zeros<dtype>(inShape));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray zerosList(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2Boost(zeros<dtype>({ inNumRows, inNumCols }));
    }
}

namespace RandomInterface
{
    template<typename dtype>
    dtype choiceSingle(const NdArray<dtype>& inArray)
    {
        return random::choice(inArray);
    }

    template<typename dtype>
    np::ndarray choiceMultiple(const NdArray<dtype>& inArray, uint32 inNum)
    {
        return nc2Boost(random::choice(inArray, inNum));
    }

    template<typename dtype>
    np::ndarray permutationScaler(dtype inValue)
    {
        return nc2Boost(random::permutation(inValue));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray permutationArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(random::permutation(inArray));
    }
}

namespace LinalgInterface
{
    template<typename dtype>
    np::ndarray hatArray(const NdArray<dtype>& inArray)
    {
        return nc2Boost(linalg::hat(inArray));
    }

    template<typename dtype>
    np::ndarray multi_dot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4)
    {
        return nc2Boost(linalg::multi_dot({ inArray1, inArray2, inArray3, inArray4 }));
    }

    template<typename dtype>
    bp::tuple pivotLU_decomposition(const NdArray<dtype>& inArray)
    {
        auto lup = linalg::pivotLU_decomposition(inArray);
        auto& l = std::get<0>(lup);
        auto& u = std::get<1>(lup);
        auto& p = std::get<2>(lup);
        return bp::make_tuple(nc2Boost(l), nc2Boost(u), nc2Boost(p));
    }
}

namespace RotationsInterface
{
    np::ndarray angleAxisRotationNdArray(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2Boost(rotations::Quaternion(inAxis, inAngle).toNdArray());
    }

    np::ndarray angleAxisRotationVec3(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2Boost(rotations::Quaternion(Vec3(inAxis), inAngle).toNdArray());
    }

    np::ndarray angularVelocity(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2, double inTime)
    {
        return nc2Boost(inQuat1.angularVelocity(inQuat2, inTime));
    }

    np::ndarray nlerp(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2, double inPercent)
    {
        return nc2Boost(inQuat1.nlerp(inQuat2, inPercent).toNdArray());
    }

    np::ndarray rotateNdArray(const rotations::Quaternion& inQuat, const NdArray<double>& inVec)
    {
        return nc2Boost(inQuat.rotate(inVec));
    }

    np::ndarray rotateVec3(const rotations::Quaternion& inQuat, const NdArray<double>& inVec)
    {
        return nc2Boost(inQuat.rotate(Vec3(inVec)).toNdArray());
    }

    np::ndarray slerp(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2, double inPercent)
    {
        return nc2Boost(inQuat1.slerp(inQuat2, inPercent).toNdArray());
    }

    np::ndarray toDCM(const rotations::Quaternion& inQuat)
    {
        return nc2Boost(inQuat.toDCM());
    }

    np::ndarray subtract(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2)
    {
        return nc2Boost((inQuat1 - inQuat2).toNdArray());
    }

    np::ndarray negative(const rotations::Quaternion& inQuat)
    {
        return nc2Boost((-inQuat).toNdArray());
    }

    np::ndarray multiplyScaler(const rotations::Quaternion& inQuat, double inScaler)
    {
        const rotations::Quaternion returnQuat = inQuat * inScaler;
        return nc2Boost(returnQuat.toNdArray());
    }

    np::ndarray multiplyArray(const rotations::Quaternion& inQuat, const NdArray<double>& inArray)
    {
        NdArray<double> returnArray = inQuat * inArray;
        return nc2Boost(returnArray);
    }

    np::ndarray multiplyQuaternion(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2)
    {
        const rotations::Quaternion returnQuat = inQuat1 * inQuat2;
        return nc2Boost(returnQuat.toNdArray());
    }

    np::ndarray eulerAnglesValues(double roll, double pitch, double yaw)
    {
        return nc2Boost(rotations::DCM::eulerAngles(roll, pitch, yaw));
    }

    np::ndarray eulerAnglesArray(const NdArray<double> angles)
    {
        return nc2Boost(rotations::DCM::eulerAngles(angles));
    }

    np::ndarray angleAxisRotationDcmNdArray(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2Boost(rotations::DCM::eulerAxisAngle(inAxis, inAngle));
    }

    np::ndarray angleAxisRotationDcmVec3(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2Boost(rotations::DCM::eulerAxisAngle(Vec3(inAxis), inAngle));
    }

    template<typename T>
    np::ndarray rodriguesRotation(np::ndarray& inK, double inTheta, np::ndarray& inV)
    {
        auto k = boost2Nc<T>(inK);
        auto v = boost2Nc<T>(inV);

        return nc2Boost(rotations::rodriguesRotation(k, inTheta, v));
    }

    template<typename T>
    np::ndarray wahbasProblem(np::ndarray& inWk, np::ndarray& inVk)
    {
        auto wk = boost2Nc<T>(inWk);
        auto vk = boost2Nc<T>(inVk);
        return nc2Boost(rotations::wahbasProblem(wk, vk));
    }

    template<typename T>
    np::ndarray wahbasProblemWeighted(np::ndarray& inWk, np::ndarray& inVk, np::ndarray& inAk)
    {
        auto wk = boost2Nc<T>(inWk);
        auto vk = boost2Nc<T>(inVk);
        auto ak = boost2Nc<T>(inAk);
        return nc2Boost(rotations::wahbasProblem(wk, vk, ak));
    }
}

namespace RaInterface
{
    void print(const coordinates::RA& inRa)
    {
        std::cout << inRa;
    }
}

namespace DecInterface
{
    void print(const coordinates::Dec& self)
    {
        std::cout << self;
    }
}

namespace CoordinateInterface
{
    void print(const coordinates::Coordinate& self)
    {
        std::cout << self;
    }

    double degreeSeperationCoordinate(const coordinates::Coordinate& self, const coordinates::Coordinate& inOtherCoordinate)
    {
        return self.degreeSeperation(inOtherCoordinate);
    }

    double degreeSeperationVector(const coordinates::Coordinate& self, const NdArray<double>& inVec)
    {
        return self.degreeSeperation(inVec);
    }

    double radianSeperationCoordinate(const coordinates::Coordinate& self, const coordinates::Coordinate& inOtherCoordinate)
    {
        return self.radianSeperation(inOtherCoordinate);
    }

    double radianSeperationVector(const coordinates::Coordinate& self, const NdArray<double>& inVec)
    {
        return self.radianSeperation(inVec);
    }
}

namespace DataCubeInterface
{
    template<typename dtype>
    NdArray<dtype>& at(DataCube<dtype>& self, uint32 inIndex)
    {
        return self.at(inIndex);
    }

    template<typename dtype>
    NdArray<dtype>& getItem(DataCube<dtype>& self, uint32 inIndex)
    {
        return self[inIndex];
    }
}

//================================================================================

namespace PolynomialInterface
{
    template<typename dtype>
    dtype chebyshev_t_Scaler(uint32 n, dtype inValue) noexcept
    {
        return polynomial::chebyshev_t(n, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray chebyshev_t_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2Boost(polynomial::chebyshev_t(n, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype chebyshev_u_Scaler(uint32 n, dtype inValue) noexcept
    {
        return polynomial::chebyshev_u(n, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray chebyshev_u_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2Boost(polynomial::chebyshev_u(n, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype hermite_Scaler(uint32 n, dtype inValue) noexcept
    {
        return polynomial::hermite(n, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray hermite_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2Boost(polynomial::hermite(n, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype laguerre_Scaler1(uint32 n, dtype inValue) noexcept
    {
        return polynomial::laguerre(n, inValue);
    }

    //================================================================================

    template<typename dtype>
    dtype laguerre_Scaler2(uint32 n, uint32 m, dtype inValue) noexcept
    {
        return polynomial::laguerre(n, m, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray laguerre_Array1(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2Boost(polynomial::laguerre(n, inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray laguerre_Array2(uint32 n, uint32 m, const NdArray<dtype>& inArray)
    {
        return nc2Boost(polynomial::laguerre(n, m, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype legendre_p_Scaler1(int32 n, dtype inValue) noexcept
    {
        return polynomial::legendre_p(n, inValue);
    }

    //================================================================================

    template<typename dtype>
    dtype legendre_p_Scaler2(int32 n, int32 m, dtype inValue) noexcept
    {
        return polynomial::legendre_p(n, m, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray legendre_p_Array1(int32 n, const NdArray<dtype>& inArray)
    {
        return nc2Boost(polynomial::legendre_p(n, inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray legendre_p_Array2(int32 n, int32 m, const NdArray<dtype>& inArray)
    {
        return nc2Boost(polynomial::legendre_p(n, m, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype legendre_q_Scaler(int32 n, dtype inValue) noexcept
    {
        return polynomial::legendre_q(n, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray legendre_q_Array(int32 n, const NdArray<dtype>& inArray)
    {
        return nc2Boost(polynomial::legendre_q(n, inArray));
    }

    //================================================================================

    template<typename dtype>
    bp::list spherical_harmonic(uint32 n, int32 m, dtype theta, dtype phi)
    {
        auto value = polynomial::spherical_harmonic(n, m, theta, phi);
        std::vector<double> valueVec = {value.real(), value.imag()};
        return vector2list(valueVec);
    }
}

namespace RootsInterface
{
    constexpr double EPSILON = 1e-10;

    //================================================================================

    double bisection(const polynomial::Poly1d<double>p, double a, double b)
    {
        auto rootFinder = roots::Bisection(EPSILON, p);
        return rootFinder.solve(a, b);
    }

    //================================================================================

    double brent(const polynomial::Poly1d<double>p, double a, double b)
    {
        auto rootFinder = roots::Brent(EPSILON, p);
        return rootFinder.solve(a, b);
    }

    //================================================================================

    double dekker(const polynomial::Poly1d<double>p, double a, double b)
    {
        auto rootFinder = roots::Dekker(EPSILON, p);
        return rootFinder.solve(a, b);
    }

    //================================================================================

    double newton(const polynomial::Poly1d<double>p, double x)
    {
        auto pPrime = p.deriv();
        auto rootFinder = roots::Newton(EPSILON, p, pPrime);
        return rootFinder.solve(x);
    }

    //================================================================================

    double secant(const polynomial::Poly1d<double>p, double a, double b)
    {
        auto rootFinder = roots::Secant(EPSILON, p);
        return rootFinder.solve(a, b);
    }
}

namespace IntegrateInterface
{
    constexpr uint32 NUM_ITERATIONS = 100;
    constexpr uint32 NUM_SUBDIVISIONS = 10000;

    //================================================================================

    double gauss_legendre(const polynomial::Poly1d<double>& p, double a, double b)
    {
        return integrate::gauss_legendre(a, b, NUM_ITERATIONS, p);
    }

    //================================================================================

    double romberg(const polynomial::Poly1d<double>& p, double a, double b)
    {
        return integrate::romberg(a, b, 10, p);
    }

    //================================================================================

    double simpson(const polynomial::Poly1d<double>& p, double a, double b)
    {
        return integrate::simpson(a, b, NUM_SUBDIVISIONS, p);
    }

    //================================================================================

    double trapazoidal(const polynomial::Poly1d<double>& p, double a, double b)
    {
        return integrate::trapazoidal(a, b, NUM_SUBDIVISIONS, p);
    }
}

namespace Vec2Interface
{
    np::ndarray toNdArray(const Vec2& self)
    {
        return nc2Boost(self.toNdArray());
    }

    Vec2& plusEqualScaler(Vec2& self, double scaler)
    {
        return self += scaler;
    }

    Vec2& plusEqualVec2(Vec2& self, const Vec2& rhs)
    {
        return self += rhs;
    }

    Vec2& minusEqualScaler(Vec2& self, double scaler)
    {
        return self -= scaler;
    }

    Vec2& minusEqualVec2(Vec2& self, const Vec2& rhs)
    {
        return self -= rhs;
    }

    Vec2 addVec2(const Vec2& vec1, const Vec2& vec2)
    {
        return vec1 + vec2;
    }

    Vec2 addVec2Scaler(const Vec2& vec1, double scaler)
    {
        return vec1 + scaler;
    }

    Vec2 addScalerVec2(const Vec2& vec1, double scaler)
    {
        return scaler + vec1;
    }

    Vec2 minusVec2(const Vec2& vec1, const Vec2& vec2)
    {
        return vec1 - vec2;
    }

    Vec2 minusVec2Scaler(const Vec2& vec1, double scaler)
    {
        return vec1 - scaler;
    }

    Vec2 minusScalerVec2(const Vec2& vec1, double scaler)
    {
        return scaler - vec1;
    }

    Vec2 multVec2Scaler(const Vec2& vec1, double scaler)
    {
        return vec1 * scaler;
    }

    Vec2 multScalerVec2(const Vec2& vec1, double scaler)
    {
        return scaler * vec1;
    }

    Vec2 divVec2Scaler(const Vec2& vec1, double scaler)
    {
        return vec1 / scaler;
    }

    void print(const Vec2& vec)
    {
        std::cout << vec;
    }
}

namespace Vec3Interface
{
    np::ndarray toNdArray(const Vec3& self)
    {
        return nc2Boost(self.toNdArray());
    }

    Vec3& plusEqualScaler(Vec3& self, double scaler)
    {
        return self += scaler;
    }

    Vec3& plusEqualVec3(Vec3& self, const Vec3& rhs)
    {
        return self += rhs;
    }

    Vec3& minusEqualScaler(Vec3& self, double scaler)
    {
        return self -= scaler;
    }

    Vec3& minusEqualVec3(Vec3& self, const Vec3& rhs)
    {
        return self -= rhs;
    }

    Vec3 addVec3(const Vec3& vec1, const Vec3& vec2)
    {
        return vec1 + vec2;
    }

    Vec3 addVec3Scaler(const Vec3& vec1, double scaler)
    {
        return vec1 + scaler;
    }

    Vec3 addScalerVec3(const Vec3& vec1, double scaler)
    {
        return vec1 + scaler;
    }

    Vec3 minusVec3(const Vec3& vec1, const Vec3& vec2)
    {
        return vec1 - vec2;
    }

    Vec3 minusVec3Scaler(const Vec3& vec1, double scaler)
    {
        return vec1 - scaler;
    }

    Vec3 minusScalerVec3(const Vec3& vec1, double scaler)
    {
        return scaler - vec1;
    }

    Vec3 multVec3Scaler(const Vec3& vec1, double scaler)
    {
        return vec1 * scaler;
    }

    Vec3 multScalerVec3(const Vec3& vec1, double scaler)
    {
        return vec1 * scaler;
    }

    Vec3 divVec3Scaler(const Vec3& vec1, double scaler)
    {
        return vec1 / scaler;
    }

    void print(const Vec3& vec)
    {
        std::cout << vec;
    }
}

//================================================================================

namespace SpecialInterface
{
    template<typename dtype>
    dtype airy_ai_Scaler(dtype inValue) noexcept
    {
        return special::airy_ai(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray airy_ai_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::airy_ai(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype airy_ai_prime_Scaler(dtype inValue) noexcept
    {
        return special::airy_ai_prime(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray airy_ai_prime_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::airy_ai_prime(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype airy_bi_Scaler(dtype inValue) noexcept
    {
        return special::airy_bi(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray airy_bi_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::airy_bi(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype airy_bi_prime_Scaler(dtype inValue) noexcept
    {
        return special::airy_bi_prime(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray airy_bi_prime_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::airy_bi_prime(inArray));
    }

    //================================================================================

    double bernoulli_Scaler(uint32 n) noexcept
    {
        return special::bernoilli(n);
    }

    //================================================================================

    np::ndarray bernoulli_Array(const NdArray<uint32>& inArray)
    {
        return nc2Boost(special::bernoilli(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype bessel_in_Scaler(dtype inV, dtype inValue) noexcept
    {
        return special::bessel_in(inV, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bessel_in_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::bessel_in(inV, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype bessel_in_prime_Scaler(dtype inV, dtype inValue) noexcept
    {
        return special::bessel_in_prime(inV, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bessel_in_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::bessel_in_prime(inV, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype bessel_jn_Scaler(dtype inV, dtype inValue) noexcept
    {
        return special::bessel_jn(inV, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bessel_jn_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::bessel_jn(inV, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype bessel_jn_prime_Scaler(dtype inV, dtype inValue) noexcept
    {
        return special::bessel_jn_prime(inV, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bessel_jn_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::bessel_jn_prime(inV, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype bessel_kn_Scaler(dtype inV, dtype inValue) noexcept
    {
        return special::bessel_kn(inV, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bessel_kn_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::bessel_kn(inV, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype bessel_kn_prime_Scaler(dtype inV, dtype inValue) noexcept
    {
        return special::bessel_kn_prime(inV, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bessel_kn_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::bessel_kn_prime(inV, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype bessel_yn_Scaler(dtype inV, dtype inValue) noexcept
    {
        return special::bessel_yn(inV, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bessel_yn_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::bessel_yn(inV, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype bessel_yn_prime_Scaler(dtype inV, dtype inValue) noexcept
    {
        return special::bessel_yn_prime(inV, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray bessel_yn_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::bessel_yn_prime(inV, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype beta_Scaler(dtype a, dtype b) noexcept
    {
        return special::beta(a, b);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray beta_Array(const NdArray<dtype>& a, const NdArray<dtype>& b)
    {
        return nc2Boost(special::beta(a, b));
    }

    //================================================================================

    template<typename dtype>
    dtype digamma_Scaler(dtype inValue) noexcept
    {
        return special::digamma(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray digamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::digamma(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype erf_Scaler(dtype inValue) noexcept
    {
        return special::erf(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray erf_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::erf(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype erf_inv_Scaler(dtype inValue) noexcept
    {
        return special::erf_inv(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray erf_inv_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::erf_inv(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype erfc_Scaler(dtype inValue) noexcept
    {
        return special::erfc(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray erfc_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::erfc(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype erfc_inv_Scaler(dtype inValue) noexcept
    {
        return special::erfc_inv(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray erfc_inv_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::erfc_inv(inArray));
    }

    //================================================================================

    double factorial_Scaler(uint32 inValue) noexcept
    {
        return special::factorial(inValue);
    }

    //================================================================================

    np::ndarray factorial_Array(const NdArray<uint32>& inArray)
    {
        return nc2Boost(special::factorial(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype gamma_Scaler(dtype inValue) noexcept
    {
        return special::gamma(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray gamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::gamma(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype gamma1pm1_Scaler(dtype inValue) noexcept
    {
        return special::gamma1pm1(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray gamma1pm1_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::gamma1pm1(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype log_gamma_Scaler(dtype inValue) noexcept
    {
        return special::log_gamma(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray log_gamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::log_gamma(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype polygamma_Scaler(uint32 n, dtype inValue) noexcept
    {
        return special::polygamma(n, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray polygamma_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::polygamma(n, inArray));
    }

    //================================================================================

    double prime_Scaler(uint32 inValue) noexcept
    {
        return special::prime(inValue);
    }

    //================================================================================

    np::ndarray prime_Array(const NdArray<uint32>& inArray)
    {
        return nc2Boost(special::prime(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype riemann_zeta_Scaler(dtype inValue) noexcept
    {
        return special::riemann_zeta(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray riemann_zeta_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::riemann_zeta(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray softmax(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return nc2Boost(special::softmax(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    dtype spherical_bessel_jn_Scaler(uint32 inV, dtype inValue) noexcept
    {
        return special::spherical_bessel_jn(inV, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray spherical_bessel_jn_Array(uint32 inV, const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::spherical_bessel_jn(inV, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype spherical_bessel_yn_Scaler(uint32 inV, dtype inValue) noexcept
    {
        return special::spherical_bessel_yn(inV, inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray spherical_bessel_yn_Array(uint32 inV, const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::spherical_bessel_yn(inV, inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype trigamma_Scaler(dtype inValue) noexcept
    {
        return special::trigamma(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray trigamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2Boost(special::trigamma(inArray));
    }
}

//================================================================================

BOOST_PYTHON_MODULE(NumCpp)
{
    Py_Initialize();
    np::initialize(); // needs to be called first thing in the BOOST_PYTHON_MODULE for numpy

    //http://www.boost.org/doc/libs/1_60_0/libs/python/doc/html/tutorial/tutorial/exposing.html

    bp::class_<std::vector<double> >("doubleVector")
        .def(bp::vector_indexing_suite<std::vector<double> >());

    typedef std::pair<NdArray<double>, NdArray<double> > doublePair;
    bp::class_<doublePair>("doublePair")
        .def_readonly("first", &doublePair::first)
        .def_readonly("second", &doublePair::second);

    typedef std::pair<NdArray<uint32>, NdArray<uint32> > uint32Pair;
    bp::class_<uint32Pair>("uint32Pair")
        .def_readonly("first", &uint32Pair::first)
        .def_readonly("second", &uint32Pair::second);

    // Constants.hpp
    bp::scope().attr("c") = constants::c;
    bp::scope().attr("e") = constants::e;
    bp::scope().attr("inf") = constants::inf;
    bp::scope().attr("pi") = constants::pi;
    bp::scope().attr("nan") = constants::nan;
    bp::scope().attr("VERSION") = constants::VERSION;

    // PythonInterface.hpp
    bp::def("list2vector", &list2vector<int>);
    bp::def("vector2list", &vector2list<int>);
    bp::def("map2dict", &map2dict<std::string, int>);

    // DtypeInfo.hpp
    typedef DtypeInfo<uint32> DtypeInfoUint32;
    bp::class_<DtypeInfoUint32>
        ("DtypeIntoUint32", bp::init<>())
        .def("bits", &DtypeInfoUint32::bits).staticmethod("bits")
        .def("epsilon", &DtypeInfoUint32::epsilon).staticmethod("epsilon")
        .def("isInteger", &DtypeInfoUint32::isInteger).staticmethod("isInteger")
        .def("min", &DtypeInfoUint32::min).staticmethod("min")
        .def("max", &DtypeInfoUint32::max).staticmethod("max");

    // Shape.hpp
    bp::class_<Shape>
        ("Shape", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("testListContructor", &ShapeInterface::testListContructor).staticmethod("testListContructor")
        .def_readwrite("rows", &Shape::rows)
        .def_readwrite("cols", &Shape::cols)
        .def("size", &Shape::size)
        .def("print", &Shape::print)
        .def("__str__", &Shape::str)
        .def("__eq__", &Shape::operator==)
        .def("__neq__", &Shape::operator!=);

    // Slice.hpp
    bp::class_<Slice>
        ("Slice", bp::init<>())
        .def(bp::init<int32>())
        .def(bp::init<int32, int32>())
        .def(bp::init<int32, int32, int32>())
        .def(bp::init<Slice>())
        .def_readwrite("start", &Slice::start)
        .def_readwrite("stop", &Slice::stop)
        .def_readwrite("step", &Slice::step)
        .def("numElements", &Slice::numElements)
        .def("print", &Slice::print)
        .def("__str__", &Shape::str)
        .def("__eq__", &Shape::operator==)
        .def("__neq__", &Shape::operator!=);

    // Timer.hpp
    typedef Timer<std::chrono::microseconds> MicroTimer;
    bp::class_<MicroTimer>
        ("Timer", bp::init<>())
        .def(bp::init<std::string>())
        .def("sleep", &MicroTimer::sleep)
        .def("tic", &MicroTimer::tic)
        .def("toc", &MicroTimer::toc);

    // Types.hpp
    bp::enum_<Axis>("Axis")
        .value("NONE", Axis::NONE)
        .value("ROW", Axis::ROW)
        .value("COL", Axis::COL);

    bp::enum_<Endian>("Endian")
        .value("NATIVE", Endian::NATIVE)
        .value("BIG", Endian::BIG)
        .value("LITTLE", Endian::LITTLE);

    // NdArray.hpp
    typedef NdArray<double> NdArrayDouble;
    bp::class_<NdArrayDouble>
        ("NdArray", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def(bp::init<NdArrayDouble>())
        .def("test1DListContructor", &NdArrayInterface::test1DListContructor<double>).staticmethod("test1DListContructor")
        .def("test2DListContructor", &NdArrayInterface::test2DListContructor<double>).staticmethod("test2DListContructor")
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<double>)
        .def("setArray", &NdArrayInterface::setArray<double>)
        .def("rSlice", &NdArrayDouble::rSlice)
        .def("cSlice", &NdArrayDouble::cSlice)
        .def("all", &NdArrayInterface::all<double>)
        .def("any", &NdArrayInterface::any<double>)
        .def("argmax", &NdArrayInterface::argmax<double>)
        .def("argmin", &NdArrayInterface::argmin<double>)
        .def("argsort", &NdArrayInterface::argsort<double>)
        .def("back", &NdArrayInterface::back<double>)
        .def("backReference", &NdArrayInterface::backReference<double>)
        .def("clip", &NdArrayInterface::clip<double>)
        .def("copy", &NdArrayInterface::copy<double>)
        .def("column", &NdArrayDouble::column)
        .def("contains", &NdArrayInterface::contains<double>)
        .def("cumprod", &NdArrayInterface::cumprod<double>)
        .def("cumsum", &NdArrayInterface::cumsum<double>)
        .def("diagonal", &NdArrayInterface::diagonal<double>)
        .def("dot", &NdArrayInterface::dot<double>)
        .def("dump", &NdArrayDouble::dump)
        .def("fill", &NdArrayInterface::fill<double>)
        .def("flatnonzero", &NdArrayInterface::flatnonzero<double>)
        .def("flatten", &NdArrayInterface::flatten<double>)
        .def("front", &NdArrayInterface::front<double>)
        .def("frontReference", &NdArrayInterface::frontReference<double>)
        .def("get", &NdArrayInterface::getValueFlat<double>)
        .def("get", &NdArrayInterface::getValueRowCol<double>)
        .def("get", &NdArrayInterface::getSlice1D<double>)
        .def("get", &NdArrayInterface::getSlice2D<double>)
        .def("get", &NdArrayInterface::getSlice2DRow<double>)
        .def("get", &NdArrayInterface::getSlice2DCol<double>)
        .def("getByIndices", &NdArrayInterface::getByIndices<double>)
        .def("getByMask", &NdArrayInterface::getByMask<double>)
        .def("isempty", &NdArrayDouble::isempty)
        .def("isflat", &NdArrayDouble::isflat)
        .def("issorted", &NdArrayInterface::issorted<double>)
        .def("issquare", &NdArrayDouble::issquare)
        .def("item", &NdArrayDouble::item)
        .def("max", &NdArrayInterface::max<double>)
        .def("min", &NdArrayInterface::min<double>)
        .def("mean", &NdArrayInterface::mean<double>)
        .def("median", &NdArrayInterface::median<double>)
        .def("nans", &NdArrayDouble::nans, bp::return_internal_reference<>())
        .def("nbytes", &NdArrayDouble::nbytes)
        .def("none", &NdArrayInterface::none<double>)
        .def("nonzero", &NdArrayInterface::nonzero<double>)
        .def("norm", &NdArrayInterface::norm<double>)
        .def("numRows", &NdArrayDouble::numCols)
        .def("numCols", &NdArrayDouble::numRows)
        .def("ones", &NdArrayInterface::ones<double>)
        .def("partition", &NdArrayInterface::partition<double>)
        .def("print", &NdArrayDouble::print)
        .def("prod", &NdArrayInterface::prod<double>)
        .def("ptp", &NdArrayInterface::ptp<double>)
        .def("put", &NdArrayInterface::putFlat<double>)
        .def("put", &NdArrayInterface::putRowCol<double>)
        .def("put", &NdArrayInterface::putSlice1DValue<double>)
        .def("put", &NdArrayInterface::putSlice1DValues<double>)
        .def("put", &NdArrayInterface::putSlice2DValue<double>)
        .def("put", &NdArrayInterface::putSlice2DValueRow<double>)
        .def("put", &NdArrayInterface::putSlice2DValueCol<double>)
        .def("put", &NdArrayInterface::putSlice2DValues<double>)
        .def("put", &NdArrayInterface::putSlice2DValuesRow<double>)
        .def("put", &NdArrayInterface::putSlice2DValuesCol<double>)
        .def("putMask", &NdArrayInterface::putMaskSingle<double>)
        .def("putMask", &NdArrayInterface::putMaskMultiple<double>)
        .def("ravel", &NdArrayInterface::ravel<double>)
        .def("repeat", &NdArrayInterface::repeat<double>)
        .def("reshape", &NdArrayInterface::reshapeInt<double>)
        .def("reshape", &NdArrayInterface::reshapeValues<double>)
        .def("reshape", &NdArrayInterface::reshapeShape<double>)
        .def("reshapeList", &NdArrayInterface::reshapeList<double>)
        .def("replace", &NdArrayInterface::replace<double>)
        .def("resizeFast", &NdArrayInterface::resizeFast<double>)
        .def("resizeFastList", &NdArrayInterface::resizeFastList<double>)
        .def("resizeSlow", &NdArrayInterface::resizeSlow<double>)
        .def("resizeSlowList", &NdArrayInterface::resizeSlowList<double>)
        .def("rms", &NdArrayInterface::rms<double>)
        .def("round", &NdArrayInterface::round<double>)
        .def("row", &NdArrayDouble::row)
        .def("shape", &NdArrayDouble::shape)
        .def("size", &NdArrayDouble::size)
        .def("sort", &NdArrayInterface::sort<double>)
        .def("stdev", &NdArrayInterface::stdev<double>)
        .def("sum", &NdArrayInterface::sum<double>)
        .def("swapaxes", &NdArrayInterface::swapaxes<double>)
        .def("tofile", &NdArrayDouble::tofile)
        .def("toStlVector", &NdArrayDouble::toStlVector)
        .def("trace", &NdArrayDouble::trace)
        .def("transpose", &NdArrayInterface::transpose<double>)
        .def("var", &NdArrayInterface::var<double>)
        .def("zeros", &NdArrayDouble::zeros, bp::return_internal_reference<>())
        .def("operatorPlusScaler", &NdArrayInterface::operatorPlusScaler<double>)
        .def("operatorPlusScalerReversed", &NdArrayInterface::operatorPlusScalerReversed<double>)
        .def("operatorPlusArray", &NdArrayInterface::operatorPlusArray<double>)
        .def("operatorNegative", &NdArrayInterface::operatorNegative<double>)
        .def("operatorMinusScaler", &NdArrayInterface::operatorMinusScaler<double>)
        .def("operatorMinusScalerReversed", &NdArrayInterface::operatorMinusScalerReversed<double>)
        .def("operatorMinusArray", &NdArrayInterface::operatorMinusArray<double>)
        .def("operatorMultiplyScaler", &NdArrayInterface::operatorMultiplyScaler<double>)
        .def("operatorMultiplyScalerReversed", &NdArrayInterface::operatorMultiplyScalerReversed<double>)
        .def("operatorMultiplyArray", &NdArrayInterface::operatorMultiplyArray<double>)
        .def("operatorDivideScaler", &NdArrayInterface::operatorDivideScaler<double>)
        .def("operatorDivideScalerReversed", &NdArrayInterface::operatorDivideScalerReversed<double>)
        .def("operatorDivideArray", &NdArrayInterface::operatorDivideArray<double>)
        .def("operatorEquality", &NdArrayInterface::operatorEqualityScaler<double>)
        .def("operatorEqualityReversed", &NdArrayInterface::operatorEqualityScalerReversed<double>)
        .def("operatorEquality", &NdArrayInterface::operatorEqualityArray<double>)
        .def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScaler<double>)
        .def("operatorNotEqualityReversed", &NdArrayInterface::operatorNotEqualityScalerReversed<double>)
        .def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityArray<double>)
        .def("operatorLess", &NdArrayInterface::operatorLessScaler<double>)
        .def("operatorLessReversed", &NdArrayInterface::operatorLessScalerReversed<double>)
        .def("operatorLess", &NdArrayInterface::operatorLessArray<double>)
        .def("operatorGreater", &NdArrayInterface::operatorGreaterScaler<double>)
        .def("operatorGreaterReversed", &NdArrayInterface::operatorGreaterScalerReversed<double>)
        .def("operatorGreater", &NdArrayInterface::operatorGreaterArray<double>)
        .def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScaler<double>)
        .def("operatorLessEqualReversed", &NdArrayInterface::operatorLessEqualScalerReversed<double>)
        .def("operatorLessEqual", &NdArrayInterface::operatorLessEqualArray<double>)
        .def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScaler<double>)
        .def("operatorGreaterEqualReversed", &NdArrayInterface::operatorGreaterEqualScalerReversed<double>)
        .def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualArray<double>)
        .def("operatorPrePlusPlus", &NdArrayInterface::operatorPrePlusPlus<double>)
        .def("operatorPostPlusPlus", &NdArrayInterface::operatorPostPlusPlus<double>)
        .def("operatorPreMinusMinus", &NdArrayInterface::operatorPreMinusMinus<double>)
        .def("operatorPostMinusMinus", &NdArrayInterface::operatorPostMinusMinus<double>);

    typedef NdArray<uint32> NdArrayInt;
    bp::class_<NdArrayInt>
        ("NdArrayInt", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayInt::item)
        .def("shape", &NdArrayInt::shape)
        .def("size", &NdArrayInt::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint32>)
        .def("endianess", &NdArrayInt::endianess)
        .def("setArray", &NdArrayInterface::setArray<uint32>)
        .def("byteswap", &NdArrayInt::byteswap, bp::return_internal_reference<>())
        .def("newbyteorder", &NdArrayInterface::newbyteorder<uint32>)
        .def("operatorModulusScaler", &NdArrayInterface::operatorModulusScaler<uint32>)
        .def("operatorModulusScalerReversed", &NdArrayInterface::operatorModulusScalerReversed<uint32>)
        .def("operatorModulusArray", &NdArrayInterface::operatorModulusArray<uint32>)
        .def("operatorBitwiseOrScaler", &NdArrayInterface::operatorBitwiseOrScaler<uint32>)
        .def("operatorBitwiseOrScalerReversed", &NdArrayInterface::operatorBitwiseOrScalerReversed<uint32>)
        .def("operatorBitwiseOrArray", &NdArrayInterface::operatorBitwiseOrArray<uint32>)
        .def("operatorBitwiseAndScaler", &NdArrayInterface::operatorBitwiseAndScaler<uint32>)
        .def("operatorBitwiseAndScalerReversed", &NdArrayInterface::operatorBitwiseAndScalerReversed<uint32>)
        .def("operatorBitwiseAndArray", &NdArrayInterface::operatorBitwiseAndArray<uint32>)
        .def("operatorBitwiseXorScaler", &NdArrayInterface::operatorBitwiseXorScaler<uint32>)
        .def("operatorBitwiseXorScalerReversed", &NdArrayInterface::operatorBitwiseXorScalerReversed<uint32>)
        .def("operatorBitwiseXorArray", &NdArrayInterface::operatorBitwiseXorArray<uint32>)
        .def("operatorBitwiseNot", &NdArrayInterface::operatorBitwiseNot<uint32>)
        .def("operatorLogicalAndArray", &NdArrayInterface::operatorLogicalAndArray<uint32>)
        .def("operatorLogicalAndScalar", &NdArrayInterface::operatorLogicalAndScalar<uint32>)
        .def("operatorLogicalAndScalarReversed", &NdArrayInterface::operatorLogicalAndScalarReversed<uint32>)
        .def("operatorLogicalOrArray", &NdArrayInterface::operatorLogicalOrArray<uint32>)
        .def("operatorLogicalOrScalar", &NdArrayInterface::operatorLogicalOrScalar<uint32>)
        .def("operatorLogicalOrScalarReversed", &NdArrayInterface::operatorLogicalOrScalarReversed<uint32>)
        .def("operatorNot", &NdArrayInterface::operatorNot<uint32>)
        .def("operatorBitshiftLeft", &NdArrayInterface::operatorBitshiftLeft<uint32>)
        .def("operatorBitshiftRight", &NdArrayInterface::operatorBitshiftRight<uint32>);

    typedef NdArray<int32> NdArrayInt32;
    bp::class_<NdArrayInt32>
        ("NdArrayInt32", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayInt32::item)
        .def("shape", &NdArrayInt32::shape)
        .def("size", &NdArrayInt32::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int32>)
        .def("endianess", &NdArrayInt32::endianess)
        .def("replace", &NdArrayInterface::replace<int32>)
        .def("setArray", &NdArrayInterface::setArray<int32>);

    typedef NdArray<uint32> NdArrayUInt32;
    bp::class_<NdArrayUInt32>
        ("NdArrayUInt32", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayUInt32::item)
        .def("shape", &NdArrayUInt32::shape)
        .def("size", &NdArrayUInt32::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint32>)
        .def("endianess", &NdArrayUInt32::endianess)
        .def("setArray", &NdArrayInterface::setArray<uint32>);

    typedef NdArray<uint64> NdArrayInt64;
    bp::class_<NdArrayInt64>
        ("NdArrayInt64", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayInt64::item)
        .def("shape", &NdArrayInt64::shape)
        .def("size", &NdArrayInt64::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint64>)
        .def("endianess", &NdArrayInt64::endianess)
        .def("setArray", &NdArrayInterface::setArray<uint64>);

    typedef NdArray<uint8> NdArrayInt8;
    bp::class_<NdArrayInt8>
        ("NdArrayInt8", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayInt8::item)
        .def("shape", &NdArrayInt8::shape)
        .def("size", &NdArrayInt8::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint8>)
        .def("endianess", &NdArrayInt8::endianess)
        .def("setArray", NdArrayInterface::setArray<uint8>);

    typedef NdArray<bool> NdArrayBool;
    bp::class_<NdArrayBool>
        ("NdArrayBool", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayBool::item)
        .def("shape", &NdArrayBool::shape)
        .def("size", &NdArrayBool::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<bool>)
        .def("endianess", &NdArrayBool::endianess)
        .def("setArray", NdArrayInterface::setArray<bool>);

    typedef NdArray<float> NdArrayFloat;
    bp::class_<NdArrayFloat>
        ("NdArrayFloat", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayFloat::item)
        .def("shape", &NdArrayFloat::shape)
        .def("size", &NdArrayFloat::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<float>)
        .def("endianess", &NdArrayFloat::endianess)
        .def("setArray", &NdArrayInterface::setArray<float>);

    // Methods.hpp
    bp::def("absScaler", &MethodsInterface::absScaler<double>);
    bp::def("absArray", &MethodsInterface::absArray<double>);
    bp::def("add", &MethodsInterface::addArrays<double>);
    bp::def("alen", &alen<double>);
    bp::def("all", &MethodsInterface::allArray<double>);
    bp::def("allclose", &allclose<double>);
    bp::def("amin", &MethodsInterface::aminArray<double>);
    bp::def("amax", &MethodsInterface::amaxArray<double>);
    bp::def("any", &MethodsInterface::anyArray<double>);
    bp::def("append", &append<double>);
    bp::def("applyPoly1d", &applyPoly1d<double>);
    bp::def("arange", &MethodsInterface::arangeArray<double>);
    bp::def("arccosScaler", &MethodsInterface::arccosScaler<double>);
    bp::def("arccosArray", &MethodsInterface::arccosArray<double>);
    bp::def("arccoshScaler", &MethodsInterface::arccoshScaler<double>);
    bp::def("arccoshArray", &MethodsInterface::arccoshArray<double>);
    bp::def("arcsinScaler", &MethodsInterface::arcsinScaler<double>);
    bp::def("arcsinArray", &MethodsInterface::arcsinArray<double>);
    bp::def("arcsinhScaler", &MethodsInterface::arcsinhScaler<double>);
    bp::def("arcsinhArray", &MethodsInterface::arcsinhArray<double>);
    bp::def("arctanScaler", &MethodsInterface::arctanScaler<double>);
    bp::def("arctanArray", &MethodsInterface::arctanArray<double>);
    bp::def("arctan2Scaler", &MethodsInterface::arctan2Scaler<double>);
    bp::def("arctan2Array", &MethodsInterface::arctan2Array<double>);
    bp::def("arctanhScaler", &MethodsInterface::arctanhScaler<double>);
    bp::def("arctanhArray", &MethodsInterface::arctanhArray<double>);
    bp::def("argmax", &MethodsInterface::argmaxArray<double>);
    bp::def("argmin", &MethodsInterface::argminArray<double>);
    bp::def("argsort", &MethodsInterface::argsortArray<double>);
    bp::def("argwhere", &MethodsInterface::argwhere<double>);
    bp::def("aroundScaler", &MethodsInterface::aroundScaler<double>);
    bp::def("aroundArray", &MethodsInterface::aroundArray<double>);
    bp::def("array_equal", &array_equal<double>);
    bp::def("array_equiv", &array_equiv<double>);
    bp::def("asarrayVector", &MethodsInterface::asarrayVector<double>);
    bp::def("asarrayList", &MethodsInterface::asarrayList<double>);
    bp::def("astype", &astype<double, uint32>);
    bp::def("average", &MethodsInterface::average<double>);
    bp::def("averageWeighted", &MethodsInterface::averageWeighted<double>);
    bp::def("binaryRepr", &binaryRepr<int8>);
    bp::def("binaryRepr", &binaryRepr<int16>);
    bp::def("binaryRepr", &binaryRepr<int32>);
    bp::def("binaryRepr", &binaryRepr<int64>);
    bp::def("binaryRepr", &binaryRepr<uint8>);
    bp::def("binaryRepr", &binaryRepr<uint16>);
    bp::def("binaryRepr", &binaryRepr<uint32>);
    bp::def("binaryRepr", &binaryRepr<uint64>);
    bp::def("bincount", &MethodsInterface::bincount<uint32>);
    bp::def("bincountWeighted", &MethodsInterface::bincountWeighted<uint32>);
    bp::def("bitwise_and", &MethodsInterface::bitwise_and<uint64>);
    bp::def("bitwise_not", &MethodsInterface::bitwise_not<uint64>);
    bp::def("bitwise_or", &MethodsInterface::bitwise_or<uint64>);
    bp::def("bitwise_xor", &MethodsInterface::bitwise_xor<uint64>);
    bp::def("andOperatorArray", &MethodsInterface::andOperatorArray<uint64>);
    bp::def("andOperatorScaler", &MethodsInterface::andOperatorScaler<uint64>);
    bp::def("orOperatorArray", &MethodsInterface::orOperatorArray<uint64>);
    bp::def("orOperatorScaler", &MethodsInterface::orOperatorScaler<uint64>);
    bp::def("byteswap", &MethodsInterface::byteswap<uint64>);
    bp::def("cbrtScaler", &MethodsInterface::cbrtScaler<double>);
    bp::def("cbrtArray", &MethodsInterface::cbrtArray<double>);
    bp::def("ceilScaler", &MethodsInterface::ceilScaler<double>);
    bp::def("centerOfMass", &MethodsInterface::centerOfMass<double>);
    bp::def("ceilArray", &MethodsInterface::ceilArray<double>);
    bp::def("clipScaler", &MethodsInterface::clipScaler<double>);
    bp::def("clipArray", &MethodsInterface::clipArray<double>);
    bp::def("column_stack", &MethodsInterface::column_stack<double>);
    bp::def("concatenate", &MethodsInterface::concatenate<double>);
    bp::def("contains", &contains<double>);
    bp::def("copy", &MethodsInterface::copy<double>);
    bp::def("copysign", &MethodsInterface::copySign<double>);
    bp::def("copyto", &MethodsInterface::copyto<double>);
    bp::def("cosScaler", &MethodsInterface::cosScaler<double>);
    bp::def("cosArray", &MethodsInterface::cosArray<double>);
    bp::def("coshScaler", &MethodsInterface::coshScaler<double>);
    bp::def("coshArray", &MethodsInterface::coshArray<double>);
    bp::def("count_nonzero", &MethodsInterface::count_nonzero<double>);
    bp::def("cross", &cross<double>);
    bp::def("cube", &MethodsInterface::cubeArray<double>);
    bp::def("cumprod", &MethodsInterface::cumprodArray<double>);
    bp::def("cumsum", &MethodsInterface::cumsumArray<double>);
    bp::def("deg2radScaler", &MethodsInterface::deg2radScaler<double>);
    bp::def("deg2radArray", &MethodsInterface::deg2radArray<double>);
    bp::def("degreesScaler", &MethodsInterface::degreesScaler<double>);
    bp::def("degreesArray", &MethodsInterface::degreesArray<double>);
    bp::def("deleteIndicesScaler", &MethodsInterface::deleteIndicesScaler<double>);
    bp::def("deleteIndicesSlice", &MethodsInterface::deleteIndicesSlice<double>);
    bp::def("diag", &MethodsInterface::diag<double>);
    bp::def("diagflat", &MethodsInterface::diagflat<double>);
    bp::def("diagonal", &MethodsInterface::diagonal<double>);
    bp::def("diff", &MethodsInterface::diff<double>);
    bp::def("divide", &MethodsInterface::divide<double>);
    bp::def("dot", &MethodsInterface::dot<double>);
    bp::def("dump", &dump<double>);
    bp::def("emptyRowCol", &MethodsInterface::emptyRowCol<double>);
    bp::def("emptyShape", &MethodsInterface::emptyShape<double>);
    bp::def("empty_like", &empty_like<double>);
    bp::def("endianess", &endianess<double>);
    bp::def("equal", &MethodsInterface::equal<double>);
    bp::def("expScaler", &MethodsInterface::expScaler<double>);
    bp::def("expArray", &MethodsInterface::expArray<double>);
    bp::def("exp2Scaler", &MethodsInterface::exp2Scaler<double>);
    bp::def("exp2Array", &MethodsInterface::exp2Array<double>);
    bp::def("expm1Scaler", &MethodsInterface::expm1Scaler<double>);
    bp::def("expm1Array", &MethodsInterface::expm1Array<double>);
    bp::def("eye1D", &MethodsInterface::eye1D<double>);
    bp::def("eye2D", &MethodsInterface::eye2D<double>);
    bp::def("eyeShape", &MethodsInterface::eyeShape<double>);
    bp::def("fillDiagonal", &fillDiagonal<double>);
    bp::def("find", &MethodsInterface::find);
    bp::def("findN", &MethodsInterface::findN);
    bp::def("fixScaler", &MethodsInterface::fixScaler<double>);
    bp::def("fixArray", &MethodsInterface::fixArray<double>);
    bp::def("flatten", &flatten<double>);
    bp::def("flatnonzero", &flatnonzero<double>);
    bp::def("flip", &flip<double>);
    bp::def("fliplr", &fliplr<double>);
    bp::def("flipud", &flipud<double>);
    bp::def("floorScaler", &MethodsInterface::floorScaler<double>);
    bp::def("floorArray", &MethodsInterface::floorArray<double>);
    bp::def("floor_divideScaler", &MethodsInterface::floor_divideScaler<double>);
    bp::def("floor_divideArray", &MethodsInterface::floor_divideArray<double>);
    bp::def("fmaxScaler", &MethodsInterface::fmaxScaler<double>);
    bp::def("fmaxArray", &MethodsInterface::fmaxArray<double>);
    bp::def("fminScaler", &MethodsInterface::fminScaler<double>);
    bp::def("fminArray", &MethodsInterface::fminArray<double>);
    bp::def("fmodScaler", &MethodsInterface::fmodScaler<uint32>);
    bp::def("fmodArray", &MethodsInterface::fmodArray<uint32>);
    bp::def("frombuffer", &frombuffer<double>);
    bp::def("fromfile", &fromfile<double>);
    bp::def("fullSquare", &MethodsInterface::fullSquare<double>);
    bp::def("fullRowCol", &MethodsInterface::fullRowCol<double>);
    bp::def("fullShape", &MethodsInterface::fullShape<double>);
    bp::def("full_like", &full_like<double>);
    bp::def("gcdScaler", &MethodsInterface::gcdScaler<int32>);
    bp::def("gcdArray", &MethodsInterface::gcdArray<int32>);
    bp::def("greater", &greater<double>);
    bp::def("greater_equal", &greater_equal<double>);
    bp::def("gradient", &gradient<double>);
    bp::def("histogram", &MethodsInterface::histogram<double>);
    bp::def("hstack", &MethodsInterface::hstack<double>);
    bp::def("hypotScaler", &MethodsInterface::hypotScaler<double>);
    bp::def("hypotScalerTriple", &MethodsInterface::hypotScalerTriple<double>);
    bp::def("hypotArray", &MethodsInterface::hypotArray<double>);
    bp::def("identity", &identity<double>);
    bp::def("interp", &MethodsInterface::interp<double>);
    bp::def("intersect1d", &intersect1d<uint32>);
    bp::def("invert", &invert<uint32>);
    bp::def("isclose", &isclose<double>);
    bp::def("isinfScaler", &MethodsInterface::isinfScaler<double>);
    bp::def("isinfArray", &MethodsInterface::isinfArray<double>);
    bp::def("isnanScaler", &MethodsInterface::isnanScaler<double>);
    bp::def("isnanArray", &MethodsInterface::isnanArray<double>);
    bp::def("lcmScaler", &MethodsInterface::lcmScaler<int32>);
    bp::def("lcmArray", &MethodsInterface::lcmArray<int32>);
    bp::def("ldexpScaler", &MethodsInterface::ldexpScaler<double>);
    bp::def("ldexpArray", &MethodsInterface::ldexpArray<double>);
    bp::def("left_shift", &left_shift<uint32>);
    bp::def("less", &less<double>);
    bp::def("less_equal", &less_equal<double>);
    bp::def("linspace", &linspace<double>);
    bp::def("load", &load<double>);
    bp::def("logScaler", &MethodsInterface::logScaler<double>);
    bp::def("logArray", &MethodsInterface::logArray<double>);
    bp::def("log10Scaler", &MethodsInterface::log10Scaler<double>);
    bp::def("log10Array", &MethodsInterface::log10Array<double>);
    bp::def("log1pScaler", &MethodsInterface::log1pScaler<double>);
    bp::def("log1pArray", &MethodsInterface::log1pArray<double>);
    bp::def("log2Scaler", &MethodsInterface::log2Scaler<double>);
    bp::def("log2Array", &MethodsInterface::log2Array<double>);
    bp::def("logical_and", &logical_and<double>);
    bp::def("logical_not", &logical_not<double>);
    bp::def("logical_or", &logical_or<double>);
    bp::def("logical_xor", &logical_xor<double>);
    bp::def("matmul", &matmul<double>);
    bp::def("max", &max<double>);
    bp::def("maximum", &maximum<double>);
    bp::def("meshgrid", &MethodsInterface::meshgrid<double>);
    bp::def("mean", &mean<double>);
    bp::def("median", &median<double>);
    bp::def("min", &min<double>);
    bp::def("minimum", &minimum<double>);
    bp::def("mod", &mod<uint32>);
    bp::def("multiply", &multiply<double>);
    bp::def("nanargmax", &nanargmax<double>);
    bp::def("nanargmin", &nanargmin<double>);
    bp::def("nancumprod", &nancumprod<double>);
    bp::def("nancumsum", &nancumsum<double>);
    bp::def("nanmax", &nanmax<double>);
    bp::def("nanmean", &nanmean<double>);
    bp::def("nanmedian", &nanmedian<double>);
    bp::def("nanmin", &nanmin<double>);
    bp::def("nanpercentile", &nanpercentile<double>);
    bp::def("nanprod", &nanprod<double>);
    bp::def("nansSquare", &MethodsInterface::nansSquare);
    bp::def("nansRowCol", &MethodsInterface::nansRowCol);
    bp::def("nansShape", &MethodsInterface::nansShape);
    bp::def("nansList", &MethodsInterface::nansList);
    bp::def("nans_like", &nans_like<double>);
    bp::def("nanstdev", &nanstdev<double>);
    bp::def("nansum", &nansum<double>);
    bp::def("nanvar", &nanvar<double>);
    bp::def("nbytes", &nbytes<double>);
    bp::def("newbyteorderScaler", &MethodsInterface::newbyteorderScaler<uint32>);
    bp::def("newbyteorderArray", &MethodsInterface::newbyteorderArray<uint32>);
    bp::def("negative", &negative<double>);
    bp::def("none", &MethodsInterface::noneArray<double>);
    bp::def("nonzero", &nonzero<double>);
    bp::def("norm", &norm<double>);
    bp::def("not_equal", &not_equal<double>);
    bp::def("onesSquare", &MethodsInterface::onesSquare<double>);
    bp::def("onesRowCol", &MethodsInterface::onesRowCol<double>);
    bp::def("onesShape", &MethodsInterface::onesShape<double>);
    bp::def("ones_like", &ones_like<double, double>);
    bp::def("outer", &MethodsInterface::outer<double>);
    bp::def("pad", &pad<double>);
    bp::def("partition", &partition<double>);
    bp::def("percentile", &percentile<double>);
    bp::def("powerArrayScaler", &MethodsInterface::powerArrayScaler<double>);
    bp::def("powerArrayArray", &MethodsInterface::powerArrayArray<double>);
    bp::def("powerfArrayScaler", &MethodsInterface::powerfArrayScaler<double>);
    bp::def("powerfArrayArray", &MethodsInterface::powerfArrayArray<double>);
    bp::def("prod", &prod<double>);
    bp::def("ptp", &ptp<double>);
    bp::def("put", &put<double>, bp::return_internal_reference<>());
    bp::def("putmask", &MethodsInterface::putmask<double>);
    bp::def("putmaskScaler", &MethodsInterface::putmaskScaler<double>);
    bp::def("rad2degScaler", &MethodsInterface::rad2degScaler<double>);
    bp::def("rad2degArray", &MethodsInterface::rad2degArray<double>);
    bp::def("radiansScaler", &MethodsInterface::radiansScaler<double>);
    bp::def("radiansArray", &MethodsInterface::radiansArray<double>);
    bp::def("ravel", &MethodsInterface::ravel<double>, bp::return_internal_reference<>());
    bp::def("reciprocal", &reciprocal<double>);
    bp::def("remainderScaler", &MethodsInterface::remainderScaler<double>);
    bp::def("remainderArray", &MethodsInterface::remainderArray<double>);
    bp::def("replace", &MethodsInterface::replace<double>);
    bp::def("reshape", &MethodsInterface::reshapeInt<double>, bp::return_internal_reference<>());
    bp::def("reshape", &MethodsInterface::reshapeShape<double>, bp::return_internal_reference<>());
    bp::def("reshape", &MethodsInterface::reshapeValues<double>, bp::return_internal_reference<>());
    bp::def("reshapeList", &MethodsInterface::reshapeList<double>, bp::return_internal_reference<>());
    bp::def("resizeFast", &MethodsInterface::resizeFast<double>, bp::return_internal_reference<>());
    bp::def("resizeFastList", &MethodsInterface::resizeFastList<double>, bp::return_internal_reference<>());
    bp::def("resizeSlow", &MethodsInterface::resizeSlow<double>, bp::return_internal_reference<>());
    bp::def("resizeSlowList", &MethodsInterface::resizeSlowList<double>, bp::return_internal_reference<>());
    bp::def("right_shift", &right_shift<uint32>);
    bp::def("rintScaler", &MethodsInterface::rintScaler<double>);
    bp::def("rintArray", &MethodsInterface::rintArray<double>);
    bp::def("rms", &rms<double>);
    bp::def("roll", &roll<double>);
    bp::def("rot90", &rot90<double>);
    bp::def("roundScaler", &MethodsInterface::roundScaler<double>);
    bp::def("roundArray", &MethodsInterface::roundArray<double>);
    bp::def("row_stack", &MethodsInterface::row_stack<double>);
    bp::def("setdiff1d", &setdiff1d<uint32>);
    bp::def("signScaler", &MethodsInterface::signScaler<double>);
    bp::def("signArray", &MethodsInterface::signArray<double>);
    bp::def("signbitScaler", &MethodsInterface::signbitScaler<double>);
    bp::def("signbitArray", &MethodsInterface::signbitArray<double>);
    bp::def("sinScaler", &MethodsInterface::sinScaler<double>);
    bp::def("sinArray", &MethodsInterface::sinArray<double>);
    bp::def("sincScaler", &MethodsInterface::sincScaler<double>);
    bp::def("sincArray", &MethodsInterface::sincArray<double>);
    bp::def("sinhScaler", &MethodsInterface::sinhScaler<double>);
    bp::def("sinhArray", &MethodsInterface::sinhArray<double>);
    bp::def("size", &size<double>);
    bp::def("sort", &sort<double>);
    bp::def("sqrtScaler", &MethodsInterface::sqrtScaler<double>);
    bp::def("sqrtArray", &MethodsInterface::sqrtArray<double>);
    bp::def("squareScaler", &MethodsInterface::squareScaler<double>);
    bp::def("squareArray", &MethodsInterface::squareArray<double>);
    bp::def("stack", &MethodsInterface::stack<double>);
    bp::def("stdev", &nc::stdev<double>);
    bp::def("subtract", &MethodsInterface::subtractArrays<double>);
    bp::def("sum", &sum<double>);
    bp::def("swapaxes", &swapaxes<double>);
    bp::def("swap", &nc::swap<double>);
    bp::def("tanScaler", &MethodsInterface::tanScaler<double>);
    bp::def("tanArray", &MethodsInterface::tanArray<double>);
    bp::def("tanhScaler", &MethodsInterface::tanhScaler<double>);
    bp::def("tanhArray", &MethodsInterface::tanhArray<double>);
    bp::def("tileRectangle", &MethodsInterface::tileRectangle<double>);
    bp::def("tileShape", &MethodsInterface::tileShape<double>);
    bp::def("tileList", &MethodsInterface::tileList<double>);
    bp::def("tofile", &tofile<double>);
    bp::def("toStlVector", &toStlVector<double>);
    bp::def("trace", &trace<double>);
    bp::def("transpose", &transpose<double>);
    bp::def("trapzDx", &MethodsInterface::trapzDx<double>);
    bp::def("trapz", &MethodsInterface::trapz<double>);
    bp::def("trilSquare", &MethodsInterface::trilSquare<double>);
    bp::def("trilRect", &MethodsInterface::trilRect<double>);
    bp::def("trilArray", &MethodsInterface::trilArray<double>);
    bp::def("triuSquare", &MethodsInterface::triuSquare<double>);
    bp::def("triuRect", &MethodsInterface::triuRect<double>);
    bp::def("triuArray", &MethodsInterface::triuArray<double>);
    bp::def("trim_zeros", &trim_zeros<double>);
    bp::def("truncScaler", &MethodsInterface::truncScaler<double>);
    bp::def("truncArray", &MethodsInterface::truncArray<double>);
    bp::def("union1d", &union1d<uint32>);
    bp::def("unique", &unique<double>);
    bp::def("unwrapScaler", &MethodsInterface::unwrapScaler<double>);
    bp::def("unwrapArray", &MethodsInterface::unwrapArray<double>);
    bp::def("var", &var<double>);
    bp::def("vstack", &MethodsInterface::vstack<double>);
    bp::def("where", &MethodsInterface::where<double>);
    bp::def("zerosSquare", &MethodsInterface::zerosSquare<double>);
    bp::def("zerosRowCol", &MethodsInterface::zerosRowCol<double>);
    bp::def("zerosShape", &MethodsInterface::zerosShape<double>);
    bp::def("zerosList", &MethodsInterface::zerosList<double>);
    bp::def("zeros_like", &zeros_like<double, double>);

    // Utils.hpp
    bp::def("num2str", &utils::num2str<double>);
    bp::def("sqr", &utils::sqr<double>);
    bp::def("cube", &utils::cube<double>);
    bp::def("power", &utils::power<double>);
    bp::def("powerf", &utils::powerf);

    bp::def("num2str", &utils::num2str<float>);
    bp::def("sqr", &utils::sqr<float>);
    bp::def("cube", &utils::cube<float>);
    bp::def("power", &utils::power<float>);
    bp::def("powerf", &utils::powerf);

    bp::def("num2str", &utils::num2str<int8>);
    bp::def("sqr", &utils::sqr<int8>);
    bp::def("cube", &utils::cube<int8>);
    bp::def("power", &utils::power<int8>);
    bp::def("powerf", &utils::powerf);

    bp::def("num2str", &utils::num2str<int16>);
    bp::def("sqr", &utils::sqr<int16>);
    bp::def("cube", &utils::cube<int16>);
    bp::def("power", &utils::power<int16>);
    bp::def("powerf", &utils::powerf);

    bp::def("num2str", &utils::num2str<int32>);
    bp::def("sqr", &utils::sqr<int32>);
    bp::def("cube", &utils::cube<int32>);
    bp::def("power", &utils::power<int32>);
    bp::def("powerf", &utils::powerf);

    bp::def("num2str", &utils::num2str<int64>);
    bp::def("sqr", &utils::sqr<int64>);
    bp::def("cube", &utils::cube<int64>);
    bp::def("power", &utils::power<int64>);
    bp::def("powerf", &utils::powerf);

    bp::def("num2str", &utils::num2str<uint8>);
    bp::def("sqr", &utils::sqr<uint8>);
    bp::def("cube", &utils::cube<uint8>);
    bp::def("power", &utils::power<uint8>);
    bp::def("powerf", &utils::powerf);

    bp::def("num2str", &utils::num2str<uint16>);
    bp::def("sqr", &utils::sqr<uint16>);
    bp::def("cube", &utils::cube<uint16>);
    bp::def("power", &utils::power<uint16>);
    bp::def("powerf", &utils::powerf);

    bp::def("num2str", &utils::num2str<uint32>);
    bp::def("sqr", &utils::sqr<uint32>);
    bp::def("cube", &utils::cube<uint32>);
    bp::def("power", &utils::power<uint32>);
    bp::def("powerf", &utils::powerf);

    bp::def("num2str", &utils::num2str<uint64>);
    bp::def("sqr", &utils::sqr<uint64>);
    bp::def("cube", &utils::cube<uint64>);
    bp::def("power", &utils::power<uint64>);
    bp::def("powerf", &utils::powerf);

    // Random.hpp
    NdArray<double> (*bernoulliArray)(const Shape&, double) = &random::bernoulli<double>;
    double (*bernoilliScalar)(double) = &random::bernoulli<double>;
    bp::def("bernoulli", bernoulliArray);
    bp::def("bernoulli", bernoilliScalar);

    NdArray<double> (*betaArray)(const Shape&, double, double) = &random::beta<double>;
    double (*betaScalar)(double, double) = &random::beta<double>;
    bp::def("beta", betaArray);
    bp::def("beta", betaScalar);

    NdArray<int32> (*binomialArray)(const Shape&, int32, double) = &random::binomial<int32>;
    int32 (*binomialScalar)(int32, double) = &random::binomial<int32>;
    bp::def("binomial", binomialArray);
    bp::def("binomial", binomialScalar);

    NdArray<double> (*cauchyArray)(const Shape&, double, double) = &random::cauchy<double>;
    double (*cauchyScalar)(double, double) = &random::cauchy<double>;
    bp::def("cauchy", cauchyArray);
    bp::def("cauchy", cauchyScalar);


    NdArray<double> (*chiSquareArray)(const Shape&, double) = &random::chiSquare<double>;
    double (*chiSquareScalar)(double) = &random::chiSquare<double>;
    bp::def("chiSquare", chiSquareArray);
    bp::def("chiSquare", chiSquareScalar);


    bp::def("choiceSingle", &RandomInterface::choiceSingle<double>);
    bp::def("choiceMultiple", &RandomInterface::choiceMultiple<double>);

    NdArray<int32> (*discreteArray)(const Shape&, const NdArray<double>&) = &random::discrete<int32>;
    int32 (*discreteScalar)(const NdArray<double>&) = &random::discrete<int32>;
    bp::def("discrete", discreteArray);
    bp::def("discrete", discreteScalar);

    NdArray<double> (*exponentialArray)(const Shape&, double) = &random::exponential<double>;
    double (*exponentialScalar)(double) = &random::exponential<double>;
    bp::def("exponential", exponentialArray);
    bp::def("exponential", exponentialScalar);

    NdArray<double> (*extremeValueArray)(const Shape&, double, double) = &random::extremeValue<double>;
    double (*extremeValueScalar)(double, double) = &random::extremeValue<double>;
    bp::def("extremeValue", extremeValueArray);
    bp::def("extremeValue", extremeValueScalar);

    NdArray<double> (*fArray)(const Shape&, double, double) = &random::f<double>;
    double (*fScalar)(double, double) = &random::f<double>;
    bp::def("f", fArray);
    bp::def("f", fScalar);

    NdArray<double> (*gammaArray)(const Shape&, double, double) = &random::gamma<double>;
    double (*gammaScalar)(double, double) = &random::gamma<double>;
    bp::def("gamma", gammaArray);
    bp::def("gamma", gammaScalar);

    NdArray<int32> (*geometricArray)(const Shape&, double) = &random::geometric<int32>;
    int32 (*geometricScalar)(double) = &random::geometric<int32>;
    bp::def("geometric", geometricArray);
    bp::def("geometric", geometricScalar);

    NdArray<double> (*laplaceArray)(const Shape&, double, double) = &random::laplace<double>;
    double (*laplaceScalar)(double, double) = &random::laplace<double>;
    bp::def("laplace", laplaceArray);
    bp::def("laplace", laplaceScalar);

    NdArray<double> (*lognormalArray)(const Shape&, double, double) = &random::lognormal<double>;
    double (*lognormalScalar)(double, double) = &random::lognormal<double>;
    bp::def("lognormal", lognormalArray);
    bp::def("lognormal", lognormalScalar);

    NdArray<int32> (*negativeBinomialArray)(const Shape&, int32, double) = &random::negativeBinomial<int32>;
    int32 (*negativeBinomialScalar)(int32, double) = &random::negativeBinomial<int32>;
    bp::def("negativeBinomial", negativeBinomialArray);
    bp::def("negativeBinomial", negativeBinomialScalar);

    NdArray<double> (*nonCentralChiSquaredArray)(const Shape&, double, double) = &random::nonCentralChiSquared<double>;
    double (*nonCentralChiSquaredScalar)(double, double) = &random::nonCentralChiSquared<double>;
    bp::def("nonCentralChiSquared", nonCentralChiSquaredArray);
    bp::def("nonCentralChiSquared", nonCentralChiSquaredScalar);

    NdArray<double> (*normalArray)(const Shape&, double, double) = &random::normal<double>;
    double (*normalScalar)(double, double) = &random::normal<double>;
    bp::def("normal", normalArray);
    bp::def("normal", normalScalar);

    bp::def("permutationScaler", &RandomInterface::permutationScaler<double>);
    bp::def("permutationArray", &RandomInterface::permutationArray<double>);

    NdArray<int32> (*poissonArray)(const Shape&, double) = &random::poisson<int32>;
    int32 (*poissonScalar)(double) = &random::poisson<int32>;
    bp::def("poisson", poissonArray);
    bp::def("poisson", poissonScalar);

    NdArray<double> (*randArray)(const Shape&) = &random::rand<double>;
    double (*randScalar)() = &random::rand<double>;
    bp::def("rand", randArray);
    bp::def("rand", randScalar);

    NdArray<double> (*randFloatArray)(const Shape&, double, double) = &random::randFloat<double>;
    double (*randFloatScalar)(double, double) = &random::randFloat<double>;
    bp::def("randFloat", randFloatArray);
    bp::def("randFloat", randFloatScalar);

    NdArray<int32> (*randIntArray)(const Shape&, int32, int32) = &random::randInt<int32>;
    int32 (*randIntScalar)(int32, int32) = &random::randInt<int32>;
    bp::def("randInt", randIntArray);
    bp::def("randInt", randIntScalar);

    NdArray<double> (*randNArray)(const Shape&) = &random::randN<double>;
    double (*randNScalar)() = &random::randN<double>;
    bp::def("randN", randNArray);
    bp::def("randN", randNScalar);

    bp::def("seed", &random::seed);
    bp::def("shuffle", &random::shuffle<double>);

    NdArray<double> (*standardNormalArray)(const Shape&) = &random::standardNormal<double>;
    double (*standardNormalScalar)() = &random::standardNormal<double>;
    bp::def("standardNormal", standardNormalArray);
    bp::def("standardNormal", standardNormalScalar);

    NdArray<double> (*studentTArray)(const Shape&, double) = &random::studentT<double>;
    double (*studentTScalar)(double) = &random::studentT<double>;
    bp::def("studentT", studentTArray);
    bp::def("studentT", studentTScalar);

    NdArray<double> (*triangleArray)(const Shape&, double, double, double) = &random::triangle<double>;
    double (*triangleScalar)(double, double, double) = &random::triangle<double>;
    bp::def("triangle", triangleArray);
    bp::def("triangle", triangleScalar);

    NdArray<double> (*uniformArray)(const Shape&, double, double) = &random::uniform<double>;
    double (*uniformScalar)(double, double) = &random::uniform<double>;
    bp::def("uniform", uniformArray);
    bp::def("uniform", uniformScalar);

    bp::def("uniformOnSphere", &random::uniformOnSphere<double>);

    NdArray<double> (*weibullArray)(const Shape&, double, double) = &random::weibull<double>;
    double (*weibullScalar)(double, double) = &random::weibull<double>;
    bp::def("weibull", weibullArray);
    bp::def("weibull", weibullScalar);

    // Linalg.hpp
    bp::def("cholesky", &linalg::cholesky<double>);
    bp::def("det", &linalg::det<double>);
    bp::def("hat", &LinalgInterface::hatArray<double>);
    bp::def("inv", &linalg::inv<double>);
    bp::def("lstsq", &linalg::lstsq<double>);
    bp::def("lu_decomposition", &linalg::lu_decomposition<double>);
    bp::def("matrix_power", &linalg::matrix_power<double>);
    bp::def("multi_dot", &LinalgInterface::multi_dot<double>);
    bp::def("pivotLU_decomposition", &LinalgInterface::pivotLU_decomposition<double>);
    bp::def("svd", &linalg::svd<double>);

    // Rotations.hpp
    bp::class_<rotations::Quaternion>
        ("Quaternion", bp::init<>())
        .def(bp::init<double, double, double>())
        .def(bp::init<double, double, double, double>())
        .def(bp::init<Vec3, double>())
        .def(bp::init<NdArray<double>, double>())
        .def(bp::init<NdArray<double> >())
        .def("angleAxisRotationNdArray", &RotationsInterface::angleAxisRotationNdArray).staticmethod("angleAxisRotationNdArray")
        .def("angleAxisRotationVec3", &RotationsInterface::angleAxisRotationVec3).staticmethod("angleAxisRotationVec3")
        .def("angularVelocity", &RotationsInterface::angularVelocity)
        .def("conjugate", &rotations::Quaternion::conjugate)
        .def("i", &rotations::Quaternion::i)
        .def("identity", &rotations::Quaternion::identity).staticmethod("identity")
        .def("inverse", &rotations::Quaternion::inverse)
        .def("j", &rotations::Quaternion::j)
        .def("k", &rotations::Quaternion::k)
        .def("nlerp", &RotationsInterface::nlerp)
        .def("pitch", &rotations::Quaternion::pitch)
        .def("print", &rotations::Quaternion::print)
        .def("roll", &rotations::Quaternion::roll)
        .def("rotateNdArray", &RotationsInterface::rotateNdArray)
        .def("rotateVec3", &RotationsInterface::rotateVec3)
        .def("s", &rotations::Quaternion::s)
        .def("slerp", &RotationsInterface::slerp)
        .def("toDCM", &RotationsInterface::toDCM)
        .def("toNdArray", &rotations::Quaternion::toNdArray)
        .def("xRotation", &rotations::Quaternion::xRotation).staticmethod("xRotation")
        .def("yaw", &rotations::Quaternion::yaw)
        .def("yRotation", &rotations::Quaternion::yRotation).staticmethod("yRotation")
        .def("zRotation", &rotations::Quaternion::zRotation).staticmethod("zRotation")
        .def("__eq__", &rotations::Quaternion::operator==)
        .def("__neq__", &rotations::Quaternion::operator!=)
        .def("__add__", &rotations::Quaternion::operator+)
        .def("__sub__", &RotationsInterface::subtract)
        .def("__neg__", &RotationsInterface::negative)
        .def("__mul__", &RotationsInterface::multiplyScaler)
        .def("__mul__", &RotationsInterface::multiplyQuaternion)
        .def("__mul__", &RotationsInterface::multiplyArray)
        .def("__truediv__", &rotations::Quaternion::operator/)
        .def("__str__", &rotations::Quaternion::str);

    bp::class_<rotations::DCM>
        ("DCM", bp::init<>())
        .def("eulerAnglesValues", &RotationsInterface::eulerAnglesValues).staticmethod("eulerAnglesValues")
        .def("eulerAnglesArray", &RotationsInterface::eulerAnglesArray).staticmethod("eulerAnglesArray")
        .def("angleAxisRotationNdArray", &RotationsInterface::angleAxisRotationDcmNdArray).staticmethod("angleAxisRotationNdArray")
        .def("angleAxisRotationVec3", &RotationsInterface::angleAxisRotationDcmVec3).staticmethod("angleAxisRotationVec3")
        .def("isValid", &rotations::DCM::isValid).staticmethod("isValid")
        .def("roll", &rotations::DCM::roll).staticmethod("roll")
        .def("pitch", &rotations::DCM::pitch).staticmethod("pitch")
        .def("yaw", &rotations::DCM::yaw).staticmethod("yaw")
        .def("xRotation", &rotations::DCM::xRotation).staticmethod("xRotation")
        .def("yRotation", &rotations::DCM::yRotation).staticmethod("yRotation")
        .def("zRotation", &rotations::DCM::zRotation).staticmethod("zRotation");

    bp::def("rodriguesRotation", &RotationsInterface::rodriguesRotation<double>);
    bp::def("wahbasProblem", &RotationsInterface::wahbasProblem<double>);
    bp::def("wahbasProblemWeighted", &RotationsInterface::wahbasProblemWeighted<double>);

    // Filters.hpp
    bp::enum_<filter::Boundary>("Mode")
        .value("REFLECT", filter::Boundary::REFLECT)
        .value("CONSTANT", filter::Boundary::CONSTANT)
        .value("NEAREST", filter::Boundary::NEAREST)
        .value("MIRROR", filter::Boundary::MIRROR)
        .value("WRAP", filter::Boundary::WRAP);

    bp::def("complementaryMedianFilter", &filter::complementaryMedianFilter<double>);
    bp::def("complementaryMedianFilter1d", &filter::complementaryMedianFilter1d<double>);
    bp::def("convolve", &filter::convolve<double>);
    bp::def("convolve1d", &filter::convolve1d<double>);
    bp::def("gaussianFilter", &filter::gaussianFilter<double>);
    bp::def("gaussianFilter1d", &filter::gaussianFilter1d<double>);
    bp::def("laplaceFilter", &filter::laplace<double>);
    bp::def("maximumFilter", &filter::maximumFilter<double>);
    bp::def("maximumFilter1d", &filter::maximumFilter1d<double>);
    bp::def("medianFilter", &filter::medianFilter<double>);
    bp::def("medianFilter1d", &filter::medianFilter1d<double>);
    bp::def("minimumFilter", &filter::minimumFilter<double>);
    bp::def("minumumFilter1d", &filter::minumumFilter1d<double>);
    bp::def("percentileFilter", &filter::percentileFilter<double>);
    bp::def("percentileFilter1d", &filter::percentileFilter1d<double>);
    bp::def("rankFilter", &filter::rankFilter<double>);
    bp::def("rankFilter1d", &filter::rankFilter1d<double>);
    bp::def("uniformFilter", &filter::uniformFilter<double>);
    bp::def("uniformFilter1d", &filter::uniformFilter1d<double>);

    // Image Processing
    typedef imageProcessing::Pixel<double> PixelDouble;
    bp::class_<PixelDouble>
        ("Pixel", bp::init<>())
        .def(bp::init<uint32, uint32, double>())
        .def(bp::init<PixelDouble>())
        .def("__eq__", &PixelDouble::operator==)
        .def("__ne__", &PixelDouble::operator!=)
        .def("__lt__", &PixelDouble::operator<)
        .def_readonly("clusterId", &PixelDouble::clusterId)
        .def_readonly("row", &PixelDouble::row)
        .def_readonly("col", &PixelDouble::col)
        .def_readonly("intensity", &PixelDouble::intensity)
        .def("__str__", &PixelDouble::str)
        .def("print", &PixelDouble::print);

    typedef imageProcessing::Cluster<double> ClusterDouble;
    bp::class_<ClusterDouble>
        ("Cluster", bp::init<uint32>())
        .def(bp::init<ClusterDouble>())
        .def("__eq__", &ClusterDouble::operator==)
        .def("__ne__", &ClusterDouble::operator!=)
        .def("__getitem__", &ClusterDouble::at, bp::return_internal_reference<>())
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

    typedef imageProcessing::Centroid<double> CentroidDouble;
    bp::class_<CentroidDouble>
        ("Centroid", bp::init<>())
        .def(bp::init<ClusterDouble>())
        .def(bp::init<CentroidDouble>())
        .def("row", &CentroidDouble::row)
        .def("col", &CentroidDouble::col)
        .def("intensity", &CentroidDouble::intensity)
        .def("eod", &CentroidDouble::eod)
        .def("__str__", &CentroidDouble::str)
        .def("print", &CentroidDouble::print)
        .def("__eq__", &CentroidDouble::operator==)
        .def("__ne__", &CentroidDouble::operator!=)
        .def("__lt__", &CentroidDouble::operator<);

    bp::class_<std::vector<ClusterDouble> >("cluster_vector")
        .def(bp::vector_indexing_suite<std::vector<ClusterDouble> >());

    bp::class_<std::vector<CentroidDouble> >("centroid_vector")
        .def(bp::vector_indexing_suite<std::vector<CentroidDouble> >());

    bp::def("applyThreshold", &imageProcessing::applyThreshold<double>);
    bp::def("centroidClusters", &imageProcessing::centroidClusters<double>);
    bp::def("clusterPixels", &imageProcessing::clusterPixels<double>);
    bp::def("generateThreshold", &imageProcessing::generateThreshold<double>);
    bp::def("generateCentroids", &imageProcessing::generateCentroids<double>);
    bp::def("windowExceedances", &imageProcessing::windowExceedances);

    // Coordinates.hpp
    bp::class_<coordinates::RA>
        ("Ra", bp::init<>())
        .def(bp::init<double>())
        .def(bp::init<uint8, uint8, double>())
        .def(bp::init<coordinates::RA>())
        .def("degrees", &coordinates::RA::degrees)
        .def("radians", &coordinates::RA::radians)
        .def("hours", &coordinates::RA::hours)
        .def("minutes", &coordinates::RA::minutes)
        .def("seconds", &coordinates::RA::seconds)
        .def("__str__", &coordinates::RA::str)
        .def("print", &coordinates::RA::print)
        .def("__eq__", &coordinates::RA::operator==)
        .def("__ne__", &coordinates::RA::operator!=)
        .def("print", &RaInterface::print);

    bp::enum_<coordinates::Sign>("Sign")
        .value("POSITIVE", coordinates::Sign::POSITIVE)
        .value("NEGATIVE", coordinates::Sign::NEGATIVE);

    bp::class_<coordinates::Dec>
        ("Dec", bp::init<>())
        .def(bp::init<double>())
        .def(bp::init<coordinates::Sign, uint8, uint8, double>())
        .def(bp::init<coordinates::Dec>())
        .def("sign", &coordinates::Dec::sign)
        .def("degrees", &coordinates::Dec::degrees)
        .def("radians", &coordinates::Dec::radians)
        .def("degreesWhole", &coordinates::Dec::degreesWhole)
        .def("minutes", &coordinates::Dec::minutes)
        .def("seconds", &coordinates::Dec::seconds)
        .def("__str__", &coordinates::Dec::str)
        .def("print", &coordinates::Dec::print)
        .def("__eq__", &coordinates::Dec::operator==)
        .def("__ne__", &coordinates::Dec::operator!=)
        .def("print", &DecInterface::print);

    bp::class_<coordinates::Coordinate>
        ("Coordinate", bp::init<>())
        .def(bp::init<double, double>())
        .def(bp::init<uint8, uint8, double, coordinates::Sign, uint8, uint8, double>())
        .def(bp::init<double, double, double>())
        .def(bp::init<coordinates::RA, coordinates::Dec>())
        .def(bp::init<NdArrayDouble>())
        .def(bp::init<coordinates::Coordinate>())
        .def("dec", &coordinates::Coordinate::dec, bp::return_internal_reference<>())
        .def("ra", &coordinates::Coordinate::ra, bp::return_internal_reference<>())
        .def("x", &coordinates::Coordinate::x)
        .def("y", &coordinates::Coordinate::y)
        .def("z", &coordinates::Coordinate::z)
        .def("xyz", &coordinates::Coordinate::xyz)
        .def("degreeSeperation", &CoordinateInterface::degreeSeperationCoordinate)
        .def("degreeSeperation", &CoordinateInterface::degreeSeperationVector)
        .def("radianSeperation", &CoordinateInterface::radianSeperationCoordinate)
        .def("radianSeperation", &CoordinateInterface::radianSeperationVector)
        .def("__str__", &coordinates::Coordinate::str)
        .def("print", &coordinates::Coordinate::print)
        .def("__eq__", &coordinates::Coordinate::operator==)
        .def("__ne__", &coordinates::Coordinate::operator!=)
        .def("print", &coordinates::Coordinate::print);

    // DataCube
    typedef DataCube<double> DataCubeDouble;
    bp::class_<DataCubeDouble>
        ("DataCube", bp::init<>())
        .def(bp::init<uint32>())
        .def("at", &DataCubeInterface::at<double>, bp::return_internal_reference<>())
        .def("__getitem__", &DataCubeInterface::getItem<double>, bp::return_internal_reference<>())
        .def("back", &DataCubeDouble::back, bp::return_internal_reference<>())
        .def("dump", &DataCubeDouble::dump)
        .def("front", &DataCubeDouble::front, bp::return_internal_reference<>())
        .def("isempty", &DataCubeDouble::isempty)
        .def("shape", &DataCubeDouble::shape, bp::return_internal_reference<>())
        .def("size", &DataCubeDouble::size)
        .def("pop_back", &DataCubeDouble::pop_back)
        .def("pop_front", &DataCubeDouble::pop_front)
        .def("push_back", &DataCubeDouble::push_back)
        .def("push_front", &DataCubeDouble::push_front);

    // Polynomial.hpp
    typedef polynomial::Poly1d<double> Poly1d;
    bp::class_<Poly1d>
        ("Poly1d", bp::init<>())
        .def(bp::init<NdArray<double>, bool>())
        .def("area", &Poly1d::area)
        .def("coefficients", &Poly1d::coefficients)
        .def("deriv", &Poly1d::deriv)
        .def("integ", &Poly1d::integ)
        .def("order", &Poly1d::order)
        .def("print", &Poly1d::print)
        .def("__str__", &Poly1d::str)
        .def("__repr__", &Poly1d::str)
        .def("__getitem__", &Poly1d::operator())
        .def("__add__", &Poly1d::operator+)
        .def("__iadd__", &Poly1d::operator+=, bp::return_internal_reference<>())
        .def("__sub__", &Poly1d::operator-)
        .def("__isub__", &Poly1d::operator-=, bp::return_internal_reference<>())
        .def("__mul__", &Poly1d::operator*)
        .def("__imul__", &Poly1d::operator*=, bp::return_internal_reference<>())
        .def("__pow__", &Poly1d::operator^)
        .def("__ipow__", &Poly1d::operator^=, bp::return_internal_reference<>());

    bp::def("chebyshev_t_Scaler", &PolynomialInterface::chebyshev_t_Scaler<double>);
    bp::def("chebyshev_t_Array", &PolynomialInterface::chebyshev_t_Array<double>);
    bp::def("chebyshev_u_Scaler", &PolynomialInterface::chebyshev_u_Scaler<double>);
    bp::def("chebyshev_u_Array", &PolynomialInterface::chebyshev_u_Array<double>);
    bp::def("hermite_Scaler", &PolynomialInterface::hermite_Scaler<double>);
    bp::def("hermite_Array", &PolynomialInterface::hermite_Array<double>);
    bp::def("laguerre_Scaler1", &PolynomialInterface::laguerre_Scaler1<double>);
    bp::def("laguerre_Array1", &PolynomialInterface::laguerre_Array1<double>);
    bp::def("laguerre_Scaler2", &PolynomialInterface::laguerre_Scaler2<double>);
    bp::def("laguerre_Array2", &PolynomialInterface::laguerre_Array2<double>);
    bp::def("legendre_p_Scaler1", &PolynomialInterface::legendre_p_Scaler1<double>);
    bp::def("legendre_p_Array1", &PolynomialInterface::legendre_p_Array1<double>);
    bp::def("legendre_p_Scaler2", &PolynomialInterface::legendre_p_Scaler2<double>);
    bp::def("legendre_p_Array2", &PolynomialInterface::legendre_p_Array2<double>);
    bp::def("legendre_q_Scaler", &PolynomialInterface::legendre_q_Scaler<double>);
    bp::def("legendre_q_Array", &PolynomialInterface::legendre_q_Array<double>);
    bp::def("spherical_harmonic", &PolynomialInterface::spherical_harmonic<double>);
    bp::def("spherical_harmonic_r", &polynomial::spherical_harmonic_r<double, double>);
    bp::def("spherical_harmonic_i", &polynomial::spherical_harmonic_i<double, double>);

    // Roots.hpp
    bp::def("bisection_roots", &RootsInterface::bisection);
    bp::def("brent_roots", &RootsInterface::brent);
    bp::def("dekker_roots", &RootsInterface::dekker);
    bp::def("newton_roots", &RootsInterface::newton);
    bp::def("secant_roots", &RootsInterface::secant);

    // Integrate.hpp
    bp::def("integrate_gauss_legendre", &IntegrateInterface::gauss_legendre);
    bp::def("integrate_romberg", &IntegrateInterface::romberg);
    bp::def("integrate_simpson", &IntegrateInterface::simpson);
    bp::def("integrate_trapazoidal", &IntegrateInterface::trapazoidal);

    // Vec2.hpp
    bp::class_<Vec2>
        ("Vec2", bp::init<>())
        .def(bp::init<double, double>())
        .def(bp::init<NdArray<double> >())
        .def_readwrite("x", &Vec2::x)
        .def_readwrite("y", &Vec2::y)
        .def("angle", &Vec2::angle)
        .def("clampMagnitude", &Vec2::clampMagnitude)
        .def("distance", &Vec2::distance)
        .def("dot", &Vec2::dot)
        .def("down", &Vec2::down).staticmethod("down")
        .def("left", &Vec2::left).staticmethod("left")
        .def("lerp", &Vec2::lerp)
        .def("norm", &Vec2::norm)
        .def("normalize", &Vec2::normalize)
        .def("project", &Vec2::project)
        .def("right", &Vec2::right).staticmethod("right")
        .def("__str__", &Vec2::toString)
        .def("toNdArray", &Vec2Interface::toNdArray)
        .def("up", &Vec2::up).staticmethod("up")
        .def("__eq__", &Vec2::operator==)
        .def("__ne__", &Vec2::operator!=)
        .def("__iadd__", &Vec2Interface::plusEqualScaler, bp::return_internal_reference<>())
        .def("__iadd__", &Vec2Interface::plusEqualVec2, bp::return_internal_reference<>())
        .def("__isub__", &Vec2Interface::minusEqualVec2, bp::return_internal_reference<>())
        .def("__isub__", &Vec2Interface::minusEqualScaler, bp::return_internal_reference<>())
        .def("__imul__", &Vec2::operator*=, bp::return_internal_reference<>())
        .def("__itruediv__", &Vec2::operator/=, bp::return_internal_reference<>());

    bp::def("Vec2_addVec2", &Vec2Interface::addVec2);
    bp::def("Vec2_addVec2Scaler", &Vec2Interface::addVec2Scaler);
    bp::def("Vec2_addScalerVec2", &Vec2Interface::addScalerVec2);
    bp::def("Vec2_minusVec2", &Vec2Interface::minusVec2);
    bp::def("Vec2_minusVec2Scaler", &Vec2Interface::minusVec2Scaler);
    bp::def("Vec2_minusScalerVec2", &Vec2Interface::minusScalerVec2);
    bp::def("Vec2_multVec2Scaler", &Vec2Interface::multVec2Scaler);
    bp::def("Vec2_multScalerVec2", &Vec2Interface::multScalerVec2);
    bp::def("Vec2_divVec2Scaler", &Vec2Interface::divVec2Scaler);
    bp::def("Vec2_print", &Vec2Interface::print);

    // Vec3.hpp
    bp::class_<Vec3>
        ("Vec3", bp::init<>())
        .def(bp::init<double, double, double>())
        .def(bp::init<NdArray<double> >())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("angle", &Vec3::angle)
        .def("back", &Vec3::back).staticmethod("back")
        .def("clampMagnitude", &Vec3::clampMagnitude)
        .def("cross", &Vec3::cross)
        .def("distance", &Vec3::distance)
        .def("dot", &Vec3::dot)
        .def("down", &Vec3::down).staticmethod("down")
        .def("forward", &Vec3::forward).staticmethod("forward")
        .def("left", &Vec3::left).staticmethod("left")
        .def("lerp", &Vec3::lerp)
        .def("norm", &Vec3::norm)
        .def("normalize", &Vec3::normalize)
        .def("project", &Vec3::project)
        .def("right", &Vec3::right).staticmethod("right")
        .def("__str__", &Vec3::toString)
        .def("toNdArray", &Vec3Interface::toNdArray)
        .def("up", &Vec3::up).staticmethod("up")
        .def("__eq__", &Vec3::operator==)
        .def("__ne__", &Vec3::operator!=)
        .def("__iadd__", &Vec3Interface::plusEqualScaler, bp::return_internal_reference<>())
        .def("__iadd__", &Vec3Interface::plusEqualVec3, bp::return_internal_reference<>())
        .def("__isub__", &Vec3Interface::minusEqualScaler, bp::return_internal_reference<>())
        .def("__isub__", &Vec3Interface::minusEqualVec3, bp::return_internal_reference<>())
        .def("__imul__", &Vec3::operator*=, bp::return_internal_reference<>())
        .def("__itruediv__", &Vec3::operator/=, bp::return_internal_reference<>());

    bp::def("Vec3_addVec3", &Vec3Interface::addVec3);
    bp::def("Vec3_addVec3Scaler", &Vec3Interface::addVec3Scaler);
    bp::def("Vec3_addScalerVec3", &Vec3Interface::addScalerVec3);
    bp::def("Vec3_minusVec3", &Vec3Interface::minusVec3);
    bp::def("Vec3_minusVec3Scaler", &Vec3Interface::minusVec3Scaler);
    bp::def("Vec3_minusScalerVec3", &Vec3Interface::minusScalerVec3);
    bp::def("Vec3_multVec3Scaler", &Vec3Interface::multVec3Scaler);
    bp::def("Vec3_multScalerVec3", &Vec3Interface::multScalerVec3);
    bp::def("Vec3_divVec3Scaler", &Vec3Interface::divVec3Scaler);
    bp::def("Vec3_print", &Vec3Interface::print);

    // Special.hpp
    bp::def("airy_ai_Scaler", &SpecialInterface::airy_ai_Scaler<double>);
    bp::def("airy_ai_Array", &SpecialInterface::airy_ai_Array<double>);
    bp::def("airy_ai_prime_Scaler", &SpecialInterface::airy_ai_prime_Scaler<double>);
    bp::def("airy_ai_prime_Array", &SpecialInterface::airy_ai_prime_Array<double>);
    bp::def("airy_bi_Scaler", &SpecialInterface::airy_bi_Scaler<double>);
    bp::def("airy_bi_Array", &SpecialInterface::airy_bi_Array<double>);
    bp::def("airy_bi_prime_Scaler", &SpecialInterface::airy_bi_prime_Scaler<double>);
    bp::def("airy_bi_prime_Array", &SpecialInterface::airy_bi_prime_Array<double>);
    bp::def("bernoulli_Scaler", &SpecialInterface::bernoulli_Scaler);
    bp::def("bernoulli_Array", &SpecialInterface::bernoulli_Array);
    bp::def("bessel_in_Scaler", &SpecialInterface::bessel_in_Scaler<double>);
    bp::def("bessel_in_Array", &SpecialInterface::bessel_in_Array<double>);
    bp::def("bessel_in_prime_Scaler", &SpecialInterface::bessel_in_prime_Scaler<double>);
    bp::def("bessel_in_prime_Array", &SpecialInterface::bessel_in_prime_Array<double>);
    bp::def("bessel_jn_Scaler", &SpecialInterface::bessel_jn_Scaler<double>);
    bp::def("bessel_jn_Array", &SpecialInterface::bessel_jn_Array<double>);
    bp::def("bessel_jn_prime_Scaler", &SpecialInterface::bessel_jn_prime_Scaler<double>);
    bp::def("bessel_jn_prime_Array", &SpecialInterface::bessel_jn_prime_Array<double>);
    bp::def("bessel_kn_Scaler", &SpecialInterface::bessel_kn_Scaler<double>);
    bp::def("bessel_kn_Array", &SpecialInterface::bessel_kn_Array<double>);
    bp::def("bessel_kn_prime_Scaler", &SpecialInterface::bessel_kn_prime_Scaler<double>);
    bp::def("bessel_kn_prime_Array", &SpecialInterface::bessel_kn_prime_Array<double>);
    bp::def("bessel_yn_Scaler", &SpecialInterface::bessel_yn_Scaler<double>);
    bp::def("bessel_yn_Array", &SpecialInterface::bessel_yn_Array<double>);
    bp::def("bessel_yn_prime_Scaler", &SpecialInterface::bessel_yn_prime_Scaler<double>);
    bp::def("bessel_yn_prime_Array", &SpecialInterface::bessel_yn_prime_Array<double>);
    bp::def("beta_Scaler", &SpecialInterface::beta_Scaler<double>);
    bp::def("beta_Array", &SpecialInterface::beta_Array<double>);
    bp::def("cnr", &special::cnr);
    bp::def("cyclic_hankel_1", &special::cyclic_hankel_1<double, double>);
    bp::def("cyclic_hankel_2", &special::cyclic_hankel_2<double, double>);
    bp::def("digamma_Scaler", &SpecialInterface::digamma_Scaler<double>);
    bp::def("digamma_Array", &SpecialInterface::digamma_Array<double>);
    bp::def("erf_Scaler", &SpecialInterface::erf_Scaler<double>);
    bp::def("erf_Array", &SpecialInterface::erf_Array<double>);
    bp::def("erf_inv_Scaler", &SpecialInterface::erf_inv_Scaler<double>);
    bp::def("erf_inv_Array", &SpecialInterface::erf_inv_Array<double>);
    bp::def("erfc_Scaler", &SpecialInterface::erfc_Scaler<double>);
    bp::def("erfc_Array", &SpecialInterface::erfc_Array<double>);
    bp::def("erfc_inv_Scaler", &SpecialInterface::erfc_inv_Scaler<double>);
    bp::def("erfc_inv_Array", &SpecialInterface::erfc_inv_Array<double>);
    bp::def("factorial_Scaler", &SpecialInterface::factorial_Scaler);
    bp::def("factorial_Array", &SpecialInterface::factorial_Array);
    bp::def("gamma_Scaler", &SpecialInterface::gamma_Scaler<double>);
    bp::def("gamma_Array", &SpecialInterface::gamma_Array<double>);
    bp::def("gamma1pm1_Scaler", &SpecialInterface::gamma1pm1_Scaler<double>);
    bp::def("gamma1pm1_Array", &SpecialInterface::gamma1pm1_Array<double>);
    bp::def("log_gamma_Scaler", &SpecialInterface::log_gamma_Scaler<double>);
    bp::def("log_gamma_Array", &SpecialInterface::log_gamma_Array<double>);
    bp::def("pnr", &special::pnr);
    bp::def("polygamma_Scaler", &SpecialInterface::polygamma_Scaler<double>);
    bp::def("polygamma_Array", &SpecialInterface::polygamma_Array<double>);
    bp::def("prime_Scaler", &SpecialInterface::prime_Scaler);
    bp::def("prime_Array", &SpecialInterface::prime_Array);
    bp::def("riemann_zeta_Scaler", &SpecialInterface::riemann_zeta_Scaler<double>);
    bp::def("riemann_zeta_Array", &SpecialInterface::riemann_zeta_Array<double>);
    bp::def("softmax", &SpecialInterface::softmax<double>);
    bp::def("spherical_bessel_jn_Scaler", &SpecialInterface::spherical_bessel_jn_Scaler<double>);
    bp::def("spherical_bessel_jn_Array", &SpecialInterface::spherical_bessel_jn_Array<double>);
    bp::def("spherical_bessel_yn_Scaler", &SpecialInterface::spherical_bessel_yn_Scaler<double>);
    bp::def("spherical_bessel_yn_Array", &SpecialInterface::spherical_bessel_yn_Array<double>);
    bp::def("spherical_hankel_1", &special::spherical_hankel_1<double, double>);
    bp::def("spherical_hankel_2", &special::spherical_hankel_2<double, double>);
    bp::def("trigamma_Scaler", &SpecialInterface::trigamma_Scaler<double>);
    bp::def("trigamma_Array", &SpecialInterface::trigamma_Array<double>);
}
