#include "NumCpp.hpp"

#include <complex>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
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
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<dtype> test = { dtype{1},
            dtype{2},
            dtype{3},
            dtype{4},
            dtype{666},
            dtype{357},
            dtype{314159} };
        if (test.size() != 7)
        {
            return false;
        }

        if (test.shape().rows != 1 || test.shape().cols != test.size())
        {
            return false;
        }

        return test[0] == dtype{ 1 } &&
            test[1] == dtype{ 2 } &&
            test[2] == dtype{ 3 } &&
            test[3] == dtype{ 4 } &&
            test[4] == dtype{ 666 } &&
            test[5] == dtype{ 357 } &&
            test[6] == dtype{ 314159 };
    }

    //================================================================================

    template<typename dtype>
    bool test2DListContructor()
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<dtype> test = { {dtype{1}, dtype{2}}, 
            {dtype{4}, dtype{666}},
            {dtype{314159}, dtype{9}},
            {dtype{0}, dtype{8}} };
        if (test.size() != 8)
        {
            return false;
        }

        if (test.shape().rows != 4 || test.shape().cols != 2)
        {
            return false;
        }

        return test[0] == dtype{ 1 } &&
            test[1] == dtype{ 2 } &&
            test[2] == dtype{ 4 } &&
            test[3] == dtype{ 666 } &&
            test[4] == dtype{ 314159 } &&
            test[5] == dtype{ 9 } &&
            test[6] == dtype{ 0 } &&
            test[7] == dtype{ 8 };
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
        BoostNdarrayHelper<dtype> newNdArrayHelper(inBoostArray);
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

    template<typename dtype> // (1)
    np::ndarray operatorPlusEqualArray(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs += rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    np::ndarray operatorPlusEqualComplexArrayArithArray(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs += rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    np::ndarray operatorPlusEqualScaler(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2Boost(lhs += rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    np::ndarray operatorPlusEqualComplexArrayArithScaler(NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2Boost(lhs += rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    np::ndarray operatorPlusArray(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    np::ndarray operatorPlusArithArrayComplexArray(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2Boost(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    np::ndarray operatorPlusComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    np::ndarray operatorPlusArrayScaler(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2Boost(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    np::ndarray operatorPlusScalerArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    np::ndarray operatorPlusArithArrayComplexScaler(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2Boost(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    np::ndarray operatorPlusComplexScalerArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    np::ndarray operatorPlusComplexArrayArithScaler(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2Boost(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    np::ndarray operatorPlusArithScalerComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2Boost(lhs + rhs);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorNegative(const NdArray<dtype>& inArray)
    {
        return nc2Boost(-inArray);
    }

    //================================================================================

    template<typename dtype> // (1)
    np::ndarray operatorMinusEqualArray(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs -= rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    np::ndarray operatorMinusEqualComplexArrayArithArray(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs -= rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    np::ndarray operatorMinusEqualScaler(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2Boost(lhs -= rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    np::ndarray operatorMinusEqualComplexArrayArithScaler(NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2Boost(lhs -= rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    np::ndarray operatorMinusArray(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    np::ndarray operatorMinusArithArrayComplexArray(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2Boost(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    np::ndarray operatorMinusComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    np::ndarray operatorMinusArrayScaler(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2Boost(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    np::ndarray operatorMinusScalerArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    np::ndarray operatorMinusArithArrayComplexScaler(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2Boost(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    np::ndarray operatorMinusComplexScalerArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    np::ndarray operatorMinusComplexArrayArithScaler(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2Boost(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    np::ndarray operatorMinusArithScalerComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2Boost(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    np::ndarray operatorMultiplyEqualArray(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs *= rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    np::ndarray operatorMultiplyEqualComplexArrayArithArray(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs *= rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    np::ndarray operatorMultiplyEqualScaler(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2Boost(lhs *= rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    np::ndarray operatorMultiplyEqualComplexArrayArithScaler(NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2Boost(lhs *= rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    np::ndarray operatorMultiplyArray(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    np::ndarray operatorMultiplyArithArrayComplexArray(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2Boost(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    np::ndarray operatorMultiplyComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    np::ndarray operatorMultiplyArrayScaler(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2Boost(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    np::ndarray operatorMultiplyScalerArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    np::ndarray operatorMultiplyArithArrayComplexScaler(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2Boost(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    np::ndarray operatorMultiplyComplexScalerArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    np::ndarray operatorMultiplyComplexArrayArithScaler(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2Boost(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    np::ndarray operatorMultiplyArithScalerComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2Boost(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    np::ndarray operatorDivideEqualArray(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs /= rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    np::ndarray operatorDivideEqualComplexArrayArithArray(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs /= rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    np::ndarray operatorDivideEqualScaler(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2Boost(lhs /= rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    np::ndarray operatorDivideEqualComplexArrayArithScaler(NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2Boost(lhs /= rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    np::ndarray operatorDivideArray(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    np::ndarray operatorDivideArithArrayComplexArray(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2Boost(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    np::ndarray operatorDivideComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    np::ndarray operatorDivideArrayScaler(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2Boost(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    np::ndarray operatorDivideScalerArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    np::ndarray operatorDivideArithArrayComplexScaler(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2Boost(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    np::ndarray operatorDivideComplexScalerArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2Boost(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    np::ndarray operatorDivideComplexArrayArithScaler(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2Boost(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    np::ndarray operatorDivideArithScalerComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2Boost(lhs / rhs);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorModulusScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2Boost(inArray % inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorModulusScalerReversed(dtype inScaler, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inScaler % inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorModulusArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 % inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseOrScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2Boost(inArray | inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseOrScalerReversed(dtype inScaler, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inScaler | inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseOrArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 | inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseAndScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2Boost(inArray & inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseAndScalerReversed(dtype inScaler, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inScaler & inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseAndArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 & inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseXorScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2Boost(inArray ^ inScaler);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseXorScalerReversed(dtype inScaler, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inScaler ^ inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseXorArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 ^ inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitwiseNot(const NdArray<dtype>& inArray)
    {
        return nc2Boost(~inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalAndArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 && inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalAndScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2Boost(inArray && inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalAndScalarReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inValue && inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalOrArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 || inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalOrScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2Boost(inArray || inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLogicalOrScalarReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inValue || inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorNot(const NdArray<dtype>& inArray)
    {
        return nc2Boost(!inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorEqualityScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2Boost(inArray == inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorEqualityScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inValue == inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorEqualityArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 == inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorNotEqualityScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2Boost(inArray != inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorNotEqualityScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inValue != inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorNotEqualityArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 != inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2Boost(inArray < inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inValue < inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 < inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2Boost(inArray > inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inValue > inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 > inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessEqualScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2Boost(inArray <= inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessEqualScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inValue <= inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorLessEqualArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 <= inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterEqualScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2Boost(inArray >= inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterEqualScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2Boost(inValue >= inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorGreaterEqualArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(inArray1 >= inArray2);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitshiftLeft(const NdArray<dtype>& inArray, uint8 inNumBits)
    {
        return nc2Boost(inArray << inNumBits);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorBitshiftRight(const NdArray<dtype>& inArray, uint8 inNumBits)
    {
        return nc2Boost(inArray >> inNumBits);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorPrePlusPlus(NdArray<dtype>& inArray)
    {
        return nc2Boost(++inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorPostPlusPlus(NdArray<dtype>& inArray)
    {
        return nc2Boost(inArray++);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorPreMinusMinus(NdArray<dtype>& inArray)
    {
        return nc2Boost(--inArray);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray operatorPostMinusMinus(NdArray<dtype>& inArray)
    {
        return nc2Boost(inArray--);
    }
}

//================================================================================

namespace FunctionsInterface
{
    template<typename dtype>
    auto absScaler(dtype inValue) noexcept -> decltype(abs(inValue)) // trailing return type to help gcc
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
    dtype angleScaler(const std::complex<dtype>& inValue)
    {
        return angle(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray angleArray(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2Boost(angle(inArray));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray arangeArray(dtype inStart, dtype inStop, dtype inStep)
    {
        return nc2Boost(arange(inStart, inStop, inStep));
    }

    //================================================================================

    template<typename dtype>
    auto arccosScaler(dtype inValue) noexcept -> decltype(arccos(inValue)) // trailing return type to help gcc
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
    auto arccoshScaler(dtype inValue) noexcept -> decltype(arccosh(inValue)) // trailing return type to help gcc
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
    auto arcsinScaler(dtype inValue) noexcept -> decltype(arcsin(inValue)) // trailing return type to help gcc
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
    auto arcsinhScaler(dtype inValue) noexcept -> decltype(arcsinh(inValue)) // trailing return type to help gcc
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
    auto arctanScaler(dtype inValue) noexcept -> decltype(arctan(inValue)) // trailing return type to help gcc
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
    auto arctanhScaler(dtype inValue) noexcept -> decltype(arctanh(inValue)) // trailing return type to help gcc
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
    std::complex<dtype> complexScalerSingle(dtype inReal)
    {
        return nc::complex(inReal);
    }

    //================================================================================

    template<typename dtype>
    std::complex<dtype> complexScaler(dtype inReal, dtype inImag)
    {
        return nc::complex(inReal, inImag);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray complexArraySingle(const NdArray<dtype>& inReal)
    {
        return nc2Boost(nc::complex(inReal));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray complexArray(const NdArray<dtype>& inReal, const NdArray<dtype>& inImag)
    {
        return nc2Boost(nc::complex(inReal, inImag));
    }

    //================================================================================

    template<typename dtype>
    std::complex<dtype> conjScaler(const std::complex<dtype>& inValue)
    {
        return nc::conj(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray conjArray(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2Boost(nc::conj(inArray));
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
    auto cosScaler(dtype inValue) noexcept -> decltype(cos(inValue)) // trailing return type to help gcc
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
    auto coshScaler(dtype inValue) noexcept -> decltype(cosh(inValue)) // trailing return type to help gcc
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
    auto expScaler(dtype inValue) noexcept -> decltype(exp(inValue)) // trailing return type to help gcc
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
    dtype imagScaler(const std::complex<dtype>& inValue)
    {
        return nc::imag(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray imagArray(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2Boost(imag(inArray));
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
    auto logScaler(dtype inValue) noexcept -> decltype(log(inValue)) // trailing return type to help gcc
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
    auto log10Scaler(dtype inValue) noexcept -> decltype(log10(inValue)) // trailing return type to help gcc
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
    std::complex<dtype> polarScaler(dtype mag, dtype angle)
    {
        return polar(mag, angle);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray polarArray(const NdArray<dtype>& mag, const NdArray<dtype>& angle)
    {
        return nc2Boost(polar(mag, angle));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray powerArrayScaler(const NdArray<dtype>& inArray, uint8 inExponent)
    {
        return nc2Boost(nc::power(inArray, inExponent));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray powerArrayArray(const NdArray<dtype>& inArray, const NdArray<uint8>& inExponents)
    {
        return nc2Boost(nc::power(inArray, inExponents));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray powerfArrayScaler(const NdArray<dtype>& inArray, dtype inExponent)
    {
        return nc2Boost(nc::powerf(inArray, inExponent));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray powerfArrayArray(const NdArray<dtype>& inArray, const NdArray<dtype>& inExponents)
    {
        return nc2Boost(nc::powerf(inArray, inExponents));
    }

    //================================================================================

    template<typename dtype>
    std::complex<dtype> projScaler(const std::complex<dtype>& inValue)
    {
        return nc::proj(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray projArray(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2Boost(proj(inArray));
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
    dtype realScaler(const std::complex<dtype>& inValue)
    {
        return nc::real(inValue);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray realArray(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2Boost(real(inArray));
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
    auto sinScaler(dtype inValue) noexcept -> decltype(sin(inValue)) // trailing return type to help gcc
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
    dtype sincScaler(dtype inValue) noexcept
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
    auto sinhScaler(dtype inValue) noexcept -> decltype(sinh(inValue)) // trailing return type to help gcc
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
    auto sqrtScaler(dtype inValue) noexcept -> decltype(sqrt(inValue)) // trailing return type to help gcc
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
    auto tanScaler(dtype inValue) noexcept -> decltype(tan(inValue)) // trailing return type to help gcc
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
    auto tanhScaler(dtype inValue) noexcept -> decltype(tanh(inValue)) // trailing return type to help gcc
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
    std::complex<dtype> cyclic_hankel_1_Scaler(dtype v, dtype x) noexcept
    {
        return special::cyclic_hankel_1(v, x);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray cyclic_hankel_1_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2Boost(special::cyclic_hankel_1(v, x));
    }

    //================================================================================

    template<typename dtype>
    std::complex<dtype> cyclic_hankel_2_Scaler(dtype v, dtype x) noexcept
    {
        return special::cyclic_hankel_2(v, x);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray cyclic_hankel_2_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2Boost(special::cyclic_hankel_2(v, x));
    }

    //================================================================================

    template<typename dtype>
    std::complex<dtype> spherical_hankel_1_Scaler(dtype v, dtype x) noexcept
    {
        return special::spherical_hankel_1(v, x);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray spherical_hankel_1_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2Boost(special::spherical_hankel_1(v, x));
    }

    //================================================================================

    template<typename dtype>
    std::complex<dtype> spherical_hankel_2_Scaler(dtype v, dtype x) noexcept
    {
        return special::spherical_hankel_2(v, x);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray spherical_hankel_2_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2Boost(special::spherical_hankel_2(v, x));
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

    bp::class_<std::vector<double>>("doubleVector")
        .def(bp::vector_indexing_suite<std::vector<double>>());

    bp::class_<std::vector<std::complex<double>>>("doubleComplexVector")
        .def(bp::vector_indexing_suite<std::vector<std::complex<double>>>());

    typedef std::pair<NdArray<double>, NdArray<double>> doublePair;
    bp::class_<doublePair>("doublePair", bp::init<>())
        .def_readonly("first", &doublePair::first)
        .def_readonly("second", &doublePair::second);

    typedef std::pair<NdArray<uint32>, NdArray<uint32>> uint32Pair;
    bp::class_<uint32Pair>("uint32Pair", bp::init<>())
        .def_readonly("first", &uint32Pair::first)
        .def_readonly("second", &uint32Pair::second);

    // Constants.hpp
    bp::scope().attr("c") = constants::c;
    bp::scope().attr("e") = constants::e;
    bp::scope().attr("inf") = constants::inf;
    bp::scope().attr("pi") = constants::pi;
    bp::scope().attr("nan") = constants::nan;
    bp::scope().attr("j") = constants::j;
    bp::scope().attr("VERSION") = VERSION;

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
        .def("median", &NdArrayInterface::median<double>)
        .def("nans", &NdArrayDouble::nans, bp::return_internal_reference<>())
        .def("nbytes", &NdArrayDouble::nbytes)
        .def("none", &NdArrayInterface::none<double>)
        .def("nonzero", &NdArrayInterface::nonzero<double>)
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
        .def("round", &NdArrayInterface::round<double>)
        .def("row", &NdArrayDouble::row)
        .def("shape", &NdArrayDouble::shape)
        .def("size", &NdArrayDouble::size)
        .def("sort", &NdArrayInterface::sort<double>)
        .def("sum", &NdArrayInterface::sum<double>)
        .def("swapaxes", &NdArrayInterface::swapaxes<double>)
        .def("tofile", &NdArrayDouble::tofile)
        .def("toStlVector", &NdArrayDouble::toStlVector)
        .def("trace", &NdArrayDouble::trace)
        .def("transpose", &NdArrayInterface::transpose<double>)
        .def("zeros", &NdArrayDouble::zeros, bp::return_internal_reference<>());

    bp::def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualArray<double>); // (1)
    bp::def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualArray<std::complex<double>>); // (1)
    bp::def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualComplexArrayArithArray<double>); // (2)
    bp::def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualScaler<double>); // (3)
    bp::def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualScaler<std::complex<double>>); // (3)
    bp::def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualComplexArrayArithScaler<double>); // (4)

    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArray<double>); // (1)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArray<std::complex<double>>); // (1)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArithArrayComplexArray<double>); // (2)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusComplexArrayArithArray<double>); // (3)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArrayScaler<double>); // (4)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArrayScaler<std::complex<double>>); // (4)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusScalerArray<double>); // (5)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusScalerArray<std::complex<double>>); // (5)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArithArrayComplexScaler<double>); // (6)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusComplexScalerArithArray<double>); // (7)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusComplexArrayArithScaler<double>); // (8)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArithScalerComplexArray<double>); // (9)

    bp::def("operatorNegative", &NdArrayInterface::operatorNegative<double>);
    bp::def("operatorNegative", &NdArrayInterface::operatorNegative<std::complex<double>>);

    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualArray<double>); // (1)
    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualArray<std::complex<double>>); // (1)
    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualComplexArrayArithArray<double>); // (2)
    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualScaler<double>); // (3)
    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualScaler<std::complex<double>>); // (3)
    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualComplexArrayArithScaler<double>); // (4)

    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArray<double>); // (1)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArray<std::complex<double>>); // (1)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArithArrayComplexArray<double>); // (2)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusComplexArrayArithArray<double>); // (3)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArrayScaler<double>); // (4)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArrayScaler<std::complex<double>>); // (4)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusScalerArray<double>); // (5)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusScalerArray<std::complex<double>>); // (5)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArithArrayComplexScaler<double>); // (6)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusComplexScalerArithArray<double>); // (7)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusComplexArrayArithScaler<double>); // (8)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArithScalerComplexArray<double>); // (9)

    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualArray<double>); // (1)
    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualArray<std::complex<double>>); // (1)
    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualComplexArrayArithArray<double>); // (2)
    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualScaler<double>); // (3)
    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualScaler<std::complex<double>>); // (3)
    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualComplexArrayArithScaler<double>); // (4)

    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArray<double>); // (1)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArray<std::complex<double>>); // (1)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithArrayComplexArray<double>); // (2)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexArrayArithArray<double>); // (3)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArrayScaler<double>); // (4)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArrayScaler<std::complex<double>>); // (4)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyScalerArray<double>); // (5)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyScalerArray<std::complex<double>>); // (5)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithArrayComplexScaler<double>); // (6)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexScalerArithArray<double>); // (7)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexArrayArithScaler<double>); // (8)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithScalerComplexArray<double>); // (9)

    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualArray<double>); // (1)
    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualArray<std::complex<double>>); // (1)
    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualComplexArrayArithArray<double>); // (2)
    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualScaler<double>); // (3)
    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualScaler<std::complex<double>>); // (3)
    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualComplexArrayArithScaler<double>); // (4)

    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArray<double>); // (1)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArray<std::complex<double>>); // (1)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArithArrayComplexArray<double>); // (2)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideComplexArrayArithArray<double>); // (3)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArrayScaler<double>); // (4)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArrayScaler<std::complex<double>>); // (4)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideScalerArray<double>); // (5)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideScalerArray<std::complex<double>>); // (5)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArithArrayComplexScaler<double>); // (6)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideComplexScalerArithArray<double>); // (7)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideComplexArrayArithScaler<double>); // (8)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArithScalerComplexArray<double>); // (9)

    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityScaler<double>);
    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityScaler<std::complex<double>>);
    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityScalerReversed<double>);
    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityScalerReversed<std::complex<double>>);
    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityArray<double>);
    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityArray<std::complex<double>>);

    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScaler<double>);
    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScaler<std::complex<double>>);
    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScalerReversed<double>);
    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScalerReversed<std::complex<double>>);
    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityArray<double>);
    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityArray<std::complex<double>>);

    bp::def("operatorLess", &NdArrayInterface::operatorLessScaler<double>);
    bp::def("operatorLess", &NdArrayInterface::operatorLessScaler<std::complex<double>>);
    bp::def("operatorLess", &NdArrayInterface::operatorLessScalerReversed<double>);
    bp::def("operatorLess", &NdArrayInterface::operatorLessScalerReversed<std::complex<double>>);
    bp::def("operatorLess", &NdArrayInterface::operatorLessArray<double>);
    bp::def("operatorLess", &NdArrayInterface::operatorLessArray<std::complex<double>>);

    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterScaler<double>);
    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterScaler<std::complex<double>>);
    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterScalerReversed<double>);
    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterScalerReversed<std::complex<double>>);
    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterArray<double>);
    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterArray<std::complex<double>>);

    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScaler<double>);
    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScaler<std::complex<double>>);
    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScalerReversed<double>);
    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScalerReversed<std::complex<double>>);
    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualArray<double>);
    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualArray<std::complex<double>>);

    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScaler<double>);
    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScaler<std::complex<double>>);
    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScalerReversed<double>);
    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScalerReversed<std::complex<double>>);
    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualArray<double>);
    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualArray<std::complex<double>>);

    bp::def("operatorPrePlusPlus", &NdArrayInterface::operatorPrePlusPlus<double>);
    bp::def("operatorPostPlusPlus", &NdArrayInterface::operatorPostPlusPlus<double>);

    bp::def("operatorPreMinusMinus", &NdArrayInterface::operatorPreMinusMinus<double>);
    bp::def("operatorPostMinusMinus", &NdArrayInterface::operatorPostMinusMinus<double>);

    bp::def("operatorModulusScaler", &NdArrayInterface::operatorModulusScaler<uint32>);
    bp::def("operatorModulusScaler", &NdArrayInterface::operatorModulusScalerReversed<uint32>);
    bp::def("operatorModulusArray", &NdArrayInterface::operatorModulusArray<uint32>);

    bp::def("operatorBitwiseOrScaler", &NdArrayInterface::operatorBitwiseOrScaler<uint32>);
    bp::def("operatorBitwiseOrScaler", &NdArrayInterface::operatorBitwiseOrScalerReversed<uint32>);
    bp::def("operatorBitwiseOrArray", &NdArrayInterface::operatorBitwiseOrArray<uint32>);

    bp::def("operatorBitwiseAndScaler", &NdArrayInterface::operatorBitwiseAndScaler<uint32>);
    bp::def("operatorBitwiseAndScaler", &NdArrayInterface::operatorBitwiseAndScalerReversed<uint32>);
    bp::def("operatorBitwiseAndArray", &NdArrayInterface::operatorBitwiseAndArray<uint32>);

    bp::def("operatorBitwiseXorScaler", &NdArrayInterface::operatorBitwiseXorScaler<uint32>);
    bp::def("operatorBitwiseXorScaler", &NdArrayInterface::operatorBitwiseXorScalerReversed<uint32>);
    bp::def("operatorBitwiseXorArray", &NdArrayInterface::operatorBitwiseXorArray<uint32>);

    bp::def("operatorBitwiseNot", &NdArrayInterface::operatorBitwiseNot<uint32>);

    bp::def("operatorLogicalAndArray", &NdArrayInterface::operatorLogicalAndArray<uint32>);
    bp::def("operatorLogicalAndScalar", &NdArrayInterface::operatorLogicalAndScalar<uint32>);
    bp::def("operatorLogicalAndScalar", &NdArrayInterface::operatorLogicalAndScalarReversed<uint32>);

    bp::def("operatorLogicalOrArray", &NdArrayInterface::operatorLogicalOrArray<uint32>);
    bp::def("operatorLogicalOrScalar", &NdArrayInterface::operatorLogicalOrScalar<uint32>);
    bp::def("operatorLogicalOrScalar", &NdArrayInterface::operatorLogicalOrScalarReversed<uint32>);

    bp::def("operatorNot", &NdArrayInterface::operatorNot<uint32>);

    bp::def("operatorBitshiftLeft", &NdArrayInterface::operatorBitshiftLeft<uint32>);
    bp::def("operatorBitshiftRight", &NdArrayInterface::operatorBitshiftRight<uint32>);

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
        .def("setArray", &NdArrayInterface::setArray<uint32>)
        .def("byteswap", &NdArrayUInt32::byteswap, bp::return_internal_reference<>())
        .def("newbyteorder", &NdArrayInterface::newbyteorder<uint32>);

    typedef NdArray<uint64> NdArrayUInt64;
    bp::class_<NdArrayUInt64>
        ("NdArrayUInt64", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayUInt64::item)
        .def("shape", &NdArrayUInt64::shape)
        .def("size", &NdArrayUInt64::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint64>)
        .def("endianess", &NdArrayUInt64::endianess)
        .def("setArray", &NdArrayInterface::setArray<uint64>);

    typedef NdArray<uint16> NdArrayUInt16;
    bp::class_<NdArrayUInt16>
        ("NdArrayUInt16", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayUInt16::item)
        .def("shape", &NdArrayUInt16::shape)
        .def("size", &NdArrayUInt16::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint16>)
        .def("endianess", &NdArrayUInt16::endianess)
        .def("setArray", NdArrayInterface::setArray<uint16>);

    typedef NdArray<uint8> NdArrayUInt8;
    bp::class_<NdArrayUInt8>
        ("NdArrayUInt8", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayUInt8::item)
        .def("shape", &NdArrayUInt8::shape)
        .def("size", &NdArrayUInt8::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint8>)
        .def("endianess", &NdArrayUInt8::endianess)
        .def("setArray", NdArrayInterface::setArray<uint8>);

    typedef NdArray<int64> NdArrayInt64;
    bp::class_<NdArrayInt64>
        ("NdArrayInt64", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayInt64::item)
        .def("shape", &NdArrayInt64::shape)
        .def("size", &NdArrayInt64::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int64>)
        .def("endianess", &NdArrayInt64::endianess)
        .def("replace", &NdArrayInterface::replace<int64>)
        .def("setArray", &NdArrayInterface::setArray<int64>);

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

    typedef NdArray<int16> NdArrayInt16;
    bp::class_<NdArrayInt16>
        ("NdArrayInt16", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayInt16::item)
        .def("shape", &NdArrayInt16::shape)
        .def("size", &NdArrayInt16::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int16>)
        .def("endianess", &NdArrayInt16::endianess)
        .def("replace", &NdArrayInterface::replace<int16>)
        .def("setArray", &NdArrayInterface::setArray<int16>);

    typedef NdArray<int8> NdArrayInt8;
    bp::class_<NdArrayInt8>
        ("NdArrayInt8", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayInt8::item)
        .def("shape", &NdArrayInt8::shape)
        .def("size", &NdArrayInt8::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int8>)
        .def("endianess", &NdArrayInt8::endianess)
        .def("replace", &NdArrayInterface::replace<int8>)
        .def("setArray", &NdArrayInterface::setArray<int8>);

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

    typedef NdArray<std::complex<long double>> NdArrayComplexLongDouble;
    bp::class_<NdArrayComplexLongDouble>
        ("NdArrayComplexLongDouble", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayComplexLongDouble::item)
        .def("shape", &NdArrayComplexLongDouble::shape)
        .def("size", &NdArrayComplexLongDouble::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<std::complex<long double>>)
        .def("setArray", NdArrayInterface::setArray<std::complex<long double>>);

    typedef NdArray<std::complex<float>> NdArrayComplexFloat;
    bp::class_<NdArrayComplexFloat>
        ("NdArrayComplexFloat", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayComplexFloat::item)
        .def("shape", &NdArrayComplexFloat::shape)
        .def("size", &NdArrayComplexFloat::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<std::complex<float>>)
        .def("setArray", NdArrayInterface::setArray<std::complex<float>>);

    typedef NdArray<std::complex<int32>> NdArrayComplexInt32;
    bp::class_<NdArrayComplexInt32>
        ("NdArrayComplexInt32", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayComplexInt32::item)
        .def("shape", &NdArrayComplexInt32::shape)
        .def("size", &NdArrayComplexInt32::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<std::complex<int32>>)
        .def("setArray", NdArrayInterface::setArray<std::complex<int32>>);

    typedef std::complex<double> ComplexDouble;
    typedef NdArray<ComplexDouble> NdArrayComplexDouble;
    bp::class_<NdArrayComplexDouble>
        ("NdArrayComplexDouble", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def(bp::init<NdArrayComplexDouble>())
        .def("test1DListContructor", &NdArrayInterface::test1DListContructor<ComplexDouble>).staticmethod("test1DListContructor")
        .def("test2DListContructor", &NdArrayInterface::test2DListContructor<ComplexDouble>).staticmethod("test2DListContructor")
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<ComplexDouble>)
        .def("setArray", &NdArrayInterface::setArray<ComplexDouble>)
        .def("rSlice", &NdArrayComplexDouble::rSlice)
        .def("cSlice", &NdArrayComplexDouble::cSlice)
        .def("all", &NdArrayInterface::all<ComplexDouble>)
        .def("any", &NdArrayInterface::any<ComplexDouble>)
        .def("argmax", &NdArrayInterface::argmax<ComplexDouble>)
        .def("argmin", &NdArrayInterface::argmin<ComplexDouble>)
        .def("argsort", &NdArrayInterface::argsort<ComplexDouble>)
        .def("back", &NdArrayInterface::back<ComplexDouble>)
        .def("backReference", &NdArrayInterface::backReference<ComplexDouble>)
        .def("clip", &NdArrayInterface::clip<ComplexDouble>)
        .def("column", &NdArrayComplexDouble::column)
        .def("copy", &NdArrayInterface::copy<ComplexDouble>)
        .def("contains", &NdArrayInterface::contains<ComplexDouble>)
        .def("cumprod", &NdArrayInterface::cumprod<ComplexDouble>)
        .def("cumsum", &NdArrayInterface::cumsum<ComplexDouble>)
        .def("diagonal", &NdArrayInterface::diagonal<ComplexDouble>)
        .def("dot", &NdArrayInterface::dot<ComplexDouble>)
        .def("dump", &NdArrayComplexDouble::dump)
        .def("fill", &NdArrayInterface::fill<ComplexDouble>)
        .def("flatnonzero", &NdArrayInterface::flatnonzero<ComplexDouble>)
        .def("flatten", &NdArrayInterface::flatten<ComplexDouble>)
        .def("front", &NdArrayInterface::front<ComplexDouble>)
        .def("frontReference", &NdArrayInterface::frontReference<ComplexDouble>)
        .def("get", &NdArrayInterface::getValueFlat<ComplexDouble>)
        .def("get", &NdArrayInterface::getValueRowCol<ComplexDouble>)
        .def("get", &NdArrayInterface::getSlice1D<ComplexDouble>)
        .def("get", &NdArrayInterface::getSlice2D<ComplexDouble>)
        .def("get", &NdArrayInterface::getSlice2DRow<ComplexDouble>)
        .def("get", &NdArrayInterface::getSlice2DCol<ComplexDouble>)
        .def("getByIndices", &NdArrayInterface::getByIndices<ComplexDouble>)
        .def("getByMask", &NdArrayInterface::getByMask<ComplexDouble>)
        .def("isempty", &NdArrayComplexDouble::isempty)
        .def("isflat", &NdArrayComplexDouble::isflat)
        .def("issorted", &NdArrayInterface::issorted<ComplexDouble>)
        .def("issquare", &NdArrayComplexDouble::issquare)
        .def("item", &NdArrayComplexDouble::item)
        .def("max", &NdArrayInterface::max<ComplexDouble>)
        .def("min", &NdArrayInterface::min<ComplexDouble>)
        .def("median", &NdArrayInterface::median<ComplexDouble>)
        .def("nbytes", &NdArrayComplexDouble::nbytes)
        .def("none", &NdArrayInterface::none<ComplexDouble>)
        .def("nonzero", &NdArrayInterface::nonzero<ComplexDouble>)
        .def("numRows", &NdArrayComplexDouble::numCols)
        .def("numCols", &NdArrayComplexDouble::numRows)
        .def("ones", &NdArrayInterface::ones<ComplexDouble>)
        .def("partition", &NdArrayInterface::partition<ComplexDouble>)
        .def("print", &NdArrayComplexDouble::print)
        .def("prod", &NdArrayInterface::prod<ComplexDouble>)
        .def("ptp", &NdArrayInterface::ptp<ComplexDouble>)
        .def("put", &NdArrayInterface::putFlat<ComplexDouble>)
        .def("put", &NdArrayInterface::putRowCol<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice1DValue<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice1DValues<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice2DValue<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice2DValueRow<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice2DValueCol<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice2DValues<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice2DValuesRow<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice2DValuesCol<ComplexDouble>)
        .def("putMask", &NdArrayInterface::putMaskSingle<ComplexDouble>)
        .def("putMask", &NdArrayInterface::putMaskMultiple<ComplexDouble>)
        .def("ravel", &NdArrayInterface::ravel<ComplexDouble>)
        .def("repeat", &NdArrayInterface::repeat<ComplexDouble>)
        .def("reshape", &NdArrayInterface::reshapeInt<ComplexDouble>)
        .def("reshape", &NdArrayInterface::reshapeValues<ComplexDouble>)
        .def("reshape", &NdArrayInterface::reshapeShape<ComplexDouble>)
        .def("reshapeList", &NdArrayInterface::reshapeList<ComplexDouble>)
        .def("replace", &NdArrayInterface::replace<ComplexDouble>)
        .def("resizeFast", &NdArrayInterface::resizeFast<ComplexDouble>)
        .def("resizeFastList", &NdArrayInterface::resizeFastList<ComplexDouble>)
        .def("resizeSlow", &NdArrayInterface::resizeSlow<ComplexDouble>)
        .def("resizeSlowList", &NdArrayInterface::resizeSlowList<ComplexDouble>)
        .def("row", &NdArrayComplexDouble::row)
        .def("shape", &NdArrayComplexDouble::shape)
        .def("size", &NdArrayComplexDouble::size)
        .def("sort", &NdArrayInterface::sort<ComplexDouble>)
        .def("sum", &NdArrayInterface::sum<ComplexDouble>)
        .def("swapaxes", &NdArrayInterface::swapaxes<ComplexDouble>)
        .def("tofile", &NdArrayComplexDouble::tofile)
        .def("toStlVector", &NdArrayComplexDouble::toStlVector)
        .def("trace", &NdArrayComplexDouble::trace)
        .def("transpose", &NdArrayInterface::transpose<ComplexDouble>)
        .def("zeros", &NdArrayComplexDouble::zeros, bp::return_internal_reference<>());

    // Functions.hpp
    bp::def("absScaler", &FunctionsInterface::absScaler<double>);
    bp::def("absArray", &FunctionsInterface::absArray<double>);
    bp::def("absScaler", &FunctionsInterface::absScaler<std::complex<double>>);
    bp::def("absArray", &FunctionsInterface::absArray<std::complex<double>>);
    bp::def("add", &FunctionsInterface::addArrays<double>);
    bp::def("alen", &alen<double>);
    bp::def("all", &FunctionsInterface::allArray<double>);
    bp::def("allclose", &allclose<double>);
    bp::def("amin", &FunctionsInterface::aminArray<double>);
    bp::def("amax", &FunctionsInterface::amaxArray<double>);
    bp::def("angleScaler", &FunctionsInterface::angleScaler<double>);
    bp::def("angleArray", &FunctionsInterface::angleArray<double>);
    bp::def("any", &FunctionsInterface::anyArray<double>);
    bp::def("append", &append<double>);
    bp::def("applyPoly1d", &applyPoly1d<double>);
    bp::def("arange", &FunctionsInterface::arangeArray<double>);
    bp::def("arccosScaler", &FunctionsInterface::arccosScaler<double>);
    bp::def("arccosArray", &FunctionsInterface::arccosArray<double>);
    bp::def("arccosScaler", &FunctionsInterface::arccosScaler<std::complex<double>>);
    bp::def("arccosArray", &FunctionsInterface::arccosArray<std::complex<double>>);
    bp::def("arccoshScaler", &FunctionsInterface::arccoshScaler<double>);
    bp::def("arccoshArray", &FunctionsInterface::arccoshArray<double>);
    bp::def("arccoshScaler", &FunctionsInterface::arccoshScaler<std::complex<double>>);
    bp::def("arccoshArray", &FunctionsInterface::arccoshArray<std::complex<double>>);
    bp::def("arcsinScaler", &FunctionsInterface::arcsinScaler<double>);
    bp::def("arcsinArray", &FunctionsInterface::arcsinArray<double>);
    bp::def("arcsinScaler", &FunctionsInterface::arcsinScaler<std::complex<double>>);
    bp::def("arcsinArray", &FunctionsInterface::arcsinArray<std::complex<double>>);
    bp::def("arcsinhScaler", &FunctionsInterface::arcsinhScaler<double>);
    bp::def("arcsinhArray", &FunctionsInterface::arcsinhArray<double>);
    bp::def("arcsinhScaler", &FunctionsInterface::arcsinhScaler<std::complex<double>>);
    bp::def("arcsinhArray", &FunctionsInterface::arcsinhArray<std::complex<double>>);
    bp::def("arctanScaler", &FunctionsInterface::arctanScaler<double>);
    bp::def("arctanArray", &FunctionsInterface::arctanArray<double>);
    bp::def("arctanScaler", &FunctionsInterface::arctanScaler<std::complex<double>>);
    bp::def("arctanArray", &FunctionsInterface::arctanArray<std::complex<double>>);
    bp::def("arctan2Scaler", &FunctionsInterface::arctan2Scaler<double>);
    bp::def("arctan2Array", &FunctionsInterface::arctan2Array<double>);
    bp::def("arctanhScaler", &FunctionsInterface::arctanhScaler<double>);
    bp::def("arctanhArray", &FunctionsInterface::arctanhArray<double>);
    bp::def("arctanhScaler", &FunctionsInterface::arctanhScaler<std::complex<double>>);
    bp::def("arctanhArray", &FunctionsInterface::arctanhArray<std::complex<double>>);
    bp::def("argmax", &FunctionsInterface::argmaxArray<double>);
    bp::def("argmin", &FunctionsInterface::argminArray<double>);
    bp::def("argsort", &FunctionsInterface::argsortArray<double>);
    bp::def("argwhere", &FunctionsInterface::argwhere<double>);
    bp::def("aroundScaler", &FunctionsInterface::aroundScaler<double>);
    bp::def("aroundArray", &FunctionsInterface::aroundArray<double>);
    bp::def("array_equal", &array_equal<double>);
    bp::def("array_equiv", &array_equiv<double>);
    bp::def("asarrayVector", &FunctionsInterface::asarrayVector<double>);
    bp::def("asarrayList", &FunctionsInterface::asarrayList<double>);
    bp::def("astype", &astype<double, uint32>);
    bp::def("average", &FunctionsInterface::average<double>);
    bp::def("averageWeighted", &FunctionsInterface::averageWeighted<double>);

    bp::def("binaryRepr", &binaryRepr<int8>);
    bp::def("binaryRepr", &binaryRepr<int16>);
    bp::def("binaryRepr", &binaryRepr<int32>);
    bp::def("binaryRepr", &binaryRepr<int64>);
    bp::def("binaryRepr", &binaryRepr<uint8>);
    bp::def("binaryRepr", &binaryRepr<uint16>);
    bp::def("binaryRepr", &binaryRepr<uint32>);
    bp::def("binaryRepr", &binaryRepr<uint64>);
    bp::def("bincount", &FunctionsInterface::bincount<uint32>);
    bp::def("bincountWeighted", &FunctionsInterface::bincountWeighted<uint32>);
    bp::def("bitwise_and", &FunctionsInterface::bitwise_and<uint64>);
    bp::def("bitwise_not", &FunctionsInterface::bitwise_not<uint64>);
    bp::def("bitwise_or", &FunctionsInterface::bitwise_or<uint64>);
    bp::def("bitwise_xor", &FunctionsInterface::bitwise_xor<uint64>);
    bp::def("andOperatorArray", &FunctionsInterface::andOperatorArray<uint64>);
    bp::def("andOperatorScaler", &FunctionsInterface::andOperatorScaler<uint64>);
    bp::def("orOperatorArray", &FunctionsInterface::orOperatorArray<uint64>);
    bp::def("orOperatorScaler", &FunctionsInterface::orOperatorScaler<uint64>);
    bp::def("byteswap", &FunctionsInterface::byteswap<uint64>);

    bp::def("cbrtScaler", &FunctionsInterface::cbrtScaler<double>);
    bp::def("cbrtArray", &FunctionsInterface::cbrtArray<double>);
    bp::def("ceilScaler", &FunctionsInterface::ceilScaler<double>);
    bp::def("centerOfMass", &FunctionsInterface::centerOfMass<double>);
    bp::def("ceilArray", &FunctionsInterface::ceilArray<double>);
    bp::def("clipScaler", &FunctionsInterface::clipScaler<double>);
    bp::def("clipArray", &FunctionsInterface::clipArray<double>);
    bp::def("column_stack", &FunctionsInterface::column_stack<double>);
    bp::def("complexScaler", &FunctionsInterface::complexScalerSingle<double>);
    bp::def("complexScaler", &FunctionsInterface::complexScaler<double>);
    bp::def("complexArray", &FunctionsInterface::complexArraySingle<double>);
    bp::def("complexArray", &FunctionsInterface::complexArray<double>);
    bp::def("conjScaler", &FunctionsInterface::conjScaler<double>);
    bp::def("conjArray", &FunctionsInterface::conjArray<double>);
    bp::def("concatenate", &FunctionsInterface::concatenate<double>);
    bp::def("contains", &contains<double>);
    bp::def("copy", &FunctionsInterface::copy<double>);
    bp::def("copysign", &FunctionsInterface::copySign<double>);
    bp::def("copyto", &FunctionsInterface::copyto<double>);
    bp::def("cosScaler", &FunctionsInterface::cosScaler<double>);
    bp::def("cosArray", &FunctionsInterface::cosArray<double>);
    bp::def("cosScaler", &FunctionsInterface::cosScaler<std::complex<double>>);
    bp::def("cosArray", &FunctionsInterface::cosArray<std::complex<double>>);
    bp::def("coshScaler", &FunctionsInterface::coshScaler<double>);
    bp::def("coshArray", &FunctionsInterface::coshArray<double>);
    bp::def("coshScaler", &FunctionsInterface::coshScaler<std::complex<double>>);
    bp::def("coshArray", &FunctionsInterface::coshArray<std::complex<double>>);
    bp::def("count_nonzero", &FunctionsInterface::count_nonzero<double>);
    bp::def("cross", &cross<double>);
    bp::def("cube", &FunctionsInterface::cubeArray<double>);
    bp::def("cumprod", &FunctionsInterface::cumprodArray<double>);
    bp::def("cumsum", &FunctionsInterface::cumsumArray<double>);

    bp::def("deg2radScaler", &FunctionsInterface::deg2radScaler<double>);
    bp::def("deg2radArray", &FunctionsInterface::deg2radArray<double>);
    bp::def("degreesScaler", &FunctionsInterface::degreesScaler<double>);
    bp::def("degreesArray", &FunctionsInterface::degreesArray<double>);
    bp::def("deleteIndicesScaler", &FunctionsInterface::deleteIndicesScaler<double>);
    bp::def("deleteIndicesSlice", &FunctionsInterface::deleteIndicesSlice<double>);
    bp::def("diag", &FunctionsInterface::diag<double>);
    bp::def("diagflat", &FunctionsInterface::diagflat<double>);
    bp::def("diagonal", &FunctionsInterface::diagonal<double>);
    bp::def("diff", &FunctionsInterface::diff<double>);
    bp::def("divide", &FunctionsInterface::divide<double>);
    bp::def("dot", &FunctionsInterface::dot<double>);
    bp::def("dump", &dump<double>);

    bp::def("emptyRowCol", &FunctionsInterface::emptyRowCol<double>);
    bp::def("emptyShape", &FunctionsInterface::emptyShape<double>);
    bp::def("empty_like", &empty_like<double>);
    bp::def("endianess", &endianess<double>);
    bp::def("equal", &FunctionsInterface::equal<double>);
    bp::def("expScaler", &FunctionsInterface::expScaler<double>);
    bp::def("expArray", &FunctionsInterface::expArray<double>);
    bp::def("expScaler", &FunctionsInterface::expScaler<std::complex<double>>);
    bp::def("expArray", &FunctionsInterface::expArray<std::complex<double>>);
    bp::def("exp2Scaler", &FunctionsInterface::exp2Scaler<double>);
    bp::def("exp2Array", &FunctionsInterface::exp2Array<double>);
    bp::def("expm1Scaler", &FunctionsInterface::expm1Scaler<double>);
    bp::def("expm1Array", &FunctionsInterface::expm1Array<double>);
    bp::def("eye1D", &FunctionsInterface::eye1D<double>);
    bp::def("eye2D", &FunctionsInterface::eye2D<double>);
    bp::def("eyeShape", &FunctionsInterface::eyeShape<double>);

    bp::def("fillDiagonal", &fillDiagonal<double>);
    bp::def("find", &FunctionsInterface::find);
    bp::def("findN", &FunctionsInterface::findN);
    bp::def("fixScaler", &FunctionsInterface::fixScaler<double>);
    bp::def("fixArray", &FunctionsInterface::fixArray<double>);
    bp::def("flatten", &flatten<double>);
    bp::def("flatnonzero", &flatnonzero<double>);
    bp::def("flip", &flip<double>);
    bp::def("fliplr", &fliplr<double>);
    bp::def("flipud", &flipud<double>);
    bp::def("floorScaler", &FunctionsInterface::floorScaler<double>);
    bp::def("floorArray", &FunctionsInterface::floorArray<double>);
    bp::def("floor_divideScaler", &FunctionsInterface::floor_divideScaler<double>);
    bp::def("floor_divideArray", &FunctionsInterface::floor_divideArray<double>);
    bp::def("fmaxScaler", &FunctionsInterface::fmaxScaler<double>);
    bp::def("fmaxArray", &FunctionsInterface::fmaxArray<double>);
    bp::def("fminScaler", &FunctionsInterface::fminScaler<double>);
    bp::def("fminArray", &FunctionsInterface::fminArray<double>);
    bp::def("fmodScaler", &FunctionsInterface::fmodScaler<uint32>);
    bp::def("fmodArray", &FunctionsInterface::fmodArray<uint32>);
    bp::def("frombuffer", &frombuffer<double>);
    bp::def("fromfile", &fromfile<double>);
    bp::def("fullSquare", &FunctionsInterface::fullSquare<double>);
    bp::def("fullRowCol", &FunctionsInterface::fullRowCol<double>);
    bp::def("fullShape", &FunctionsInterface::fullShape<double>);
    bp::def("full_like", &full_like<double>);

    bp::def("gcdScaler", &FunctionsInterface::gcdScaler<uint32>);
    bp::def("gcdArray", &FunctionsInterface::gcdArray<uint32>);
    bp::def("greater", &greater<double>);
    bp::def("greater_equal", &greater_equal<double>);
    bp::def("gradient", &gradient<double>);

    bp::def("histogram", &FunctionsInterface::histogram<double>);
    bp::def("hstack", &FunctionsInterface::hstack<double>);
    bp::def("hypotScaler", &FunctionsInterface::hypotScaler<double>);
    bp::def("hypotScalerTriple", &FunctionsInterface::hypotScalerTriple<double>);
    bp::def("hypotArray", &FunctionsInterface::hypotArray<double>);

    bp::def("identity", &identity<double>);
    bp::def("imagScaler", &FunctionsInterface::imagScaler<double>);
    bp::def("imagArray", &FunctionsInterface::imagArray<double>);
    bp::def("interp", &FunctionsInterface::interp<double>);
    bp::def("intersect1d", &intersect1d<uint32>);
    bp::def("invert", &invert<uint32>);
    bp::def("isclose", &isclose<double>);
    bp::def("isinfScaler", &FunctionsInterface::isinfScaler<double>);
    bp::def("isinfArray", &FunctionsInterface::isinfArray<double>);
    bp::def("isnanScaler", &FunctionsInterface::isnanScaler<double>);
    bp::def("isnanArray", &FunctionsInterface::isnanArray<double>);

    bp::def("lcmScaler", &FunctionsInterface::lcmScaler<uint32>);
    bp::def("lcmArray", &FunctionsInterface::lcmArray<uint32>);
    bp::def("ldexpScaler", &FunctionsInterface::ldexpScaler<double>);
    bp::def("ldexpArray", &FunctionsInterface::ldexpArray<double>);
    bp::def("left_shift", &left_shift<uint32>);
    bp::def("less", &less<double>);
    bp::def("less_equal", &less_equal<double>);
    bp::def("linspace", &linspace<double>);
    bp::def("load", &load<double>);
    bp::def("logScaler", &FunctionsInterface::logScaler<double>);
    bp::def("logArray", &FunctionsInterface::logArray<double>);
    bp::def("logScaler", &FunctionsInterface::logScaler<std::complex<double>>);
    bp::def("logArray", &FunctionsInterface::logArray<std::complex<double>>);
    bp::def("log10Scaler", &FunctionsInterface::log10Scaler<double>);
    bp::def("log10Array", &FunctionsInterface::log10Array<std::complex<double>>);
    bp::def("log10Scaler", &FunctionsInterface::log10Scaler<std::complex<double>>);
    bp::def("log10Array", &FunctionsInterface::log10Array<double>);
    bp::def("log1pScaler", &FunctionsInterface::log1pScaler<double>);
    bp::def("log1pArray", &FunctionsInterface::log1pArray<double>);
    bp::def("log2Scaler", &FunctionsInterface::log2Scaler<double>);
    bp::def("log2Array", &FunctionsInterface::log2Array<double>);
    bp::def("logical_and", &logical_and<double>);
    bp::def("logical_not", &logical_not<double>);
    bp::def("logical_or", &logical_or<double>);
    bp::def("logical_xor", &logical_xor<double>);

    bp::def("matmul", &matmul<double>);
    bp::def("max", &max<double>);
    bp::def("maximum", &maximum<double>);
    bp::def("meshgrid", &FunctionsInterface::meshgrid<double>);
    NdArray<double> (*meanDouble)(const NdArray<double>&, Axis) = &mean<double>; 
    bp::def("mean", meanDouble);
    NdArray<std::complex<double>> (*meanComplexDouble)(const NdArray<std::complex<double>>&, Axis) = &mean<double>; 
    bp::def("mean", meanComplexDouble);
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
    bp::def("nansSquare", &FunctionsInterface::nansSquare);
    bp::def("nansRowCol", &FunctionsInterface::nansRowCol);
    bp::def("nansShape", &FunctionsInterface::nansShape);
    bp::def("nansList", &FunctionsInterface::nansList);
    bp::def("nans_like", &nans_like<double>);
    bp::def("nanstdev", &nanstdev<double>);
    bp::def("nansum", &nansum<double>);
    bp::def("nanvar", &nanvar<double>);
    bp::def("nbytes", &nbytes<double>);
    bp::def("newbyteorderScaler", &FunctionsInterface::newbyteorderScaler<uint32>);
    bp::def("newbyteorderArray", &FunctionsInterface::newbyteorderArray<uint32>);
    bp::def("negative", &negative<double>);
    bp::def("none", &FunctionsInterface::noneArray<double>);
    bp::def("nonzero", &nonzero<double>);
    NdArray<double> (*normDouble)(const NdArray<double>&, Axis) = &norm<double>; 
    bp::def("norm", normDouble);
    NdArray<std::complex<double>> (*normComplexDouble)(const NdArray<std::complex<double>>&, Axis) = &norm<double>; 
    bp::def("norm", normComplexDouble);
    bp::def("not_equal", &not_equal<double>);

    bp::def("onesSquare", &FunctionsInterface::onesSquare<double>);
    bp::def("onesRowCol", &FunctionsInterface::onesRowCol<double>);
    bp::def("onesShape", &FunctionsInterface::onesShape<double>);
    bp::def("ones_like", &ones_like<double, double>);
    bp::def("outer", &FunctionsInterface::outer<double>);

    bp::def("pad", &pad<double>);
    bp::def("partition", &partition<double>);
    bp::def("percentile", &percentile<double>);
    bp::def("polarScaler", &FunctionsInterface::polarScaler<double>);
    bp::def("polarArray", &FunctionsInterface::polarArray<double>);
    bp::def("powerArrayScaler", &FunctionsInterface::powerArrayScaler<double>);
    bp::def("powerArrayArray", &FunctionsInterface::powerArrayArray<double>);
    bp::def("powerArrayScaler", &FunctionsInterface::powerArrayScaler<std::complex<double>>);
    bp::def("powerArrayArray", &FunctionsInterface::powerArrayArray<std::complex<double>>);
    bp::def("powerfArrayScaler", &FunctionsInterface::powerfArrayScaler<double>);
    bp::def("powerfArrayArray", &FunctionsInterface::powerfArrayArray<double>);
    bp::def("powerfArrayScaler", &FunctionsInterface::powerfArrayScaler<std::complex<double>>);
    bp::def("powerfArrayArray", &FunctionsInterface::powerfArrayArray<std::complex<double>>);
    bp::def("prod", &prod<double>);
    bp::def("projScaler", &FunctionsInterface::projScaler<double>);
    bp::def("projArray", &FunctionsInterface::projArray<double>);
    bp::def("ptp", &ptp<double>);
    bp::def("put", &put<double>, bp::return_internal_reference<>());
    bp::def("putmask", &FunctionsInterface::putmask<double>);
    bp::def("putmaskScaler", &FunctionsInterface::putmaskScaler<double>);

    bp::def("rad2degScaler", &FunctionsInterface::rad2degScaler<double>);
    bp::def("rad2degArray", &FunctionsInterface::rad2degArray<double>);
    bp::def("radiansScaler", &FunctionsInterface::radiansScaler<double>);
    bp::def("radiansArray", &FunctionsInterface::radiansArray<double>);
    bp::def("ravel", &FunctionsInterface::ravel<double>, bp::return_internal_reference<>());
    bp::def("reciprocal", &reciprocal<double>);
    bp::def("realScaler", &FunctionsInterface::realScaler<double>);
    bp::def("realArray", &FunctionsInterface::realArray<double>);
    bp::def("remainderScaler", &FunctionsInterface::remainderScaler<double>);
    bp::def("remainderArray", &FunctionsInterface::remainderArray<double>);
    bp::def("replace", &FunctionsInterface::replace<double>);
    bp::def("reshape", &FunctionsInterface::reshapeInt<double>, bp::return_internal_reference<>());
    bp::def("reshape", &FunctionsInterface::reshapeShape<double>, bp::return_internal_reference<>());
    bp::def("reshape", &FunctionsInterface::reshapeValues<double>, bp::return_internal_reference<>());
    bp::def("reshapeList", &FunctionsInterface::reshapeList<double>, bp::return_internal_reference<>());
    bp::def("resizeFast", &FunctionsInterface::resizeFast<double>, bp::return_internal_reference<>());
    bp::def("resizeFastList", &FunctionsInterface::resizeFastList<double>, bp::return_internal_reference<>());
    bp::def("resizeSlow", &FunctionsInterface::resizeSlow<double>, bp::return_internal_reference<>());
    bp::def("resizeSlowList", &FunctionsInterface::resizeSlowList<double>, bp::return_internal_reference<>());
    bp::def("right_shift", &right_shift<uint32>);
    bp::def("rintScaler", &FunctionsInterface::rintScaler<double>);
    bp::def("rintArray", &FunctionsInterface::rintArray<double>);
    NdArray<double> (*rmsDouble)(const NdArray<double>&, Axis) = &rms<double>; 
    bp::def("rms", rmsDouble);
    NdArray<std::complex<double>> (*rmsComplexDouble)(const NdArray<std::complex<double>>&, Axis) = &rms<double>; 
    bp::def("rms", rmsComplexDouble);
    bp::def("roll", &roll<double>);
    bp::def("rot90", &rot90<double>);
    bp::def("roundScaler", &FunctionsInterface::roundScaler<double>);
    bp::def("roundArray", &FunctionsInterface::roundArray<double>);
    bp::def("row_stack", &FunctionsInterface::row_stack<double>);

    bp::def("setdiff1d", &setdiff1d<uint32>);
    bp::def("signScaler", &FunctionsInterface::signScaler<double>);
    bp::def("signArray", &FunctionsInterface::signArray<double>);
    bp::def("signbitScaler", &FunctionsInterface::signbitScaler<double>);
    bp::def("signbitArray", &FunctionsInterface::signbitArray<double>);
    bp::def("sinScaler", &FunctionsInterface::sinScaler<double>);
    bp::def("sinArray", &FunctionsInterface::sinArray<double>);
    bp::def("sinScaler", &FunctionsInterface::sinScaler<std::complex<double>>);
    bp::def("sinArray", &FunctionsInterface::sinArray<std::complex<double>>);
    bp::def("sincScaler", &FunctionsInterface::sincScaler<double>);
    bp::def("sincArray", &FunctionsInterface::sincArray<double>);
    bp::def("sinhScaler", &FunctionsInterface::sinhScaler<double>);
    bp::def("sinhArray", &FunctionsInterface::sinhArray<double>);
    bp::def("sinhScaler", &FunctionsInterface::sinhScaler<std::complex<double>>);
    bp::def("sinhArray", &FunctionsInterface::sinhArray<std::complex<double>>);
    bp::def("size", &size<double>);
    bp::def("sort", &sort<double>);
    bp::def("sqrtScaler", &FunctionsInterface::sqrtScaler<double>);
    bp::def("sqrtArray", &FunctionsInterface::sqrtArray<double>);
    bp::def("sqrtScaler", &FunctionsInterface::sqrtScaler<std::complex<double>>);
    bp::def("sqrtArray", &FunctionsInterface::sqrtArray<std::complex<double>>);
    bp::def("squareScaler", &FunctionsInterface::squareScaler<double>);
    bp::def("squareArray", &FunctionsInterface::squareArray<double>);
    bp::def("stack", &FunctionsInterface::stack<double>);
    NdArray<double> (*stdevDouble)(const NdArray<double>&, Axis) = &stdev<double>; 
    bp::def("stdev", stdevDouble);
    NdArray<std::complex<double>> (*stdevComplexDouble)(const NdArray<std::complex<double>>&, Axis) = &stdev<double>; 
    bp::def("stdev", stdevComplexDouble);
    bp::def("subtract", &FunctionsInterface::subtractArrays<double>);
    bp::def("sum", &sum<double>);
    bp::def("swapaxes", &swapaxes<double>);
    bp::def("swap", &nc::swap<double>);

    bp::def("tanScaler", &FunctionsInterface::tanScaler<double>);
    bp::def("tanArray", &FunctionsInterface::tanArray<double>);
    bp::def("tanScaler", &FunctionsInterface::tanScaler<std::complex<double>>);
    bp::def("tanArray", &FunctionsInterface::tanArray<std::complex<double>>);
    bp::def("tanhScaler", &FunctionsInterface::tanhScaler<double>);
    bp::def("tanhArray", &FunctionsInterface::tanhArray<double>);
    bp::def("tanhScaler", &FunctionsInterface::tanhScaler<std::complex<double>>);
    bp::def("tanhArray", &FunctionsInterface::tanhArray<std::complex<double>>);
    bp::def("tileRectangle", &FunctionsInterface::tileRectangle<double>);
    bp::def("tileShape", &FunctionsInterface::tileShape<double>);
    bp::def("tileList", &FunctionsInterface::tileList<double>);
    bp::def("tofile", &tofile<double>);
    bp::def("toStlVector", &toStlVector<double>);
    bp::def("trace", &trace<double>);
    bp::def("transpose", &transpose<double>);
    bp::def("trapzDx", &FunctionsInterface::trapzDx<double>);
    bp::def("trapz", &FunctionsInterface::trapz<double>);
    bp::def("trilSquare", &FunctionsInterface::trilSquare<double>);
    bp::def("trilRect", &FunctionsInterface::trilRect<double>);
    bp::def("trilArray", &FunctionsInterface::trilArray<double>);
    bp::def("triuSquare", &FunctionsInterface::triuSquare<double>);
    bp::def("triuRect", &FunctionsInterface::triuRect<double>);
    bp::def("triuArray", &FunctionsInterface::triuArray<double>);
    bp::def("trim_zeros", &trim_zeros<double>);
    bp::def("truncScaler", &FunctionsInterface::truncScaler<double>);
    bp::def("truncArray", &FunctionsInterface::truncArray<double>);

    bp::def("union1d", &union1d<uint32>);
    bp::def("unique", &unique<double>);
    bp::def("unwrapScaler", &FunctionsInterface::unwrapScaler<double>);
    bp::def("unwrapArray", &FunctionsInterface::unwrapArray<double>);

    NdArray<double> (*varDouble)(const NdArray<double>&, Axis) = &var<double>;
    bp::def("var", varDouble);
    NdArray<std::complex<double>> (*varComplexDouble)(const NdArray<std::complex<double>>&, Axis) = &var<double>; 
    bp::def("var", varComplexDouble);
    bp::def("vstack", &FunctionsInterface::vstack<double>);

    bp::def("where", &FunctionsInterface::where<double>);

    bp::def("zerosSquare", &FunctionsInterface::zerosSquare<double>);
    bp::def("zerosRowCol", &FunctionsInterface::zerosRowCol<double>);
    bp::def("zerosShape", &FunctionsInterface::zerosShape<double>);
    bp::def("zerosList", &FunctionsInterface::zerosList<double>);
    bp::def("zeros_like", &zeros_like<double, double>);

    // Utils.hpp
    bp::def("num2str", &utils::num2str<double>);
    bp::def("sqr", &utils::sqr<double>);
    bp::def("cube", &utils::cube<double>);
    bp::def("power", &utils::power<double>);
    bp::def("power", &utils::power<std::complex<double>>);
    decltype(utils::powerf<double, double>(double{ 0 }, double{ 0 }))(*powerf_double)(double, double) = &utils::powerf<double, double>;
    bp::def("powerf", powerf_double);
    decltype(utils::powerf<std::complex<double>, std::complex<double>>(std::complex<double>{ 0 }, std::complex<double>{ 0 }))(*powerf_complexDouble)
        (std::complex<double>, std::complex<double>) = &utils::powerf<std::complex<double>, std::complex<double>>;
    bp::def("powerf_complex", powerf_complexDouble);

    bp::def("num2str", &utils::num2str<float>);
    bp::def("sqr", &utils::sqr<float>);
    bp::def("cube", &utils::cube<float>);
    bp::def("power", &utils::power<float>);
    bp::def("power", &utils::power<std::complex<float>>);
    decltype(utils::powerf<float, float>(float{ 0 }, float{ 0 }))(*powerf_float)(float, float) = &utils::powerf<float, float>;
    bp::def("powerf", powerf_float);
    decltype(utils::powerf<std::complex<float>, std::complex<float>>(std::complex<float>{ 0 }, std::complex<float>{ 0 }))(*powerf_complexFloat)
        (std::complex<float>, std::complex<float>) = &utils::powerf<std::complex<float>, std::complex<float>>;
    bp::def("powerf_complex", powerf_complexFloat);

    bp::def("num2str", &utils::num2str<int8>);
    bp::def("sqr", &utils::sqr<int8>);
    bp::def("cube", &utils::cube<int8>);
    bp::def("power", &utils::power<int8>);
    decltype(utils::powerf<int8, double>(int8{ 0 }, double{ 0 }))(*powerf_int8)(int8, double) = &utils::powerf<int8, double>;
    bp::def("powerf", powerf_int8);

    bp::def("num2str", &utils::num2str<int16>);
    bp::def("sqr", &utils::sqr<int16>);
    bp::def("cube", &utils::cube<int16>);
    bp::def("power", &utils::power<int16>);
    decltype(utils::powerf<int16, double>(int16{ 0 }, double{ 0 }))(*powerf_int16)(int16, double) = &utils::powerf<int16, double>;
    bp::def("powerf", powerf_int16);

    bp::def("num2str", &utils::num2str<int32>);
    bp::def("sqr", &utils::sqr<int32>);
    bp::def("cube", &utils::cube<int32>);
    bp::def("power", &utils::power<int32>);
    decltype(utils::powerf<int32, double>(int32{ 0 }, double{ 0 }))(*powerf_int32)(int32, double) = &utils::powerf<int32, double>;
    bp::def("powerf", powerf_int32);

    bp::def("num2str", &utils::num2str<int64>);
    bp::def("sqr", &utils::sqr<int64>);
    bp::def("cube", &utils::cube<int64>);
    bp::def("power", &utils::power<int64>);
    decltype(utils::powerf<int64, double>(int64{ 0 }, double{ 0 }))(*powerf_int64)(int64, double) = &utils::powerf<int64, double>;
    bp::def("powerf", powerf_int64);

    bp::def("num2str", &utils::num2str<uint8>);
    bp::def("sqr", &utils::sqr<uint8>);
    bp::def("cube", &utils::cube<uint8>);
    bp::def("power", &utils::power<uint8>);
    decltype(utils::powerf<uint8, double>(uint8{ 0 }, double{ 0 }))(*powerf_uint8)(uint8, double) = &utils::powerf<uint8, double>;
    bp::def("powerf", powerf_uint8);

    bp::def("num2str", &utils::num2str<uint16>);
    bp::def("sqr", &utils::sqr<uint16>);
    bp::def("cube", &utils::cube<uint16>);
    bp::def("power", &utils::power<uint16>);
    decltype(utils::powerf<uint16, double>(uint16{ 0 }, double{ 0 }))(*powerf_uint16)(uint16, double) = &utils::powerf<uint16, double>;
    bp::def("powerf", powerf_uint16);

    bp::def("num2str", &utils::num2str<uint32>);
    bp::def("sqr", &utils::sqr<uint32>);
    bp::def("cube", &utils::cube<uint32>);
    bp::def("power", &utils::power<uint32>);
    decltype(utils::powerf<uint32, double>(uint32{ 0 }, double{ 0 }))(*powerf_uint32)(uint32, double) = &utils::powerf<uint32, double>;
    bp::def("powerf", powerf_uint32);

    bp::def("num2str", &utils::num2str<uint64>);
    bp::def("sqr", &utils::sqr<uint64>);
    bp::def("cube", &utils::cube<uint64>);
    bp::def("power", &utils::power<uint64>);
    decltype(utils::powerf<uint64, double>(uint64{ 0 }, double{ 0 }))(*powerf_uint64)(uint64, double) = &utils::powerf<uint64, double>;
    bp::def("powerf", powerf_uint64);

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
    bp::def("cyclic_hankel_1_Scaler", &SpecialInterface::cyclic_hankel_1_Scaler<double>);
    bp::def("cyclic_hankel_1_Array", &SpecialInterface::cyclic_hankel_1_Array<double>);
    bp::def("cyclic_hankel_2_Scaler", &SpecialInterface::cyclic_hankel_2_Scaler<double>);
    bp::def("cyclic_hankel_2_Array", &SpecialInterface::cyclic_hankel_2_Array<double>);
    bp::def("spherical_hankel_1_Scaler", &SpecialInterface::spherical_hankel_1_Scaler<double>);
    bp::def("spherical_hankel_1_Array", &SpecialInterface::spherical_hankel_1_Array<double>);
    bp::def("spherical_hankel_2_Scaler", &SpecialInterface::spherical_hankel_2_Scaler<double>);
    bp::def("spherical_hankel_2_Array", &SpecialInterface::spherical_hankel_2_Array<double>);
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
    bp::def("trigamma_Scaler", &SpecialInterface::trigamma_Scaler<double>);
    bp::def("trigamma_Array", &SpecialInterface::trigamma_Array<double>);
}
