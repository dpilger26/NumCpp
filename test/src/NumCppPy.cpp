#include "NumCpp.hpp"
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <complex>
#include <deque>
#include <forward_list>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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
    bool testListContructor() 
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

namespace IteratorInterface
{
    template<typename Iterator>
    typename Iterator::value_type dereference(Iterator& self)
    {
        return *self;
    }

    //================================================================================

    template<typename Iterator>
    Iterator operatorPlusPlusPre(Iterator& self)
    {
        return ++self;
    }

    //================================================================================

    template<typename Iterator>
    Iterator operatorMinusMinusPre(Iterator& self)
    {
        return --self;
    }

    //================================================================================

    template<typename Iterator>
    Iterator operatorPlusEqual(Iterator& self, typename Iterator::difference_type offset)
    {
        return self += offset;
    }

    //================================================================================

    template<typename Iterator>
    Iterator operatorMinusEqual(Iterator& self, typename Iterator::difference_type offset)
    {
        return self -= offset;
    }

    //================================================================================

    template<typename Iterator>
    typename Iterator::value_type access(Iterator& self, typename Iterator::difference_type offset)
    {
        return self[offset];
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

    template<typename T>
    np::ndarray test1dArrayConstructor(T value1, T value2)
    {
        std::array<T, 2> arr = {value1, value2};
        auto newNcArray = NdArray<T>(arr);

        return nc2Boost(newNcArray);
    }

    //================================================================================

    template<typename T>
    np::ndarray test2dArrayConstructor(T value1, T value2)
    {
        std::array<std::array<T, 2>, 2> arr2d;
        arr2d[0][0] = value1;
        arr2d[0][1] = value2;
        arr2d[1][0] = value1;
        arr2d[1][1] = value2;
        auto newNcArray = NdArray<T>(arr2d);

        return nc2Boost(newNcArray);
    }

    //================================================================================

    template<typename T>
    np::ndarray test1dVectorConstructor(np::ndarray inArray)
    {
        const auto ncArray = boost2Nc<T>(inArray);

        std::vector<T> vec(ncArray.size());
        std::copy(ncArray.cbegin(), ncArray.cend(), vec.begin());

        auto newNcArray = NdArray<T>(vec);

        return nc2Boost(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    np::ndarray test2dVectorConstructor(np::ndarray inArray)
    {
        const auto ncArray = boost2Nc<T>(inArray);

        std::vector<std::vector<T>> vec2d(ncArray.numRows(), std::vector<T>(ncArray.numCols()));
        for (uint32 row = 0; row < ncArray.numRows(); ++row)
        {
            std::copy(ncArray.cbegin(row), ncArray.cend(row), vec2d[row].begin());
        }

        auto newNcArray = NdArray<T>(vec2d);

        return nc2Boost(newNcArray);
    }

    //================================================================================

    template<typename T>
    np::ndarray test2dVectorArrayConstructor(np::ndarray inArray)
    {
        const auto ncArray = boost2Nc<T>(inArray);
        if (ncArray.numCols() != 2)
        {
            throw std::invalid_argument("Input array must be [n, 2] shape.");
        }

        std::vector<std::array<T, 2>> vec2d(ncArray.numRows());
        for (uint32 row = 0; row < ncArray.numRows(); ++row)
        {
            std::copy(ncArray.cbegin(row), ncArray.cend(row), vec2d[row].begin());
        }

        auto newNcArray = NdArray<T>(vec2d);

        return nc2Boost(newNcArray);
    }

    //================================================================================

    template<typename T>
    np::ndarray test1dDequeConstructor(np::ndarray inArray)
    {
        const auto ncArray = boost2Nc<T>(inArray);

        std::deque<T> deq(ncArray.size());
        std::copy(ncArray.cbegin(), ncArray.cend(), deq.begin());

        auto newNcArray = NdArray<T>(deq);

        return nc2Boost(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    np::ndarray test2dDequeConstructor(np::ndarray inArray)
    {
        const auto ncArray = boost2Nc<T>(inArray);

        std::deque<std::deque<T>> deq2d(ncArray.numRows(), std::deque<T>(ncArray.numCols()));
        for (uint32 row = 0; row < ncArray.numRows(); ++row)
        {
            std::copy(ncArray.cbegin(row), ncArray.cend(row), deq2d[row].begin());
        }

        auto newNcArray = NdArray<T>(deq2d);

        return nc2Boost(newNcArray);
    }

    //================================================================================

    template<typename T>
    np::ndarray test1dListConstructor(np::ndarray inArray)
    {
        const auto ncArray = boost2Nc<T>(inArray);

        std::list<T> list(ncArray.size());
        std::copy(ncArray.cbegin(), ncArray.cend(), list.begin());

        auto newNcArray = NdArray<T>(list);

        return nc2Boost(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    np::ndarray test1dIteratorConstructor(np::ndarray inArray)
    {
        const auto ncArray = boost2Nc<T>(inArray);

        std::vector<T> vec(ncArray.size());
        std::copy(ncArray.cbegin(), ncArray.cend(), vec.begin());

        auto newNcArray = NdArray<T>(vec.cbegin(), vec.cend());

        return nc2Boost(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    np::ndarray test1dIteratorConstructor2(np::ndarray inArray)
    {
        const auto ncArray = boost2Nc<T>(inArray);
        auto newNcArray = NdArray<T>(ncArray.cbegin(), ncArray.cend());

        return nc2Boost(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    np::ndarray test1dPointerConstructor(np::ndarray inArray)
    {
        const auto ncArray = boost2Nc<T>(inArray);
        auto newNcArray = NdArray<T>(ncArray.data(), ncArray.size());

        return nc2Boost(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    np::ndarray test2dPointerConstructor(np::ndarray inArray)
    {
        const auto ncArray = boost2Nc<T>(inArray);
        auto newNcArray = NdArray<T>(ncArray.data(), ncArray.numRows(), ncArray.numCols());

        return nc2Boost(newNcArray);
    }

    //================================================================================

    template<typename T>
    np::ndarray test1dPointerShellConstructor(np::ndarray inArray)
    {
        auto ncArray = boost2Nc<T>(inArray);
        auto newNcArray = NdArray<T>(ncArray.dataRelease(), ncArray.size(), true);

        return nc2Boost(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    np::ndarray test2dPointerShellConstructor(np::ndarray inArray)
    {
        auto ncArray = boost2Nc<T>(inArray);
        auto newNcArray = NdArray<T>(ncArray.dataRelease(), ncArray.numRows(), ncArray.numCols(), true);

        return nc2Boost(newNcArray);
    }

    //================================================================================

    template<typename T>
    np::ndarray testCopyConstructor(np::ndarray inArray)
    {
        const auto ncArray = boost2Nc<T>(inArray);
        auto newNcArray = NdArray<T>(ncArray);

        return nc2Boost(newNcArray);
    }

    //================================================================================

    template<typename T>
    np::ndarray testMoveConstructor(np::ndarray inArray)
    {
        auto ncArray = boost2Nc<T>(inArray);
        auto newNcArray = NdArray<T>(std::move(ncArray));

        return nc2Boost(newNcArray);
    }

    //================================================================================

    template<typename T>
    np::ndarray testAssignementOperator(np::ndarray inArray)
    {
        auto ncArray = boost2Nc<T>(inArray);
        NdArray<T> newNcArray;
        newNcArray = ncArray;

        return nc2Boost(newNcArray);
    }

    //================================================================================

    template<typename T>
    np::ndarray testAssignementScalerOperator(np::ndarray inArray, T value)
    {
        auto ncArray = boost2Nc<T>(inArray);
        ncArray = value;

        return nc2Boost(ncArray);
    }

    //================================================================================

    template<typename T>
    np::ndarray testMoveAssignementOperator(np::ndarray inArray)
    {
        auto ncArray = boost2Nc<T>(inArray);
        NdArray<T> newNcArray;
        newNcArray = std::move(ncArray);

        return nc2Boost(newNcArray);
    }

    //================================================================================

    struct TestStruct
    {
        int member1{ 0 };
        int member2{ 0 };
        double member3{ 0.0 };
        bool member4{ true };
    };

    //================================================================================

    bool testStructuredArray()
    {
        NdArray<TestStruct> test1;
        NdArray<TestStruct> test2(5);
        NdArray<TestStruct> test3(5, 5);
        NdArray<TestStruct> test4(test3.shape());
        NdArray<TestStruct> test5_1(test2);
        NdArray<TestStruct> test5_2(std::move(test4));
        NdArray<TestStruct> test6 = { TestStruct{666, 357, 3.14519, true},
            TestStruct{666, 357, 3.14519, true} };
        NdArray<TestStruct> test7 = { {TestStruct{666, 357, 3.14519, true}, TestStruct{667, 377, 3.7519, false}},
            {TestStruct{665, 357, 3.15519, false}, TestStruct{69, 359, 3.19519, true}} };

        auto testStruct = TestStruct{ 666, 357, 3.14519, true };
        test6 = testStruct;
        test1.resizeFast({10, 10});
        test1 = std::move(testStruct);

        test7.begin();
        test5_1.begin(0);
        test5_2.end();
        test2.end(0);

        test2.resizeFast({10, 10});
        test2 = TestStruct{ 666, 357, 3.14519, true };
        test2.rSlice();
        test2.cSlice();
        test2.back();
        test2.column(0);
        test2.copy();
        test2.data();
        test2.diagonal();
        test2.dump("test.bin");
        remove("test.bin");
        test2.fill(TestStruct{0, 1, 6.5, false});
        test2.flatten();
        test2.front();
        test2[0];
        test2(0, 0);
        test2[0] = TestStruct{0, 1, 6.5, false};
        test2(0, 0) = TestStruct{0, 1, 6.5, false};
        test2.isempty();
        test2.isflat();
        test2.issquare();
        test2.nbytes();
        test2.numRows();
        test2.numCols();
        test2.put(0, TestStruct{ 0, 1, 6.5, false });
        test2.ravel();
        test2.repeat({2, 2});
        test2.reshape(test2.size(), 1);
        test2.resizeFast(1, 1);
        test2.resizeSlow(10, 10);
        test2.row(0);
        test2.shape();
        test2.size();
        test2.swapaxes();
        test2.transpose();

        return true;
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
    dtype backRow(const NdArray<dtype>& self, typename NdArray<dtype>::size_type row)
    {
        return self.back(row);
    }

    //================================================================================

    template<typename dtype>
    dtype backRowReference(NdArray<dtype>& self, typename NdArray<dtype>::size_type row)
    {
        return self.back(row);
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
    dtype frontRow(const NdArray<dtype>& self, typename NdArray<dtype>::size_type row)
    {
        return self.front(row);
    }

    //================================================================================

    template<typename dtype>
    dtype frontRowReference(NdArray<dtype>& self, typename NdArray<dtype>::size_type row)
    {
        return self.front(row);
    }

    //================================================================================

    template<typename dtype>
    dtype getValueFlat(NdArray<dtype>& self, int32 inIndex)
    {
        return self[inIndex];
    }

    //================================================================================

    template<typename dtype>
    dtype getValueFlatConst(const NdArray<dtype>& self, int32 inIndex)
    {
        return self[inIndex];
    }

    //================================================================================

    template<typename dtype>
    dtype getValueRowCol(NdArray<dtype>& self, int32 inRow, int32 inCol)
    {
        return self(inRow, inCol);
    }

    //================================================================================

    template<typename dtype>
    dtype getValueRowColConst(const NdArray<dtype>& self, int32 inRow, int32 inCol)
    {
        return self(inRow, inCol);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray getMask(const NdArray<dtype>& self, const NdArray<bool>& mask)
    {
        return nc2Boost(self[mask]);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray getIndices(const NdArray<dtype>& self, const NdArray<typename NdArray<dtype>::size_type>& indices)
    {
        return nc2Boost(self[indices]);
    }

    //================================================================================


    template<typename dtype>
    np::ndarray getSlice1D(const NdArray<dtype>& self, const Slice& inSlice)
    {
        return nc2Boost(self[inSlice]);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray getSlice2D(const NdArray<dtype>& self, const Slice& inRowSlice, const Slice& inColSlice)
    {
        return nc2Boost(self(inRowSlice, inColSlice));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray getSlice2DCol(const NdArray<dtype>& self, const Slice& inRowSlice, int32 inColIndex)
    {
        return nc2Boost(self(inRowSlice, inColIndex));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray getSlice2DRow(const NdArray<dtype>& self, int32 inRowIndex, const Slice& inColSlice)
    {
        return nc2Boost(self(inRowIndex, inColSlice));
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
    auto absScaler(dtype inValue)  -> decltype(abs(inValue)) // trailing return type to help gcc
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

    template<typename Type1, typename Type2>
    np::ndarray add(const Type1& in1, const Type2& in2)
    {
        return nc2Boost(nc::add(in1, in2));
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
    auto arccosScaler(dtype inValue)  -> decltype(arccos(inValue)) // trailing return type to help gcc
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
    auto arccoshScaler(dtype inValue)  -> decltype(arccosh(inValue)) // trailing return type to help gcc
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
    auto arcsinScaler(dtype inValue)  -> decltype(arcsin(inValue)) // trailing return type to help gcc
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
    auto arcsinhScaler(dtype inValue)  -> decltype(arcsinh(inValue)) // trailing return type to help gcc
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
    auto arctanScaler(dtype inValue)  -> decltype(arctan(inValue)) // trailing return type to help gcc
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
    dtype arctan2Scaler(dtype inY, dtype inX) 
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
    auto arctanhScaler(dtype inValue)  -> decltype(arctanh(inValue)) // trailing return type to help gcc
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
    np::ndarray aroundArray(NdArray<dtype>& inArray, uint8 inNumDecimals)
    {
        return nc2Boost(around(inArray, inNumDecimals));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayInitializerList(dtype inValue1, dtype inValue2)
    {
        auto a = asarray({ inValue1, inValue2 });
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayInitializerList2D(dtype inValue1, dtype inValue2)
    {
        auto a = asarray({ {inValue1, inValue2}, {inValue1, inValue2} });
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayArray1D(dtype inValue1, dtype inValue2)
    {
        std::array<dtype, 2> arr = { inValue1, inValue2 };
        auto a = asarray(arr, false);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayArray1DCopy(dtype inValue1, dtype inValue2)
    {
        std::array<dtype, 2> arr = { inValue1, inValue2 };
        auto a = asarray(arr, true);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayArray2D(dtype inValue1, dtype inValue2)
    {
        std::array<std::array<dtype, 2>, 2> arr;
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr, false);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayArray2DCopy(dtype inValue1, dtype inValue2)
    {
        std::array<std::array<dtype, 2>, 2> arr;
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr, true);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayVector1D(dtype inValue1, dtype inValue2)
    {
        std::vector<dtype> arr = { inValue1, inValue2 };
        auto a = asarray(arr, false);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayVector1DCopy(dtype inValue1, dtype inValue2)
    {
        std::vector<dtype> arr = { inValue1, inValue2 };
        auto a = asarray(arr, true);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayVector2D(dtype inValue1, dtype inValue2)
    {
        std::vector<std::vector<dtype>> arr(2, std::vector<dtype>(2));
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayVectorArray2D(dtype inValue1, dtype inValue2)
    {
        std::vector<std::array<dtype, 2>> arr(2);
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr, false);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayVectorArray2DCopy(dtype inValue1, dtype inValue2)
    {
        std::vector<std::array<dtype, 2>> arr(2);
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr, true);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayDeque1D(dtype inValue1, dtype inValue2)
    {
        std::deque<dtype> arr = { inValue1, inValue2 };
        auto a = asarray(arr);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayDeque2D(dtype inValue1, dtype inValue2)
    {
        std::deque<std::deque<dtype>> arr(2, std::deque<dtype>(2));
        for (auto& row : arr)
        {
            row[0] = inValue1;
            row[1] = inValue2;
        }
        auto a = asarray(arr);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayList(dtype inValue1, dtype inValue2)
    {
        std::list<dtype> arr = { inValue1, inValue2 };
        auto a = asarray(arr);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayIterators(dtype inValue1, dtype inValue2)
    {
        std::vector<dtype> arr = { inValue1, inValue2 };
        auto a = asarray(arr.begin(), arr.end());
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayPointerIterators(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        auto a = asarray(ptr.get(), ptr.get() + 2);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayPointer(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        auto a = asarray(ptr.release(), uint32{ 2 });
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayPointer2D(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(4);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        ptr[2] = inValue1;
        ptr[3] = inValue2;
        auto a = asarray(ptr.release(), uint32{ 2 }, uint32{ 2 });
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayPointerShell(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        auto a = asarray(ptr.get(), uint32{ 2 }, false);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayPointerShell2D(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(4);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        ptr[2] = inValue1;
        ptr[3] = inValue2;
        auto a = asarray(ptr.get(), uint32{ 2 }, uint32{ 2 }, false);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayPointerShellTakeOwnership(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        auto a = asarray(ptr.release(), 2, true);
        return nc2Boost<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray asarrayPointerShell2DTakeOwnership(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(4);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        ptr[2] = inValue1;
        ptr[3] = inValue2;
        auto a = asarray(ptr.release(), 2, 2, true);
        return nc2Boost<dtype>(a);
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
    np::ndarray averageWeightedComplex(const NdArray<std::complex<dtype>>& inArray, 
        const NdArray<dtype>& inWeights, Axis inAxis = Axis::NONE)
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
    dtype cbrtScaler(dtype inValue) 
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
    dtype ceilScaler(dtype inValue) 
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
    np::ndarray centerOfMass(const NdArray<dtype>& inArray, const Axis inAxis = Axis::NONE) 
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
    auto cosScaler(dtype inValue)  -> decltype(cos(inValue)) // trailing return type to help gcc
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
    auto coshScaler(dtype inValue)  -> decltype(cosh(inValue)) // trailing return type to help gcc
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
    dtype deg2radScaler(dtype inValue) 
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
    dtype degreesScaler(dtype inValue) 
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

    template<typename Type1, typename Type2>
    np::ndarray divide(const Type1& in1, const Type2& in2)
    {
        return nc2Boost(nc::divide(in1, in2));
    }

    //================================================================================

    template<typename dtype1, typename dtype2>
    np::ndarray dot(const NdArray<dtype1>& inArray1, const NdArray<dtype2>& inArray2)
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
    auto expScaler(dtype inValue)  -> decltype(exp(inValue)) // trailing return type to help gcc
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
    dtype exp2Scaler(dtype inValue) 
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
    dtype expm1Scaler(dtype inValue) 
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

    np::ndarray find(const NdArray<bool>& inArray) 
    {
        return nc2Boost(nc::find(inArray));
    }

    //================================================================================

    np::ndarray findN(const NdArray<bool>& inArray, uint32 n) 
    {
        return nc2Boost(nc::find(inArray, n));
    }

    //================================================================================

    template<typename dtype>
    dtype fixScaler(dtype inValue) 
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
    dtype floorScaler(dtype inValue) 
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
    dtype floor_divideScaler(dtype inValue1, dtype inValue2) 
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
    dtype fmaxScaler(dtype inValue1, dtype inValue2) 
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
    dtype fminScaler(dtype inValue1, dtype inValue2) 
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
    dtype fmodScaler(dtype inValue1, dtype inValue2) 
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
    np::ndarray frombuffer(const NdArray<dtype>& inArray)
    {
        auto buffer = reinterpret_cast<const char*>(inArray.data());
        return nc2Boost(nc::frombuffer<dtype>(buffer, static_cast<uint32>(inArray.nbytes())));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray fromiter(const NdArray<dtype>& inArray)
    {
        std::vector<dtype> vec(inArray.begin(), inArray.end());
        return nc2Boost(nc::fromiter<dtype>(vec.begin(), vec.end()));
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
    dtype gcdScaler(dtype inValue1, dtype inValue2) 
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
    np::ndarray gradient(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(nc::gradient(inArray, inAxis));
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
    dtype hypotScaler(dtype inValue1, dtype inValue2) 
    {
        return hypot(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    dtype hypotScalerTriple(dtype inValue1, dtype inValue2, dtype inValue3) 
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
    bool isinfScaler(dtype inValue) 
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
    bool isnanScaler(dtype inValue) 
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
    dtype ldexpScaler(dtype inValue1, uint8 inValue2) 
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
    dtype lcmScaler(dtype inValue1, dtype inValue2) 
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
    auto logScaler(dtype inValue)  -> decltype(log(inValue)) // trailing return type to help gcc
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
    auto log10Scaler(dtype inValue)  -> decltype(log10(inValue)) // trailing return type to help gcc
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
    dtype log1pScaler(dtype inValue) 
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
    dtype log2Scaler(dtype inValue) 
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

    template<typename dtype1, typename dtype2>
    np::ndarray matmul(const NdArray<dtype1>& inArray1, const NdArray<dtype2>& inArray2)
    {
        return nc2Boost(nc::matmul(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray max(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(nc::max(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray maximum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(nc::maximum(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    std::pair<NdArray<dtype>, NdArray<dtype> > meshgrid(const Slice& inISlice, const Slice& inJSlice)
    {
        return nc::meshgrid<dtype>(inISlice, inJSlice);
    }

    //================================================================================

    template<typename dtype>
    np::ndarray min(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return nc2Boost(nc::min(inArray, inAxis));
    }

    //================================================================================

    template<typename dtype>
    np::ndarray minimum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2Boost(nc::minimum(inArray1, inArray2));
    }

    //================================================================================

    template<typename Type1, typename Type2>
    np::ndarray multiply(const Type1& in1, const Type2& in2)
    {
        return nc2Boost(nc::multiply(in1, in2));
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
    dtype rad2degScaler(dtype inValue) 
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
    dtype radiansScaler(dtype inValue) 
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
    np::ndarray reciprocal(NdArray<dtype>& inArray)
    {
        return nc2Boost(nc::reciprocal(inArray));
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
    dtype remainderScaler(dtype inValue1, dtype inValue2) 
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
    dtype rintScaler(dtype inValue) 
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
    int8 signScaler(dtype inValue) 
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
    bool signbitScaler(dtype inValue) 
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
    auto sinScaler(dtype inValue)  -> decltype(sin(inValue)) // trailing return type to help gcc
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
    dtype sincScaler(dtype inValue) 
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
    auto sinhScaler(dtype inValue)  -> decltype(sinh(inValue)) // trailing return type to help gcc
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
    auto sqrtScaler(dtype inValue)  -> decltype(sqrt(inValue)) // trailing return type to help gcc
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
    dtype squareScaler(dtype inValue) 
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

    template<typename Type1, typename Type2>
    np::ndarray subtract(const Type1& in1, const Type2& in2)
    {
        return nc2Boost(nc::subtract(in1, in2));
    }

    //================================================================================

    template<typename dtype>
    auto tanScaler(dtype inValue)  -> decltype(tan(inValue)) // trailing return type to help gcc
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
    auto tanhScaler(dtype inValue)  -> decltype(tanh(inValue)) // trailing return type to help gcc
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
    dtype unwrapScaler(dtype inValue) 
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
    double truncScaler(dtype inValue) 
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
    dtype chebyshev_t_Scaler(uint32 n, dtype inValue) 
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
    dtype chebyshev_u_Scaler(uint32 n, dtype inValue) 
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
    dtype hermite_Scaler(uint32 n, dtype inValue) 
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
    dtype laguerre_Scaler1(uint32 n, dtype inValue) 
    {
        return polynomial::laguerre(n, inValue);
    }

    //================================================================================

    template<typename dtype>
    dtype laguerre_Scaler2(uint32 n, uint32 m, dtype inValue) 
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
    dtype legendre_p_Scaler1(int32 n, dtype inValue) 
    {
        return polynomial::legendre_p(n, inValue);
    }

    //================================================================================

    template<typename dtype>
    dtype legendre_p_Scaler2(int32 n, int32 m, dtype inValue) 
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
    dtype legendre_q_Scaler(int32 n, dtype inValue) 
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
    dtype airy_ai_Scaler(dtype inValue) 
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
    dtype airy_ai_prime_Scaler(dtype inValue) 
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
    dtype airy_bi_Scaler(dtype inValue) 
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
    dtype airy_bi_prime_Scaler(dtype inValue) 
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

    double bernoulli_Scaler(uint32 n) 
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
    dtype bessel_in_Scaler(dtype inV, dtype inValue) 
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
    dtype bessel_in_prime_Scaler(dtype inV, dtype inValue) 
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
    dtype bessel_jn_Scaler(dtype inV, dtype inValue) 
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
    dtype bessel_jn_prime_Scaler(dtype inV, dtype inValue) 
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
    dtype bessel_kn_Scaler(dtype inV, dtype inValue) 
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
    dtype bessel_kn_prime_Scaler(dtype inV, dtype inValue) 
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
    dtype bessel_yn_Scaler(dtype inV, dtype inValue) 
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
    dtype bessel_yn_prime_Scaler(dtype inV, dtype inValue) 
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
    dtype beta_Scaler(dtype a, dtype b) 
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
    std::complex<dtype> cyclic_hankel_1_Scaler(dtype v, dtype x) 
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
    std::complex<dtype> cyclic_hankel_2_Scaler(dtype v, dtype x) 
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
    std::complex<dtype> spherical_hankel_1_Scaler(dtype v, dtype x) 
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
    std::complex<dtype> spherical_hankel_2_Scaler(dtype v, dtype x) 
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
    dtype digamma_Scaler(dtype inValue) 
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
    dtype erf_Scaler(dtype inValue) 
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
    dtype erf_inv_Scaler(dtype inValue) 
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
    dtype erfc_Scaler(dtype inValue) 
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
    dtype erfc_inv_Scaler(dtype inValue) 
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

    double factorial_Scaler(uint32 inValue) 
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
    dtype gamma_Scaler(dtype inValue) 
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
    dtype gamma1pm1_Scaler(dtype inValue) 
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
    dtype log_gamma_Scaler(dtype inValue) 
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
    dtype polygamma_Scaler(uint32 n, dtype inValue) 
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

    double prime_Scaler(uint32 inValue) 
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
    dtype riemann_zeta_Scaler(dtype inValue) 
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
    dtype spherical_bessel_jn_Scaler(uint32 inV, dtype inValue) 
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
    dtype spherical_bessel_yn_Scaler(uint32 inV, dtype inValue) 
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
    dtype trigamma_Scaler(dtype inValue) 
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

    // Version.hpp
    bp::scope().attr("VERSION") = VERSION;

    // Constants.hpp
    bp::scope().attr("c") = constants::c;
    bp::scope().attr("e") = constants::e;
    bp::scope().attr("inf") = constants::inf;
    bp::scope().attr("pi") = constants::pi;
    bp::scope().attr("nan") = constants::nan;
    bp::scope().attr("j") = constants::j;
    bp::scope().attr("DAYS_PER_WEEK") = constants::DAYS_PER_WEEK;
    bp::scope().attr("MINUTES_PER_HOUR") = constants::MINUTES_PER_HOUR;
    bp::scope().attr("SECONDS_PER_MINUTE") = constants::SECONDS_PER_MINUTE;
    bp::scope().attr("MILLISECONDS_PER_SECOND") = constants::MILLISECONDS_PER_SECOND;
    bp::scope().attr("SECONDS_PER_HOUR") = constants::SECONDS_PER_HOUR;
    bp::scope().attr("HOURS_PER_DAY") = constants::HOURS_PER_DAY;
    bp::scope().attr("MINUTES_PER_DAY") = constants::MINUTES_PER_DAY;
    bp::scope().attr("SECONDS_PER_DAY") = constants::SECONDS_PER_DAY;
    bp::scope().attr("MILLISECONDS_PER_DAY") = constants::MILLISECONDS_PER_DAY;
    bp::scope().attr("SECONDS_PER_WEEK") = constants::SECONDS_PER_WEEK;

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
        .def("isSigned", &DtypeInfoUint32::isSigned).staticmethod("isSigned")
        .def("min", &DtypeInfoUint32::min).staticmethod("min")
        .def("max", &DtypeInfoUint32::max).staticmethod("max");

    typedef DtypeInfo<std::complex<double>> DtypeInfoComplexDouble;
    bp::class_<DtypeInfoComplexDouble>
        ("DtypeInfoComplexDouble", bp::init<>())
        .def("bits", &DtypeInfoComplexDouble::bits).staticmethod("bits")
        .def("epsilon", &DtypeInfoComplexDouble::epsilon).staticmethod("epsilon")
        .def("isInteger", &DtypeInfoComplexDouble::isInteger).staticmethod("isInteger")
        .def("isSigned", &DtypeInfoComplexDouble::isSigned).staticmethod("isSigned")
        .def("min", &DtypeInfoComplexDouble::min).staticmethod("min")
        .def("max", &DtypeInfoComplexDouble::max).staticmethod("max");

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
    using NdArrayDouble = NdArray<double>;
    using NdArrayDoubleIterator = NdArrayDouble::iterator;
    using NdArrayDoubleConstIterator = NdArrayDouble::const_iterator;
    using NdArrayDoubleReverseIterator = NdArrayDouble::reverse_iterator;
    using NdArrayDoubleConstReverseIterator = NdArrayDouble::const_reverse_iterator;
    using NdArrayDoubleColumnIterator = NdArrayDouble::column_iterator;
    using NdArrayDoubleConstColumnIterator = NdArrayDouble::const_column_iterator;
    using NdArrayDoubleReverseColumnIterator = NdArrayDouble::reverse_column_iterator;
    using NdArrayDoubleConstReverseColumnIterator = NdArrayDouble::const_reverse_column_iterator;

    using ComplexDouble = std::complex<double>;
    using NdArrayComplexDouble = NdArray<ComplexDouble>;
    using NdArrayComplexDoubleIterator = NdArrayComplexDouble::iterator;
    using NdArrayComplexDoubleConstIterator = NdArrayComplexDouble::const_iterator;
    using NdArrayComplexDoubleReverseIterator = NdArrayComplexDouble::reverse_iterator;
    using NdArrayComplexDoubleConstReverseIterator = NdArrayComplexDouble::const_reverse_iterator;
    using NdArrayComplexDoubleColumnIterator = NdArrayComplexDouble::column_iterator;
    using NdArrayComplexDoubleConstColumnIterator = NdArrayComplexDouble::const_column_iterator;
    using NdArrayComplexDoubleReverseColumnIterator = NdArrayComplexDouble::reverse_column_iterator;
    using NdArrayComplexDoubleConstReverseColumnIterator = NdArrayComplexDouble::const_reverse_column_iterator;

    NdArrayDoubleIterator& (NdArrayDoubleIterator::*iteratorOperatorPlusPlusPre)()  = &NdArrayDoubleIterator::operator++;
    NdArrayDoubleIterator (NdArrayDoubleIterator::*iteratorOperatorPlusPlusPost)(int)  = &NdArrayDoubleIterator::operator++;
    NdArrayDoubleIterator& (NdArrayDoubleIterator::*iteratorOperatorMinusMinusPre)()  = &NdArrayDoubleIterator::operator--;
    NdArrayDoubleIterator (NdArrayDoubleIterator::*iteratorOperatorMinusMinusPost)(int)  = &NdArrayDoubleIterator::operator--;
    NdArrayDoubleIterator& (NdArrayDoubleIterator::*iteratorOperatorPlusEqual)(NdArrayDoubleIterator::difference_type)  = &NdArrayDoubleIterator::operator+=;
    NdArrayDoubleIterator (NdArrayDoubleIterator::*iteratorOperatorPlus)(NdArrayDoubleIterator::difference_type) const  = &NdArrayDoubleIterator::operator+;
    NdArrayDoubleIterator& (NdArrayDoubleIterator::*iteratorOperatorMinusEqual)(NdArrayDoubleIterator::difference_type)  = &NdArrayDoubleIterator::operator-=;
    NdArrayDoubleIterator(NdArrayDoubleIterator::*iteratorOperatorMinus)(NdArrayDoubleIterator::difference_type) const  = &NdArrayDoubleIterator::operator-;
    //NdArrayDoubleIterator::difference_type(NdArrayDoubleIterator::*iteratorOperatorDiff)(const NdArrayDoubleIterator&) const = &NdArrayDoubleIterator::operator-;
    //bool(NdArrayDoubleIterator::*iteratorOperatorEqual)(const NdArrayDoubleIterator&) const = &NdArrayDoubleIterator::operator==;
    //bool(NdArrayDoubleConstIterator::*iteratorOperatorNotEqual)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleIterator::operator!=;
    //bool(NdArrayDoubleConstIterator::*iteratorOperatorLT)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleIterator::operator<;
    //bool(NdArrayDoubleConstIterator::*iteratorOperatorLTE)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleIterator::operator<=;
    //bool(NdArrayDoubleConstIterator::*iteratorOperatorGT)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleIterator::operator>;
    //bool(NdArrayDoubleConstIterator::*iteratorOperatorGTE)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleIterator::operator>=;

    NdArrayComplexDoubleIterator& (NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoublePlusPlusPre)()  = &NdArrayComplexDoubleIterator::operator++;
    NdArrayComplexDoubleIterator (NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoublePlusPlusPost)(int)  = &NdArrayComplexDoubleIterator::operator++;
    NdArrayComplexDoubleIterator& (NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoubleMinusMinusPre)()  = &NdArrayComplexDoubleIterator::operator--;
    NdArrayComplexDoubleIterator (NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoubleMinusMinusPost)(int)  = &NdArrayComplexDoubleIterator::operator--;
    NdArrayComplexDoubleIterator& (NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoublePlusEqual)(NdArrayComplexDoubleIterator::difference_type)  = &NdArrayComplexDoubleIterator::operator+=;
    NdArrayComplexDoubleIterator (NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoublePlus)(NdArrayComplexDoubleIterator::difference_type) const  = &NdArrayComplexDoubleIterator::operator+;
    NdArrayComplexDoubleIterator& (NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoubleMinusEqual)(NdArrayComplexDoubleIterator::difference_type)  = &NdArrayComplexDoubleIterator::operator-=;
    NdArrayComplexDoubleIterator(NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoubleMinus)(NdArrayComplexDoubleIterator::difference_type) const  = &NdArrayComplexDoubleIterator::operator-;
    //NdArrayComplexDoubleIterator::difference_type(NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoubleDiff)(const NdArrayComplexDoubleIterator&) const = &NdArrayComplexDoubleIterator::operator-;
    //bool(NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoubleEqual)(const NdArrayComplexDoubleIterator&) const = &NdArrayComplexDoubleIterator::operator==;
    //bool(NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoubleNotEqual)(const NdArrayComplexDoubleIterator&) const = &NdArrayComplexDoubleIterator::operator!=;
    //bool(NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoubleLT)(const NdArrayComplexDoubleIterator&) const = &NdArrayComplexDoubleIterator::operator<;
    //bool(NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoubleLTE)(const NdArrayComplexDoubleIterator&) const = &NdArrayComplexDoubleIterator::operator<=;
    //bool(NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoubleGT)(const NdArrayComplexDoubleIterator&) const = &NdArrayComplexDoubleIterator::operator>;
    //bool(NdArrayComplexDoubleIterator::*iteratorOperatorComplexDoubleGTE)(const NdArrayComplexDoubleIterator&) const = &NdArrayComplexDoubleIterator::operator>=;

    bp::class_<NdArrayDoubleIterator>
        ("NdArrayDoubleIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayDoubleIterator>)
        .def("operatorPlusPlusPost", iteratorOperatorPlusPlusPost)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayDoubleIterator>)
        .def("operatorMinusMinusPost", iteratorOperatorMinusMinusPost)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayDoubleIterator>)
        .def("__add__", iteratorOperatorPlus)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayDoubleIterator>)
        .def("__sub__", iteratorOperatorMinus)
        //.def("__sub__", iteratorOperatorDiff)
        //.def("__eq__", iteratorOperatorEqual)
        //.def("__ne__", iteratorOperatorNotEqual)
        //.def("__lt__", iteratorOperatorLT)
        //.def("__le__", iteratorOperatorLTE)
        //.def("__gt__", iteratorOperatorGT)
        //.def("__ge__", iteratorOperatorGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleIterator>);

    bp::class_<NdArrayComplexDoubleIterator>
        ("NdArrayComplexDoubleIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleIterator>)
        .def("operatorPlusPlusPre", iteratorOperatorComplexDoublePlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", iteratorOperatorComplexDoublePlusPlusPost)
        .def("operatorMinusMinusPre", iteratorOperatorComplexDoubleMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", iteratorOperatorComplexDoubleMinusMinusPost)
        .def("__iadd__", iteratorOperatorComplexDoublePlusEqual, bp::return_internal_reference<>())
        .def("__add__", iteratorOperatorComplexDoublePlus)
        .def("__isub__", iteratorOperatorComplexDoubleMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", iteratorOperatorComplexDoubleMinus)
        //.def("__sub__", iteratorOperatorComplexDoubleDiff)
        //.def("__eq__", iteratorOperatorComplexDoubleEqual)
        //.def("__ne__", iteratorOperatorComplexDoubleNotEqual)
        //.def("__lt__", iteratorOperatorComplexDoubleLT)
        //.def("__le__", iteratorOperatorComplexDoubleLTE)
        //.def("__gt__", iteratorOperatorComplexDoubleGT)
        //.def("__ge__", iteratorOperatorComplexDoubleGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleIterator>);

    NdArrayDoubleConstIterator& (NdArrayDoubleConstIterator::*constIteratorOperatorPlusPlusPre)()  = &NdArrayDoubleConstIterator::operator++;
    NdArrayDoubleConstIterator (NdArrayDoubleConstIterator::*constIteratorOperatorPlusPlusPost)(int)  = &NdArrayDoubleConstIterator::operator++;
    NdArrayDoubleConstIterator& (NdArrayDoubleConstIterator::*constIteratorOperatorMinusMinusPre)()  = &NdArrayDoubleConstIterator::operator--;
    NdArrayDoubleConstIterator (NdArrayDoubleConstIterator::*constIteratorOperatorMinusMinusPost)(int)  = &NdArrayDoubleConstIterator::operator--;
    NdArrayDoubleConstIterator& (NdArrayDoubleConstIterator::*constIteratorOperatorPlusEqual)(NdArrayDoubleConstIterator::difference_type)  = &NdArrayDoubleConstIterator::operator+=;
    NdArrayDoubleConstIterator (NdArrayDoubleConstIterator::*constIteratorOperatorPlus)(NdArrayDoubleConstIterator::difference_type) const  = &NdArrayDoubleConstIterator::operator+;
    NdArrayDoubleConstIterator& (NdArrayDoubleConstIterator::*constIteratorOperatorMinusEqual)(NdArrayDoubleConstIterator::difference_type)  = &NdArrayDoubleConstIterator::operator-=;
    NdArrayDoubleConstIterator(NdArrayDoubleConstIterator::*constIteratorOperatorMinus)(NdArrayDoubleConstIterator::difference_type) const  = &NdArrayDoubleConstIterator::operator-;
    NdArrayDoubleConstIterator::difference_type(NdArrayDoubleConstIterator::*constIteratorOperatorDiff)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleConstIterator::operator-;
    bool(NdArrayDoubleConstIterator::*constIteratorOperatorEqual)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleConstIterator::operator==;
    bool(NdArrayDoubleConstIterator::*constIteratorOperatorNotEqual)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleConstIterator::operator!=;
    bool(NdArrayDoubleConstIterator::*constIteratorOperatorLT)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleConstIterator::operator<;
    bool(NdArrayDoubleConstIterator::*constIteratorOperatorLTE)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleConstIterator::operator<=;
    bool(NdArrayDoubleConstIterator::*constIteratorOperatorGT)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleConstIterator::operator>;
    bool(NdArrayDoubleConstIterator::*constIteratorOperatorGTE)(const NdArrayDoubleConstIterator&) const = &NdArrayDoubleConstIterator::operator>=;

    NdArrayComplexDoubleConstIterator& (NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexPlusPlusPre)()  = &NdArrayComplexDoubleConstIterator::operator++;
    NdArrayComplexDoubleConstIterator (NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexPlusPlusPost)(int)  = &NdArrayComplexDoubleConstIterator::operator++;
    NdArrayComplexDoubleConstIterator& (NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexMinusMinusPre)()  = &NdArrayComplexDoubleConstIterator::operator--;
    NdArrayComplexDoubleConstIterator (NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexMinusMinusPost)(int)  = &NdArrayComplexDoubleConstIterator::operator--;
    NdArrayComplexDoubleConstIterator& (NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexPlusEqual)(NdArrayComplexDoubleConstIterator::difference_type)  = &NdArrayComplexDoubleConstIterator::operator+=;
    NdArrayComplexDoubleConstIterator (NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexPlus)(NdArrayComplexDoubleConstIterator::difference_type) const  = &NdArrayComplexDoubleConstIterator::operator+;
    NdArrayComplexDoubleConstIterator& (NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexMinusEqual)(NdArrayComplexDoubleConstIterator::difference_type)  = &NdArrayComplexDoubleConstIterator::operator-=;
    NdArrayComplexDoubleConstIterator(NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexMinus)(NdArrayComplexDoubleConstIterator::difference_type) const  = &NdArrayComplexDoubleConstIterator::operator-;
    NdArrayComplexDoubleConstIterator::difference_type(NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexDiff)(const NdArrayComplexDoubleConstIterator&) const = &NdArrayComplexDoubleConstIterator::operator-;
    bool(NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexEqual)(const NdArrayComplexDoubleConstIterator&) const = &NdArrayComplexDoubleConstIterator::operator==;
    bool(NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexNotEqual)(const NdArrayComplexDoubleConstIterator&) const = &NdArrayComplexDoubleConstIterator::operator!=;
    bool(NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexLT)(const NdArrayComplexDoubleConstIterator&) const = &NdArrayComplexDoubleConstIterator::operator<;
    bool(NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexLTE)(const NdArrayComplexDoubleConstIterator&) const = &NdArrayComplexDoubleConstIterator::operator<=;
    bool(NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexGT)(const NdArrayComplexDoubleConstIterator&) const = &NdArrayComplexDoubleConstIterator::operator>;
    bool(NdArrayComplexDoubleConstIterator::*constIteratorOperatorComplexGTE)(const NdArrayComplexDoubleConstIterator&) const = &NdArrayComplexDoubleConstIterator::operator>=;

    bp::class_<NdArrayDoubleConstIterator>
        ("NdArrayDoubleConstIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleConstIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayDoubleConstIterator>)
        .def("operatorPlusPlusPost", constIteratorOperatorPlusPlusPost)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayDoubleConstIterator>)
        .def("operatorMinusMinusPost", constIteratorOperatorMinusMinusPost)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayDoubleConstIterator>)
        .def("__add__", constIteratorOperatorPlus)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayDoubleConstIterator>)
        .def("__sub__", constIteratorOperatorMinus)
        .def("__sub__", constIteratorOperatorDiff)
        .def("__eq__", constIteratorOperatorEqual)
        .def("__ne__", constIteratorOperatorNotEqual)
        .def("__lt__", constIteratorOperatorLT)
        .def("__le__", constIteratorOperatorLTE)
        .def("__gt__", constIteratorOperatorGT)
        .def("__ge__", constIteratorOperatorGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleConstIterator>);


    bp::class_<NdArrayComplexDoubleConstIterator>
        ("NdArrayComplexDoubleConstIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleConstIterator>)
        .def("operatorPlusPlusPre", constIteratorOperatorComplexPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", constIteratorOperatorComplexPlusPlusPost)
        .def("operatorMinusMinusPre", constIteratorOperatorComplexMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", constIteratorOperatorComplexMinusMinusPost)
        .def("__iadd__", constIteratorOperatorComplexPlusEqual, bp::return_internal_reference<>())
        .def("__add__", constIteratorOperatorComplexPlus)
        .def("__isub__", constIteratorOperatorComplexMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", constIteratorOperatorComplexMinus)
        .def("__sub__", constIteratorOperatorComplexDiff)
        .def("__eq__", constIteratorOperatorComplexEqual)
        .def("__ne__", constIteratorOperatorComplexNotEqual)
        .def("__lt__", constIteratorOperatorComplexLT)
        .def("__le__", constIteratorOperatorComplexLTE)
        .def("__gt__", constIteratorOperatorComplexGT)
        .def("__ge__", constIteratorOperatorComplexGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleConstIterator>);

    NdArrayDoubleReverseIterator& (NdArrayDoubleReverseIterator::*reverseIteratorOperatorPlusPlusPre)() = &NdArrayDoubleReverseIterator::operator++;
    NdArrayDoubleReverseIterator (NdArrayDoubleReverseIterator::*reverseIteratorOperatorPlusPlusPost)(int) = &NdArrayDoubleReverseIterator::operator++;
    NdArrayDoubleReverseIterator& (NdArrayDoubleReverseIterator::*reverseIteratorOperatorMinusMinusPre)() = &NdArrayDoubleReverseIterator::operator--;
    NdArrayDoubleReverseIterator (NdArrayDoubleReverseIterator::*reverseIteratorOperatorMinusMinusPost)(int) = &NdArrayDoubleReverseIterator::operator--;
    NdArrayDoubleReverseIterator& (NdArrayDoubleReverseIterator::*reverseIteratorOperatorPlusEqual)(NdArrayDoubleReverseIterator::difference_type) = &NdArrayDoubleReverseIterator::operator+=;
    NdArrayDoubleReverseIterator (NdArrayDoubleReverseIterator::*reverseIteratorOperatorPlus)(NdArrayDoubleReverseIterator::difference_type) const = &NdArrayDoubleReverseIterator::operator+;
    NdArrayDoubleReverseIterator& (NdArrayDoubleReverseIterator::*reverseIteratorOperatorMinusEqual)(NdArrayDoubleReverseIterator::difference_type) = &NdArrayDoubleReverseIterator::operator-=;
    NdArrayDoubleReverseIterator(NdArrayDoubleReverseIterator::*reverseIteratorOperatorMinus)(NdArrayDoubleReverseIterator::difference_type) const = &NdArrayDoubleReverseIterator::operator-;
    //NdArrayDoubleReverseIterator::difference_type(NdArrayDoubleReverseIterator::*reverseIteratorOperatorDiff)(const NdArrayDoubleReverseIterator&) const = &NdArrayDoubleReverseIterator::operator-;
    //bool(NdArrayDoubleReverseIterator::*reverseIteratorOperatorEqual)(const NdArrayDoubleReverseIterator&) const = &NdArrayDoubleReverseIterator::operator==;
    //bool(NdArrayDoubleReverseIterator::*reverseIteratorOperatorNotEqual)(const NdArrayDoubleReverseIterator&) const = &NdArrayDoubleReverseIterator::operator!=;
    //bool(NdArrayDoubleReverseIterator::*reverseIteratorOperatorLT)(const NdArrayDoubleReverseIterator&) const = &NdArrayDoubleReverseIterator::operator<;
    //bool(NdArrayDoubleReverseIterator::*reverseIteratorOperatorLTE)(const NdArrayDoubleReverseIterator&) const = &NdArrayDoubleReverseIterator::operator<=;
    //bool(NdArrayDoubleReverseIterator::*reverseIteratorOperatorGT)(const NdArrayDoubleReverseIterator&) const = &NdArrayDoubleReverseIterator::operator>;
    //bool(NdArrayDoubleReverseIterator::*reverseIteratorOperatorGTE)(const NdArrayDoubleReverseIterator&) const = &NdArrayDoubleReverseIterator::operator>=;

    NdArrayComplexDoubleReverseIterator& (NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexPlusPlusPre)() = &NdArrayComplexDoubleReverseIterator::operator++;
    NdArrayComplexDoubleReverseIterator (NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexPlusPlusPost)(int) = &NdArrayComplexDoubleReverseIterator::operator++;
    NdArrayComplexDoubleReverseIterator& (NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexMinusMinusPre)() = &NdArrayComplexDoubleReverseIterator::operator--;
    NdArrayComplexDoubleReverseIterator (NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexMinusMinusPost)(int) = &NdArrayComplexDoubleReverseIterator::operator--;
    NdArrayComplexDoubleReverseIterator& (NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexPlusEqual)(NdArrayComplexDoubleReverseIterator::difference_type) = &NdArrayComplexDoubleReverseIterator::operator+=;
    NdArrayComplexDoubleReverseIterator (NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexPlus)(NdArrayComplexDoubleReverseIterator::difference_type) const = &NdArrayComplexDoubleReverseIterator::operator+;
    NdArrayComplexDoubleReverseIterator& (NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexMinusEqual)(NdArrayComplexDoubleReverseIterator::difference_type) = &NdArrayComplexDoubleReverseIterator::operator-=;
    NdArrayComplexDoubleReverseIterator(NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexMinus)(NdArrayComplexDoubleReverseIterator::difference_type) const = &NdArrayComplexDoubleReverseIterator::operator-;
    //NdArrayComplexDoubleReverseIterator::difference_type(NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexDiff)(const NdArrayComplexDoubleReverseIterator&) const = &NdArrayComplexDoubleReverseIterator::operator-;
    //bool(NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexEqual)(const NdArrayComplexDoubleReverseIterator&) const = &NdArrayComplexDoubleReverseIterator::operator==;
    //bool(NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexNotEqual)(const NdArrayComplexDoubleReverseIterator&) const = &NdArrayComplexDoubleReverseIterator::operator!=;
    //bool(NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexLT)(const NdArrayComplexDoubleReverseIterator&) const = &NdArrayComplexDoubleReverseIterator::operator<;
    //bool(NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexLTE)(const NdArrayComplexDoubleReverseIterator&) const = &NdArrayComplexDoubleReverseIterator::operator<=;
    //bool(NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexGT)(const NdArrayComplexDoubleReverseIterator&) const = &NdArrayComplexDoubleReverseIterator::operator>;
    //bool(NdArrayComplexDoubleReverseIterator::*reverseIteratorOperatorComplexGTE)(const NdArrayComplexDoubleReverseIterator&) const = &NdArrayComplexDoubleReverseIterator::operator>=;

    bp::class_<NdArrayDoubleReverseIterator>
        ("NdArrayDoubleReverseIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleReverseIterator>)
        .def("operatorPlusPlusPre", reverseIteratorOperatorPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", reverseIteratorOperatorPlusPlusPost)
        .def("operatorMinusMinusPre", reverseIteratorOperatorMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", reverseIteratorOperatorMinusMinusPost)
        .def("__iadd__", reverseIteratorOperatorPlusEqual, bp::return_internal_reference<>())
        .def("__add__", reverseIteratorOperatorPlus)
        .def("__isub__", reverseIteratorOperatorMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", reverseIteratorOperatorMinus)
        //.def("__sub__", reverseIteratorOperatorDiff)
        //.def("__eq__", reverseIteratorOperatorEqual)
        //.def("__ne__", reverseIteratorOperatorNotEqual)
        //.def("__lt__", reverseIteratorOperatorLT)
        //.def("__le__", reverseIteratorOperatorLTE)
        //.def("__gt__", reverseIteratorOperatorGT)
        //.def("__ge__", reverseIteratorOperatorGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleReverseIterator>);

    bp::class_<NdArrayComplexDoubleReverseIterator>
        ("NdArrayComplexDoubleReverseIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleReverseIterator>)
        .def("operatorPlusPlusPre", reverseIteratorOperatorComplexPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", reverseIteratorOperatorComplexPlusPlusPost)
        .def("operatorMinusMinusPre", reverseIteratorOperatorComplexMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", reverseIteratorOperatorComplexMinusMinusPost)
        .def("__iadd__", reverseIteratorOperatorComplexPlusEqual, bp::return_internal_reference<>())
        .def("__add__", reverseIteratorOperatorComplexPlus)
        .def("__isub__", reverseIteratorOperatorComplexMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", reverseIteratorOperatorComplexMinus)
        //.def("__sub__", reverseIteratorOperatorComplexDiff)
        //.def("__eq__", reverseIteratorOperatorComplexEqual)
        //.def("__ne__", reverseIteratorOperatorComplexNotEqual)
        //.def("__lt__", reverseIteratorOperatorComplexLT)
        //.def("__le__", reverseIteratorOperatorComplexLTE)
        //.def("__gt__", reverseIteratorOperatorComplexGT)
        //.def("__ge__", reverseIteratorOperatorComplexGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleReverseIterator>);

    NdArrayDoubleConstReverseIterator& (NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorPlusPlusPre)() = &NdArrayDoubleConstReverseIterator::operator++;
    NdArrayDoubleConstReverseIterator (NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorPlusPlusPost)(int) = &NdArrayDoubleConstReverseIterator::operator++;
    NdArrayDoubleConstReverseIterator& (NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorMinusMinusPre)() = &NdArrayDoubleConstReverseIterator::operator--;
    NdArrayDoubleConstReverseIterator (NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorMinusMinusPost)(int) = &NdArrayDoubleConstReverseIterator::operator--;
    NdArrayDoubleConstReverseIterator& (NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorPlusEqual)(NdArrayDoubleConstReverseIterator::difference_type) = &NdArrayDoubleConstReverseIterator::operator+=;
    NdArrayDoubleConstReverseIterator (NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorPlus)(NdArrayDoubleConstReverseIterator::difference_type) const = &NdArrayDoubleConstReverseIterator::operator+;
    NdArrayDoubleConstReverseIterator& (NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorMinusEqual)(NdArrayDoubleConstReverseIterator::difference_type) = &NdArrayDoubleConstReverseIterator::operator-=;
    NdArrayDoubleConstReverseIterator(NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorMinus)(NdArrayDoubleConstReverseIterator::difference_type) const = &NdArrayDoubleConstReverseIterator::operator-;
    //NdArrayDoubleConstReverseIterator::difference_type(NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorDiff)(const NdArrayDoubleConstReverseIterator&) const = &NdArrayDoubleConstReverseIterator::operator-;
    //bool(NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorEqual)(const NdArrayDoubleConstReverseIterator&) const = &NdArrayDoubleConstReverseIterator::operator==;
    //bool(NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorNotEqual)(const NdArrayDoubleConstReverseIterator&) const = &NdArrayDoubleConstReverseIterator::operator!=;
    //bool(NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorLT)(const NdArrayDoubleConstReverseIterator&) const = &NdArrayDoubleConstReverseIterator::operator<;
    //bool(NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorLTE)(const NdArrayDoubleConstReverseIterator&) const = &NdArrayDoubleConstReverseIterator::operator<=;
    //bool(NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorGT)(const NdArrayDoubleConstReverseIterator&) const = &NdArrayDoubleConstReverseIterator::operator>;
    //bool(NdArrayDoubleConstReverseIterator::*constReverseIteratorOperatorGTE)(const NdArrayDoubleConstReverseIterator&) const = &NdArrayDoubleConstReverseIterator::operator>=;

    NdArrayComplexDoubleConstReverseIterator& (NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexPlusPlusPre)() = &NdArrayComplexDoubleConstReverseIterator::operator++;
    NdArrayComplexDoubleConstReverseIterator (NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexPlusPlusPost)(int) = &NdArrayComplexDoubleConstReverseIterator::operator++;
    NdArrayComplexDoubleConstReverseIterator& (NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexMinusMinusPre)() = &NdArrayComplexDoubleConstReverseIterator::operator--;
    NdArrayComplexDoubleConstReverseIterator (NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexMinusMinusPost)(int) = &NdArrayComplexDoubleConstReverseIterator::operator--;
    NdArrayComplexDoubleConstReverseIterator& (NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexPlusEqual)(NdArrayComplexDoubleConstReverseIterator::difference_type) = &NdArrayComplexDoubleConstReverseIterator::operator+=;
    NdArrayComplexDoubleConstReverseIterator (NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexPlus)(NdArrayComplexDoubleConstReverseIterator::difference_type) const = &NdArrayComplexDoubleConstReverseIterator::operator+;
    NdArrayComplexDoubleConstReverseIterator& (NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexMinusEqual)(NdArrayComplexDoubleConstReverseIterator::difference_type) = &NdArrayComplexDoubleConstReverseIterator::operator-=;
    NdArrayComplexDoubleConstReverseIterator(NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexMinus)(NdArrayComplexDoubleConstReverseIterator::difference_type) const = &NdArrayComplexDoubleConstReverseIterator::operator-;
    //NdArrayComplexDoubleConstReverseIterator::difference_type(NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexDiff)(const NdArrayComplexDoubleConstReverseIterator&) const = &NdArrayComplexDoubleConstReverseIterator::operator-;
    //bool(NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexEqual)(const NdArrayComplexDoubleConstReverseIterator&) const = &NdArrayComplexDoubleConstReverseIterator::operator==;
    //bool(NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexNotEqual)(const NdArrayComplexDoubleConstReverseIterator&) const = &NdArrayComplexDoubleConstReverseIterator::operator!=;
    //bool(NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexLT)(const NdArrayComplexDoubleConstReverseIterator&) const = &NdArrayComplexDoubleConstReverseIterator::operator<;
    //bool(NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexLTE)(const NdArrayComplexDoubleConstReverseIterator&) const = &NdArrayComplexDoubleConstReverseIterator::operator<=;
    //bool(NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexGT)(const NdArrayComplexDoubleConstReverseIterator&) const = &NdArrayComplexDoubleConstReverseIterator::operator>;
    //bool(NdArrayComplexDoubleConstReverseIterator::*constReverseIteratorOperatorComplexGTE)(const NdArrayComplexDoubleConstReverseIterator&) const = &NdArrayComplexDoubleConstReverseIterator::operator>=;

    bp::class_<NdArrayDoubleConstReverseIterator>
        ("NdArrayDoubleConstReverseIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleConstReverseIterator>)
        .def("operatorPlusPlusPre", constReverseIteratorOperatorPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", constReverseIteratorOperatorPlusPlusPost)
        .def("operatorMinusMinusPre", constReverseIteratorOperatorMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", constReverseIteratorOperatorMinusMinusPost)
        .def("__iadd__", constReverseIteratorOperatorPlusEqual, bp::return_internal_reference<>())
        .def("__add__", constReverseIteratorOperatorPlus)
        .def("__isub__", constReverseIteratorOperatorMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", constReverseIteratorOperatorMinus)
        //.def("__sub__", constReverseIteratorOperatorDiff)
        //.def("__eq__", constReverseIteratorOperatorEqual)
        //.def("__ne__", constReverseIteratorOperatorNotEqual)
        //.def("__lt__", constReverseIteratorOperatorLT)
        //.def("__le__", constReverseIteratorOperatorLTE)
        //.def("__gt__", constReverseIteratorOperatorGT)
        //.def("__ge__", constReverseIteratorOperatorGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleConstReverseIterator>);

    bp::class_<NdArrayComplexDoubleConstReverseIterator>
        ("NdArrayComplexDoubleConstReverseIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleConstReverseIterator>)
        .def("operatorPlusPlusPre", constReverseIteratorOperatorComplexPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", constReverseIteratorOperatorComplexPlusPlusPost)
        .def("operatorMinusMinusPre", constReverseIteratorOperatorComplexMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", constReverseIteratorOperatorComplexMinusMinusPost)
        .def("__iadd__", constReverseIteratorOperatorComplexPlusEqual, bp::return_internal_reference<>())
        .def("__add__", constReverseIteratorOperatorComplexPlus)
        .def("__isub__", constReverseIteratorOperatorComplexMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", constReverseIteratorOperatorComplexMinus)
        //.def("__sub__", constReverseIteratorOperatorComplexDiff)
        //.def("__eq__", constReverseIteratorOperatorComplexEqual)
        //.def("__ne__", constReverseIteratorOperatorComplexNotEqual)
        //.def("__lt__", constReverseIteratorOperatorComplexLT)
        //.def("__le__", constReverseIteratorOperatorComplexLTE)
        //.def("__gt__", constReverseIteratorOperatorComplexGT)
        //.def("__ge__", constReverseIteratorOperatorComplexGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleConstReverseIterator>);

    NdArrayDoubleColumnIterator& (NdArrayDoubleColumnIterator::*columnIteratorOperatorPlusPlusPre)()  = &NdArrayDoubleColumnIterator::operator++;
    NdArrayDoubleColumnIterator (NdArrayDoubleColumnIterator::*columnIteratorOperatorPlusPlusPost)(int)  = &NdArrayDoubleColumnIterator::operator++;
    NdArrayDoubleColumnIterator& (NdArrayDoubleColumnIterator::*columnIteratorOperatorMinusMinusPre)()  = &NdArrayDoubleColumnIterator::operator--;
    NdArrayDoubleColumnIterator (NdArrayDoubleColumnIterator::*columnIteratorOperatorMinusMinusPost)(int)  = &NdArrayDoubleColumnIterator::operator--;
    NdArrayDoubleColumnIterator& (NdArrayDoubleColumnIterator::*columnIteratorOperatorPlusEqual)(NdArrayDoubleColumnIterator::difference_type)  = &NdArrayDoubleColumnIterator::operator+=;
    NdArrayDoubleColumnIterator (NdArrayDoubleColumnIterator::*columnIteratorOperatorPlus)(NdArrayDoubleColumnIterator::difference_type) const  = &NdArrayDoubleColumnIterator::operator+;
    NdArrayDoubleColumnIterator& (NdArrayDoubleColumnIterator::*columnIteratorOperatorMinusEqual)(NdArrayDoubleColumnIterator::difference_type)  = &NdArrayDoubleColumnIterator::operator-=;
    NdArrayDoubleColumnIterator(NdArrayDoubleColumnIterator::*columnIteratorOperatorMinus)(NdArrayDoubleColumnIterator::difference_type) const  = &NdArrayDoubleColumnIterator::operator-;
    //NdArrayDoubleColumnIterator::difference_type(NdArrayDoubleColumnIterator::*columnIteratorOperatorDiff)(const NdArrayDoubleColumnIterator&) const = &NdArrayDoubleColumnIterator::operator-;
    //bool(NdArrayDoubleColumnIterator::*columnIteratorOperatorEqual)(const NdArrayDoubleColumnIterator&) const = &NdArrayDoubleColumnIterator::operator==;
    //bool(NdArrayDoubleColumnIterator::*columnIteratorOperatorNotEqual)(const NdArrayDoubleColumnIterator&) const = &NdArrayDoubleColumnIterator::operator!=;
    //bool(NdArrayDoubleColumnIterator::*columnIteratorOperatorLT)(const NdArrayDoubleColumnIterator&) const = &NdArrayDoubleColumnIterator::operator<;
    //bool(NdArrayDoubleColumnIterator::*columnIteratorOperatorLTE)(const NdArrayDoubleColumnIterator&) const = &NdArrayDoubleColumnIterator::operator<=;
    //bool(NdArrayDoubleColumnIterator::*columnIteratorOperatorGT)(const NdArrayDoubleColumnIterator&) const = &NdArrayDoubleColumnIterator::operator>;
    //bool(NdArrayDoubleColumnIterator::*columnIteratorOperatorGTE)(const NdArrayDoubleColumnIterator&) const = &NdArrayDoubleColumnIterator::operator>=;

    NdArrayComplexDoubleColumnIterator& (NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoublePlusPlusPre)()  = &NdArrayComplexDoubleColumnIterator::operator++;
    NdArrayComplexDoubleColumnIterator (NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoublePlusPlusPost)(int)  = &NdArrayComplexDoubleColumnIterator::operator++;
    NdArrayComplexDoubleColumnIterator& (NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoubleMinusMinusPre)()  = &NdArrayComplexDoubleColumnIterator::operator--;
    NdArrayComplexDoubleColumnIterator (NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoubleMinusMinusPost)(int)  = &NdArrayComplexDoubleColumnIterator::operator--;
    NdArrayComplexDoubleColumnIterator& (NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoublePlusEqual)(NdArrayComplexDoubleColumnIterator::difference_type)  = &NdArrayComplexDoubleColumnIterator::operator+=;
    NdArrayComplexDoubleColumnIterator (NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoublePlus)(NdArrayComplexDoubleColumnIterator::difference_type) const  = &NdArrayComplexDoubleColumnIterator::operator+;
    NdArrayComplexDoubleColumnIterator& (NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoubleMinusEqual)(NdArrayComplexDoubleColumnIterator::difference_type)  = &NdArrayComplexDoubleColumnIterator::operator-=;
    NdArrayComplexDoubleColumnIterator(NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoubleMinus)(NdArrayComplexDoubleColumnIterator::difference_type) const  = &NdArrayComplexDoubleColumnIterator::operator-;
    //NdArrayComplexDoubleColumnIterator::difference_type(NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoubleDiff)(const NdArrayComplexDoubleColumnIterator&) const = &NdArrayComplexDoubleColumnIterator::operator-;
    //bool(NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoubleEqual)(const NdArrayComplexDoubleColumnIterator&) const = &NdArrayComplexDoubleColumnIterator::operator==;
    //bool(NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoubleNotEqual)(const NdArrayComplexDoubleColumnIterator&) const = &NdArrayComplexDoubleColumnIterator::operator!=;
    //bool(NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoubleLT)(const NdArrayComplexDoubleColumnIterator&) const = &NdArrayComplexDoubleColumnIterator::operator<;
    //bool(NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoubleLTE)(const NdArrayComplexDoubleColumnIterator&) const = &NdArrayComplexDoubleColumnIterator::operator<=;
    //bool(NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoubleGT)(const NdArrayComplexDoubleColumnIterator&) const = &NdArrayComplexDoubleColumnIterator::operator>;
    //bool(NdArrayComplexDoubleColumnIterator::*columnIteratorOperatorComplexDoubleGTE)(const NdArrayComplexDoubleColumnIterator&) const = &NdArrayComplexDoubleColumnIterator::operator>=;

    bp::class_<NdArrayDoubleColumnIterator>
        ("NdArrayDoubleColumnIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleColumnIterator>)
        .def("operatorPlusPlusPre", columnIteratorOperatorPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", columnIteratorOperatorPlusPlusPost)
        .def("operatorMinusMinusPre", columnIteratorOperatorMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", columnIteratorOperatorMinusMinusPost)
        .def("__iadd__", columnIteratorOperatorPlusEqual, bp::return_internal_reference<>())
        .def("__add__", columnIteratorOperatorPlus)
        .def("__isub__", columnIteratorOperatorMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", columnIteratorOperatorMinus)
        //.def("__sub__", columnIteratorOperatorDiff)
        //.def("__eq__", columnIteratorOperatorEqual)
        //.def("__ne__", columnIteratorOperatorNotEqual)
        //.def("__lt__", columnIteratorOperatorLT)
        //.def("__le__", columnIteratorOperatorLTE)
        //.def("__gt__", columnIteratorOperatorGT)
        //.def("__ge__", columnIteratorOperatorGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleColumnIterator>);

    bp::class_<NdArrayComplexDoubleColumnIterator>
        ("NdArrayComplexDoubleColumnIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleColumnIterator>)
        .def("operatorPlusPlusPre", columnIteratorOperatorComplexDoublePlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", columnIteratorOperatorComplexDoublePlusPlusPost)
        .def("operatorMinusMinusPre", columnIteratorOperatorComplexDoubleMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", columnIteratorOperatorComplexDoubleMinusMinusPost)
        .def("__iadd__", columnIteratorOperatorComplexDoublePlusEqual, bp::return_internal_reference<>())
        .def("__add__", columnIteratorOperatorComplexDoublePlus)
        .def("__isub__", columnIteratorOperatorComplexDoubleMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", columnIteratorOperatorComplexDoubleMinus)
        //.def("__sub__", columnIteratorOperatorComplexDoubleDiff)
        //.def("__eq__", columnIteratorOperatorComplexDoubleEqual)
        //.def("__ne__", columnIteratorOperatorComplexDoubleNotEqual)
        //.def("__lt__", columnIteratorOperatorComplexDoubleLT)
        //.def("__le__", columnIteratorOperatorComplexDoubleLTE)
        //.def("__gt__", columnIteratorOperatorComplexDoubleGT)
        //.def("__ge__", columnIteratorOperatorComplexDoubleGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleColumnIterator>);

    NdArrayDoubleConstColumnIterator& (NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorPlusPlusPre)()  = &NdArrayDoubleConstColumnIterator::operator++;
    NdArrayDoubleConstColumnIterator (NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorPlusPlusPost)(int)  = &NdArrayDoubleConstColumnIterator::operator++;
    NdArrayDoubleConstColumnIterator& (NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorMinusMinusPre)()  = &NdArrayDoubleConstColumnIterator::operator--;
    NdArrayDoubleConstColumnIterator (NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorMinusMinusPost)(int)  = &NdArrayDoubleConstColumnIterator::operator--;
    NdArrayDoubleConstColumnIterator& (NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorPlusEqual)(NdArrayDoubleConstColumnIterator::difference_type)  = &NdArrayDoubleConstColumnIterator::operator+=;
    NdArrayDoubleConstColumnIterator (NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorPlus)(NdArrayDoubleConstColumnIterator::difference_type) const  = &NdArrayDoubleConstColumnIterator::operator+;
    NdArrayDoubleConstColumnIterator& (NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorMinusEqual)(NdArrayDoubleConstColumnIterator::difference_type)  = &NdArrayDoubleConstColumnIterator::operator-=;
    NdArrayDoubleConstColumnIterator(NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorMinus)(NdArrayDoubleConstColumnIterator::difference_type) const  = &NdArrayDoubleConstColumnIterator::operator-;
    NdArrayDoubleConstColumnIterator::difference_type(NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorDiff)(const NdArrayDoubleConstColumnIterator&) const = &NdArrayDoubleConstColumnIterator::operator-;
    bool(NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorEqual)(const NdArrayDoubleConstColumnIterator&) const = &NdArrayDoubleConstColumnIterator::operator==;
    bool(NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorNotEqual)(const NdArrayDoubleConstColumnIterator&) const = &NdArrayDoubleConstColumnIterator::operator!=;
    bool(NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorLT)(const NdArrayDoubleConstColumnIterator&) const = &NdArrayDoubleConstColumnIterator::operator<;
    bool(NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorLTE)(const NdArrayDoubleConstColumnIterator&) const = &NdArrayDoubleConstColumnIterator::operator<=;
    bool(NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorGT)(const NdArrayDoubleConstColumnIterator&) const = &NdArrayDoubleConstColumnIterator::operator>;
    bool(NdArrayDoubleConstColumnIterator::*constColumnIteratorOperatorGTE)(const NdArrayDoubleConstColumnIterator&) const = &NdArrayDoubleConstColumnIterator::operator>=;

    NdArrayComplexDoubleConstColumnIterator& (NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexPlusPlusPre)()  = &NdArrayComplexDoubleConstColumnIterator::operator++;
    NdArrayComplexDoubleConstColumnIterator (NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexPlusPlusPost)(int)  = &NdArrayComplexDoubleConstColumnIterator::operator++;
    NdArrayComplexDoubleConstColumnIterator& (NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexMinusMinusPre)()  = &NdArrayComplexDoubleConstColumnIterator::operator--;
    NdArrayComplexDoubleConstColumnIterator (NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexMinusMinusPost)(int)  = &NdArrayComplexDoubleConstColumnIterator::operator--;
    NdArrayComplexDoubleConstColumnIterator& (NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexPlusEqual)(NdArrayComplexDoubleConstColumnIterator::difference_type)  = &NdArrayComplexDoubleConstColumnIterator::operator+=;
    NdArrayComplexDoubleConstColumnIterator (NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexPlus)(NdArrayComplexDoubleConstColumnIterator::difference_type) const  = &NdArrayComplexDoubleConstColumnIterator::operator+;
    NdArrayComplexDoubleConstColumnIterator& (NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexMinusEqual)(NdArrayComplexDoubleConstColumnIterator::difference_type)  = &NdArrayComplexDoubleConstColumnIterator::operator-=;
    NdArrayComplexDoubleConstColumnIterator(NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexMinus)(NdArrayComplexDoubleConstColumnIterator::difference_type) const  = &NdArrayComplexDoubleConstColumnIterator::operator-;
    NdArrayComplexDoubleConstColumnIterator::difference_type(NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexDiff)(const NdArrayComplexDoubleConstColumnIterator&) const = &NdArrayComplexDoubleConstColumnIterator::operator-;
    bool(NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexEqual)(const NdArrayComplexDoubleConstColumnIterator&) const = &NdArrayComplexDoubleConstColumnIterator::operator==;
    bool(NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexNotEqual)(const NdArrayComplexDoubleConstColumnIterator&) const = &NdArrayComplexDoubleConstColumnIterator::operator!=;
    bool(NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexLT)(const NdArrayComplexDoubleConstColumnIterator&) const = &NdArrayComplexDoubleConstColumnIterator::operator<;
    bool(NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexLTE)(const NdArrayComplexDoubleConstColumnIterator&) const = &NdArrayComplexDoubleConstColumnIterator::operator<=;
    bool(NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexGT)(const NdArrayComplexDoubleConstColumnIterator&) const = &NdArrayComplexDoubleConstColumnIterator::operator>;
    bool(NdArrayComplexDoubleConstColumnIterator::*constColumnIteratorOperatorComplexGTE)(const NdArrayComplexDoubleConstColumnIterator&) const = &NdArrayComplexDoubleConstColumnIterator::operator>=;

    bp::class_<NdArrayDoubleConstColumnIterator>
        ("NdArrayDoubleConstColumnIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleConstColumnIterator>)
        .def("operatorPlusPlusPre", constColumnIteratorOperatorPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", constColumnIteratorOperatorPlusPlusPost)
        .def("operatorMinusMinusPre", constColumnIteratorOperatorMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", constColumnIteratorOperatorMinusMinusPost)
        .def("__iadd__", constColumnIteratorOperatorPlusEqual, bp::return_internal_reference<>())
        .def("__add__", constColumnIteratorOperatorPlus)
        .def("__isub__", constColumnIteratorOperatorMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", constColumnIteratorOperatorMinus)
        .def("__sub__", constColumnIteratorOperatorDiff)
        .def("__eq__", constColumnIteratorOperatorEqual)
        .def("__ne__", constColumnIteratorOperatorNotEqual)
        .def("__lt__", constColumnIteratorOperatorLT)
        .def("__le__", constColumnIteratorOperatorLTE)
        .def("__gt__", constColumnIteratorOperatorGT)
        .def("__ge__", constColumnIteratorOperatorGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleConstColumnIterator>);


    bp::class_<NdArrayComplexDoubleConstColumnIterator>
        ("NdArrayComplexDoubleConstColumnIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleConstColumnIterator>)
        .def("operatorPlusPlusPre", constColumnIteratorOperatorComplexPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", constColumnIteratorOperatorComplexPlusPlusPost)
        .def("operatorMinusMinusPre", constColumnIteratorOperatorComplexMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", constColumnIteratorOperatorComplexMinusMinusPost)
        .def("__iadd__", constColumnIteratorOperatorComplexPlusEqual, bp::return_internal_reference<>())
        .def("__add__", constColumnIteratorOperatorComplexPlus)
        .def("__isub__", constColumnIteratorOperatorComplexMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", constColumnIteratorOperatorComplexMinus)
        .def("__sub__", constColumnIteratorOperatorComplexDiff)
        .def("__eq__", constColumnIteratorOperatorComplexEqual)
        .def("__ne__", constColumnIteratorOperatorComplexNotEqual)
        .def("__lt__", constColumnIteratorOperatorComplexLT)
        .def("__le__", constColumnIteratorOperatorComplexLTE)
        .def("__gt__", constColumnIteratorOperatorComplexGT)
        .def("__ge__", constColumnIteratorOperatorComplexGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleConstColumnIterator>);

    NdArrayDoubleReverseColumnIterator& (NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorPlusPlusPre)() = &NdArrayDoubleReverseColumnIterator::operator++;
    NdArrayDoubleReverseColumnIterator (NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorPlusPlusPost)(int) = &NdArrayDoubleReverseColumnIterator::operator++;
    NdArrayDoubleReverseColumnIterator& (NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorMinusMinusPre)() = &NdArrayDoubleReverseColumnIterator::operator--;
    NdArrayDoubleReverseColumnIterator (NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorMinusMinusPost)(int) = &NdArrayDoubleReverseColumnIterator::operator--;
    NdArrayDoubleReverseColumnIterator& (NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorPlusEqual)(NdArrayDoubleReverseColumnIterator::difference_type) = &NdArrayDoubleReverseColumnIterator::operator+=;
    NdArrayDoubleReverseColumnIterator (NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorPlus)(NdArrayDoubleReverseColumnIterator::difference_type) const = &NdArrayDoubleReverseColumnIterator::operator+;
    NdArrayDoubleReverseColumnIterator& (NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorMinusEqual)(NdArrayDoubleReverseColumnIterator::difference_type) = &NdArrayDoubleReverseColumnIterator::operator-=;
    NdArrayDoubleReverseColumnIterator(NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorMinus)(NdArrayDoubleReverseColumnIterator::difference_type) const = &NdArrayDoubleReverseColumnIterator::operator-;
    //NdArrayDoubleReverseColumnIterator::difference_type(NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorDiff)(const NdArrayDoubleReverseColumnIterator&) const = &NdArrayDoubleReverseColumnIterator::operator-;
    //bool(NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorEqual)(const NdArrayDoubleReverseColumnIterator&) const = &NdArrayDoubleReverseColumnIterator::operator==;
    //bool(NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorNotEqual)(const NdArrayDoubleReverseColumnIterator&) const = &NdArrayDoubleReverseColumnIterator::operator!=;
    //bool(NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorLT)(const NdArrayDoubleReverseColumnIterator&) const = &NdArrayDoubleReverseColumnIterator::operator<;
    //bool(NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorLTE)(const NdArrayDoubleReverseColumnIterator&) const = &NdArrayDoubleReverseColumnIterator::operator<=;
    //bool(NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorGT)(const NdArrayDoubleReverseColumnIterator&) const = &NdArrayDoubleReverseColumnIterator::operator>;
    //bool(NdArrayDoubleReverseColumnIterator::*reverseColumnIteratorOperatorGTE)(const NdArrayDoubleReverseColumnIterator&) const = &NdArrayDoubleReverseColumnIterator::operator>=;

    NdArrayComplexDoubleReverseColumnIterator& (NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexPlusPlusPre)() = &NdArrayComplexDoubleReverseColumnIterator::operator++;
    NdArrayComplexDoubleReverseColumnIterator (NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexPlusPlusPost)(int) = &NdArrayComplexDoubleReverseColumnIterator::operator++;
    NdArrayComplexDoubleReverseColumnIterator& (NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexMinusMinusPre)() = &NdArrayComplexDoubleReverseColumnIterator::operator--;
    NdArrayComplexDoubleReverseColumnIterator (NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexMinusMinusPost)(int) = &NdArrayComplexDoubleReverseColumnIterator::operator--;
    NdArrayComplexDoubleReverseColumnIterator& (NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexPlusEqual)(NdArrayComplexDoubleReverseColumnIterator::difference_type) = &NdArrayComplexDoubleReverseColumnIterator::operator+=;
    NdArrayComplexDoubleReverseColumnIterator (NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexPlus)(NdArrayComplexDoubleReverseColumnIterator::difference_type) const = &NdArrayComplexDoubleReverseColumnIterator::operator+;
    NdArrayComplexDoubleReverseColumnIterator& (NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexMinusEqual)(NdArrayComplexDoubleReverseColumnIterator::difference_type) = &NdArrayComplexDoubleReverseColumnIterator::operator-=;
    NdArrayComplexDoubleReverseColumnIterator(NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexMinus)(NdArrayComplexDoubleReverseColumnIterator::difference_type) const = &NdArrayComplexDoubleReverseColumnIterator::operator-;
    //NdArrayComplexDoubleReverseColumnIterator::difference_type(NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexDiff)(const NdArrayComplexDoubleReverseColumnIterator&) const = &NdArrayComplexDoubleReverseColumnIterator::operator-;
    //bool(NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexEqual)(const NdArrayComplexDoubleReverseColumnIterator&) const = &NdArrayComplexDoubleReverseColumnIterator::operator==;
    //bool(NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexNotEqual)(const NdArrayComplexDoubleReverseColumnIterator&) const = &NdArrayComplexDoubleReverseColumnIterator::operator!=;
    //bool(NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexLT)(const NdArrayComplexDoubleReverseColumnIterator&) const = &NdArrayComplexDoubleReverseColumnIterator::operator<;
    //bool(NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexLTE)(const NdArrayComplexDoubleReverseColumnIterator&) const = &NdArrayComplexDoubleReverseColumnIterator::operator<=;
    //bool(NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexGT)(const NdArrayComplexDoubleReverseColumnIterator&) const = &NdArrayComplexDoubleReverseColumnIterator::operator>;
    //bool(NdArrayComplexDoubleReverseColumnIterator::*reverseColumnIteratorOperatorComplexGTE)(const NdArrayComplexDoubleReverseColumnIterator&) const = &NdArrayComplexDoubleReverseColumnIterator::operator>=;

    bp::class_<NdArrayDoubleReverseColumnIterator>
        ("NdArrayDoubleReverseColumnIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleReverseColumnIterator>)
        .def("operatorPlusPlusPre", reverseColumnIteratorOperatorPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", reverseColumnIteratorOperatorPlusPlusPost)
        .def("operatorMinusMinusPre", reverseColumnIteratorOperatorMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", reverseColumnIteratorOperatorMinusMinusPost)
        .def("__iadd__", reverseColumnIteratorOperatorPlusEqual, bp::return_internal_reference<>())
        .def("__add__", reverseColumnIteratorOperatorPlus)
        .def("__isub__", reverseColumnIteratorOperatorMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", reverseColumnIteratorOperatorMinus)
        //.def("__sub__", reverseColumnIteratorOperatorComplexDiff)
        //.def("__eq__", reverseColumnIteratorOperatorComplexEqual)
        //.def("__ne__", reverseColumnIteratorOperatorComplexNotEqual)
        //.def("__lt__", reverseColumnIteratorOperatorComplexLT)
        //.def("__le__", reverseColumnIteratorOperatorComplexLTE)
        //.def("__gt__", reverseColumnIteratorOperatorComplexGT)
        //.def("__ge__", reverseColumnIteratorOperatorComplexGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleReverseColumnIterator>);

    bp::class_<NdArrayComplexDoubleReverseColumnIterator>
        ("NdArrayComplexDoubleReverseColumnIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleReverseColumnIterator>)
        .def("operatorPlusPlusPre", reverseColumnIteratorOperatorComplexPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", reverseColumnIteratorOperatorComplexPlusPlusPost)
        .def("operatorMinusMinusPre", reverseColumnIteratorOperatorComplexMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", reverseColumnIteratorOperatorComplexMinusMinusPost)
        .def("__iadd__", reverseColumnIteratorOperatorComplexPlusEqual, bp::return_internal_reference<>())
        .def("__add__", reverseColumnIteratorOperatorComplexPlus)
        .def("__isub__", reverseColumnIteratorOperatorComplexMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", reverseColumnIteratorOperatorComplexMinus)
        //.def("__sub__", reverseColumnIteratorOperatorComplexDiff)
        //.def("__eq__", reverseColumnIteratorOperatorComplexEqual)
        //.def("__ne__", reverseColumnIteratorOperatorComplexNotEqual)
        //.def("__lt__", reverseColumnIteratorOperatorComplexLT)
        //.def("__le__", reverseColumnIteratorOperatorComplexLTE)
        //.def("__gt__", reverseColumnIteratorOperatorComplexGT)
        //.def("__ge__", reverseColumnIteratorOperatorComplexGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleReverseColumnIterator>);

    NdArrayDoubleConstReverseColumnIterator& (NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorPlusPlusPre)() = &NdArrayDoubleConstReverseColumnIterator::operator++;
    NdArrayDoubleConstReverseColumnIterator (NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorPlusPlusPost)(int) = &NdArrayDoubleConstReverseColumnIterator::operator++;
    NdArrayDoubleConstReverseColumnIterator& (NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorMinusMinusPre)() = &NdArrayDoubleConstReverseColumnIterator::operator--;
    NdArrayDoubleConstReverseColumnIterator (NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorMinusMinusPost)(int) = &NdArrayDoubleConstReverseColumnIterator::operator--;
    NdArrayDoubleConstReverseColumnIterator& (NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorPlusEqual)(NdArrayDoubleConstReverseColumnIterator::difference_type) = &NdArrayDoubleConstReverseColumnIterator::operator+=;
    NdArrayDoubleConstReverseColumnIterator (NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorPlus)(NdArrayDoubleConstReverseColumnIterator::difference_type) const = &NdArrayDoubleConstReverseColumnIterator::operator+;
    NdArrayDoubleConstReverseColumnIterator& (NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorMinusEqual)(NdArrayDoubleConstReverseColumnIterator::difference_type) = &NdArrayDoubleConstReverseColumnIterator::operator-=;
    NdArrayDoubleConstReverseColumnIterator(NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorMinus)(NdArrayDoubleConstReverseColumnIterator::difference_type) const = &NdArrayDoubleConstReverseColumnIterator::operator-;
    //NdArrayDoubleConstReverseColumnIterator::difference_type(NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorDiff)(const NdArrayDoubleConstReverseColumnIterator&) const = &NdArrayDoubleConstReverseColumnIterator::operator-;
    //bool(NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorEqual)(const NdArrayDoubleConstReverseColumnIterator&) const = &NdArrayDoubleConstReverseColumnIterator::operator==;
    //bool(NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorNotEqual)(const NdArrayDoubleConstReverseColumnIterator&) const = &NdArrayDoubleConstReverseColumnIterator::operator!=;
    //bool(NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorLT)(const NdArrayDoubleConstReverseColumnIterator&) const = &NdArrayDoubleConstReverseColumnIterator::operator<;
    //bool(NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorLTE)(const NdArrayDoubleConstReverseColumnIterator&) const = &NdArrayDoubleConstReverseColumnIterator::operator<=;
    //bool(NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorGT)(const NdArrayDoubleConstReverseColumnIterator&) const = &NdArrayDoubleConstReverseColumnIterator::operator>;
    //bool(NdArrayDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorGTE)(const NdArrayDoubleConstReverseColumnIterator&) const = &NdArrayDoubleConstReverseColumnIterator::operator>=;

    NdArrayComplexDoubleConstReverseColumnIterator& (NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexPlusPlusPre)() = &NdArrayComplexDoubleConstReverseColumnIterator::operator++;
    NdArrayComplexDoubleConstReverseColumnIterator (NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexPlusPlusPost)(int) = &NdArrayComplexDoubleConstReverseColumnIterator::operator++;
    NdArrayComplexDoubleConstReverseColumnIterator& (NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexMinusMinusPre)() = &NdArrayComplexDoubleConstReverseColumnIterator::operator--;
    NdArrayComplexDoubleConstReverseColumnIterator (NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexMinusMinusPost)(int) = &NdArrayComplexDoubleConstReverseColumnIterator::operator--;
    NdArrayComplexDoubleConstReverseColumnIterator& (NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexPlusEqual)(NdArrayComplexDoubleConstReverseColumnIterator::difference_type) = &NdArrayComplexDoubleConstReverseColumnIterator::operator+=;
    NdArrayComplexDoubleConstReverseColumnIterator (NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexPlus)(NdArrayComplexDoubleConstReverseColumnIterator::difference_type) const = &NdArrayComplexDoubleConstReverseColumnIterator::operator+;
    NdArrayComplexDoubleConstReverseColumnIterator& (NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexMinusEqual)(NdArrayComplexDoubleConstReverseColumnIterator::difference_type) = &NdArrayComplexDoubleConstReverseColumnIterator::operator-=;
    NdArrayComplexDoubleConstReverseColumnIterator(NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexMinus)(NdArrayComplexDoubleConstReverseColumnIterator::difference_type) const = &NdArrayComplexDoubleConstReverseColumnIterator::operator-;
    //NdArrayComplexDoubleConstReverseColumnIterator::difference_type(NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexDiff)(const NdArrayComplexDoubleConstReverseColumnIterator&) const = &NdArrayComplexDoubleConstReverseColumnIterator::operator-;
    //bool(NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexEqual)(const NdArrayComplexDoubleConstReverseColumnIterator&) const = &NdArrayComplexDoubleConstReverseColumnIterator::operator==;
    //bool(NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexNotEqual)(const NdArrayComplexDoubleConstReverseColumnIterator&) const = &NdArrayComplexDoubleConstReverseColumnIterator::operator!=;
    //bool(NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexLT)(const NdArrayComplexDoubleConstReverseColumnIterator&) const = &NdArrayComplexDoubleConstReverseColumnIterator::operator<;
    //bool(NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexLTE)(const NdArrayComplexDoubleConstReverseColumnIterator&) const = &NdArrayComplexDoubleConstReverseColumnIterator::operator<=;
    //bool(NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexGT)(const NdArrayComplexDoubleConstReverseColumnIterator&) const = &NdArrayComplexDoubleConstReverseColumnIterator::operator>;
    //bool(NdArrayComplexDoubleConstReverseColumnIterator::*constReverseColumnIteratorOperatorComplexGTE)(const NdArrayComplexDoubleConstReverseColumnIterator&) const = &NdArrayComplexDoubleConstReverseColumnIterator::operator>=;

    bp::class_<NdArrayDoubleConstReverseColumnIterator>
        ("NdArrayDoubleConstReverseColumnIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleConstReverseColumnIterator>)
        .def("operatorPlusPlusPre", constReverseColumnIteratorOperatorPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", constReverseColumnIteratorOperatorPlusPlusPost)
        .def("operatorMinusMinusPre", constReverseColumnIteratorOperatorMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", constReverseColumnIteratorOperatorMinusMinusPost)
        .def("__iadd__", constReverseColumnIteratorOperatorPlusEqual, bp::return_internal_reference<>())
        .def("__add__", constReverseColumnIteratorOperatorPlus)
        .def("__isub__", constReverseColumnIteratorOperatorMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", constReverseColumnIteratorOperatorMinus)
        //.def("__sub__", constReverseColumnIteratorOperatorDiff)
        //.def("__eq__", constReverseColumnIteratorOperatorEqual)
        //.def("__ne__", constReverseColumnIteratorOperatorNotEqual)
        //.def("__lt__", constReverseColumnIteratorOperatorLT)
        //.def("__le__", constReverseColumnIteratorOperatorLTE)
        //.def("__gt__", constReverseColumnIteratorOperatorGT)
        //.def("__ge__", constReverseColumnIteratorOperatorGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleConstReverseColumnIterator>);

    bp::class_<NdArrayComplexDoubleConstReverseColumnIterator>
        ("NdArrayComplexDoubleConstReverseColumnIterator", bp::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("operatorPlusPlusPre", constReverseColumnIteratorOperatorComplexPlusPlusPre, bp::return_internal_reference<>())
        .def("operatorPlusPlusPost", constReverseColumnIteratorOperatorComplexPlusPlusPost)
        .def("operatorMinusMinusPre", constReverseColumnIteratorOperatorComplexMinusMinusPre, bp::return_internal_reference<>())
        .def("operatorMinusMinusPost", constReverseColumnIteratorOperatorComplexMinusMinusPost)
        .def("__iadd__", constReverseColumnIteratorOperatorComplexPlusEqual, bp::return_internal_reference<>())
        .def("__add__", constReverseColumnIteratorOperatorComplexPlus)
        .def("__isub__", constReverseColumnIteratorOperatorComplexMinusEqual, bp::return_internal_reference<>())
        .def("__sub__", constReverseColumnIteratorOperatorComplexMinus)
        //.def("__sub__", constReverseColumnIteratorOperatorComplexDiff)
        //.def("__eq__", constReverseColumnIteratorOperatorComplexEqual)
        //.def("__ne__", constReverseColumnIteratorOperatorComplexNotEqual)
        //.def("__lt__", constReverseColumnIteratorOperatorComplexLT)
        //.def("__le__", constReverseColumnIteratorOperatorComplexLTE)
        //.def("__gt__", constReverseColumnIteratorOperatorComplexGT)
        //.def("__ge__", constReverseColumnIteratorOperatorComplexGTE)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleConstReverseColumnIterator>);

    NdArrayDouble::reference(NdArrayDouble::*atSingleScaler)(int32) = &NdArrayDouble::at;
    NdArrayDouble::const_reference(NdArrayDouble::*atSingleScalerConst)(int32) const= &NdArrayDouble::at;
    NdArrayDouble::reference(NdArrayDouble::*atRowColScalers)(int32, int32) = &NdArrayDouble::at;
    NdArrayDouble::const_reference(NdArrayDouble::*atRowColScalersConst)(int32, int32) const = &NdArrayDouble::at;
    NdArrayDouble(NdArrayDouble::*atSlice)(const Slice&) const = &NdArrayDouble::at;
    NdArrayDouble(NdArrayDouble::*atSliceSlice)(const Slice&, const Slice&) const = &NdArrayDouble::at;
    NdArrayDouble(NdArrayDouble::*atSliceInt)(const Slice&, int32) const = &NdArrayDouble::at;
    NdArrayDouble(NdArrayDouble::*atIntSlice)(int32, const Slice&) const = &NdArrayDouble::at;

    NdArrayComplexDouble::reference(NdArrayComplexDouble::*atComplexSingleScaler)(int32) = &NdArrayComplexDouble::at;
    NdArrayComplexDouble::const_reference(NdArrayComplexDouble::*atComplexSingleScalerConst)(int32) const= &NdArrayComplexDouble::at;
    NdArrayComplexDouble::reference(NdArrayComplexDouble::*atComplexRowColScalers)(int32, int32) = &NdArrayComplexDouble::at;
    NdArrayComplexDouble::const_reference(NdArrayComplexDouble::*atComplexRowColScalersConst)(int32, int32) const = &NdArrayComplexDouble::at;
    NdArrayComplexDouble(NdArrayComplexDouble::*atComplexSlice)(const Slice&) const = &NdArrayComplexDouble::at;
    NdArrayComplexDouble(NdArrayComplexDouble::*atComplexSliceSlice)(const Slice&, const Slice&) const = &NdArrayComplexDouble::at;
    NdArrayComplexDouble(NdArrayComplexDouble::*atComplexSliceInt)(const Slice&, int32) const = &NdArrayComplexDouble::at;
    NdArrayComplexDouble(NdArrayComplexDouble::*atComplexIntSlice)(int32, const Slice&) const = &NdArrayComplexDouble::at;

    NdArrayDoubleIterator(NdArrayDouble::*begin)()  = &NdArrayDouble::begin;
    NdArrayDoubleIterator(NdArrayDouble::*beginRow)(NdArrayDouble::size_type) = &NdArrayDouble::begin;
    NdArrayDoubleConstIterator(NdArrayDouble::*beginConst)() const  = &NdArrayDouble::cbegin;
    NdArrayDoubleConstIterator(NdArrayDouble::*beginRowConst)(NdArrayDouble::size_type) const = &NdArrayDouble::cbegin;

    NdArrayComplexDoubleIterator(NdArrayComplexDouble::*beginComplex)()  = &NdArrayComplexDouble::begin;
    NdArrayComplexDoubleIterator(NdArrayComplexDouble::*beginRowComplex)(NdArrayComplexDouble::size_type) = &NdArrayComplexDouble::begin;
    NdArrayComplexDoubleConstIterator(NdArrayComplexDouble::*beginConstComplex)() const  = &NdArrayComplexDouble::cbegin;
    NdArrayComplexDoubleConstIterator(NdArrayComplexDouble::*beginRowConstComplex)(NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::cbegin;

    NdArrayDoubleColumnIterator(NdArrayDouble::*colbegin)()  = &NdArrayDouble::colbegin;
    NdArrayDoubleColumnIterator(NdArrayDouble::*colbeginCol)(NdArrayDouble::size_type) = &NdArrayDouble::colbegin;
    NdArrayDoubleConstColumnIterator(NdArrayDouble::*colbeginConst)() const  = &NdArrayDouble::ccolbegin;
    NdArrayDoubleConstColumnIterator(NdArrayDouble::*colbeginColConst)(NdArrayDouble::size_type) const = &NdArrayDouble::ccolbegin;

    NdArrayComplexDoubleColumnIterator(NdArrayComplexDouble::*colbeginComplex)()  = &NdArrayComplexDouble::colbegin;
    NdArrayComplexDoubleColumnIterator(NdArrayComplexDouble::*colbeginColComplex)(NdArrayComplexDouble::size_type) = &NdArrayComplexDouble::colbegin;
    NdArrayComplexDoubleConstColumnIterator(NdArrayComplexDouble::*colbeginConstComplex)() const  = &NdArrayComplexDouble::ccolbegin;
    NdArrayComplexDoubleConstColumnIterator(NdArrayComplexDouble::*colbeginColConstComplex)(NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::ccolbegin;

    NdArrayDoubleReverseIterator(NdArrayDouble::*rbegin)()  = &NdArrayDouble::rbegin;
    NdArrayDoubleReverseIterator(NdArrayDouble::*rbeginRow)(NdArrayDouble::size_type) = &NdArrayDouble::rbegin;
    NdArrayDoubleConstReverseIterator(NdArrayDouble::*rbeginConst)() const  = &NdArrayDouble::crbegin;
    NdArrayDoubleConstReverseIterator(NdArrayDouble::*rbeginRowConst)(NdArrayDouble::size_type) const = &NdArrayDouble::crbegin;

    NdArrayComplexDoubleReverseIterator(NdArrayComplexDouble::*rbeginComplex)()  = &NdArrayComplexDouble::rbegin;
    NdArrayComplexDoubleReverseIterator(NdArrayComplexDouble::*rbeginRowComplex)(NdArrayComplexDouble::size_type) = &NdArrayComplexDouble::rbegin;
    NdArrayComplexDoubleConstReverseIterator(NdArrayComplexDouble::*rbeginConstComplex)() const  = &NdArrayComplexDouble::crbegin;
    NdArrayComplexDoubleConstReverseIterator(NdArrayComplexDouble::*rbeginRowConstComplex)(NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::crbegin;

    NdArrayDoubleReverseColumnIterator(NdArrayDouble::*rcolbegin)()  = &NdArrayDouble::rcolbegin;
    NdArrayDoubleReverseColumnIterator(NdArrayDouble::*rcolbeginCol)(NdArrayDouble::size_type) = &NdArrayDouble::rcolbegin;
    NdArrayDoubleConstReverseColumnIterator(NdArrayDouble::*rcolbeginConst)() const  = &NdArrayDouble::crcolbegin;
    NdArrayDoubleConstReverseColumnIterator(NdArrayDouble::*rcolbeginColConst)(NdArrayDouble::size_type) const = &NdArrayDouble::crcolbegin;

    NdArrayComplexDoubleReverseColumnIterator(NdArrayComplexDouble::*rcolbeginComplex)()  = &NdArrayComplexDouble::rcolbegin;
    NdArrayComplexDoubleReverseColumnIterator(NdArrayComplexDouble::*rcolbeginColComplex)(NdArrayComplexDouble::size_type) = &NdArrayComplexDouble::rcolbegin;
    NdArrayComplexDoubleConstReverseColumnIterator(NdArrayComplexDouble::*rcolbeginConstComplex)() const  = &NdArrayComplexDouble::crcolbegin;
    NdArrayComplexDoubleConstReverseColumnIterator(NdArrayComplexDouble::*rcolbeginColConstComplex)(NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::crcolbegin;

    NdArrayDoubleIterator(NdArrayDouble::*end)()  = &NdArrayDouble::end;
    NdArrayDoubleIterator(NdArrayDouble::*endRow)(NdArrayDouble::size_type) = &NdArrayDouble::end;
    NdArrayDoubleConstIterator(NdArrayDouble::*endConst)() const  = &NdArrayDouble::cend;
    NdArrayDoubleConstIterator(NdArrayDouble::*endRowConst)(NdArrayDouble::size_type) const = &NdArrayDouble::cend;

    NdArrayComplexDoubleIterator(NdArrayComplexDouble::*endComplex)()  = &NdArrayComplexDouble::end;
    NdArrayComplexDoubleIterator(NdArrayComplexDouble::*endRowComplex)(NdArrayComplexDouble::size_type) = &NdArrayComplexDouble::end;
    NdArrayComplexDoubleConstIterator(NdArrayComplexDouble::*endConstComplex)() const  = &NdArrayComplexDouble::cend;
    NdArrayComplexDoubleConstIterator(NdArrayComplexDouble::*endRowConstComplex)(NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::cend;

    NdArrayDoubleColumnIterator(NdArrayDouble::*colend)()  = &NdArrayDouble::colend;
    NdArrayDoubleColumnIterator(NdArrayDouble::*colendCol)(NdArrayDouble::size_type) = &NdArrayDouble::colend;
    NdArrayDoubleConstColumnIterator(NdArrayDouble::*colendConst)() const  = &NdArrayDouble::ccolend;
    NdArrayDoubleConstColumnIterator(NdArrayDouble::*colendColConst)(NdArrayDouble::size_type) const = &NdArrayDouble::ccolend;

    NdArrayComplexDoubleColumnIterator(NdArrayComplexDouble::*colendComplex)()  = &NdArrayComplexDouble::colend;
    NdArrayComplexDoubleColumnIterator(NdArrayComplexDouble::*colendColComplex)(NdArrayComplexDouble::size_type) = &NdArrayComplexDouble::colend;
    NdArrayComplexDoubleConstColumnIterator(NdArrayComplexDouble::*colendConstComplex)() const  = &NdArrayComplexDouble::ccolend;
    NdArrayComplexDoubleConstColumnIterator(NdArrayComplexDouble::*colendColConstComplex)(NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::ccolend;

    NdArrayDoubleReverseIterator(NdArrayDouble::*rend)()  = &NdArrayDouble::rend;
    NdArrayDoubleReverseIterator(NdArrayDouble::*rendRow)(NdArrayDouble::size_type) = &NdArrayDouble::rend;
    NdArrayDoubleConstReverseIterator(NdArrayDouble::*rendConst)() const  = &NdArrayDouble::crend;
    NdArrayDoubleConstReverseIterator(NdArrayDouble::*rendRowConst)(NdArrayDouble::size_type) const = &NdArrayDouble::crend;

    NdArrayComplexDoubleReverseIterator(NdArrayComplexDouble::*rendComplex)()  = &NdArrayComplexDouble::rend;
    NdArrayComplexDoubleReverseIterator(NdArrayComplexDouble::*rendRowComplex)(NdArrayComplexDouble::size_type) = &NdArrayComplexDouble::rend;
    NdArrayComplexDoubleConstReverseIterator(NdArrayComplexDouble::*rendConstComplex)() const  = &NdArrayComplexDouble::crend;
    NdArrayComplexDoubleConstReverseIterator(NdArrayComplexDouble::*rendRowConstComplex)(NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::crend;

    NdArrayDoubleReverseColumnIterator(NdArrayDouble::*rcolend)()  = &NdArrayDouble::rcolend;
    NdArrayDoubleReverseColumnIterator(NdArrayDouble::*rcolendCol)(NdArrayDouble::size_type) = &NdArrayDouble::rcolend;
    NdArrayDoubleConstReverseColumnIterator(NdArrayDouble::*rcolendConst)() const  = &NdArrayDouble::crcolend;
    NdArrayDoubleConstReverseColumnIterator(NdArrayDouble::*rcolendColConst)(NdArrayDouble::size_type) const = &NdArrayDouble::crcolend;

    NdArrayComplexDoubleReverseColumnIterator(NdArrayComplexDouble::*rcolendComplex)()  = &NdArrayComplexDouble::rcolend;
    NdArrayComplexDoubleReverseColumnIterator(NdArrayComplexDouble::*rcolendColComplex)(NdArrayComplexDouble::size_type) = &NdArrayComplexDouble::rcolend;
    NdArrayComplexDoubleConstReverseColumnIterator(NdArrayComplexDouble::*rcolendConstComplex)() const  = &NdArrayComplexDouble::crcolend;
    NdArrayComplexDoubleConstReverseColumnIterator(NdArrayComplexDouble::*rcolendColConstComplex)(NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::crcolend;

    bp::def("test1DListContructor", &NdArrayInterface::test1DListContructor<double>);
    bp::def("test1DListContructor", &NdArrayInterface::test1DListContructor<ComplexDouble>);
    bp::def("test2DListContructor", &NdArrayInterface::test2DListContructor<double>);
    bp::def("test2DListContructor", &NdArrayInterface::test2DListContructor<ComplexDouble>);
    bp::def("test1dArrayConstructor", &NdArrayInterface::test1dArrayConstructor<double>);
    bp::def("test1dArrayConstructor", &NdArrayInterface::test1dArrayConstructor<ComplexDouble>);
    bp::def("test2dArrayConstructor", &NdArrayInterface::test2dArrayConstructor<double>);
    bp::def("test2dArrayConstructor", &NdArrayInterface::test2dArrayConstructor<ComplexDouble>);
    bp::def("test1dVectorConstructor", &NdArrayInterface::test1dVectorConstructor<double>);
    bp::def("test1dVectorConstructor", &NdArrayInterface::test1dVectorConstructor<ComplexDouble>);
    bp::def("test2dVectorConstructor", &NdArrayInterface::test2dVectorConstructor<double>);
    bp::def("test2dVectorConstructor", &NdArrayInterface::test2dVectorConstructor<ComplexDouble>);
    bp::def("test2dVectorArrayConstructor", &NdArrayInterface::test2dVectorArrayConstructor<double>);
    bp::def("test2dVectorArrayConstructor", &NdArrayInterface::test2dVectorArrayConstructor<ComplexDouble>);
    bp::def("test1dDequeConstructor", &NdArrayInterface::test1dDequeConstructor<double>);
    bp::def("test1dDequeConstructor", &NdArrayInterface::test1dDequeConstructor<ComplexDouble>);
    bp::def("test2dDequeConstructor", &NdArrayInterface::test2dDequeConstructor<double>);
    bp::def("test2dDequeConstructor", &NdArrayInterface::test2dDequeConstructor<ComplexDouble>);
    bp::def("test1dListConstructor", &NdArrayInterface::test1dListConstructor<double>);
    bp::def("test1dListConstructor", &NdArrayInterface::test1dListConstructor<ComplexDouble>);
    bp::def("test1dIteratorConstructor", &NdArrayInterface::test1dIteratorConstructor<double>);
    bp::def("test1dIteratorConstructor", &NdArrayInterface::test1dIteratorConstructor<ComplexDouble>);
    bp::def("test1dIteratorConstructor2", &NdArrayInterface::test1dIteratorConstructor<double>);
    bp::def("test1dIteratorConstructor2", &NdArrayInterface::test1dIteratorConstructor<ComplexDouble>);
    bp::def("test1dPointerConstructor", &NdArrayInterface::test1dPointerConstructor<double>);
    bp::def("test1dPointerConstructor", &NdArrayInterface::test1dPointerConstructor<ComplexDouble>);
    bp::def("test2dPointerConstructor", &NdArrayInterface::test2dPointerConstructor<double>);
    bp::def("test2dPointerConstructor", &NdArrayInterface::test2dPointerConstructor<ComplexDouble>);
    bp::def("test1dPointerShellConstructor", &NdArrayInterface::test1dPointerShellConstructor<double>);
    bp::def("test1dPointerShellConstructor", &NdArrayInterface::test1dPointerShellConstructor<ComplexDouble>);
    bp::def("test2dPointerShellConstructor", &NdArrayInterface::test2dPointerShellConstructor<double>);
    bp::def("test2dPointerShellConstructor", &NdArrayInterface::test2dPointerShellConstructor<ComplexDouble>);
    bp::def("testCopyConstructor", &NdArrayInterface::testCopyConstructor<double>);
    bp::def("testCopyConstructor", &NdArrayInterface::testCopyConstructor<ComplexDouble>);
    bp::def("testMoveConstructor", &NdArrayInterface::testMoveConstructor<double>);
    bp::def("testMoveConstructor", &NdArrayInterface::testMoveConstructor<ComplexDouble>);
    bp::def("testAssignementOperator", &NdArrayInterface::testAssignementOperator<double>);
    bp::def("testAssignementOperator", &NdArrayInterface::testAssignementOperator<ComplexDouble>);
    bp::def("testAssignementScalerOperator", &NdArrayInterface::testAssignementScalerOperator<double>);
    bp::def("testAssignementScalerOperator", &NdArrayInterface::testAssignementScalerOperator<ComplexDouble>);
    bp::def("testMoveAssignementOperator", &NdArrayInterface::testMoveAssignementOperator<double>);
    bp::def("testMoveAssignementOperator", &NdArrayInterface::testMoveAssignementOperator<ComplexDouble>);

    bp::class_<NdArrayDouble>
        ("NdArray", bp::init<>())
        .def(bp::init<NdArrayDouble::size_type>())
        .def(bp::init<NdArrayDouble::size_type, NdArrayDouble::size_type>())
        .def(bp::init<Shape>())
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<double>)
        .def("setArray", &NdArrayInterface::setArray<double>)
        .def("rSlice", &NdArrayDouble::rSlice)
        .def("cSlice", &NdArrayDouble::cSlice)
        .def("get", &NdArrayInterface::getValueFlat<double>)
        .def("getConst", &NdArrayInterface::getValueFlatConst<double>)
        .def("get", &NdArrayInterface::getValueRowCol<double>)
        .def("getConst", &NdArrayInterface::getValueRowColConst<double>)
        .def("get", &NdArrayInterface::getMask<double>)
        .def("get", &NdArrayInterface::getIndices<double>)
        .def("get", &NdArrayInterface::getSlice1D<double>)
        .def("get", &NdArrayInterface::getSlice2D<double>)
        .def("get", &NdArrayInterface::getSlice2DRow<double>)
        .def("get", &NdArrayInterface::getSlice2DCol<double>)
        .def("at", atSingleScaler, bp::return_value_policy<bp::return_by_value>())
        .def("atConst", atSingleScalerConst, bp::return_value_policy<bp::return_by_value>())
        .def("at", atRowColScalers, bp::return_value_policy<bp::return_by_value>())
        .def("atConst", atRowColScalersConst, bp::return_value_policy<bp::return_by_value>())
        .def("at", atSlice)
        .def("at", atSliceSlice)
        .def("at", atSliceInt)
        .def("at", atIntSlice)
        .def("begin", begin)
        .def("begin", beginRow)
        .def("colbegin", colbegin)
        .def("colbegin", colbeginCol)
        .def("beginConst", beginConst)
        .def("beginConst", beginRowConst)
        .def("colbeginConst", colbeginConst)
        .def("colbeginConst", colbeginColConst)
        .def("rbegin", rbegin)
        .def("rbegin", rbeginRow)
        .def("rcolbegin", rcolbegin)
        .def("rcolbegin", rcolbeginCol)
        .def("rbeginConst", rbeginConst)
        .def("rbeginConst", rbeginRowConst)
        .def("rcolbeginConst", rcolbeginConst)
        .def("rcolbeginConst", rcolbeginColConst)
        .def("end", end)
        .def("end", endRow)
        .def("colend", colend)
        .def("colend", colendCol)
        .def("endConst", endConst)
        .def("endConst", endRowConst)
        .def("colendConst", colendConst)
        .def("colendConst", colendColConst)
        .def("rend", rend)
        .def("rend", rendRow)
        .def("rcolend", rcolend)
        .def("rcolend", rcolendCol)
        .def("rendConst", rendConst)
        .def("rendConst", rendRowConst)
        .def("rcolendConst", rcolendConst)
        .def("rcolendConst", rcolendColConst)
        .def("all", &NdArrayInterface::all<double>)
        .def("any", &NdArrayInterface::any<double>)
        .def("argmax", &NdArrayInterface::argmax<double>)
        .def("argmin", &NdArrayInterface::argmin<double>)
        .def("argsort", &NdArrayInterface::argsort<double>)
        .def("astypeUint32", &NdArrayDouble::astype<uint32>)
        .def("astypeComplex", &NdArrayDouble::astype<ComplexDouble>)
        .def("back", &NdArrayInterface::back<double>)
        .def("backReference", &NdArrayInterface::backReference<double>)
        .def("back", &NdArrayInterface::backRow<double>)
        .def("backReference", &NdArrayInterface::backRowReference<double>)
        .def("clip", &NdArrayInterface::clip<double>)
        .def("column", &NdArrayDouble::column)
        .def("contains", &NdArrayInterface::contains<double>)
        .def("copy", &NdArrayInterface::copy<double>)
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
        .def("front", &NdArrayInterface::frontRow<double>)
        .def("frontReference", &NdArrayInterface::frontRowReference<double>)
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
        .def("numRows", &NdArrayDouble::numRows)
        .def("numCols", &NdArrayDouble::numCols)
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
    bp::def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualArray<ComplexDouble>); // (1)
    bp::def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualComplexArrayArithArray<double>); // (2)
    bp::def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualScaler<double>); // (3)
    bp::def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualScaler<ComplexDouble>); // (3)
    bp::def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualComplexArrayArithScaler<double>); // (4)

    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArray<double>); // (1)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArray<ComplexDouble>); // (1)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArithArrayComplexArray<double>); // (2)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusComplexArrayArithArray<double>); // (3)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArrayScaler<double>); // (4)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArrayScaler<ComplexDouble>); // (4)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusScalerArray<double>); // (5)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusScalerArray<ComplexDouble>); // (5)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArithArrayComplexScaler<double>); // (6)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusComplexScalerArithArray<double>); // (7)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusComplexArrayArithScaler<double>); // (8)
    bp::def("operatorPlus", &NdArrayInterface::operatorPlusArithScalerComplexArray<double>); // (9)

    bp::def("operatorNegative", &NdArrayInterface::operatorNegative<double>);
    bp::def("operatorNegative", &NdArrayInterface::operatorNegative<ComplexDouble>);

    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualArray<double>); // (1)
    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualArray<ComplexDouble>); // (1)
    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualComplexArrayArithArray<double>); // (2)
    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualScaler<double>); // (3)
    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualScaler<ComplexDouble>); // (3)
    bp::def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualComplexArrayArithScaler<double>); // (4)

    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArray<double>); // (1)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArray<ComplexDouble>); // (1)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArithArrayComplexArray<double>); // (2)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusComplexArrayArithArray<double>); // (3)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArrayScaler<double>); // (4)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArrayScaler<ComplexDouble>); // (4)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusScalerArray<double>); // (5)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusScalerArray<ComplexDouble>); // (5)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArithArrayComplexScaler<double>); // (6)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusComplexScalerArithArray<double>); // (7)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusComplexArrayArithScaler<double>); // (8)
    bp::def("operatorMinus", &NdArrayInterface::operatorMinusArithScalerComplexArray<double>); // (9)

    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualArray<double>); // (1)
    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualArray<ComplexDouble>); // (1)
    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualComplexArrayArithArray<double>); // (2)
    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualScaler<double>); // (3)
    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualScaler<ComplexDouble>); // (3)
    bp::def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualComplexArrayArithScaler<double>); // (4)

    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArray<double>); // (1)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArray<ComplexDouble>); // (1)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithArrayComplexArray<double>); // (2)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexArrayArithArray<double>); // (3)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArrayScaler<double>); // (4)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArrayScaler<ComplexDouble>); // (4)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyScalerArray<double>); // (5)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyScalerArray<ComplexDouble>); // (5)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithArrayComplexScaler<double>); // (6)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexScalerArithArray<double>); // (7)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexArrayArithScaler<double>); // (8)
    bp::def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithScalerComplexArray<double>); // (9)

    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualArray<double>); // (1)
    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualArray<ComplexDouble>); // (1)
    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualComplexArrayArithArray<double>); // (2)
    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualScaler<double>); // (3)
    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualScaler<ComplexDouble>); // (3)
    bp::def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualComplexArrayArithScaler<double>); // (4)

    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArray<double>); // (1)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArray<ComplexDouble>); // (1)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArithArrayComplexArray<double>); // (2)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideComplexArrayArithArray<double>); // (3)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArrayScaler<double>); // (4)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArrayScaler<ComplexDouble>); // (4)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideScalerArray<double>); // (5)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideScalerArray<ComplexDouble>); // (5)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArithArrayComplexScaler<double>); // (6)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideComplexScalerArithArray<double>); // (7)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideComplexArrayArithScaler<double>); // (8)
    bp::def("operatorDivide", &NdArrayInterface::operatorDivideArithScalerComplexArray<double>); // (9)

    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityScaler<double>);
    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityScaler<ComplexDouble>);
    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityScalerReversed<double>);
    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityScalerReversed<ComplexDouble>);
    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityArray<double>);
    bp::def("operatorEquality", &NdArrayInterface::operatorEqualityArray<ComplexDouble>);

    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScaler<double>);
    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScaler<ComplexDouble>);
    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScalerReversed<double>);
    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScalerReversed<ComplexDouble>);
    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityArray<double>);
    bp::def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityArray<ComplexDouble>);

    bp::def("operatorLess", &NdArrayInterface::operatorLessScaler<double>);
    bp::def("operatorLess", &NdArrayInterface::operatorLessScaler<ComplexDouble>);
    bp::def("operatorLess", &NdArrayInterface::operatorLessScalerReversed<double>);
    bp::def("operatorLess", &NdArrayInterface::operatorLessScalerReversed<ComplexDouble>);
    bp::def("operatorLess", &NdArrayInterface::operatorLessArray<double>);
    bp::def("operatorLess", &NdArrayInterface::operatorLessArray<ComplexDouble>);

    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterScaler<double>);
    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterScaler<ComplexDouble>);
    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterScalerReversed<double>);
    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterScalerReversed<ComplexDouble>);
    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterArray<double>);
    bp::def("operatorGreater", &NdArrayInterface::operatorGreaterArray<ComplexDouble>);

    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScaler<double>);
    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScaler<ComplexDouble>);
    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScalerReversed<double>);
    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScalerReversed<ComplexDouble>);
    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualArray<double>);
    bp::def("operatorLessEqual", &NdArrayInterface::operatorLessEqualArray<ComplexDouble>);

    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScaler<double>);
    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScaler<ComplexDouble>);
    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScalerReversed<double>);
    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScalerReversed<ComplexDouble>);
    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualArray<double>);
    bp::def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualArray<ComplexDouble>);

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

    typedef NdArray<std::complex<uint32>> NdArrayComplexUint32;
    bp::class_<NdArrayComplexUint32>
        ("NdArrayComplexUint32", bp::init<>())
        .def(bp::init<uint32>())
        .def(bp::init<uint32, uint32>())
        .def(bp::init<Shape>())
        .def("item", &NdArrayComplexUint32::item)
        .def("shape", &NdArrayComplexUint32::shape)
        .def("size", &NdArrayComplexUint32::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<std::complex<uint32>>)
        .def("setArray", NdArrayInterface::setArray<std::complex<uint32>>);

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
        .def("get", &NdArrayInterface::getValueFlat<ComplexDouble>)
        .def("getConst", &NdArrayInterface::getValueFlatConst<ComplexDouble>)
        .def("get", &NdArrayInterface::getValueRowCol<ComplexDouble>)
        .def("getConst", &NdArrayInterface::getValueRowColConst<ComplexDouble>)
        .def("get", &NdArrayInterface::getMask<ComplexDouble>)
        .def("get", &NdArrayInterface::getIndices<ComplexDouble>)
        .def("get", &NdArrayInterface::getSlice1D<ComplexDouble>)
        .def("get", &NdArrayInterface::getSlice2D<ComplexDouble>)
        .def("get", &NdArrayInterface::getSlice2DRow<ComplexDouble>)
        .def("get", &NdArrayInterface::getSlice2DCol<ComplexDouble>)
        .def("at", atComplexSingleScaler, bp::return_value_policy<bp::return_by_value>())
        .def("atConst", atComplexSingleScalerConst, bp::return_value_policy<bp::return_by_value>())
        .def("at", atComplexRowColScalers, bp::return_value_policy<bp::return_by_value>())
        .def("atConst", atComplexRowColScalersConst, bp::return_value_policy<bp::return_by_value>())
        .def("at", atComplexSlice)
        .def("at", atComplexSliceSlice)
        .def("at", atComplexSliceInt)
        .def("at", atComplexIntSlice)
        .def("begin", beginComplex)
        .def("begin", beginRowComplex)
        .def("colbegin", colbeginComplex)
        .def("colbegin", colbeginColComplex)
        .def("beginConst", beginConstComplex)
        .def("beginConst", beginRowConstComplex)
        .def("colbeginConst", colbeginConstComplex)
        .def("colbeginConst", colbeginColConstComplex)
        .def("rbegin", rbeginComplex)
        .def("rbegin", rbeginRowComplex)
        .def("rcolbegin", rcolbeginComplex)
        .def("rcolbegin", rcolbeginColComplex)
        .def("rbeginConst", rbeginConstComplex)
        .def("rbeginConst", rbeginRowConstComplex)
        .def("rcolbeginConst", rcolbeginConstComplex)
        .def("rcolbeginConst", rcolbeginColConstComplex)
        .def("end", endComplex)
        .def("end", endRowComplex)
        .def("colend", colendComplex)
        .def("colend", colendColComplex)
        .def("endConst", endConstComplex)
        .def("endConst", endRowConstComplex)
        .def("colendConst", colendConstComplex)
        .def("colendConst", colendColConstComplex)
        .def("rend", rendComplex)
        .def("rend", rendRowComplex)
        .def("rcolend", rcolendComplex)
        .def("rcolend", rcolendColComplex)
        .def("rendConst", rendConstComplex)
        .def("rendConst", rendRowConstComplex)
        .def("rcolendConst", rcolendConstComplex)
        .def("rcolendConst", rcolendColConstComplex)
        .def("rSlice", &NdArrayComplexDouble::rSlice)
        .def("cSlice", &NdArrayComplexDouble::cSlice)
        .def("all", &NdArrayInterface::all<ComplexDouble>)
        .def("any", &NdArrayInterface::any<ComplexDouble>)
        .def("argmax", &NdArrayInterface::argmax<ComplexDouble>)
        .def("argmin", &NdArrayInterface::argmin<ComplexDouble>)
        .def("argsort", &NdArrayInterface::argsort<ComplexDouble>)
        .def("astypeDouble", &NdArrayComplexDouble::astype<double>)
        .def("astypeComplexFloat", &NdArrayComplexDouble::astype<std::complex<float>>)
        .def("back", &NdArrayInterface::back<ComplexDouble>)
        .def("backReference", &NdArrayInterface::backReference<ComplexDouble>)
        .def("back", &NdArrayInterface::backRow<ComplexDouble>)
        .def("backReference", &NdArrayInterface::backRowReference<ComplexDouble>)
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
        .def("front", &NdArrayInterface::frontRow<ComplexDouble>)
        .def("frontReference", &NdArrayInterface::frontRowReference<ComplexDouble>)
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

    bp::def("testStructuredArray", &NdArrayInterface::testStructuredArray);

    // Functions.hpp
    bp::def("absScaler", &FunctionsInterface::absScaler<double>);
    bp::def("absArray", &FunctionsInterface::absArray<double>);
    bp::def("absScaler", &FunctionsInterface::absScaler<ComplexDouble>);
    bp::def("absArray", &FunctionsInterface::absArray<ComplexDouble>);
    bp::def("add", &FunctionsInterface::add<NdArray<double>, NdArray<double>>);
    bp::def("add", &FunctionsInterface::add<NdArray<double>, double>);
    bp::def("add", &FunctionsInterface::add<double, NdArray<double>>);
    bp::def("add", &FunctionsInterface::add<NdArray<ComplexDouble>, NdArray<ComplexDouble>>);
    bp::def("add", &FunctionsInterface::add<NdArray<ComplexDouble>, ComplexDouble>);
    bp::def("add", &FunctionsInterface::add<ComplexDouble, NdArray<ComplexDouble>>);
    bp::def("add", &FunctionsInterface::add<NdArray<double>, NdArray<ComplexDouble>>);
    bp::def("add", &FunctionsInterface::add<NdArray<ComplexDouble>, NdArray<double>>);
    bp::def("add", &FunctionsInterface::add<NdArray<double>, ComplexDouble>);
    bp::def("add", &FunctionsInterface::add<ComplexDouble, NdArray<double>>);
    bp::def("add", &FunctionsInterface::add<NdArray<ComplexDouble>, double>);
    bp::def("add", &FunctionsInterface::add<double, NdArray<ComplexDouble>>);
    bp::def("alen", &alen<double>);
    bp::def("all", &FunctionsInterface::allArray<double>);
    bp::def("all", &FunctionsInterface::allArray<ComplexDouble>);
    bp::def("allclose", &allclose<double>);
    bp::def("amax", &FunctionsInterface::amaxArray<double>);
    bp::def("amax", &FunctionsInterface::amaxArray<ComplexDouble>);
    bp::def("amin", &FunctionsInterface::aminArray<double>);
    bp::def("amin", &FunctionsInterface::aminArray<ComplexDouble>);
    bp::def("angleScaler", &FunctionsInterface::angleScaler<double>);
    bp::def("angleArray", &FunctionsInterface::angleArray<double>);
    bp::def("any", &FunctionsInterface::anyArray<double>);
    bp::def("any", &FunctionsInterface::anyArray<ComplexDouble>);
    bp::def("append", &append<double>);
    bp::def("applyPoly1d", &applyPoly1d<double>);
    bp::def("arange", &FunctionsInterface::arangeArray<double>);
    bp::def("arccosScaler", &FunctionsInterface::arccosScaler<double>);
    bp::def("arccosArray", &FunctionsInterface::arccosArray<double>);
    bp::def("arccosScaler", &FunctionsInterface::arccosScaler<ComplexDouble>);
    bp::def("arccosArray", &FunctionsInterface::arccosArray<ComplexDouble>);
    bp::def("arccoshScaler", &FunctionsInterface::arccoshScaler<double>);
    bp::def("arccoshArray", &FunctionsInterface::arccoshArray<double>);
    bp::def("arccoshScaler", &FunctionsInterface::arccoshScaler<ComplexDouble>);
    bp::def("arccoshArray", &FunctionsInterface::arccoshArray<ComplexDouble>);
    bp::def("arcsinScaler", &FunctionsInterface::arcsinScaler<double>);
    bp::def("arcsinArray", &FunctionsInterface::arcsinArray<double>);
    bp::def("arcsinScaler", &FunctionsInterface::arcsinScaler<ComplexDouble>);
    bp::def("arcsinArray", &FunctionsInterface::arcsinArray<ComplexDouble>);
    bp::def("arcsinhScaler", &FunctionsInterface::arcsinhScaler<double>);
    bp::def("arcsinhArray", &FunctionsInterface::arcsinhArray<double>);
    bp::def("arcsinhScaler", &FunctionsInterface::arcsinhScaler<ComplexDouble>);
    bp::def("arcsinhArray", &FunctionsInterface::arcsinhArray<ComplexDouble>);
    bp::def("arctanScaler", &FunctionsInterface::arctanScaler<double>);
    bp::def("arctanArray", &FunctionsInterface::arctanArray<double>);
    bp::def("arctanScaler", &FunctionsInterface::arctanScaler<ComplexDouble>);
    bp::def("arctanArray", &FunctionsInterface::arctanArray<ComplexDouble>);
    bp::def("arctan2Scaler", &FunctionsInterface::arctan2Scaler<double>);
    bp::def("arctan2Array", &FunctionsInterface::arctan2Array<double>);
    bp::def("arctanhScaler", &FunctionsInterface::arctanhScaler<double>);
    bp::def("arctanhArray", &FunctionsInterface::arctanhArray<double>);
    bp::def("arctanhScaler", &FunctionsInterface::arctanhScaler<ComplexDouble>);
    bp::def("arctanhArray", &FunctionsInterface::arctanhArray<ComplexDouble>);
    bp::def("argmax", &FunctionsInterface::argmaxArray<double>);
    bp::def("argmax", &FunctionsInterface::argmaxArray<ComplexDouble>);
    bp::def("argmin", &FunctionsInterface::argminArray<double>);
    bp::def("argmin", &FunctionsInterface::argminArray<ComplexDouble>);
    bp::def("argsort", &FunctionsInterface::argsortArray<double>);
    bp::def("argsort", &FunctionsInterface::argsortArray<ComplexDouble>);
    bp::def("argwhere", &FunctionsInterface::argwhere<double>);
    bp::def("argwhere", &FunctionsInterface::argwhere<ComplexDouble>);
    bp::def("aroundScaler", &FunctionsInterface::aroundScaler<double>);
    bp::def("aroundArray", &FunctionsInterface::aroundArray<double>);
    bp::def("array_equal", &array_equal<double>);
    bp::def("array_equal", &array_equal<ComplexDouble>);
    bp::def("array_equiv", &array_equiv<double>);
    bp::def("array_equiv", &array_equiv<ComplexDouble>);
    bp::def("asarrayInitializerList", &FunctionsInterface::asarrayInitializerList<double>);
    bp::def("asarrayInitializerList", &FunctionsInterface::asarrayInitializerList<ComplexDouble>);
    bp::def("asarrayInitializerList2D", &FunctionsInterface::asarrayInitializerList2D<double>);
    bp::def("asarrayInitializerList2D", &FunctionsInterface::asarrayInitializerList2D<ComplexDouble>);
    bp::def("asarrayArray1D", &FunctionsInterface::asarrayArray1D<double>);
    bp::def("asarrayArray1D", &FunctionsInterface::asarrayArray1D<ComplexDouble>);
    bp::def("asarrayArray1DCopy", &FunctionsInterface::asarrayArray1DCopy<double>);
    bp::def("asarrayArray1DCopy", &FunctionsInterface::asarrayArray1DCopy<ComplexDouble>);
    bp::def("asarrayArray2D", &FunctionsInterface::asarrayArray2D<double>);
    bp::def("asarrayArray2D", &FunctionsInterface::asarrayArray2D<ComplexDouble>);
    bp::def("asarrayArray2DCopy", &FunctionsInterface::asarrayArray2DCopy<double>);
    bp::def("asarrayArray2DCopy", &FunctionsInterface::asarrayArray2DCopy<ComplexDouble>);
    bp::def("asarrayVector1D", &FunctionsInterface::asarrayVector1D<double>);
    bp::def("asarrayVector1D", &FunctionsInterface::asarrayVector1D<ComplexDouble>);
    bp::def("asarrayVector1DCopy", &FunctionsInterface::asarrayVector1DCopy<double>);
    bp::def("asarrayVector1DCopy", &FunctionsInterface::asarrayVector1DCopy<ComplexDouble>);
    bp::def("asarrayVector2D", &FunctionsInterface::asarrayVector2D<double>);
    bp::def("asarrayVector2D", &FunctionsInterface::asarrayVector2D<ComplexDouble>);
    bp::def("asarrayVectorArray2D", &FunctionsInterface::asarrayVectorArray2D<double>);
    bp::def("asarrayVectorArray2D", &FunctionsInterface::asarrayVectorArray2D<ComplexDouble>);
    bp::def("asarrayVectorArray2DCopy", &FunctionsInterface::asarrayVectorArray2DCopy<double>);
    bp::def("asarrayVectorArray2DCopy", &FunctionsInterface::asarrayVectorArray2DCopy<ComplexDouble>);
    bp::def("asarrayDeque1D", &FunctionsInterface::asarrayDeque1D<double>);
    bp::def("asarrayDeque1D", &FunctionsInterface::asarrayDeque1D<ComplexDouble>);
    bp::def("asarrayDeque2D", &FunctionsInterface::asarrayDeque2D<double>);
    bp::def("asarrayDeque2D", &FunctionsInterface::asarrayDeque2D<ComplexDouble>);
    bp::def("asarrayList", &FunctionsInterface::asarrayList<double>);
    bp::def("asarrayList", &FunctionsInterface::asarrayList<ComplexDouble>);
    bp::def("asarrayIterators", &FunctionsInterface::asarrayIterators<double>);
    bp::def("asarrayIterators", &FunctionsInterface::asarrayIterators<ComplexDouble>);
    bp::def("asarrayPointerIterators", &FunctionsInterface::asarrayPointerIterators<double>);
    bp::def("asarrayPointerIterators", &FunctionsInterface::asarrayPointerIterators<ComplexDouble>);
    bp::def("asarrayPointer", &FunctionsInterface::asarrayPointer<double>);
    bp::def("asarrayPointer", &FunctionsInterface::asarrayPointer<ComplexDouble>);
    bp::def("asarrayPointer2D", &FunctionsInterface::asarrayPointer2D<double>);
    bp::def("asarrayPointer2D", &FunctionsInterface::asarrayPointer2D<ComplexDouble>);
    bp::def("asarrayPointerShell", &FunctionsInterface::asarrayPointerShell<double>);
    bp::def("asarrayPointerShell", &FunctionsInterface::asarrayPointerShell<ComplexDouble>);
    bp::def("asarrayPointerShell2D", &FunctionsInterface::asarrayPointerShell2D<double>);
    bp::def("asarrayPointerShell2D", &FunctionsInterface::asarrayPointerShell2D<ComplexDouble>);
    bp::def("asarrayPointerShellTakeOwnership", &FunctionsInterface::asarrayPointerShellTakeOwnership<double>);
    bp::def("asarrayPointerShellTakeOwnership", &FunctionsInterface::asarrayPointerShellTakeOwnership<ComplexDouble>);
    bp::def("asarrayPointerShell2DTakeOwnership", &FunctionsInterface::asarrayPointerShell2DTakeOwnership<double>);
    bp::def("asarrayPointerShell2DTakeOwnership", &FunctionsInterface::asarrayPointerShell2DTakeOwnership<ComplexDouble>);
    bp::def("astypeDoubleToUint32", &astype<uint32, double>);
    bp::def("astypeDoubleToComplex", &astype<ComplexDouble, double>);
    bp::def("astypeComplexToComplex", &astype<std::complex<float>, ComplexDouble>);
    bp::def("astypeComplexToDouble", &astype<double, ComplexDouble>);
    bp::def("average", &FunctionsInterface::average<double>);
    bp::def("average", &FunctionsInterface::average<ComplexDouble>);
    bp::def("averageWeighted", &FunctionsInterface::averageWeighted<double>);
    bp::def("averageWeighted", &FunctionsInterface::averageWeightedComplex<double>);

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
    bp::def("clipScaler", &FunctionsInterface::clipScaler<ComplexDouble>);
    bp::def("clipArray", &FunctionsInterface::clipArray<double>);
    bp::def("clipArray", &FunctionsInterface::clipArray<ComplexDouble>);
    bp::def("column_stack", &FunctionsInterface::column_stack<double>);
    bp::def("complexScaler", &FunctionsInterface::complexScalerSingle<double>);
    bp::def("complexScaler", &FunctionsInterface::complexScaler<double>);
    bp::def("complexArray", &FunctionsInterface::complexArraySingle<double>);
    bp::def("complexArray", &FunctionsInterface::complexArray<double>);
    bp::def("conjScaler", &FunctionsInterface::conjScaler<double>);
    bp::def("conjArray", &FunctionsInterface::conjArray<double>);
    bp::def("concatenate", &FunctionsInterface::concatenate<double>);
    bp::def("contains", &contains<double>);
    bp::def("contains", &contains<ComplexDouble>);
    bp::def("copy", &FunctionsInterface::copy<double>);
    bp::def("copysign", &FunctionsInterface::copySign<double>);
    bp::def("copyto", &FunctionsInterface::copyto<double>);
    bp::def("cosScaler", &FunctionsInterface::cosScaler<double>);
    bp::def("cosScaler", &FunctionsInterface::cosScaler<ComplexDouble>);
    bp::def("cosArray", &FunctionsInterface::cosArray<double>);
    bp::def("cosArray", &FunctionsInterface::cosArray<ComplexDouble>);
    bp::def("coshScaler", &FunctionsInterface::coshScaler<double>);
    bp::def("coshScaler", &FunctionsInterface::coshScaler<ComplexDouble>);
    bp::def("coshArray", &FunctionsInterface::coshArray<double>);
    bp::def("coshArray", &FunctionsInterface::coshArray<ComplexDouble>);
    bp::def("count_nonzero", &FunctionsInterface::count_nonzero<double>);
    bp::def("count_nonzero", &FunctionsInterface::count_nonzero<ComplexDouble>);
    bp::def("cross", &cross<double>);
    bp::def("cross", &cross<ComplexDouble>);
    bp::def("cube", &FunctionsInterface::cubeArray<double>);
    bp::def("cube", &FunctionsInterface::cubeArray<ComplexDouble>);
    bp::def("cumprod", &FunctionsInterface::cumprodArray<double>);
    bp::def("cumprod", &FunctionsInterface::cumprodArray<ComplexDouble>);
    bp::def("cumsum", &FunctionsInterface::cumsumArray<double>);
    bp::def("cumsum", &FunctionsInterface::cumsumArray<ComplexDouble>);

    bp::def("deg2radScaler", &FunctionsInterface::deg2radScaler<double>);
    bp::def("deg2radArray", &FunctionsInterface::deg2radArray<double>);
    bp::def("degreesScaler", &FunctionsInterface::degreesScaler<double>);
    bp::def("degreesArray", &FunctionsInterface::degreesArray<double>);
    bp::def("deleteIndicesScaler", &FunctionsInterface::deleteIndicesScaler<double>);
    bp::def("deleteIndicesSlice", &FunctionsInterface::deleteIndicesSlice<double>);
    bp::def("diag", &FunctionsInterface::diag<double>);
    bp::def("diag", &FunctionsInterface::diag<ComplexDouble>);
    bp::def("diagflat", &FunctionsInterface::diagflat<double>);
    bp::def("diagflat", &FunctionsInterface::diagflat<ComplexDouble>);
    bp::def("diagonal", &FunctionsInterface::diagonal<double>);
    bp::def("diagonal", &FunctionsInterface::diagonal<ComplexDouble>);
    bp::def("diff", &FunctionsInterface::diff<double>);
    bp::def("diff", &FunctionsInterface::diff<ComplexDouble>);
    bp::def("divide", &FunctionsInterface::divide<NdArray<double>, NdArray<double>>);
    bp::def("divide", &FunctionsInterface::divide<NdArray<double>, double>);
    bp::def("divide", &FunctionsInterface::divide<double, NdArray<double>>);
    bp::def("divide", &FunctionsInterface::divide<NdArray<ComplexDouble>, NdArray<ComplexDouble>>);
    bp::def("divide", &FunctionsInterface::divide<NdArray<ComplexDouble>, ComplexDouble>);
    bp::def("divide", &FunctionsInterface::divide<ComplexDouble, NdArray<ComplexDouble>>);
    bp::def("divide", &FunctionsInterface::divide<NdArray<double>, NdArray<ComplexDouble>>);
    bp::def("divide", &FunctionsInterface::divide<NdArray<ComplexDouble>, NdArray<double>>);
    bp::def("divide", &FunctionsInterface::divide<NdArray<double>, ComplexDouble>);
    bp::def("divide", &FunctionsInterface::divide<ComplexDouble, NdArray<double>>);
    bp::def("divide", &FunctionsInterface::divide<NdArray<ComplexDouble>, double>);
    bp::def("divide", &FunctionsInterface::divide<double, NdArray<ComplexDouble>>);
    bp::def("dot", &FunctionsInterface::dot<double, double>);
    bp::def("dot", &FunctionsInterface::dot<ComplexDouble, ComplexDouble>);
    bp::def("dot", &FunctionsInterface::dot<double, ComplexDouble>);
    bp::def("dot", &FunctionsInterface::dot<ComplexDouble, double>);
    bp::def("dump", &dump<double>);
    bp::def("dump", &dump<ComplexDouble>);

    bp::def("emptyRowCol", &FunctionsInterface::emptyRowCol<double>);
    bp::def("emptyShape", &FunctionsInterface::emptyShape<double>);
    bp::def("empty_like", &empty_like<double>);
    bp::def("endianess", &endianess<double>);
    bp::def("equal", &FunctionsInterface::equal<double>);
    bp::def("equal", &FunctionsInterface::equal<ComplexDouble>);
    bp::def("expScaler", &FunctionsInterface::expScaler<double>);
    bp::def("expScaler", &FunctionsInterface::expScaler<ComplexDouble>);
    bp::def("expArray", &FunctionsInterface::expArray<double>);
    bp::def("expArray", &FunctionsInterface::expArray<ComplexDouble>);
    bp::def("exp2Scaler", &FunctionsInterface::exp2Scaler<double>);
    bp::def("exp2Array", &FunctionsInterface::exp2Array<double>);
    bp::def("expm1Scaler", &FunctionsInterface::expm1Scaler<double>);
    bp::def("expm1Scaler", &FunctionsInterface::expm1Scaler<ComplexDouble>);
    bp::def("expm1Array", &FunctionsInterface::expm1Array<double>);
    bp::def("expm1Array", &FunctionsInterface::expm1Array<ComplexDouble>);
    bp::def("eye1D", &FunctionsInterface::eye1D<double>);
    bp::def("eye1DComplex", &FunctionsInterface::eye1D<ComplexDouble>);
    bp::def("eye2D", &FunctionsInterface::eye2D<double>);
    bp::def("eye2DComplex", &FunctionsInterface::eye2D<ComplexDouble>);
    bp::def("eyeShape", &FunctionsInterface::eyeShape<double>);
    bp::def("eyeShapeComplex", &FunctionsInterface::eyeShape<ComplexDouble>);

    bp::def("fillDiagonal", &fillDiagonal<double>);
    bp::def("find", &FunctionsInterface::find);
    bp::def("findN", &FunctionsInterface::findN);
    bp::def("fixScaler", &FunctionsInterface::fixScaler<double>);
    bp::def("fixArray", &FunctionsInterface::fixArray<double>);
    bp::def("flatten", &flatten<double>);
    bp::def("flatnonzero", &flatnonzero<double>);
    bp::def("flatnonzero", &flatnonzero<ComplexDouble>);
    bp::def("flip", &flip<double>);
    bp::def("fliplr", &fliplr<double>);
    bp::def("flipud", &flipud<double>);
    bp::def("floorScaler", &FunctionsInterface::floorScaler<double>);
    bp::def("floorArray", &FunctionsInterface::floorArray<double>);
    bp::def("floor_divideScaler", &FunctionsInterface::floor_divideScaler<double>);
    bp::def("floor_divideArray", &FunctionsInterface::floor_divideArray<double>);
    bp::def("fmaxScaler", &FunctionsInterface::fmaxScaler<double>);
    bp::def("fmaxScaler", &FunctionsInterface::fmaxScaler<ComplexDouble>);
    bp::def("fmaxArray", &FunctionsInterface::fmaxArray<double>);
    bp::def("fmaxArray", &FunctionsInterface::fmaxArray<ComplexDouble>);
    bp::def("fminScaler", &FunctionsInterface::fminScaler<double>);
    bp::def("fminScaler", &FunctionsInterface::fminScaler<ComplexDouble>);
    bp::def("fminArray", &FunctionsInterface::fminArray<double>);
    bp::def("fminArray", &FunctionsInterface::fminArray<ComplexDouble>);
    bp::def("fmodScaler", &FunctionsInterface::fmodScaler<uint32>);
    bp::def("fmodArray", &FunctionsInterface::fmodArray<uint32>);
    bp::def("frombuffer", &FunctionsInterface::frombuffer<double>);
    bp::def("frombuffer", &FunctionsInterface::frombuffer<ComplexDouble>);
    bp::def("fromfile", &fromfile<double>);
    bp::def("fromiter", &FunctionsInterface::fromiter<double>);
    bp::def("fromiter", &FunctionsInterface::fromiter<ComplexDouble>);
    bp::def("fullSquare", &FunctionsInterface::fullSquare<double>);
    bp::def("fullSquareComplex", &FunctionsInterface::fullSquare<ComplexDouble>);
    bp::def("fullRowCol", &FunctionsInterface::fullRowCol<double>);
    bp::def("fullRowColComplex", &FunctionsInterface::fullRowCol<ComplexDouble>);
    bp::def("fullShape", &FunctionsInterface::fullShape<double>);
    bp::def("fullShapeComplex", &FunctionsInterface::fullShape<ComplexDouble>);
    bp::def("full_like", &full_like<double>);
    bp::def("full_likeComplex", &full_like<ComplexDouble>);

    bp::def("gcdScaler", &FunctionsInterface::gcdScaler<uint32>);
    bp::def("gcdArray", &FunctionsInterface::gcdArray<uint32>);
    bp::def("greater", &greater<double>);
    bp::def("greater", &greater<ComplexDouble>);
    bp::def("greater_equal", &greater_equal<double>);
    bp::def("greater_equal", &greater_equal<ComplexDouble>);
    bp::def("gradient", &FunctionsInterface::gradient<double>);
    bp::def("gradient", &FunctionsInterface::gradient<ComplexDouble>);

    bp::def("histogram", &FunctionsInterface::histogram<double>);
    bp::def("hstack", &FunctionsInterface::hstack<double>);
    bp::def("hypotScaler", &FunctionsInterface::hypotScaler<double>);
    bp::def("hypotScalerTriple", &FunctionsInterface::hypotScalerTriple<double>);
    bp::def("hypotArray", &FunctionsInterface::hypotArray<double>);

    bp::def("identity", &identity<double>);
    bp::def("identityComplex", &identity<ComplexDouble>);
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
    bp::def("less", &less<ComplexDouble>);
    bp::def("less_equal", &less_equal<double>);
    bp::def("less_equal", &less_equal<ComplexDouble>);
    bp::def("linspace", &linspace<double>);
    bp::def("load", &load<double>);
    bp::def("logScaler", &FunctionsInterface::logScaler<double>);
    bp::def("logArray", &FunctionsInterface::logArray<double>);
    bp::def("logScaler", &FunctionsInterface::logScaler<ComplexDouble>);
    bp::def("logArray", &FunctionsInterface::logArray<ComplexDouble>);
    bp::def("log10Scaler", &FunctionsInterface::log10Scaler<double>);
    bp::def("log10Array", &FunctionsInterface::log10Array<ComplexDouble>);
    bp::def("log10Scaler", &FunctionsInterface::log10Scaler<ComplexDouble>);
    bp::def("log10Array", &FunctionsInterface::log10Array<double>);
    bp::def("log1pScaler", &FunctionsInterface::log1pScaler<double>);
    bp::def("log1pArray", &FunctionsInterface::log1pArray<double>);
    bp::def("log2Scaler", &FunctionsInterface::log2Scaler<double>);
    bp::def("log2Array", &FunctionsInterface::log2Array<double>);
    bp::def("logical_and", &logical_and<double>);
    bp::def("logical_not", &logical_not<double>);
    bp::def("logical_or", &logical_or<double>);
    bp::def("logical_xor", &logical_xor<double>);

    bp::def("matmul", &FunctionsInterface::matmul<double, double>);
    bp::def("matmul", &FunctionsInterface::matmul<ComplexDouble, ComplexDouble>);
    bp::def("matmul", &FunctionsInterface::matmul<double, ComplexDouble>);
    bp::def("matmul", &FunctionsInterface::matmul<ComplexDouble, double>);
    bp::def("max", &FunctionsInterface::max<double>);
    bp::def("max", &FunctionsInterface::max<ComplexDouble>);
    bp::def("maximum", &FunctionsInterface::maximum<double>);
    bp::def("maximum", &FunctionsInterface::maximum<ComplexDouble>);
    NdArray<double> (*meanDouble)(const NdArray<double>&, Axis) = &mean<double>; 
    bp::def("mean", meanDouble);
    NdArray<ComplexDouble> (*meanComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &mean<double>; 
    bp::def("mean", meanComplexDouble);
    bp::def("median", &median<double>);
    bp::def("meshgrid", &FunctionsInterface::meshgrid<double>);
    bp::def("min", &FunctionsInterface::min<double>);
    bp::def("min", &FunctionsInterface::min<ComplexDouble>);
    bp::def("minimum", &FunctionsInterface::minimum<double>);
    bp::def("minimum", &FunctionsInterface::minimum<ComplexDouble>);
    bp::def("mod", &mod<uint32>);
    bp::def("multiply", &FunctionsInterface::multiply<NdArray<double>, NdArray<double>>);
    bp::def("multiply", &FunctionsInterface::multiply<NdArray<double>, double>);
    bp::def("multiply", &FunctionsInterface::multiply<double, NdArray<double>>);
    bp::def("multiply", &FunctionsInterface::multiply<NdArray<ComplexDouble>, NdArray<ComplexDouble>>);
    bp::def("multiply", &FunctionsInterface::multiply<NdArray<ComplexDouble>, ComplexDouble>);
    bp::def("multiply", &FunctionsInterface::multiply<ComplexDouble, NdArray<ComplexDouble>>);
    bp::def("multiply", &FunctionsInterface::multiply<NdArray<double>, NdArray<ComplexDouble>>);
    bp::def("multiply", &FunctionsInterface::multiply<NdArray<ComplexDouble>, NdArray<double>>);
    bp::def("multiply", &FunctionsInterface::multiply<NdArray<double>, ComplexDouble>);
    bp::def("multiply", &FunctionsInterface::multiply<ComplexDouble, NdArray<double>>);
    bp::def("multiply", &FunctionsInterface::multiply<NdArray<ComplexDouble>, double>);
    bp::def("multiply", &FunctionsInterface::multiply<double, NdArray<ComplexDouble>>);

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
    bp::def("nbytes", &nbytes<ComplexDouble>);
    bp::def("negative", &negative<double>);
    bp::def("negative", &negative<ComplexDouble>);
    bp::def("newbyteorderScaler", &FunctionsInterface::newbyteorderScaler<uint32>);
    bp::def("newbyteorderArray", &FunctionsInterface::newbyteorderArray<uint32>);
    bp::def("none", &FunctionsInterface::noneArray<double>);
    bp::def("none", &FunctionsInterface::noneArray<ComplexDouble>);
    bp::def("nonzero", &nonzero<double>);
    bp::def("nonzero", &nonzero<ComplexDouble>);
    NdArray<double> (*normDouble)(const NdArray<double>&, Axis) = &norm<double>; 
    bp::def("norm", normDouble);
    NdArray<ComplexDouble> (*normComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &norm<double>; 
    bp::def("norm", normComplexDouble);
    bp::def("not_equal", &not_equal<double>);
    bp::def("not_equal", &not_equal<ComplexDouble>);

    bp::def("onesSquare", &FunctionsInterface::onesSquare<double>);
    bp::def("onesSquareComplex", &FunctionsInterface::onesSquare<ComplexDouble>);
    bp::def("onesRowCol", &FunctionsInterface::onesRowCol<double>);
    bp::def("onesRowColComplex", &FunctionsInterface::onesRowCol<ComplexDouble>);
    bp::def("onesShape", &FunctionsInterface::onesShape<double>);
    bp::def("onesShapeComplex", &FunctionsInterface::onesShape<ComplexDouble>);
    bp::def("ones_like", &ones_like<double, double>);
    bp::def("ones_likeComplex", &ones_like<ComplexDouble, double>);
    bp::def("outer", &FunctionsInterface::outer<double>);
    bp::def("outer", &FunctionsInterface::outer<ComplexDouble>);

    bp::def("pad", &pad<double>);
    bp::def("pad", &pad<ComplexDouble>);
    bp::def("partition", &partition<double>);
    bp::def("partition", &partition<ComplexDouble>);
    bp::def("percentile", &percentile<double>);
    bp::def("polarScaler", &FunctionsInterface::polarScaler<double>);
    bp::def("polarArray", &FunctionsInterface::polarArray<double>);
    bp::def("powerArrayScaler", &FunctionsInterface::powerArrayScaler<double>);
    bp::def("powerArrayArray", &FunctionsInterface::powerArrayArray<double>);
    bp::def("powerArrayScaler", &FunctionsInterface::powerArrayScaler<ComplexDouble>);
    bp::def("powerArrayArray", &FunctionsInterface::powerArrayArray<ComplexDouble>);
    bp::def("powerfArrayScaler", &FunctionsInterface::powerfArrayScaler<double>);
    bp::def("powerfArrayArray", &FunctionsInterface::powerfArrayArray<double>);
    bp::def("powerfArrayScaler", &FunctionsInterface::powerfArrayScaler<ComplexDouble>);
    bp::def("powerfArrayArray", &FunctionsInterface::powerfArrayArray<ComplexDouble>);
    bp::def("prod", &prod<double>);
    bp::def("prod", &prod<ComplexDouble>);
    bp::def("projScaler", &FunctionsInterface::projScaler<double>);
    bp::def("projArray", &FunctionsInterface::projArray<double>);
    bp::def("ptp", &ptp<double>);
    bp::def("ptp", &ptp<ComplexDouble>);
    NdArray<double>& (*putDoubleValue)(NdArray<double>&, const NdArray<uint32>&, double) = &put<double>;
    bp::def("put", putDoubleValue, bp::return_internal_reference<>());
    NdArray<double>& (*putDoubleArray)(NdArray<double>&, const NdArray<uint32>&, const NdArray<double>&) = &put<double>;
    bp::def("put", putDoubleArray, bp::return_internal_reference<>());
    bp::def("putmask", &FunctionsInterface::putmask<double>);
    bp::def("putmaskScaler", &FunctionsInterface::putmaskScaler<double>);

    bp::def("rad2degScaler", &FunctionsInterface::rad2degScaler<double>);
    bp::def("rad2degArray", &FunctionsInterface::rad2degArray<double>);
    bp::def("radiansScaler", &FunctionsInterface::radiansScaler<double>);
    bp::def("radiansArray", &FunctionsInterface::radiansArray<double>);
    bp::def("ravel", &FunctionsInterface::ravel<double>, bp::return_internal_reference<>());
    bp::def("reciprocal", &FunctionsInterface::reciprocal<double>);
    bp::def("reciprocal", &FunctionsInterface::reciprocal<ComplexDouble>);
    bp::def("realScaler", &FunctionsInterface::realScaler<double>);
    bp::def("realArray", &FunctionsInterface::realArray<double>);
    bp::def("remainderScaler", &FunctionsInterface::remainderScaler<double>);
    bp::def("remainderArray", &FunctionsInterface::remainderArray<double>);
    bp::def("replace", &FunctionsInterface::replace<double>);
    bp::def("replace", &FunctionsInterface::replace<ComplexDouble>);
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
    NdArray<ComplexDouble> (*rmsComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &rms<double>; 
    bp::def("rms", rmsComplexDouble);
    bp::def("roll", &roll<double>);
    bp::def("rot90", &rot90<double>);
    bp::def("roundScaler", &FunctionsInterface::roundScaler<double>);
    bp::def("roundArray", &FunctionsInterface::roundArray<double>);
    bp::def("row_stack", &FunctionsInterface::row_stack<double>);

    bp::def("setdiff1d", &setdiff1d<uint32>);
    bp::def("setdiff1d", &setdiff1d<std::complex<uint32>>);
    bp::def("signScaler", &FunctionsInterface::signScaler<double>);
    bp::def("signScaler", &FunctionsInterface::signScaler<ComplexDouble>);
    bp::def("signArray", &FunctionsInterface::signArray<double>);
    bp::def("signArray", &FunctionsInterface::signArray<ComplexDouble>);
    bp::def("signbitScaler", &FunctionsInterface::signbitScaler<double>);
    bp::def("signbitArray", &FunctionsInterface::signbitArray<double>);
    bp::def("sinScaler", &FunctionsInterface::sinScaler<double>);
    bp::def("sinScaler", &FunctionsInterface::sinScaler<ComplexDouble>);
    bp::def("sinArray", &FunctionsInterface::sinArray<double>);
    bp::def("sinArray", &FunctionsInterface::sinArray<ComplexDouble>);
    bp::def("sincScaler", &FunctionsInterface::sincScaler<double>);
    bp::def("sincArray", &FunctionsInterface::sincArray<double>);
    bp::def("sinhScaler", &FunctionsInterface::sinhScaler<ComplexDouble>);
    bp::def("sinhScaler", &FunctionsInterface::sinhScaler<double>);
    bp::def("sinhArray", &FunctionsInterface::sinhArray<double>);
    bp::def("sinhArray", &FunctionsInterface::sinhArray<ComplexDouble>);
    bp::def("size", &size<double>);
    bp::def("sort", &sort<double>);
    bp::def("sort", &sort<ComplexDouble>);
    bp::def("sqrtScaler", &FunctionsInterface::sqrtScaler<double>);
    bp::def("sqrtScaler", &FunctionsInterface::sqrtScaler<ComplexDouble>);
    bp::def("sqrtArray", &FunctionsInterface::sqrtArray<double>);
    bp::def("sqrtArray", &FunctionsInterface::sqrtArray<ComplexDouble>);
    bp::def("squareScaler", &FunctionsInterface::squareScaler<double>);
    bp::def("squareScaler", &FunctionsInterface::squareScaler<ComplexDouble>);
    bp::def("squareArray", &FunctionsInterface::squareArray<double>);
    bp::def("squareArray", &FunctionsInterface::squareArray<ComplexDouble>);
    bp::def("stack", &FunctionsInterface::stack<double>);
    NdArray<double> (*stdevDouble)(const NdArray<double>&, Axis) = &stdev<double>; 
    bp::def("stdev", stdevDouble);
    NdArray<ComplexDouble> (*stdevComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &stdev<double>; 
    bp::def("stdev", stdevComplexDouble);
    bp::def("subtract", &FunctionsInterface::subtract<NdArray<double>, NdArray<double>>);
    bp::def("subtract", &FunctionsInterface::subtract<NdArray<double>, double>);
    bp::def("subtract", &FunctionsInterface::subtract<double, NdArray<double>>);
    bp::def("subtract", &FunctionsInterface::subtract<NdArray<ComplexDouble>, NdArray<ComplexDouble>>);
    bp::def("subtract", &FunctionsInterface::subtract<NdArray<ComplexDouble>, ComplexDouble>);
    bp::def("subtract", &FunctionsInterface::subtract<ComplexDouble, NdArray<ComplexDouble>>);
    bp::def("subtract", &FunctionsInterface::subtract<NdArray<double>, NdArray<ComplexDouble>>);
    bp::def("subtract", &FunctionsInterface::subtract<NdArray<ComplexDouble>, NdArray<double>>);
    bp::def("subtract", &FunctionsInterface::subtract<NdArray<double>, ComplexDouble>);
    bp::def("subtract", &FunctionsInterface::subtract<ComplexDouble, NdArray<double>>);
    bp::def("subtract", &FunctionsInterface::subtract<NdArray<ComplexDouble>, double>);
    bp::def("subtract", &FunctionsInterface::subtract<double, NdArray<ComplexDouble>>);
    bp::def("sum", &sum<double>);
    bp::def("sum", &sum<ComplexDouble>);
    bp::def("swapaxes", &swapaxes<double>);
    bp::def("swap", &nc::swap<double>);

    bp::def("tanScaler", &FunctionsInterface::tanScaler<double>);
    bp::def("tanScaler", &FunctionsInterface::tanScaler<ComplexDouble>);
    bp::def("tanArray", &FunctionsInterface::tanArray<double>);
    bp::def("tanArray", &FunctionsInterface::tanArray<ComplexDouble>);
    bp::def("tanhScaler", &FunctionsInterface::tanhScaler<double>);
    bp::def("tanhScaler", &FunctionsInterface::tanhScaler<ComplexDouble>);
    bp::def("tanhArray", &FunctionsInterface::tanhArray<double>);
    bp::def("tanhArray", &FunctionsInterface::tanhArray<ComplexDouble>);
    bp::def("tileRectangle", &FunctionsInterface::tileRectangle<double>);
    bp::def("tileShape", &FunctionsInterface::tileShape<double>);
    bp::def("tileList", &FunctionsInterface::tileList<double>);
    bp::def("tofile", &tofile<double>);
    bp::def("toStlVector", &toStlVector<double>);
    bp::def("trace", &trace<double>);
    bp::def("trace", &trace<ComplexDouble>);
    bp::def("transpose", &transpose<double>);
    bp::def("trapzDx", &FunctionsInterface::trapzDx<double>);
    bp::def("trapz", &FunctionsInterface::trapz<double>);
    bp::def("trilSquare", &FunctionsInterface::trilSquare<double>);
    bp::def("trilSquareComplex", &FunctionsInterface::trilSquare<ComplexDouble>);
    bp::def("trilRect", &FunctionsInterface::trilRect<double>);
    bp::def("trilRectComplex", &FunctionsInterface::trilRect<ComplexDouble>);
    bp::def("trilArray", &FunctionsInterface::trilArray<double>);
    bp::def("trilArray", &FunctionsInterface::trilArray<ComplexDouble>);
    bp::def("triuSquare", &FunctionsInterface::triuSquare<double>);
    bp::def("triuSquareComplex", &FunctionsInterface::triuSquare<ComplexDouble>);
    bp::def("triuRect", &FunctionsInterface::triuRect<double>);
    bp::def("triuRectComplex", &FunctionsInterface::triuRect<ComplexDouble>);
    bp::def("triuArray", &FunctionsInterface::triuArray<double>);
    bp::def("triuArray", &FunctionsInterface::triuArray<ComplexDouble>);
    bp::def("trim_zeros", &trim_zeros<double>);
    bp::def("trim_zeros", &trim_zeros<ComplexDouble>);
    bp::def("truncScaler", &FunctionsInterface::truncScaler<double>);
    bp::def("truncArray", &FunctionsInterface::truncArray<double>);

    bp::def("union1d", &union1d<uint32>);
    bp::def("union1d", &union1d<std::complex<uint32>>);
    bp::def("unique", &unique<uint32>);
    bp::def("unique", &unique<std::complex<uint32>>);
    bp::def("unwrapScaler", &FunctionsInterface::unwrapScaler<double>);
    bp::def("unwrapArray", &FunctionsInterface::unwrapArray<double>);

    NdArray<double> (*varDouble)(const NdArray<double>&, Axis) = &var<double>;
    bp::def("var", varDouble);
    NdArray<ComplexDouble> (*varComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &var<double>; 
    bp::def("var", varComplexDouble);
    bp::def("vstack", &FunctionsInterface::vstack<double>);

    bp::def("where", &FunctionsInterface::where<double>);
    bp::def("where", &FunctionsInterface::where<ComplexDouble>);

    bp::def("zerosSquare", &FunctionsInterface::zerosSquare<double>);
    bp::def("zerosSquareComplex", &FunctionsInterface::zerosSquare<ComplexDouble>);
    bp::def("zerosRowCol", &FunctionsInterface::zerosRowCol<double>);
    bp::def("zerosRowColComplex", &FunctionsInterface::zerosRowCol<ComplexDouble>);
    bp::def("zerosShape", &FunctionsInterface::zerosShape<double>);
    bp::def("zerosShapeComplex", &FunctionsInterface::zerosShape<ComplexDouble>);
    bp::def("zerosList", &FunctionsInterface::zerosList<double>);
    bp::def("zerosListComplex", &FunctionsInterface::zerosList<ComplexDouble>);
    bp::def("zeros_like", &zeros_like<double, double>);
    bp::def("zeros_likeComplex", &zeros_like<ComplexDouble, double>);

    // Utils.hpp
    bp::def("num2str", &utils::num2str<double>);
    bp::def("sqr", &utils::sqr<double>);
    bp::def("cube", &utils::cube<double>);
    bp::def("power", &utils::power<double>);
    bp::def("power", &utils::power<ComplexDouble>);
    decltype(utils::powerf<double, double>(double{ 0 }, double{ 0 }))(*powerf_double)(double, double) = &utils::powerf<double, double>;
    bp::def("powerf", powerf_double);
    decltype(utils::powerf<ComplexDouble, ComplexDouble>(ComplexDouble{ 0 }, ComplexDouble{ 0 }))(*powerf_complexDouble)
        (ComplexDouble, ComplexDouble) = &utils::powerf<ComplexDouble, ComplexDouble>;
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
    bp::def("multi_dot", &LinalgInterface::multi_dot<ComplexDouble>);
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

    Poly1d(*fit)(const NdArrayDouble&, const NdArrayDouble&, uint8) = &Poly1d::fit;
    Poly1d (*fitWeighted)(const NdArrayDouble&, const NdArrayDouble&, const NdArrayDouble&, uint8) = &Poly1d::fit;

    bp::class_<Poly1d>
        ("Poly1d", bp::init<>())
        .def(bp::init<NdArray<double>, bool>())
        .def("area", &Poly1d::area)
        .def("coefficients", &Poly1d::coefficients)
        .def("deriv", &Poly1d::deriv)
        .def("fit", fit).staticmethod("fit")
        .def("fitWeighted", fitWeighted).staticmethod("fitWeighted")
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
