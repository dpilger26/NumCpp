#include "NumCpp.hpp"
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstdio>
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

#include "pybind11/pybind11.h"
#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

using namespace nc;
using namespace nc::pybindInterface;
namespace pb11 = pybind11;

//================================================================================

namespace ShapeInterface
{
    bool testListContructor() 
    {
        const Shape test = { 357, 666 };
        return test.rows == 357 && test.cols == 666;
    }
}  // namespace ShapeInterface

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
    Iterator operatorPlusPlusPost(Iterator& self)
    {
        return self++;
    }

    //================================================================================

    template<typename Iterator>
    Iterator operatorMinusMinusPre(Iterator& self)
    {
        return --self;
    }

    //================================================================================

    template<typename Iterator>
    Iterator operatorMinusMinusPost(Iterator& self)
    {
        return self--;
    }

    //================================================================================

    template<typename Iterator>
    Iterator operatorPlusEqual(Iterator& self, typename Iterator::difference_type offset)
    {
        return self += offset;
    }

    //================================================================================

    template<typename Iterator>
    Iterator operatorPlus(Iterator& self, typename Iterator::difference_type offset)
    {
        return self + offset;
    }

    //================================================================================

    template<typename Iterator>
    Iterator operatorMinusEqual(Iterator& self, typename Iterator::difference_type offset)
    {
        return self -= offset;
    }

    //================================================================================

    template<typename Iterator>
    Iterator operatorMinus(Iterator& self, typename Iterator::difference_type offset)
    {
        return self - offset;
    }

    //================================================================================

    template<typename Iterator>
    typename Iterator::difference_type operatorDiff(const Iterator& self, const Iterator& rhs)
    {
        return self - rhs;
    }

    //================================================================================

    template<typename Iterator>
    bool operatorEqual(Iterator& self, const Iterator& rhs)
    {
        return self == rhs;
    }

    //================================================================================

    template<typename Iterator>
    bool operatorNotEqual(Iterator& self, const Iterator& rhs)
    {
        return self != rhs;
    }

    //================================================================================

    template<typename Iterator>
    bool operatorLess(Iterator& self, const Iterator& rhs)
    {
        return self < rhs;
    }

    //================================================================================

    template<typename Iterator>
    bool operatorLessEqual(Iterator& self, const Iterator& rhs)
    {
        return self <= rhs;
    }

    //================================================================================

    template<typename Iterator>
    bool operatorGreater(Iterator& self, const Iterator& rhs)
    {
        return self > rhs;
    }

    //================================================================================

    template<typename Iterator>
    bool operatorGreaterEqual(Iterator& self, const Iterator& rhs)
    {
        return self >= rhs;
    }

    //================================================================================

    template<typename Iterator>
    typename Iterator::value_type access(Iterator& self, typename Iterator::difference_type offset)
    {
        return self[offset];
    }
}  // namespace IteratorInterface

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
    pbArrayGeneric test1dArrayConstructor(T value1, T value2)
    {
        std::array<T, 2> arr = {value1, value2};
        auto newNcArray = NdArray<T>(arr);

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test2dArrayConstructor(T value1, T value2)
    {
        std::array<std::array<T, 2>, 2> arr2d{};
        arr2d[0][0] = value1;
        arr2d[0][1] = value2;
        arr2d[1][0] = value1;
        arr2d[1][1] = value2;
        auto newNcArray = NdArray<T>(arr2d);

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test1dVectorConstructor(pbArray<T> inArray)
    {
        const auto ncArray = pybind2nc(inArray);

        std::vector<T> vec(ncArray.size());
        std::copy(ncArray.cbegin(), ncArray.cend(), vec.begin());

        auto newNcArray = NdArray<T>(vec);

        return nc2pybind(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test2dVectorConstructor(pbArray<T> inArray)
    {
        const auto ncArray = pybind2nc(inArray);

        std::vector<std::vector<T>> vec2d(ncArray.numRows(), std::vector<T>(ncArray.numCols()));
        for (uint32 row = 0; row < ncArray.numRows(); ++row)
        {
            std::copy(ncArray.cbegin(row), ncArray.cend(row), vec2d[row].begin());
        }

        auto newNcArray = NdArray<T>(vec2d);

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test2dVectorArrayConstructor(pbArray<T> inArray)
    {
        const auto ncArray = pybind2nc(inArray);
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

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test1dDequeConstructor(pbArray<T> inArray)
    {
        const auto ncArray = pybind2nc(inArray);

        std::deque<T> deq(ncArray.size());
        std::copy(ncArray.cbegin(), ncArray.cend(), deq.begin());

        auto newNcArray = NdArray<T>(deq);

        return nc2pybind(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test2dDequeConstructor(pbArray<T> inArray)
    {
        const auto ncArray = pybind2nc(inArray);

        std::deque<std::deque<T>> deq2d(ncArray.numRows(), std::deque<T>(ncArray.numCols()));
        for (uint32 row = 0; row < ncArray.numRows(); ++row)
        {
            std::copy(ncArray.cbegin(row), ncArray.cend(row), deq2d[row].begin());
        }

        auto newNcArray = NdArray<T>(deq2d);

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test1dListConstructor(pbArray<T> inArray)
    {
        const auto ncArray = pybind2nc(inArray);

        std::list<T> list(ncArray.size());
        std::copy(ncArray.cbegin(), ncArray.cend(), list.begin());

        auto newNcArray = NdArray<T>(list);

        return nc2pybind(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test1dIteratorConstructor(pbArray<T> inArray)
    {
        const auto ncArray = pybind2nc(inArray);

        std::vector<T> vec(ncArray.size());
        std::copy(ncArray.cbegin(), ncArray.cend(), vec.begin());

        auto newNcArray = NdArray<T>(vec.cbegin(), vec.cend());

        return nc2pybind(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test1dIteratorConstructor2(pbArray<T> inArray)
    {
        const auto ncArray = pybind2nc(inArray);
        auto newNcArray = NdArray<T>(ncArray.cbegin(), ncArray.cend());

        return nc2pybind(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test1dPointerConstructor(pbArray<T> inArray)
    {
        const auto ncArray = pybind2nc(inArray);
        auto newNcArray = NdArray<T>(ncArray.data(), ncArray.size());

        return nc2pybind(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test2dPointerConstructor(pbArray<T> inArray)
    {
        const auto ncArray = pybind2nc(inArray);
        auto newNcArray = NdArray<T>(ncArray.data(), ncArray.numRows(), ncArray.numCols());

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test1dPointerShellConstructor(pbArray<T> inArray)
    {
        auto ncArray = pybind2nc(inArray).copy();
        auto newNcArray = NdArray<T>(ncArray.dataRelease(), ncArray.size(), true);

        return nc2pybind(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test2dPointerShellConstructor(pbArray<T> inArray)
    {
        auto ncArray = pybind2nc(inArray).copy();
        auto newNcArray = NdArray<T>(ncArray.dataRelease(), ncArray.numRows(), ncArray.numCols(), true);

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric testCopyConstructor(pbArray<T> inArray)
    {
        const auto ncArray = pybind2nc(inArray);
        auto newNcArray = NdArray<T>(ncArray);

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric testMoveConstructor(pbArray<T> inArray)
    {
        auto ncArray = pybind2nc(inArray);
        auto newNcArray = NdArray<T>(std::move(ncArray));

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric testAssignementOperator(pbArray<T> inArray)
    {
        auto ncArray = pybind2nc(inArray);
        NdArray<T> newNcArray;
        newNcArray = ncArray;

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric testAssignementScalerOperator(pbArray<T> inArray, T value)
    {
        auto ncArray = pybind2nc(inArray);
        ncArray = value;

        return nc2pybind(ncArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric testMoveAssignementOperator(pbArray<T> inArray)
    {
        auto ncArray = pybind2nc(inArray);
        NdArray<T> newNcArray;
        newNcArray = std::move(ncArray);

        return nc2pybind(newNcArray);
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
        test1 = testStruct;

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
    pbArrayGeneric getNumpyArray(NdArray<dtype>& inArray)
    {
        return nc2pybind(inArray);
    }

    //================================================================================

    template<typename dtype>
    void setArray(NdArray<dtype>& self, pbArray<dtype>& inPyArray)
    {
        const auto numDims = inPyArray.ndim();
        if (numDims > 2)
        {
            std::string errorString = "ERROR: Input array can only have up to 2 dimensions!";
            PyErr_SetString(PyExc_RuntimeError, errorString.c_str());
        }

        self = pybind2nc_copy(inPyArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric all(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.all(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric any(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.any(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric argmax(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.argmax(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric argmin(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.argmin(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric argsort(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.argsort(inAxis));
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
    pbArrayGeneric clip(const NdArray<dtype>& self, dtype inMin, dtype inMax)
    {
        return nc2pybind(self.clip(inMin, inMax));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric copy(const NdArray<dtype>& self)
    {
        return nc2pybind(self.copy());
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric contains(const NdArray<dtype>& self, dtype inValue, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.contains(inValue, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric cumprod(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.cumprod(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric cumsum(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.cumsum(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric diagonal(const NdArray<dtype>& self, int32 inOffset = 0, Axis inAxis = Axis::ROW)
    {
        return nc2pybind(self.diagonal(inOffset, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric dot(const NdArray<dtype>& self, const NdArray<dtype>& inOtherArray)
    {
        return nc2pybind(self.dot(inOtherArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fill(NdArray<dtype>& self, dtype inFillValue)
    {
        self.fill(inFillValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric flatnonzero(const NdArray<dtype>& self)
    {
        return nc2pybind(self.flatnonzero());
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric flatten(const NdArray<dtype>& self)
    {
        return nc2pybind(self.flatten());
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
    pbArrayGeneric getMask(const NdArray<dtype>& self, const NdArray<bool>& mask)
    {
        return nc2pybind(self[mask]);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getIndices(const NdArray<dtype>& self, const NdArray<typename NdArray<dtype>::size_type>& indices)
    {
        return nc2pybind(self[indices]);
    }

    //================================================================================


    template<typename dtype>
    pbArrayGeneric getSlice1D(const NdArray<dtype>& self, const Slice& inSlice)
    {
        return nc2pybind(self[inSlice]);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getSlice2D(const NdArray<dtype>& self, const Slice& inRowSlice, const Slice& inColSlice)
    {
        return nc2pybind(self(inRowSlice, inColSlice));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getSlice2DCol(const NdArray<dtype>& self, const Slice& inRowSlice, int32 inColIndex)
    {
        return nc2pybind(self(inRowSlice, inColIndex));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getSlice2DRow(const NdArray<dtype>& self, int32 inRowIndex, const Slice& inColSlice)
    {
        return nc2pybind(self(inRowIndex, inColSlice));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getIndicesScaler(const NdArray<dtype>& self, const NdArray<int32>& rowIndices, 
        int32 colIndex)
    {
        return nc2pybind(self(rowIndices, colIndex));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getIndicesSlice(const NdArray<dtype>& self, const NdArray<int32>& rowIndices, 
        Slice colSlice)
    {
        return nc2pybind(self(rowIndices, colSlice));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getScalerIndices(const NdArray<dtype>& self, int32 rowIndex,
        const NdArray<int32>& colIndices)
    {
        return nc2pybind(self(rowIndex, colIndices));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getSliceIndices(const NdArray<dtype>& self, Slice rowSlice,
        const NdArray<int32>& colIndices)
    {
        return nc2pybind(self(rowSlice, colIndices));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getIndices2D(const NdArray<dtype>& self, const NdArray<int32>& rowIndices, 
        const NdArray<int32>& colIndices)
    {
        return nc2pybind(self(rowIndices, colIndices));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getByIndices(const NdArray<dtype>& self, const NdArray<uint32>& inIndices)
    {
        return nc2pybind(self.getByIndices(inIndices));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getByMask(const NdArray<dtype>& self, const NdArray<bool>& inMask)
    {
        return nc2pybind(self.getByMask(inMask));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric issorted(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.issorted(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric max(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.max(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric min(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.min(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric median(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.median(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric newbyteorder(const NdArray<dtype>& self, Endian inEndiness = Endian::NATIVE)
    {
        return nc2pybind(self.newbyteorder(inEndiness));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric none(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.none(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pb11::tuple nonzero(const NdArray<dtype>& self)
    {
        auto rowCol = self.nonzero();
        return pb11::make_tuple(nc2pybind(rowCol.first), nc2pybind(rowCol.second));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ones(NdArray<dtype>& self)
    {
        self.ones();
        return nc2pybind<dtype>(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric partition(NdArray<dtype>& self, uint32 inKth, Axis inAxis = Axis::NONE)
    {
        self.partition(inKth, inAxis);
        return nc2pybind<dtype>(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric prod(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.prod(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric ptp(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.ptp(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putFlat(NdArray<dtype>& self, int32 inIndex, dtype inValue)
    {
        self.put(inIndex, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putRowCol(NdArray<dtype>& self, int32 inRow, int32 inCol, dtype inValue)
    {
        self.put(inRow, inCol, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice1DValue(NdArray<dtype>& self, const Slice& inSlice, dtype inValue)
    {
        self.put(inSlice, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice1DValues(NdArray<dtype>& self, const Slice& inSlice, pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inSlice, inValues);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValue(NdArray<dtype>& self, const Slice& inSliceRow, const Slice& inSliceCol, dtype inValue)
    {
        self.put(inSliceRow, inSliceCol, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValueRow(NdArray<dtype>& self, int32 inRowIndex, const Slice& inSliceCol, dtype inValue)
    {
        self.put(inRowIndex, inSliceCol, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValueCol(NdArray<dtype>& self, const Slice& inSliceRow, int32 inColIndex, dtype inValue)
    {
        self.put(inSliceRow, inColIndex, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValues(NdArray<dtype>& self, const Slice& inSliceRow, const Slice& inSliceCol, pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inSliceRow, inSliceCol, inValues);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValuesRow(NdArray<dtype>& self, int32 inRowIndex, const Slice& inSliceCol, pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inRowIndex, inSliceCol, inValues);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValuesCol(NdArray<dtype>& self, const Slice& inSliceRow, int32 inColIndex, pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inSliceRow, inColIndex, inValues);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putMaskSingle(NdArray<dtype>& self, pbArray<bool>& inMask, dtype inValue)
    {
        auto mask = pybind2nc(inMask);
        self.putMask(mask, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putMaskMultiple(NdArray<dtype>& self, pbArray<bool>& inMask, pbArray<dtype>& inArrayValues)
    {
        auto mask = pybind2nc(inMask);
        auto inValues = pybind2nc(inArrayValues);
        self.putMask(mask, inValues);
        return nc2pybind(self);
    }

    template<typename dtype>
    pbArrayGeneric ravel(NdArray<dtype>& self)
    {
        self.ravel();
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric repeat(const NdArray<dtype>& self, const Shape& inRepeatShape)
    {
        return nc2pybind(self.repeat(inRepeatShape));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric reshapeInt(NdArray<dtype>& self, uint32 size)
    {
        self.reshape(size);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric reshapeValues(NdArray<dtype>& self, int32 inNumRows, int32 inNumCols)
    {
        self.reshape(inNumRows, inNumCols);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric reshapeShape(NdArray<dtype>& self, const Shape& inShape)
    {
        self.reshape(inShape);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric reshapeList(NdArray<dtype>& self, const Shape& inShape)
    {
        self.reshape({ inShape.rows, inShape.cols });
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric replace(NdArray<dtype>& self, dtype oldValue, dtype newValue)
    {
        self.replace(oldValue, newValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric resizeFast(NdArray<dtype>& self, const Shape& inShape)
    {
        self.resizeFast(inShape);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric resizeFastList(NdArray<dtype>& self, const Shape& inShape)
    {
        self.resizeFast({ inShape.rows, inShape.cols });
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric resizeSlow(NdArray<dtype>& self, const Shape& inShape)
    {
        self.resizeSlow(inShape);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric resizeSlowList(NdArray<dtype>& self, const Shape& inShape)
    {
        self.resizeSlow({ inShape.rows, inShape.cols });
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric round(const NdArray<dtype>& self, uint8 inNumDecimals)
    {
        return nc2pybind(self.round(inNumDecimals));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric sort(NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        self.sort(inAxis);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric sum(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.sum(inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric swapaxes(const NdArray<dtype>& self)
    {
        return nc2pybind(self.swapaxes());
    }

    //================================================================================

    template<typename dtype>
    void tofileBinary(const NdArray<dtype>& self, const std::string& filename)
    {
        return self.tofile(filename);
    }

    //================================================================================

    template<typename dtype>
    void tofileTxt(const NdArray<dtype>& self, const std::string& filename, const char sep)
    {
        return self.tofile(filename, sep);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric transpose(const NdArray<dtype>& self)
    {
        return nc2pybind(self.transpose());
    }

    //================================================================================

    template<typename dtype> // (1)
    pbArrayGeneric operatorPlusEqualArray(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs += rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    pbArrayGeneric operatorPlusEqualComplexArrayArithArray(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs += rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorPlusEqualScaler(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs += rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorPlusEqualComplexArrayArithScaler(NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs += rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    pbArrayGeneric operatorPlusArray(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    pbArrayGeneric operatorPlusArithArrayComplexArray(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorPlusComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorPlusArrayScaler(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    pbArrayGeneric operatorPlusScalerArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    pbArrayGeneric operatorPlusArithArrayComplexScaler(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    pbArrayGeneric operatorPlusComplexScalerArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    pbArrayGeneric operatorPlusComplexArrayArithScaler(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    pbArrayGeneric operatorPlusArithScalerComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorNegative(const NdArray<dtype>& inArray)
    {
        return nc2pybind(-inArray);
    }

    //================================================================================

    template<typename dtype> // (1)
    pbArrayGeneric operatorMinusEqualArray(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs -= rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    pbArrayGeneric operatorMinusEqualComplexArrayArithArray(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs -= rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorMinusEqualScaler(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs -= rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorMinusEqualComplexArrayArithScaler(NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs -= rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    pbArrayGeneric operatorMinusArray(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    pbArrayGeneric operatorMinusArithArrayComplexArray(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorMinusComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorMinusArrayScaler(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    pbArrayGeneric operatorMinusScalerArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    pbArrayGeneric operatorMinusArithArrayComplexScaler(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    pbArrayGeneric operatorMinusComplexScalerArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    pbArrayGeneric operatorMinusComplexArrayArithScaler(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    pbArrayGeneric operatorMinusArithScalerComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    pbArrayGeneric operatorMultiplyEqualArray(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs *= rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    pbArrayGeneric operatorMultiplyEqualComplexArrayArithArray(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs *= rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorMultiplyEqualScaler(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs *= rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorMultiplyEqualComplexArrayArithScaler(NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs *= rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    pbArrayGeneric operatorMultiplyArray(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    pbArrayGeneric operatorMultiplyArithArrayComplexArray(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorMultiplyComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorMultiplyArrayScaler(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    pbArrayGeneric operatorMultiplyScalerArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    pbArrayGeneric operatorMultiplyArithArrayComplexScaler(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    pbArrayGeneric operatorMultiplyComplexScalerArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    pbArrayGeneric operatorMultiplyComplexArrayArithScaler(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    pbArrayGeneric operatorMultiplyArithScalerComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    pbArrayGeneric operatorDivideEqualArray(NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs /= rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    pbArrayGeneric operatorDivideEqualComplexArrayArithArray(NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs /= rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorDivideEqualScaler(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs /= rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorDivideEqualComplexArrayArithScaler(NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs /= rhs);
    }

    //================================================================================

    template<typename dtype> // (1)
    pbArrayGeneric operatorDivideArray(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (2)
    pbArrayGeneric operatorDivideArithArrayComplexArray(const NdArray<dtype>& lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorDivideComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorDivideArrayScaler(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    pbArrayGeneric operatorDivideScalerArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    pbArrayGeneric operatorDivideArithArrayComplexScaler(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    pbArrayGeneric operatorDivideComplexScalerArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    pbArrayGeneric operatorDivideComplexArrayArithScaler(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    pbArrayGeneric operatorDivideArithScalerComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorModulusScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2pybind(inArray % inScaler);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorModulusScalerReversed(dtype inScaler, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inScaler % inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorModulusArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 % inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseOrScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2pybind(inArray | inScaler);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseOrScalerReversed(dtype inScaler, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inScaler | inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseOrArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 | inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseAndScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2pybind(inArray & inScaler);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseAndScalerReversed(dtype inScaler, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inScaler & inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseAndArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 & inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseXorScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2pybind(inArray ^ inScaler);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseXorScalerReversed(dtype inScaler, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inScaler ^ inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseXorArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 ^ inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseNot(const NdArray<dtype>& inArray)
    {
        return nc2pybind(~inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLogicalAndArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 && inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLogicalAndScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray && inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLogicalAndScalarReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inValue && inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLogicalOrArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 || inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLogicalOrScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray || inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLogicalOrScalarReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inValue || inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorNot(const NdArray<dtype>& inArray)
    {
        return nc2pybind(!inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorEqualityScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray == inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorEqualityScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inValue == inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorEqualityArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 == inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorNotEqualityScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray != inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorNotEqualityScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inValue != inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorNotEqualityArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 != inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLessScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray < inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLessScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inValue < inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLessArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 < inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorGreaterScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray > inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorGreaterScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inValue > inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorGreaterArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 > inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLessEqualScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray <= inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLessEqualScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inValue <= inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLessEqualArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 <= inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorGreaterEqualScaler(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray >= inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorGreaterEqualScalerReversed(dtype inValue, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inValue >= inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorGreaterEqualArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 >= inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitshiftLeft(const NdArray<dtype>& inArray, uint8 inNumBits)
    {
        return nc2pybind(inArray << inNumBits);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitshiftRight(const NdArray<dtype>& inArray, uint8 inNumBits)
    {
        return nc2pybind(inArray >> inNumBits);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorPrePlusPlus(NdArray<dtype>& inArray)
    {
        return nc2pybind(++inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorPostPlusPlus(NdArray<dtype>& inArray)
    {
        return nc2pybind(inArray++);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorPreMinusMinus(NdArray<dtype>& inArray)
    {
        return nc2pybind(--inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorPostMinusMinus(NdArray<dtype>& inArray)
    {
        return nc2pybind(inArray--);
    }
}  // namespace NdArrayInterface

//================================================================================

namespace FunctionsInterface
{
    template<typename dtype>
    auto absScaler(dtype inValue) -> decltype(abs(inValue)) // trailing return type to help gcc
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
    dtype angleScaler(const std::complex<dtype>& inValue)
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
    auto arccosScaler(dtype inValue) -> decltype(arccos(inValue)) // trailing return type to help gcc
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
    auto arccoshScaler(dtype inValue) -> decltype(arccosh(inValue)) // trailing return type to help gcc
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
    auto arcsinScaler(dtype inValue) -> decltype(arcsin(inValue)) // trailing return type to help gcc
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
    auto arcsinhScaler(dtype inValue) -> decltype(arcsinh(inValue)) // trailing return type to help gcc
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
    auto arctanScaler(dtype inValue) -> decltype(arctan(inValue)) // trailing return type to help gcc
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
    dtype arctan2Scaler(dtype inY, dtype inX) 
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
    auto arctanhScaler(dtype inValue) -> decltype(arctanh(inValue)) // trailing return type to help gcc
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
    dtype aroundScaler(dtype inValue, uint8 inNumDecimals)
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
        auto a = asarray({ {inValue1, inValue2}, {inValue1, inValue2} });
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayArray1D(dtype inValue1, dtype inValue2)
    {
        std::array<dtype, 2> arr = { inValue1, inValue2 };
        auto a = asarray(arr, false);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayArray1DCopy(dtype inValue1, dtype inValue2)
    {
        std::array<dtype, 2> arr = { inValue1, inValue2 };
        auto a = asarray(arr, true);
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
        auto a = asarray(arr, false);
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
        auto a = asarray(arr, true);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayVector1D(dtype inValue1, dtype inValue2)
    {
        std::vector<dtype> arr = { inValue1, inValue2 };
        auto a = asarray(arr, false);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayVector1DCopy(dtype inValue1, dtype inValue2)
    {
        std::vector<dtype> arr = { inValue1, inValue2 };
        auto a = asarray(arr, true);
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
        auto a = asarray(arr, false);
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
        auto a = asarray(arr, true);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayDeque1D(dtype inValue1, dtype inValue2)
    {
        std::deque<dtype> arr = { inValue1, inValue2 };
        auto a = asarray(arr);
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
        auto a = asarray(arr);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayIterators(dtype inValue1, dtype inValue2)
    {
        std::vector<dtype> arr = { inValue1, inValue2 };
        auto a = asarray(arr.begin(), arr.end());
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointerIterators(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        auto a = asarray(ptr.get(), ptr.get() + 2);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointer(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        auto a = asarray(ptr.release(), uint32{ 2 });
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointer2D(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(4);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        ptr[2] = inValue1;
        ptr[3] = inValue2;
        auto a = asarray(ptr.release(), uint32{ 2 }, uint32{ 2 });
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointerShell(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        auto a = asarray(ptr.get(), uint32{ 2 }, false);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointerShell2D(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(4);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        ptr[2] = inValue1;
        ptr[3] = inValue2;
        auto a = asarray(ptr.get(), uint32{ 2 }, uint32{ 2 }, false);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointerShellTakeOwnership(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(2);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        auto a = asarray(ptr.release(), 2, true);
        return nc2pybind<dtype>(a);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric asarrayPointerShell2DTakeOwnership(dtype inValue1, dtype inValue2)
    {
        auto ptr = std::make_unique<dtype[]>(4);
        ptr[0] = inValue1;
        ptr[1] = inValue2;
        ptr[2] = inValue1;
        ptr[3] = inValue2;
        auto a = asarray(ptr.release(), 2, 2, true);
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
    pbArrayGeneric averageWeighted(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(average(inArray, inWeights, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric averageWeightedComplex(const NdArray<std::complex<dtype>>& inArray, 
        const NdArray<dtype>& inWeights, Axis inAxis = Axis::NONE)
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
    pbArrayGeneric bincountWeighted(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0)
    {
        return nc2pybind(bincount(inArray, inWeights, inMinLength));
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
    pbArrayGeneric andOperatorScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2pybind(inArray && inScaler);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric orOperatorArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 || inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric orOperatorScaler(const NdArray<dtype>& inArray, dtype inScaler)
    {
        return nc2pybind(inArray || inScaler);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric byteswap(const NdArray<dtype>& inArray)
    {
        return nc2pybind(nc::byteswap(inArray));
    }

    //================================================================================

    template<typename dtype>
    dtype cbrtScaler(dtype inValue) 
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
    dtype ceilScaler(dtype inValue) 
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
    dtype clipScaler(dtype inValue, dtype inMinValue, dtype inMaxValue)
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
    pbArrayGeneric column_stack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::column_stack({ inArray1, inArray2, inArray3, inArray4 }));
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
    std::complex<dtype> conjScaler(const std::complex<dtype>& inValue)
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
    pbArrayGeneric concatenate(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4, Axis inAxis)
    {
        return nc2pybind(nc::concatenate({ inArray1, inArray2, inArray3, inArray4 }, inAxis));
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
    auto cosScaler(dtype inValue) -> decltype(cos(inValue)) // trailing return type to help gcc
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
    auto coshScaler(dtype inValue) -> decltype(cosh(inValue)) // trailing return type to help gcc
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
    dtype deg2radScaler(dtype inValue) 
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
    dtype degreesScaler(dtype inValue) 
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
    pbArrayGeneric deleteIndicesScaler(const NdArray<dtype>& inArray, uint32 inIndex, Axis inAxis)
    {
        return nc2pybind(deleteIndices(inArray, inIndex, inAxis));
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
    auto expScaler(dtype inValue) -> decltype(exp(inValue)) // trailing return type to help gcc
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
    dtype exp2Scaler(dtype inValue) 
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
    dtype expm1Scaler(dtype inValue) 
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
    dtype fixScaler(dtype inValue) 
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
    dtype floorScaler(dtype inValue) 
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
    dtype floor_divideScaler(dtype inValue1, dtype inValue2) 
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
    dtype fmaxScaler(dtype inValue1, dtype inValue2) 
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
    dtype fminScaler(dtype inValue1, dtype inValue2) 
    {
        return fmin(inValue1, inValue2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric fminArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(fmin(inArray1, inArray2));
    }

    template<typename dtype>
    dtype fmodScaler(dtype inValue1, dtype inValue2) 
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
    pbArrayGeneric fromiter(const NdArray<dtype>& inArray)
    {
        std::vector<dtype> vec(inArray.begin(), inArray.end());
        return nc2pybind(nc::fromiter<dtype>(vec.begin(), vec.end()));
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

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_gcd_lcm)
    template<typename dtype>
    dtype gcdScaler(dtype inValue1, dtype inValue2) 
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
    pb11::tuple histogram(const NdArray<dtype>& inArray, uint32 inNumBins = 10)
    {
        std::pair<NdArray<uint32>, NdArray<double> > output = nc::histogram(inArray, inNumBins);
        return pb11::make_tuple(output.first, output.second);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric hstack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::hstack({ inArray1, inArray2, inArray3, inArray4 }));
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
    pbArrayGeneric hypotArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(hypot(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    dtype imagScaler(const std::complex<dtype>& inValue)
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

    //================================================================================

    template<typename dtype>
    pbArrayGeneric interp(const NdArray<dtype>& inX, const NdArray<dtype>& inXp, const NdArray<dtype>& inFp)
    {
        return nc2pybind(nc::interp(inX, inXp, inFp));
    }

    //================================================================================

    template<typename dtype>
    bool isinfScaler(dtype inValue) 
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
    bool isposinfScaler(dtype inValue) 
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
    bool isneginfScaler(dtype inValue) 
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
    bool isnanScaler(dtype inValue) 
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

    pbArrayGeneric kaiser(nc::int32 m, double beta)
    {
        return nc2pybind(nc::kaiser(m, beta));
    }

    //================================================================================

    template<typename dtype>
    dtype ldexpScaler(dtype inValue1, uint8 inValue2) 
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

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_gcd_lcm)
    template<typename dtype>
    dtype lcmScaler(dtype inValue1, dtype inValue2) 
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
    auto logScaler(dtype inValue) -> decltype(log(inValue)) // trailing return type to help gcc
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
    auto logbScaler(dtype inValue, dtype base) -> decltype(logb(inValue)) // trailing return type to help gcc
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
    pbArrayGeneric logspace(dtype start, dtype stop, uint32 num, bool endPoint, double base)
    {
        return nc2pybind(nc::logspace(start, stop, num, endPoint, base));
    }

    //================================================================================

    template<typename dtype>
    auto log10Scaler(dtype inValue) -> decltype(log10(inValue)) // trailing return type to help gcc
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
    dtype log1pScaler(dtype inValue) 
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
    dtype log2Scaler(dtype inValue) 
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
    pbArrayGeneric maximum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(nc::maximum(inArray1, inArray2));
    }

    //================================================================================

    template<typename dtype>
    std::pair<NdArray<dtype>, NdArray<dtype> > meshgrid(const Slice& inISlice, const Slice& inJSlice)
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
    pbArrayGeneric minimum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(nc::minimum(inArray1, inArray2));
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
    dtype newbyteorderScaler(dtype inValue, Endian inEndianess)
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
    pbArrayGeneric select(std::vector<pbArray<bool>> condlist, 
        std::vector<pbArray<dtype>> choicelist, dtype defaultValue)
    {
        std::vector<NdArray<bool>> condVec{};
        std::vector<const NdArray<bool>*> condVecPtr{};
        condVec.reserve(condlist.size());
        condVecPtr.reserve(condlist.size());
        for (auto& cond : condlist)
        {
            condVec.push_back(pybind2nc(cond));
            condVecPtr.push_back(&condVec.back());
        }

        std::vector<NdArray<dtype>> choiceVec{};
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
    pbArrayGeneric selectVector(std::vector<pbArray<bool>> condlist, 
        std::vector<pbArray<dtype>> choicelist, dtype defaultValue)
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
    pbArrayGeneric selectInitializerList(pbArray<bool> cond1, 
        pbArray<bool> cond2, 
        pbArray<bool> cond3,
        pbArray<dtype> choice1,
        pbArray<dtype> choice2,
        pbArray<dtype> choice3,
        dtype defaultValue)
    {
        return nc2pybind(nc::select({pybind2nc(cond1), pybind2nc(cond2), pybind2nc(cond3)},
            {pybind2nc(choice1), pybind2nc(choice2), pybind2nc(choice3)}, 
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
    std::complex<dtype> polarScaler(dtype mag, dtype angle)
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
    pbArrayGeneric powerArrayScaler(const NdArray<dtype>& inArray, uint8 inExponent)
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
    pbArrayGeneric powerfArrayScaler(const NdArray<dtype>& inArray, dtype inExponent)
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
    std::complex<dtype> projScaler(const std::complex<dtype>& inValue)
    {
        return nc::proj(inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric projArray(const NdArray<std::complex<dtype>>& inArray)
    {
        return nc2pybind(proj(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putmask(NdArray<dtype>& inArray, const NdArray<bool>& inMask, const NdArray<dtype>& inValues)
    {
        return nc2pybind(nc::putmask(inArray, inMask, inValues));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putmaskScaler(NdArray<dtype>& inArray, const NdArray<bool>& inMask, dtype inValue)
    {
        return nc2pybind(putmask(inArray, inMask, inValue));
    }

    //================================================================================

    template<typename dtype>
    dtype rad2degScaler(dtype inValue) 
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
    dtype radiansScaler(dtype inValue) 
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
    dtype realScaler(const std::complex<dtype>& inValue)
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
    dtype remainderScaler(dtype inValue1, dtype inValue2) 
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
    dtype rintScaler(dtype inValue) 
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
    dtype roundScaler(dtype inValue, uint8 inDecimals)
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
    pbArrayGeneric row_stack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::row_stack({ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    int8 signScaler(dtype inValue) 
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
    bool signbitScaler(dtype inValue) 
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
    auto sinScaler(dtype inValue) -> decltype(sin(inValue)) // trailing return type to help gcc
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
    dtype sincScaler(dtype inValue) 
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
    auto sinhScaler(dtype inValue) -> decltype(sinh(inValue)) // trailing return type to help gcc
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
    auto sqrtScaler(dtype inValue) -> decltype(sqrt(inValue)) // trailing return type to help gcc
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
    dtype squareScaler(dtype inValue) 
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

    template<typename Type1, typename Type2>
    pbArrayGeneric subtract(const Type1& in1, const Type2& in2)
    {
        return nc2pybind(nc::subtract(in1, in2));
    }

    //================================================================================

    template<typename dtype>
    auto tanScaler(dtype inValue) -> decltype(tan(inValue)) // trailing return type to help gcc
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
    auto tanhScaler(dtype inValue) -> decltype(tanh(inValue)) // trailing return type to help gcc
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
    dtype unwrapScaler(dtype inValue) 
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
    double truncScaler(dtype inValue) 
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

    template<typename dtype>
    pbArrayGeneric stack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4, nc::Axis inAxis)
    {
        return nc2pybind(nc::stack({ inArray1, inArray2, inArray3, inArray4 }, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric vstack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2,
        const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4)
    {
        return nc2pybind(nc::vstack({ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric whereArrayArray(const NdArray<bool>& inMask, const NdArray<dtype>& inA, const NdArray<dtype>& inB)
    {
        return nc2pybind(nc::where(inMask, inA, inB));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric whereScalerArray(const NdArray<bool>& inMask, const NdArray<dtype>& inA, dtype inB)
    {
        return nc2pybind(nc::where(inMask, inA, inB));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric whereArrayScaler(const NdArray<bool>& inMask, dtype inA, const NdArray<dtype>& inB)
    {
        return nc2pybind(nc::where(inMask, inA, inB));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric whereScalerScaler(const NdArray<bool>& inMask, dtype inA, dtype inB)
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

namespace RandomInterface
{
    template<typename dtype>
    dtype choiceSingle(const NdArray<dtype>& inArray)
    {
        return random::choice(inArray);
    }

    template<typename dtype>
    pbArrayGeneric choiceMultiple(const NdArray<dtype>& inArray, uint32 inNum)
    {
        return nc2pybind(random::choice(inArray, inNum));
    }

    template<typename dtype>
    pbArrayGeneric permutationScaler(dtype inValue)
    {
        return nc2pybind(random::permutation(inValue));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric permutationArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(random::permutation(inArray));
    }
}  // namespace RandomInterface

namespace LinalgInterface
{
    template<typename dtype>
    pbArrayGeneric hatArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(linalg::hat(inArray));
    }

    template<typename dtype>
    pbArrayGeneric multi_dot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, const NdArray<dtype>& inArray3, const NdArray<dtype>& inArray4)
    {
        return nc2pybind(linalg::multi_dot({ inArray1, inArray2, inArray3, inArray4 }));
    }

    template<typename dtype>
    pb11::tuple pivotLU_decomposition(const NdArray<dtype>& inArray)
    {
        auto lup = linalg::pivotLU_decomposition(inArray);
        auto& l = std::get<0>(lup);
        auto& u = std::get<1>(lup);
        auto& p = std::get<2>(lup);
        return pb11::make_tuple(nc2pybind(l), nc2pybind(u), nc2pybind(p));
    }

    template<typename dtype>
    pbArray<double> solve(const NdArray<dtype>& inA, const NdArray<dtype>& inB)
    {
        return nc2pybind(linalg::solve(inA, inB));
    }
}  // namespace LinalgInterface

namespace RotationsInterface
{
    pbArray<double> angleAxisRotationNdArray(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2pybind(rotations::Quaternion(inAxis, inAngle).toNdArray());
    }

    pbArray<double> angleAxisRotationVec3(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2pybind(rotations::Quaternion(Vec3(inAxis), inAngle).toNdArray());
    }

    pbArray<double> angularVelocity(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2, double inTime)
    {
        return nc2pybind(inQuat1.angularVelocity(inQuat2, inTime));
    }

    pbArray<double> nlerp(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2, double inPercent)
    {
        return nc2pybind(inQuat1.nlerp(inQuat2, inPercent).toNdArray());
    }

    pbArray<double> rotateNdArray(const rotations::Quaternion& inQuat, const NdArray<double>& inVec)
    {
        return nc2pybind(inQuat.rotate(inVec));
    }

    pbArray<double> rotateVec3(const rotations::Quaternion& inQuat, const NdArray<double>& inVec)
    {
        return nc2pybind(inQuat.rotate(Vec3(inVec)).toNdArray());
    }

    pbArray<double> slerp(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2, double inPercent)
    {
        return nc2pybind(inQuat1.slerp(inQuat2, inPercent).toNdArray());
    }

    pbArray<double> toDCM(const rotations::Quaternion& inQuat)
    {
        return nc2pybind(inQuat.toDCM());
    }

    pbArray<double> subtract(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2)
    {
        return nc2pybind((inQuat1 - inQuat2).toNdArray());
    }

    pbArray<double> negative(const rotations::Quaternion& inQuat)
    {
        return nc2pybind((-inQuat).toNdArray());
    }

    pbArray<double> multiplyScaler(const rotations::Quaternion& inQuat, double inScaler)
    {
        const rotations::Quaternion returnQuat = inQuat * inScaler;
        return nc2pybind(returnQuat.toNdArray());
    }

    pbArray<double> multiplyArray(const rotations::Quaternion& inQuat, const NdArray<double>& inArray)
    {
        NdArray<double> returnArray = inQuat * inArray;
        return nc2pybind(returnArray);
    }

    pbArray<double> multiplyQuaternion(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2)
    {
        const rotations::Quaternion returnQuat = inQuat1 * inQuat2;
        return nc2pybind(returnQuat.toNdArray());
    }

    pbArray<double> eulerAnglesValues(double roll, double pitch, double yaw)
    {
        return nc2pybind(rotations::DCM::eulerAngles(roll, pitch, yaw));
    }

    pbArray<double> eulerAnglesArray(const NdArray<double>& angles)
    {
        return nc2pybind(rotations::DCM::eulerAngles(angles));
    }

    pbArray<double> angleAxisRotationDcmNdArray(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2pybind(rotations::DCM::eulerAxisAngle(inAxis, inAngle));
    }

    pbArray<double> angleAxisRotationDcmVec3(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2pybind(rotations::DCM::eulerAxisAngle(Vec3(inAxis), inAngle));
    }

    template<typename T>
    pbArrayGeneric rodriguesRotation(pbArray<T>& inK, double inTheta, pbArray<T>& inV)
    {
        auto k = pybind2nc(inK);
        auto v = pybind2nc(inV);

        return nc2pybind(rotations::rodriguesRotation(k, inTheta, v));
    }

    template<typename T>
    pbArrayGeneric wahbasProblem(pbArray<T>& inWk, pbArray<T>& inVk)
    {
        auto wk = pybind2nc(inWk);
        auto vk = pybind2nc(inVk);
        return nc2pybind(rotations::wahbasProblem(wk, vk));
    }

    template<typename T>
    pbArrayGeneric wahbasProblemWeighted(pbArray<T>& inWk, pbArray<T>& inVk, pbArray<T>& inAk)
    {
        auto wk = pybind2nc(inWk);
        auto vk = pybind2nc(inVk);
        auto ak = pybind2nc(inAk);
        return nc2pybind(rotations::wahbasProblem(wk, vk, ak));
    }
} // namespace RotationsInterface

namespace RaInterface
{
    void print(const coordinates::RA& inRa)
    {
        std::cout << inRa;
    }
} // namespace RaInterface

namespace DecInterface
{
    void print(const coordinates::Dec& self)
    {
        std::cout << self;
    }
} // namespace DecInterface

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
}  // namespace CoordinateInterface

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

    template<typename dtype>
    pbArrayGeneric sliceZIndexAll(const DataCube<dtype>& self, int32 inIndex)
    {
        return nc2pybind(self.sliceZAll(inIndex));
    }

    template<typename dtype>
    pbArrayGeneric sliceZIndex(const DataCube<dtype>& self, int32 inIndex, const Slice& inSlice)
    {
        return nc2pybind(self.sliceZ(inIndex, inSlice));
    }

    template<typename dtype>
    pbArrayGeneric sliceZRowColAll(const DataCube<dtype>& self, int32 inRow, int32 inCol)
    {
        return nc2pybind(self.sliceZAll(inRow, inCol));
    }

    template<typename dtype>
    pbArrayGeneric sliceZRowCol(const DataCube<dtype>& self, int32 inRow, int32 inCol, const Slice& inSlice)
    {
        return nc2pybind(self.sliceZ(inRow, inCol, inSlice));
    }

    template<typename dtype>
    pbArrayGeneric sliceZSliceScalerAll(const DataCube<dtype>& self, const Slice& inRow, int32 inCol)
    {
        return nc2pybind(self.sliceZAll(inRow, inCol));
    }

    template<typename dtype>
    pbArrayGeneric sliceZSliceScaler(const DataCube<dtype>& self, const Slice& inRow, int32 inCol, const Slice& inSlice)
    {
        return nc2pybind(self.sliceZ(inRow, inCol, inSlice));
    }

    template<typename dtype>
    pbArrayGeneric sliceZScalerSliceAll(const DataCube<dtype>& self, int32 inRow, const Slice& inCol)
    {
        return nc2pybind(self.sliceZAll(inRow, inCol));
    }

    template<typename dtype>
    pbArrayGeneric sliceZScalerSlice(const DataCube<dtype>& self, int32 inRow, const Slice& inCol, const Slice& inSlice)
    {
        return nc2pybind(self.sliceZ(inRow, inCol, inSlice));
    }

    template<typename dtype>
    DataCube<dtype> sliceZSliceSliceAll(const DataCube<dtype>& self, const Slice& inRow, const Slice& inCol)
    {
        return self.sliceZAll(inRow, inCol);
    }

    template<typename dtype>
    DataCube<dtype> sliceZSliceSlice(const DataCube<dtype>& self, const Slice& inRow, const Slice& inCol, const Slice& inSlice)
    {
        return self.sliceZ(inRow, inCol, inSlice);
    }

    template<typename dtype>
    pbArrayGeneric sliceZAtIndexAll(const DataCube<dtype>& self, int32 inIndex)
    {
        return nc2pybind(self.sliceZAllat(inIndex));
    }

    template<typename dtype>
    pbArrayGeneric sliceZAtIndex(const DataCube<dtype>& self, int32 inIndex, const Slice& inSlice)
    {
        return nc2pybind(self.sliceZat(inIndex, inSlice));
    }

    template<typename dtype>
    pbArrayGeneric sliceZAtRowColAll(const DataCube<dtype>& self, int32 inRow, int32 inCol)
    {
        return nc2pybind(self.sliceZAllat(inRow, inCol));
    }

    template<typename dtype>
    pbArrayGeneric sliceZAtRowCol(const DataCube<dtype>& self, int32 inRow, int32 inCol, const Slice& inSlice)
    {
        return nc2pybind(self.sliceZat(inRow, inCol, inSlice));
    }

    template<typename dtype>
    pbArrayGeneric sliceZAtSliceScalerAll(const DataCube<dtype>& self, const Slice& inRow, int32 inCol)
    {
        return nc2pybind(self.sliceZAllat(inRow, inCol));
    }

    template<typename dtype>
    pbArrayGeneric sliceZAtSliceScaler(const DataCube<dtype>& self, const Slice& inRow, int32 inCol, const Slice& inSlice)
    {
        return nc2pybind(self.sliceZat(inRow, inCol, inSlice));
    }

    template<typename dtype>
    pbArrayGeneric sliceZAtScalerSliceAll(const DataCube<dtype>& self, int32 inRow, const Slice& inCol)
    {
        return nc2pybind(self.sliceZAllat(inRow, inCol));
    }

    template<typename dtype>
    pbArrayGeneric sliceZAtScalerSlice(const DataCube<dtype>& self, int32 inRow, const Slice& inCol, const Slice& inSlice)
    {
        return nc2pybind(self.sliceZat(inRow, inCol, inSlice));
    }

    template<typename dtype>
    DataCube<dtype> sliceZAtSliceSliceAll(const DataCube<dtype>& self, const Slice& inRow, const Slice& inCol)
    {
        return self.sliceZAllat(inRow, inCol);
    }

    template<typename dtype>
    DataCube<dtype> sliceZAtSliceSlice(const DataCube<dtype>& self, const Slice& inRow, const Slice& inCol, const Slice& inSlice)
    {
        return self.sliceZat(inRow, inCol, inSlice);
    }

}  // namespace DataCubeInterface

//================================================================================

namespace PolynomialInterface
{
#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype chebyshev_t_Scaler(uint32 n, dtype inValue) 
    {
        return polynomial::chebyshev_t(n, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric chebyshev_t_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::chebyshev_t(n, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype chebyshev_u_Scaler(uint32 n, dtype inValue) 
    {
        return polynomial::chebyshev_u(n, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric chebyshev_u_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::chebyshev_u(n, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype hermite_Scaler(uint32 n, dtype inValue) 
    {
        return polynomial::hermite(n, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric hermite_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::hermite(n, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype laguerre_Scaler1(uint32 n, dtype inValue) 
    {
        return polynomial::laguerre(n, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype laguerre_Scaler2(uint32 n, uint32 m, dtype inValue) 
    {
        return polynomial::laguerre(n, m, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric laguerre_Array1(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::laguerre(n, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric laguerre_Array2(uint32 n, uint32 m, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::laguerre(n, m, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype legendre_p_Scaler1(int32 n, dtype inValue) 
    {
        return polynomial::legendre_p(n, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype legendre_p_Scaler2(int32 n, int32 m, dtype inValue) 
    {
        return polynomial::legendre_p(n, m, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric legendre_p_Array1(int32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::legendre_p(n, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric legendre_p_Array2(int32 n, int32 m, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::legendre_p(n, m, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype legendre_q_Scaler(int32 n, dtype inValue) 
    {
        return polynomial::legendre_q(n, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric legendre_q_Array(int32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::legendre_q(n, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    std::vector<double> spherical_harmonic(uint32 n, int32 m, dtype theta, dtype phi)
    {
        auto value = polynomial::spherical_harmonic(n, m, theta, phi);
        std::vector<double> valueVec = {value.real(), value.imag()};
        return valueVec;
    }
#endif
}  // namespace PolynomialInterface

namespace RootsInterface
{
    constexpr double EPSILON = 1e-10;

    //================================================================================

    double bisection(const polynomial::Poly1d<double>&p, double a, double b)
    {
        auto rootFinder = roots::Bisection(EPSILON, p);
        return rootFinder.solve(a, b);
    }

    //================================================================================

    double brent(const polynomial::Poly1d<double>&p, double a, double b)
    {
        auto rootFinder = roots::Brent(EPSILON, p);
        return rootFinder.solve(a, b);
    }

    //================================================================================

    double dekker(const polynomial::Poly1d<double>&p, double a, double b)
    {
        auto rootFinder = roots::Dekker(EPSILON, p);
        return rootFinder.solve(a, b);
    }

    //================================================================================

    double newton(const polynomial::Poly1d<double>&p, double x)
    {
        auto pPrime = p.deriv();
        auto rootFinder = roots::Newton(EPSILON, p, pPrime);
        return rootFinder.solve(x);
    }

    //================================================================================

    double secant(const polynomial::Poly1d<double>&p, double a, double b)
    {
        auto rootFinder = roots::Secant(EPSILON, p);
        return rootFinder.solve(a, b);
    }
}  // namespace RootsInterface

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
}  // namespace IntegrateInterface

namespace Vec2Interface
{
    pbArray<double> toNdArray(const Vec2& self)
    {
        return nc2pybind(self.toNdArray());
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
}  // namespace Vec2Interface

namespace Vec3Interface
{
    pbArray<double> toNdArray(const Vec3& self)
    {
        return nc2pybind(self.toNdArray());
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
} // namespace Vec3Interface

//================================================================================

namespace SpecialInterface
{
#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype airy_ai_Scaler(dtype inValue) 
    {
        return special::airy_ai(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric airy_ai_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::airy_ai(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype airy_ai_prime_Scaler(dtype inValue) 
    {
        return special::airy_ai_prime(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric airy_ai_prime_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::airy_ai_prime(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype airy_bi_Scaler(dtype inValue) 
    {
        return special::airy_bi(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric airy_bi_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::airy_bi(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype airy_bi_prime_Scaler(dtype inValue) 
    {
        return special::airy_bi_prime(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric airy_bi_prime_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::airy_bi_prime(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    double bernoulli_Scaler(uint32 n) 
    {
        return special::bernoilli(n);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    pbArray<double> bernoulli_Array(const NdArray<uint32>& inArray)
    {
        return nc2pybind(special::bernoilli(inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype bessel_in_Scaler(dtype inV, dtype inValue) 
    {
        return special::bessel_in(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric bessel_in_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_in(inV, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype bessel_in_prime_Scaler(dtype inV, dtype inValue) 
    {
        return special::bessel_in_prime(inV, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric bessel_in_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_in_prime(inV, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype bessel_jn_Scaler(dtype inV, dtype inValue) 
    {
        return special::bessel_jn(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric bessel_jn_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_jn(inV, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype bessel_jn_prime_Scaler(dtype inV, dtype inValue) 
    {
        return special::bessel_jn_prime(inV, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric bessel_jn_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_jn_prime(inV, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype bessel_kn_Scaler(dtype inV, dtype inValue) 
    {
        return special::bessel_kn(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric bessel_kn_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_kn(inV, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype bessel_kn_prime_Scaler(dtype inV, dtype inValue) 
    {
        return special::bessel_kn_prime(inV, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric bessel_kn_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_kn_prime(inV, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype bessel_yn_Scaler(dtype inV, dtype inValue) 
    {
        return special::bessel_yn(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric bessel_yn_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_yn(inV, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype bessel_yn_prime_Scaler(dtype inV, dtype inValue) 
    {
        return special::bessel_yn_prime(inV, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric bessel_yn_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_yn_prime(inV, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype beta_Scaler(dtype a, dtype b) 
    {
        return special::beta(a, b);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric beta_Array(const NdArray<dtype>& a, const NdArray<dtype>& b)
    {
        return nc2pybind(special::beta(a, b));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype comp_ellint_1_Scaler(dtype k) 
    {
        return special::comp_ellint_1(k);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric comp_ellint_1_Array(const NdArray<dtype>& k) 
    {
        return nc2pybind(special::comp_ellint_1(k));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype comp_ellint_2_Scaler(dtype k) 
    {
        return special::comp_ellint_2(k);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric comp_ellint_2_Array(const NdArray<dtype>& k) 
    {
        return nc2pybind(special::comp_ellint_2(k));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype comp_ellint_3_Scaler(dtype k, dtype v) 
    {
        return special::comp_ellint_3(k, v);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype1, typename dtype2>
    pbArrayGeneric comp_ellint_3_Array(const NdArray<dtype1>& k, const NdArray<dtype2>& v) 
    {
        return nc2pybind(special::comp_ellint_3(k, v));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    std::complex<dtype> cyclic_hankel_1_Scaler(dtype v, dtype x) 
    {
        return special::cyclic_hankel_1(v, x);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric cyclic_hankel_1_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2pybind(special::cyclic_hankel_1(v, x));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    std::complex<dtype> cyclic_hankel_2_Scaler(dtype v, dtype x) 
    {
        return special::cyclic_hankel_2(v, x);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric cyclic_hankel_2_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2pybind(special::cyclic_hankel_2(v, x));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype ellint_1_Scaler(dtype k, dtype p) 
    {
        return special::ellint_1(k, p);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype1, typename dtype2>
    pbArrayGeneric ellint_1_Array(const NdArray<dtype1>& k, const NdArray<dtype2>& p) 
    {
        return nc2pybind(special::ellint_1(k, p));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype ellint_2_Scaler(dtype k, dtype p) 
    {
        return special::ellint_2(k, p);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype1, typename dtype2>
    pbArrayGeneric ellint_2_Array(const NdArray<dtype1>& k, const NdArray<dtype2>& p) 
    {
        return nc2pybind(special::ellint_2(k, p));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype ellint_3_Scaler(dtype k, dtype v, dtype p) 
    {
        return special::ellint_3(k, v, p);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype1, typename dtype2, typename dtype3>
    pbArrayGeneric ellint_3_Array(const NdArray<dtype1>& k, const NdArray<dtype2>& v, const NdArray<dtype3>& p) 
    {
        return nc2pybind(special::ellint_3(k, v, p));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype expint_Scaler(dtype k) 
    {
        return special::expint(k);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric expint_Array(const NdArray<dtype>& k) 
    {
        return nc2pybind(special::expint(k));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype digamma_Scaler(dtype inValue) 
    {
        return special::digamma(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric digamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::digamma(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype erf_Scaler(dtype inValue) 
    {
        return special::erf(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric erf_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::erf(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype erf_inv_Scaler(dtype inValue) 
    {
        return special::erf_inv(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric erf_inv_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::erf_inv(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype erfc_Scaler(dtype inValue) 
    {
        return special::erfc(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric erfc_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::erfc(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype erfc_inv_Scaler(dtype inValue) 
    {
        return special::erfc_inv(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric erfc_inv_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::erfc_inv(inArray));
    }
#endif

    //================================================================================

    double factorial_Scaler(uint32 inValue) 
    {
        return special::factorial(inValue);
    }

    //================================================================================

    pbArray<double> factorial_Array(const NdArray<uint32>& inArray)
    {
        return nc2pybind(special::factorial(inArray));
    }

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype gamma_Scaler(dtype inValue) 
    {
        return special::gamma(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric gamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::gamma(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype gamma1pm1_Scaler(dtype inValue) 
    {
        return special::gamma1pm1(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric gamma1pm1_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::gamma1pm1(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype log_gamma_Scaler(dtype inValue) 
    {
        return special::log_gamma(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric log_gamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::log_gamma(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype polygamma_Scaler(uint32 n, dtype inValue) 
    {
        return special::polygamma(n, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric polygamma_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::polygamma(n, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    double prime_Scaler(uint32 inValue) 
    {
        return special::prime(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    pbArray<uint32> prime_Array(const NdArray<uint32>& inArray)
    {
        return nc2pybind(special::prime(inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype riemann_zeta_Scaler(dtype inValue) 
    {
        return special::riemann_zeta(inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric riemann_zeta_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::riemann_zeta(inArray));
    }
#endif

    //================================================================================

    template<typename dtype>
    pbArrayGeneric softmax(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return nc2pybind(special::softmax(inArray, inAxis));
    }

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype spherical_bessel_jn_Scaler(uint32 inV, dtype inValue) 
    {
        return special::spherical_bessel_jn(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric spherical_bessel_jn_Array(uint32 inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::spherical_bessel_jn(inV, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype spherical_bessel_yn_Scaler(uint32 inV, dtype inValue) 
    {
        return special::spherical_bessel_yn(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric spherical_bessel_yn_Array(uint32 inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::spherical_bessel_yn(inV, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    std::complex<dtype> spherical_hankel_1_Scaler(dtype v, dtype x) 
    {
        return special::spherical_hankel_1(v, x);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric spherical_hankel_1_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2pybind(special::spherical_hankel_1(v, x));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    std::complex<dtype> spherical_hankel_2_Scaler(dtype v, dtype x) 
    {
        return special::spherical_hankel_2(v, x);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric spherical_hankel_2_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2pybind(special::spherical_hankel_2(v, x));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype trigamma_Scaler(dtype inValue) 
    {
        return special::trigamma(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric trigamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::trigamma(inArray));
    }
#endif
}  // namespace SpecialInterface

//================================================================================

PYBIND11_MODULE(NumCppPy, m)
{
    m.doc() = "NumCpp unit test bindings";

    typedef std::pair<NdArray<double>, NdArray<double>> doublePair;
    pb11::class_<doublePair>(m, "doublePair")
        .def(pb11::init<>())
        .def_readonly("first", &doublePair::first)
        .def_readonly("second", &doublePair::second);

    typedef std::pair<NdArray<uint32>, NdArray<uint32>> uint32Pair;
    pb11::class_<uint32Pair>(m, "uint32Pair")
        .def(pb11::init<>())
        .def_readonly("first", &uint32Pair::first)
        .def_readonly("second", &uint32Pair::second);

#ifdef NUMCPP_NO_USE_BOOST
    m.attr("NUMCPP_NO_USE_BOOST") = true;
#else
    m.attr("NUMCPP_NO_USE_BOOST") = false;
#endif

#ifdef __cpp_lib_gcd_lcm
    m.attr("STL_GCD_LCM") = true;
#else
    m.attr("STL_GCD_LCM") = false;
#endif

#ifdef __cpp_lib_clamp
    m.attr("STL_CLAMP") = true;
#else
    m.attr("STL_CLAMP") = false;
#endif

#ifdef __cpp_lib_hypot
    m.attr("STL_HYPOT") = true;
#else
    m.attr("STL_HYPOT") = false;
#endif

#ifdef __cpp_lib_math_special_functions
    m.attr("STL_SPECIAL_FUNCTIONS") = true;
#else
    m.attr("STL_SPECIAL_FUNCTIONS") = false;
#endif

#ifdef __cpp_lib_execution
    m.attr("STL_LIB_EXECUTION") = true;
#else
    m.attr("STL_LIB_EXECUTION") = false;
#endif

#ifdef __cpp_lib_parallel_algorithm
    m.attr("STL_LIB_PARALLEL_ALGORITHM") = true;
#else
    m.attr("STL_LIB_PARALLEL_ALGORITHM") = false;
#endif

    // Version.hpp
    m.attr("VERSION") = VERSION;

    // Constants.hpp
    m.attr("c") = constants::c;
    m.attr("e") = constants::e;
    m.attr("inf") = constants::inf;
    m.attr("pi") = constants::pi;
    m.attr("nan") = constants::nan;
    m.attr("j") = constants::j;
    m.attr("DAYS_PER_WEEK") = constants::DAYS_PER_WEEK;
    m.attr("MINUTES_PER_HOUR") = constants::MINUTES_PER_HOUR;
    m.attr("SECONDS_PER_MINUTE") = constants::SECONDS_PER_MINUTE;
    m.attr("MILLISECONDS_PER_SECOND") = constants::MILLISECONDS_PER_SECOND;
    m.attr("SECONDS_PER_HOUR") = constants::SECONDS_PER_HOUR;
    m.attr("HOURS_PER_DAY") = constants::HOURS_PER_DAY;
    m.attr("MINUTES_PER_DAY") = constants::MINUTES_PER_DAY;
    m.attr("SECONDS_PER_DAY") = constants::SECONDS_PER_DAY;
    m.attr("MILLISECONDS_PER_DAY") = constants::MILLISECONDS_PER_DAY;
    m.attr("SECONDS_PER_WEEK") = constants::SECONDS_PER_WEEK;

    // DtypeInfo.hpp
    using DtypeInfoUint32 = DtypeInfo<uint32>;
    pb11::class_<DtypeInfoUint32>
        (m, "DtypeIntoUint32")
        .def(pb11::init<>())
        .def_static("bits", &DtypeInfoUint32::bits)
        .def_static("epsilon", &DtypeInfoUint32::epsilon)
        .def_static("isInteger", &DtypeInfoUint32::isInteger)
        .def_static("isSigned", &DtypeInfoUint32::isSigned)
        .def_static("min", &DtypeInfoUint32::min)
        .def_static("max", &DtypeInfoUint32::max);

    using DtypeInfoComplexDouble = DtypeInfo<std::complex<double> >;
    pb11::class_<DtypeInfoComplexDouble>
        (m, "DtypeInfoComplexDouble")
        .def(pb11::init<>())
        .def_static("bits", &DtypeInfoComplexDouble::bits)
        .def_static("epsilon", &DtypeInfoComplexDouble::epsilon)
        .def_static("isInteger", &DtypeInfoComplexDouble::isInteger)
        .def_static("isSigned", &DtypeInfoComplexDouble::isSigned)
        .def_static("min", &DtypeInfoComplexDouble::min)
        .def_static("max", &DtypeInfoComplexDouble::max);

    // Shape.hpp
    pb11::class_<Shape>
        (m, "Shape")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def_static("testListContructor", &ShapeInterface::testListContructor)
        .def_readwrite("rows", &Shape::rows)
        .def_readwrite("cols", &Shape::cols)
        .def("size", &Shape::size)
        .def("print", &Shape::print)
        .def("__str__", &Shape::str)
        .def("__eq__", &Shape::operator==)
        .def("__neq__", &Shape::operator!=);

    // Slice.hpp
    pb11::class_<Slice>
        (m, "Slice")
        .def(pb11::init<>())
        .def(pb11::init<int32>())
        .def(pb11::init<int32, int32>())
        .def(pb11::init<int32, int32, int32>())
        .def(pb11::init<Slice>())
        .def_readwrite("start", &Slice::start)
        .def_readwrite("stop", &Slice::stop)
        .def_readwrite("step", &Slice::step)
        .def("numElements", &Slice::numElements)
        .def("print", &Slice::print)
        .def("__str__", &Slice::str)
        .def("__eq__", &Slice::operator==)
        .def("__neq__", &Slice::operator!=);

    // Timer.hpp
    using MicroTimer = Timer<std::chrono::microseconds>;
    pb11::class_<MicroTimer>
        (m, "Timer")
        .def(pb11::init<>())
        .def(pb11::init<std::string>())
        .def("sleep", &MicroTimer::sleep)
        .def("tic", &MicroTimer::tic)
        .def("toc", &MicroTimer::toc);

    // Types.hpp
    pb11::enum_<Axis>(m, "Axis")
        .value("NONE", Axis::NONE)
        .value("ROW", Axis::ROW)
        .value("COL", Axis::COL);

    pb11::enum_<Endian>(m, "Endian")
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

    pb11::class_<NdArrayDoubleIterator>
        (m, "NdArrayDoubleIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayDoubleIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayDoubleIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayDoubleIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayDoubleIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayDoubleIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayDoubleIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayDoubleIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayDoubleIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayDoubleIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayDoubleIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayDoubleIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayDoubleIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayDoubleIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayDoubleIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayDoubleIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleIterator>);

    pb11::class_<NdArrayComplexDoubleIterator>
        (m, "NdArrayComplexDoubleIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayComplexDoubleIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayComplexDoubleIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayComplexDoubleIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayComplexDoubleIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayComplexDoubleIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayComplexDoubleIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayComplexDoubleIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayComplexDoubleIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayComplexDoubleIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayComplexDoubleIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayComplexDoubleIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleIterator>);

    pb11::class_<NdArrayDoubleConstIterator>
        (m, "NdArrayDoubleConstIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleConstIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayDoubleConstIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayDoubleConstIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayDoubleConstIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayDoubleConstIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayDoubleConstIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayDoubleConstIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayDoubleConstIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayDoubleConstIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayDoubleConstIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayDoubleConstIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayDoubleConstIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayDoubleConstIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayDoubleConstIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayDoubleConstIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayDoubleConstIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleConstIterator>);


    pb11::class_<NdArrayComplexDoubleConstIterator>
        (m, "NdArrayComplexDoubleConstIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleConstIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleConstIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleConstIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleConstIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleConstIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayComplexDoubleConstIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayComplexDoubleConstIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayComplexDoubleConstIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayComplexDoubleConstIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayComplexDoubleConstIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayComplexDoubleConstIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayComplexDoubleConstIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayComplexDoubleConstIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayComplexDoubleConstIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayComplexDoubleConstIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayComplexDoubleConstIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleConstIterator>);

    pb11::class_<NdArrayDoubleReverseIterator>
        (m, "NdArrayDoubleReverseIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleReverseIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayDoubleReverseIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayDoubleReverseIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayDoubleReverseIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayDoubleReverseIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayDoubleReverseIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayDoubleReverseIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayDoubleReverseIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayDoubleReverseIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayDoubleReverseIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayDoubleReverseIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayDoubleReverseIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayDoubleReverseIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayDoubleReverseIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayDoubleReverseIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayDoubleReverseIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleReverseIterator>);

    pb11::class_<NdArrayComplexDoubleReverseIterator>
        (m, "NdArrayComplexDoubleReverseIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleReverseIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleReverseIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleReverseIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleReverseIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleReverseIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayComplexDoubleReverseIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayComplexDoubleReverseIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayComplexDoubleReverseIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayComplexDoubleReverseIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayComplexDoubleReverseIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayComplexDoubleReverseIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayComplexDoubleReverseIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayComplexDoubleReverseIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayComplexDoubleReverseIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayComplexDoubleReverseIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayComplexDoubleReverseIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleReverseIterator>);

    pb11::class_<NdArrayDoubleConstReverseIterator>
        (m, "NdArrayDoubleConstReverseIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleConstReverseIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayDoubleConstReverseIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayDoubleConstReverseIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayDoubleConstReverseIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayDoubleConstReverseIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayDoubleConstReverseIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayDoubleConstReverseIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayDoubleConstReverseIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayDoubleConstReverseIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayDoubleConstReverseIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayDoubleConstReverseIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayDoubleConstReverseIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayDoubleConstReverseIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayDoubleConstReverseIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayDoubleConstReverseIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayDoubleConstReverseIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleConstReverseIterator>);

    pb11::class_<NdArrayComplexDoubleConstReverseIterator>
        (m, "NdArrayComplexDoubleConstReverseIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleConstReverseIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleConstReverseIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleConstReverseIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleConstReverseIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleConstReverseIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayComplexDoubleConstReverseIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayComplexDoubleConstReverseIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayComplexDoubleConstReverseIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayComplexDoubleConstReverseIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayComplexDoubleConstReverseIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayComplexDoubleConstReverseIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayComplexDoubleConstReverseIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayComplexDoubleConstReverseIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayComplexDoubleConstReverseIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayComplexDoubleConstReverseIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayComplexDoubleConstReverseIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleConstReverseIterator>);

    pb11::class_<NdArrayDoubleColumnIterator>
        (m, "NdArrayDoubleColumnIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleColumnIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayDoubleColumnIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayDoubleColumnIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayDoubleColumnIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayDoubleColumnIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayDoubleColumnIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayDoubleColumnIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayDoubleColumnIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayDoubleColumnIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayDoubleColumnIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayDoubleColumnIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayDoubleColumnIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayDoubleColumnIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayDoubleColumnIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayDoubleColumnIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayDoubleColumnIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleColumnIterator>);

    pb11::class_<NdArrayComplexDoubleColumnIterator>
        (m, "NdArrayComplexDoubleColumnIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleColumnIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleColumnIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleColumnIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleColumnIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleColumnIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayComplexDoubleColumnIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayComplexDoubleColumnIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayComplexDoubleColumnIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayComplexDoubleColumnIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayComplexDoubleColumnIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayComplexDoubleColumnIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayComplexDoubleColumnIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayComplexDoubleColumnIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayComplexDoubleColumnIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayComplexDoubleColumnIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayComplexDoubleColumnIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleColumnIterator>);

    pb11::class_<NdArrayDoubleConstColumnIterator>
        (m, "NdArrayDoubleConstColumnIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleConstColumnIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayDoubleConstColumnIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayDoubleConstColumnIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayDoubleConstColumnIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayDoubleConstColumnIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayDoubleConstColumnIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayDoubleConstColumnIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayDoubleConstColumnIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayDoubleConstColumnIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayDoubleConstColumnIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayDoubleConstColumnIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayDoubleConstColumnIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayDoubleConstColumnIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayDoubleConstColumnIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayDoubleConstColumnIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayDoubleConstColumnIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleConstColumnIterator>);

    pb11::class_<NdArrayComplexDoubleConstColumnIterator>
        (m, "NdArrayComplexDoubleConstColumnIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleConstColumnIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleConstColumnIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleConstColumnIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleConstColumnIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleConstColumnIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayComplexDoubleConstColumnIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayComplexDoubleConstColumnIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayComplexDoubleConstColumnIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayComplexDoubleConstColumnIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayComplexDoubleConstColumnIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayComplexDoubleConstColumnIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayComplexDoubleConstColumnIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayComplexDoubleConstColumnIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayComplexDoubleConstColumnIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayComplexDoubleConstColumnIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayComplexDoubleConstColumnIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleConstColumnIterator>);

    pb11::class_<NdArrayDoubleReverseColumnIterator>
        (m, "NdArrayDoubleReverseColumnIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleReverseColumnIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayDoubleReverseColumnIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayDoubleReverseColumnIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayDoubleReverseColumnIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayDoubleReverseColumnIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayDoubleReverseColumnIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayDoubleReverseColumnIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayDoubleReverseColumnIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayDoubleReverseColumnIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayDoubleReverseColumnIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayDoubleReverseColumnIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayDoubleReverseColumnIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayDoubleReverseColumnIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayDoubleReverseColumnIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayDoubleReverseColumnIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayDoubleReverseColumnIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleReverseColumnIterator>);

    pb11::class_<NdArrayComplexDoubleReverseColumnIterator>
        (m, "NdArrayComplexDoubleReverseColumnIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleReverseColumnIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleReverseColumnIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleReverseColumnIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleReverseColumnIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayComplexDoubleReverseColumnIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayComplexDoubleReverseColumnIterator>);

    pb11::class_<NdArrayDoubleConstReverseColumnIterator>
        (m, "NdArrayDoubleConstReverseColumnIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleConstReverseColumnIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayDoubleConstReverseColumnIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayDoubleConstReverseColumnIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayDoubleConstReverseColumnIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayDoubleConstReverseColumnIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayDoubleConstReverseColumnIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayDoubleConstReverseColumnIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayDoubleConstReverseColumnIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayDoubleConstReverseColumnIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayDoubleConstReverseColumnIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayDoubleConstReverseColumnIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayDoubleConstReverseColumnIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayDoubleConstReverseColumnIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayDoubleConstReverseColumnIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayDoubleConstReverseColumnIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayDoubleConstReverseColumnIterator>)
        .def("__getitem__", &IteratorInterface::access<NdArrayDoubleConstReverseColumnIterator>);

    pb11::class_<NdArrayComplexDoubleConstReverseColumnIterator>
        (m, "NdArrayComplexDoubleConstReverseColumnIterator")
        .def(pb11::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("operatorMinusMinusPost", IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("__iadd__", IteratorInterface::operatorPlusEqual<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("__add__", IteratorInterface::operatorPlus<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("__isub__", IteratorInterface::operatorMinusEqual<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("__sub__", IteratorInterface::operatorMinus<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("__sub__", IteratorInterface::operatorDiff<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("__eq__", IteratorInterface::operatorEqual<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("__ne__", IteratorInterface::operatorNotEqual<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("__lt__", IteratorInterface::operatorLess<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("__le__", IteratorInterface::operatorLessEqual<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("__gt__", IteratorInterface::operatorGreater<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("__ge__", IteratorInterface::operatorGreaterEqual<NdArrayComplexDoubleConstReverseColumnIterator>)
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

    m.def("test1DListContructor", &NdArrayInterface::test1DListContructor<double>);
    m.def("test1DListContructor", &NdArrayInterface::test1DListContructor<ComplexDouble>);
    m.def("test2DListContructor", &NdArrayInterface::test2DListContructor<double>);
    m.def("test2DListContructor", &NdArrayInterface::test2DListContructor<ComplexDouble>);
    m.def("test1dArrayConstructor", &NdArrayInterface::test1dArrayConstructor<double>);
    m.def("test1dArrayConstructor", &NdArrayInterface::test1dArrayConstructor<ComplexDouble>);
    m.def("test2dArrayConstructor", &NdArrayInterface::test2dArrayConstructor<double>);
    m.def("test2dArrayConstructor", &NdArrayInterface::test2dArrayConstructor<ComplexDouble>);
    m.def("test1dVectorConstructor", &NdArrayInterface::test1dVectorConstructor<double>);
    m.def("test1dVectorConstructor", &NdArrayInterface::test1dVectorConstructor<ComplexDouble>);
    m.def("test2dVectorConstructor", &NdArrayInterface::test2dVectorConstructor<double>);
    m.def("test2dVectorConstructor", &NdArrayInterface::test2dVectorConstructor<ComplexDouble>);
    m.def("test2dVectorArrayConstructor", &NdArrayInterface::test2dVectorArrayConstructor<double>);
    m.def("test2dVectorArrayConstructor", &NdArrayInterface::test2dVectorArrayConstructor<ComplexDouble>);
    m.def("test1dDequeConstructor", &NdArrayInterface::test1dDequeConstructor<double>);
    m.def("test1dDequeConstructor", &NdArrayInterface::test1dDequeConstructor<ComplexDouble>);
    m.def("test2dDequeConstructor", &NdArrayInterface::test2dDequeConstructor<double>);
    m.def("test2dDequeConstructor", &NdArrayInterface::test2dDequeConstructor<ComplexDouble>);
    m.def("test1dListConstructor", &NdArrayInterface::test1dListConstructor<double>);
    m.def("test1dListConstructor", &NdArrayInterface::test1dListConstructor<ComplexDouble>);
    m.def("test1dIteratorConstructor", &NdArrayInterface::test1dIteratorConstructor<double>);
    m.def("test1dIteratorConstructor", &NdArrayInterface::test1dIteratorConstructor<ComplexDouble>);
    m.def("test1dIteratorConstructor2", &NdArrayInterface::test1dIteratorConstructor<double>);
    m.def("test1dIteratorConstructor2", &NdArrayInterface::test1dIteratorConstructor<ComplexDouble>);
    m.def("test1dPointerConstructor", &NdArrayInterface::test1dPointerConstructor<double>);
    m.def("test1dPointerConstructor", &NdArrayInterface::test1dPointerConstructor<ComplexDouble>);
    m.def("test2dPointerConstructor", &NdArrayInterface::test2dPointerConstructor<double>);
    m.def("test2dPointerConstructor", &NdArrayInterface::test2dPointerConstructor<ComplexDouble>);
    m.def("test1dPointerShellConstructor", &NdArrayInterface::test1dPointerShellConstructor<double>);
    m.def("test1dPointerShellConstructor", &NdArrayInterface::test1dPointerShellConstructor<ComplexDouble>);
    m.def("test2dPointerShellConstructor", &NdArrayInterface::test2dPointerShellConstructor<double>);
    m.def("test2dPointerShellConstructor", &NdArrayInterface::test2dPointerShellConstructor<ComplexDouble>);
    m.def("testCopyConstructor", &NdArrayInterface::testCopyConstructor<double>);
    m.def("testCopyConstructor", &NdArrayInterface::testCopyConstructor<ComplexDouble>);
    m.def("testMoveConstructor", &NdArrayInterface::testMoveConstructor<double>);
    m.def("testMoveConstructor", &NdArrayInterface::testMoveConstructor<ComplexDouble>);
    m.def("testAssignementOperator", &NdArrayInterface::testAssignementOperator<double>);
    m.def("testAssignementOperator", &NdArrayInterface::testAssignementOperator<ComplexDouble>);
    m.def("testAssignementScalerOperator", &NdArrayInterface::testAssignementScalerOperator<double>);
    m.def("testAssignementScalerOperator", &NdArrayInterface::testAssignementScalerOperator<ComplexDouble>);
    m.def("testMoveAssignementOperator", &NdArrayInterface::testMoveAssignementOperator<double>);
    m.def("testMoveAssignementOperator", &NdArrayInterface::testMoveAssignementOperator<ComplexDouble>);

    pb11::class_<NdArrayDouble>
        (m, "NdArray")
        .def(pb11::init<>())
        .def(pb11::init<NdArrayDouble::size_type>())
        .def(pb11::init<NdArrayDouble::size_type, NdArrayDouble::size_type>())
        .def(pb11::init<Shape>())
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
        .def("get", &NdArrayInterface::getIndicesScaler<double>)
        .def("get", &NdArrayInterface::getIndicesSlice<double>)
        .def("get", &NdArrayInterface::getScalerIndices<double>)
        .def("get", &NdArrayInterface::getSliceIndices<double>)
        .def("get", &NdArrayInterface::getIndices2D<double>)
        .def("at", atSingleScaler, pb11::return_value_policy::copy)
        .def("atConst", atSingleScalerConst, pb11::return_value_policy::copy)
        .def("at", atRowColScalers, pb11::return_value_policy::copy)
        .def("atConst", atRowColScalersConst, pb11::return_value_policy::copy)
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
        .def("nans", &NdArrayDouble::nans, pb11::return_value_policy::reference)
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
        .def("tofile", &NdArrayInterface::tofileBinary<double>)
        .def("tofile", &NdArrayInterface::tofileTxt<double>)
        .def("toIndices", &NdArrayDouble::toIndices)
        .def("toStlVector", &NdArrayDouble::toStlVector)
        .def("trace", &NdArrayDouble::trace)
        .def("transpose", &NdArrayInterface::transpose<double>)
        .def("zeros", &NdArrayDouble::zeros, pb11::return_value_policy::reference);

    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualArray<double>); // (1)
    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualArray<ComplexDouble>); // (1)
    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualComplexArrayArithArray<double>); // (2)
    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualScaler<double>); // (3)
    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualScaler<ComplexDouble>); // (3)
    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualComplexArrayArithScaler<double>); // (4)

    m.def("operatorPlus", &NdArrayInterface::operatorPlusArray<double>); // (1)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArray<ComplexDouble>); // (1)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArithArrayComplexArray<double>); // (2)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusComplexArrayArithArray<double>); // (3)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArrayScaler<double>); // (4)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArrayScaler<ComplexDouble>); // (4)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusScalerArray<double>); // (5)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusScalerArray<ComplexDouble>); // (5)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArithArrayComplexScaler<double>); // (6)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusComplexScalerArithArray<double>); // (7)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusComplexArrayArithScaler<double>); // (8)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArithScalerComplexArray<double>); // (9)

    m.def("operatorNegative", &NdArrayInterface::operatorNegative<double>);
    m.def("operatorNegative", &NdArrayInterface::operatorNegative<ComplexDouble>);

    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualArray<double>); // (1)
    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualArray<ComplexDouble>); // (1)
    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualComplexArrayArithArray<double>); // (2)
    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualScaler<double>); // (3)
    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualScaler<ComplexDouble>); // (3)
    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualComplexArrayArithScaler<double>); // (4)

    m.def("operatorMinus", &NdArrayInterface::operatorMinusArray<double>); // (1)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArray<ComplexDouble>); // (1)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArithArrayComplexArray<double>); // (2)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusComplexArrayArithArray<double>); // (3)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArrayScaler<double>); // (4)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArrayScaler<ComplexDouble>); // (4)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusScalerArray<double>); // (5)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusScalerArray<ComplexDouble>); // (5)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArithArrayComplexScaler<double>); // (6)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusComplexScalerArithArray<double>); // (7)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusComplexArrayArithScaler<double>); // (8)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArithScalerComplexArray<double>); // (9)

    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualArray<double>); // (1)
    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualArray<ComplexDouble>); // (1)
    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualComplexArrayArithArray<double>); // (2)
    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualScaler<double>); // (3)
    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualScaler<ComplexDouble>); // (3)
    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualComplexArrayArithScaler<double>); // (4)

    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArray<double>); // (1)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArray<ComplexDouble>); // (1)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithArrayComplexArray<double>); // (2)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexArrayArithArray<double>); // (3)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArrayScaler<double>); // (4)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArrayScaler<ComplexDouble>); // (4)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyScalerArray<double>); // (5)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyScalerArray<ComplexDouble>); // (5)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithArrayComplexScaler<double>); // (6)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexScalerArithArray<double>); // (7)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexArrayArithScaler<double>); // (8)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithScalerComplexArray<double>); // (9)

    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualArray<double>); // (1)
    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualArray<ComplexDouble>); // (1)
    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualComplexArrayArithArray<double>); // (2)
    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualScaler<double>); // (3)
    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualScaler<ComplexDouble>); // (3)
    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualComplexArrayArithScaler<double>); // (4)

    m.def("operatorDivide", &NdArrayInterface::operatorDivideArray<double>); // (1)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArray<ComplexDouble>); // (1)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArithArrayComplexArray<double>); // (2)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideComplexArrayArithArray<double>); // (3)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArrayScaler<double>); // (4)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArrayScaler<ComplexDouble>); // (4)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideScalerArray<double>); // (5)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideScalerArray<ComplexDouble>); // (5)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArithArrayComplexScaler<double>); // (6)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideComplexScalerArithArray<double>); // (7)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideComplexArrayArithScaler<double>); // (8)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArithScalerComplexArray<double>); // (9)

    m.def("operatorEquality", &NdArrayInterface::operatorEqualityScaler<double>);
    m.def("operatorEquality", &NdArrayInterface::operatorEqualityScaler<ComplexDouble>);
    m.def("operatorEquality", &NdArrayInterface::operatorEqualityScalerReversed<double>);
    m.def("operatorEquality", &NdArrayInterface::operatorEqualityScalerReversed<ComplexDouble>);
    m.def("operatorEquality", &NdArrayInterface::operatorEqualityArray<double>);
    m.def("operatorEquality", &NdArrayInterface::operatorEqualityArray<ComplexDouble>);

    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScaler<double>);
    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScaler<ComplexDouble>);
    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScalerReversed<double>);
    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScalerReversed<ComplexDouble>);
    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityArray<double>);
    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityArray<ComplexDouble>);

    m.def("operatorLess", &NdArrayInterface::operatorLessScaler<double>);
    m.def("operatorLess", &NdArrayInterface::operatorLessScaler<ComplexDouble>);
    m.def("operatorLess", &NdArrayInterface::operatorLessScalerReversed<double>);
    m.def("operatorLess", &NdArrayInterface::operatorLessScalerReversed<ComplexDouble>);
    m.def("operatorLess", &NdArrayInterface::operatorLessArray<double>);
    m.def("operatorLess", &NdArrayInterface::operatorLessArray<ComplexDouble>);

    m.def("operatorGreater", &NdArrayInterface::operatorGreaterScaler<double>);
    m.def("operatorGreater", &NdArrayInterface::operatorGreaterScaler<ComplexDouble>);
    m.def("operatorGreater", &NdArrayInterface::operatorGreaterScalerReversed<double>);
    m.def("operatorGreater", &NdArrayInterface::operatorGreaterScalerReversed<ComplexDouble>);
    m.def("operatorGreater", &NdArrayInterface::operatorGreaterArray<double>);
    m.def("operatorGreater", &NdArrayInterface::operatorGreaterArray<ComplexDouble>);

    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScaler<double>);
    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScaler<ComplexDouble>);
    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScalerReversed<double>);
    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScalerReversed<ComplexDouble>);
    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualArray<double>);
    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualArray<ComplexDouble>);

    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScaler<double>);
    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScaler<ComplexDouble>);
    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScalerReversed<double>);
    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScalerReversed<ComplexDouble>);
    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualArray<double>);
    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualArray<ComplexDouble>);

    m.def("operatorPrePlusPlus", &NdArrayInterface::operatorPrePlusPlus<double>);
    m.def("operatorPostPlusPlus", &NdArrayInterface::operatorPostPlusPlus<double>);

    m.def("operatorPreMinusMinus", &NdArrayInterface::operatorPreMinusMinus<double>);
    m.def("operatorPostMinusMinus", &NdArrayInterface::operatorPostMinusMinus<double>);

    m.def("operatorModulusScaler", &NdArrayInterface::operatorModulusScaler<uint32>);
    m.def("operatorModulusScaler", &NdArrayInterface::operatorModulusScalerReversed<uint32>);
    m.def("operatorModulusArray", &NdArrayInterface::operatorModulusArray<uint32>);

    m.def("operatorModulusScaler", &NdArrayInterface::operatorModulusScaler<double>);
    m.def("operatorModulusScaler", &NdArrayInterface::operatorModulusScalerReversed<double>);
    m.def("operatorModulusArray", &NdArrayInterface::operatorModulusArray<double>);

    m.def("operatorBitwiseOrScaler", &NdArrayInterface::operatorBitwiseOrScaler<uint32>);
    m.def("operatorBitwiseOrScaler", &NdArrayInterface::operatorBitwiseOrScalerReversed<uint32>);
    m.def("operatorBitwiseOrArray", &NdArrayInterface::operatorBitwiseOrArray<uint32>);

    m.def("operatorBitwiseAndScaler", &NdArrayInterface::operatorBitwiseAndScaler<uint32>);
    m.def("operatorBitwiseAndScaler", &NdArrayInterface::operatorBitwiseAndScalerReversed<uint32>);
    m.def("operatorBitwiseAndArray", &NdArrayInterface::operatorBitwiseAndArray<uint32>);

    m.def("operatorBitwiseXorScaler", &NdArrayInterface::operatorBitwiseXorScaler<uint32>);
    m.def("operatorBitwiseXorScaler", &NdArrayInterface::operatorBitwiseXorScalerReversed<uint32>);
    m.def("operatorBitwiseXorArray", &NdArrayInterface::operatorBitwiseXorArray<uint32>);

    m.def("operatorBitwiseNot", &NdArrayInterface::operatorBitwiseNot<uint32>);

    m.def("operatorLogicalAndArray", &NdArrayInterface::operatorLogicalAndArray<uint32>);
    m.def("operatorLogicalAndScalar", &NdArrayInterface::operatorLogicalAndScalar<uint32>);
    m.def("operatorLogicalAndScalar", &NdArrayInterface::operatorLogicalAndScalarReversed<uint32>);

    m.def("operatorLogicalOrArray", &NdArrayInterface::operatorLogicalOrArray<uint32>);
    m.def("operatorLogicalOrScalar", &NdArrayInterface::operatorLogicalOrScalar<uint32>);
    m.def("operatorLogicalOrScalar", &NdArrayInterface::operatorLogicalOrScalarReversed<uint32>);

    m.def("operatorNot", &NdArrayInterface::operatorNot<uint32>);

    m.def("operatorBitshiftLeft", &NdArrayInterface::operatorBitshiftLeft<uint32>);
    m.def("operatorBitshiftRight", &NdArrayInterface::operatorBitshiftRight<uint32>);

    using NdArrayUInt32 = NdArray<uint32>;
    pb11::class_<NdArrayUInt32>
        (m, "NdArrayUInt32")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayUInt32::item)
        .def("shape", &NdArrayUInt32::shape)
        .def("size", &NdArrayUInt32::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint32>)
        .def("endianess", &NdArrayUInt32::endianess)
        .def("setArray", &NdArrayInterface::setArray<uint32>)
        .def("byteswap", &NdArrayUInt32::byteswap, pb11::return_value_policy::reference)
        .def("newbyteorder", &NdArrayInterface::newbyteorder<uint32>);

    using NdArrayUInt64 = NdArray<uint64>;
    pb11::class_<NdArrayUInt64>
        (m, "NdArrayUInt64")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayUInt64::item)
        .def("shape", &NdArrayUInt64::shape)
        .def("size", &NdArrayUInt64::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint64>)
        .def("endianess", &NdArrayUInt64::endianess)
        .def("setArray", &NdArrayInterface::setArray<uint64>);

    using NdArrayUInt16 = NdArray<uint16>;
    pb11::class_<NdArrayUInt16>
        (m, "NdArrayUInt16")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayUInt16::item)
        .def("shape", &NdArrayUInt16::shape)
        .def("size", &NdArrayUInt16::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint16>)
        .def("endianess", &NdArrayUInt16::endianess)
        .def("setArray", NdArrayInterface::setArray<uint16>);

    using NdArrayUInt8 = NdArray<uint8>;
    pb11::class_<NdArrayUInt8>
        (m, "NdArrayUInt8")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayUInt8::item)
        .def("shape", &NdArrayUInt8::shape)
        .def("size", &NdArrayUInt8::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint8>)
        .def("endianess", &NdArrayUInt8::endianess)
        .def("setArray", NdArrayInterface::setArray<uint8>);

    using NdArrayInt64 = NdArray<int64>;
    pb11::class_<NdArrayInt64>
        (m, "NdArrayInt64")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayInt64::item)
        .def("shape", &NdArrayInt64::shape)
        .def("size", &NdArrayInt64::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int64>)
        .def("endianess", &NdArrayInt64::endianess)
        .def("replace", &NdArrayInterface::replace<int64>)
        .def("setArray", &NdArrayInterface::setArray<int64>);

    using NdArrayInt32 = NdArray<int32>;
    pb11::class_<NdArrayInt32>
        (m, "NdArrayInt32")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayInt32::item)
        .def("shape", &NdArrayInt32::shape)
        .def("size", &NdArrayInt32::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int32>)
        .def("endianess", &NdArrayInt32::endianess)
        .def("replace", &NdArrayInterface::replace<int32>)
        .def("setArray", &NdArrayInterface::setArray<int32>);

    using NdArrayInt16 = NdArray<int16>;
    pb11::class_<NdArrayInt16>
        (m, "NdArrayInt16")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayInt16::item)
        .def("shape", &NdArrayInt16::shape)
        .def("size", &NdArrayInt16::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int16>)
        .def("endianess", &NdArrayInt16::endianess)
        .def("replace", &NdArrayInterface::replace<int16>)
        .def("setArray", &NdArrayInterface::setArray<int16>);

    using NdArrayInt8 = NdArray<int8>;
    pb11::class_<NdArrayInt8>
        (m, "NdArrayInt8")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayInt8::item)
        .def("shape", &NdArrayInt8::shape)
        .def("size", &NdArrayInt8::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int8>)
        .def("endianess", &NdArrayInt8::endianess)
        .def("replace", &NdArrayInterface::replace<int8>)
        .def("setArray", &NdArrayInterface::setArray<int8>);

    using NdArrayFloat = NdArray<float>;
    pb11::class_<NdArrayFloat>
        (m, "NdArrayFloat")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayFloat::item)
        .def("shape", &NdArrayFloat::shape)
        .def("size", &NdArrayFloat::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<float>)
        .def("endianess", &NdArrayFloat::endianess)
        .def("setArray", &NdArrayInterface::setArray<float>);

    using NdArrayBool = NdArray<bool>;
    pb11::class_<NdArrayBool>
        (m, "NdArrayBool")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayBool::item)
        .def("shape", &NdArrayBool::shape)
        .def("size", &NdArrayBool::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<bool>)
        .def("endianess", &NdArrayBool::endianess)
        .def("setArray", NdArrayInterface::setArray<bool>);

    using NdArrayComplexLongDouble = NdArray<std::complex<long double> >;
    pb11::class_<NdArrayComplexLongDouble>
        (m, "NdArrayComplexLongDouble")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayComplexLongDouble::item)
        .def("shape", &NdArrayComplexLongDouble::shape)
        .def("size", &NdArrayComplexLongDouble::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<std::complex<long double>>)
        .def("setArray", NdArrayInterface::setArray<std::complex<long double>>);

    using NdArrayComplexFloat = NdArray<std::complex<float> >;
    pb11::class_<NdArrayComplexFloat>
        (m, "NdArrayComplexFloat")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def("item", &NdArrayComplexFloat::item)
        .def("shape", &NdArrayComplexFloat::shape)
        .def("size", &NdArrayComplexFloat::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<std::complex<float>>)
        .def("setArray", NdArrayInterface::setArray<std::complex<float>>);

    pb11::class_<NdArrayComplexDouble>
        (m, "NdArrayComplexDouble")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def(pb11::init<NdArrayComplexDouble>())
        .def_static("test1DListContructor", &NdArrayInterface::test1DListContructor<ComplexDouble>)
        .def_static("test2DListContructor", &NdArrayInterface::test2DListContructor<ComplexDouble>)
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
        .def("at", atComplexSingleScaler, pb11::return_value_policy::copy)
        .def("atConst", atComplexSingleScalerConst, pb11::return_value_policy::copy)
        .def("at", atComplexRowColScalers, pb11::return_value_policy::copy)
        .def("atConst", atComplexRowColScalersConst, pb11::return_value_policy::copy)
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
        .def("tofile", &NdArrayInterface::tofileBinary<ComplexDouble>)
        .def("tofile", &NdArrayInterface::tofileTxt<ComplexDouble>)
        .def("toStlVector", &NdArrayComplexDouble::toStlVector)
        .def("trace", &NdArrayComplexDouble::trace)
        .def("transpose", &NdArrayInterface::transpose<ComplexDouble>)
        .def("zeros", &NdArrayComplexDouble::zeros, pb11::return_value_policy::reference);

    m.def("testStructuredArray", &NdArrayInterface::testStructuredArray);

    // Functions.hpp
    m.def("absScaler", &FunctionsInterface::absScaler<double>);
    m.def("absArray", &FunctionsInterface::absArray<double>);
    m.def("absScaler", &FunctionsInterface::absScaler<ComplexDouble>);
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
    m.def("angleScaler", &FunctionsInterface::angleScaler<double>);
    m.def("angleArray", &FunctionsInterface::angleArray<double>);
    m.def("any", &FunctionsInterface::anyArray<double>);
    m.def("any", &FunctionsInterface::anyArray<ComplexDouble>);
    m.def("append", &append<double>);
    m.def("applyPoly1d", &applyPoly1d<double>);
    m.def("arange", &FunctionsInterface::arangeArray<double>);
    m.def("arccosScaler", &FunctionsInterface::arccosScaler<double>);
    m.def("arccosArray", &FunctionsInterface::arccosArray<double>);
    m.def("arccosScaler", &FunctionsInterface::arccosScaler<ComplexDouble>);
    m.def("arccosArray", &FunctionsInterface::arccosArray<ComplexDouble>);
    m.def("arccoshScaler", &FunctionsInterface::arccoshScaler<double>);
    m.def("arccoshArray", &FunctionsInterface::arccoshArray<double>);
    m.def("arccoshScaler", &FunctionsInterface::arccoshScaler<ComplexDouble>);
    m.def("arccoshArray", &FunctionsInterface::arccoshArray<ComplexDouble>);
    m.def("arcsinScaler", &FunctionsInterface::arcsinScaler<double>);
    m.def("arcsinArray", &FunctionsInterface::arcsinArray<double>);
    m.def("arcsinScaler", &FunctionsInterface::arcsinScaler<ComplexDouble>);
    m.def("arcsinArray", &FunctionsInterface::arcsinArray<ComplexDouble>);
    m.def("arcsinhScaler", &FunctionsInterface::arcsinhScaler<double>);
    m.def("arcsinhArray", &FunctionsInterface::arcsinhArray<double>);
    m.def("arcsinhScaler", &FunctionsInterface::arcsinhScaler<ComplexDouble>);
    m.def("arcsinhArray", &FunctionsInterface::arcsinhArray<ComplexDouble>);
    m.def("arctanScaler", &FunctionsInterface::arctanScaler<double>);
    m.def("arctanArray", &FunctionsInterface::arctanArray<double>);
    m.def("arctanScaler", &FunctionsInterface::arctanScaler<ComplexDouble>);
    m.def("arctanArray", &FunctionsInterface::arctanArray<ComplexDouble>);
    m.def("arctan2Scaler", &FunctionsInterface::arctan2Scaler<double>);
    m.def("arctan2Array", &FunctionsInterface::arctan2Array<double>);
    m.def("arctanhScaler", &FunctionsInterface::arctanhScaler<double>);
    m.def("arctanhArray", &FunctionsInterface::arctanhArray<double>);
    m.def("arctanhScaler", &FunctionsInterface::arctanhScaler<ComplexDouble>);
    m.def("arctanhArray", &FunctionsInterface::arctanhArray<ComplexDouble>);
    m.def("argmax", &FunctionsInterface::argmaxArray<double>);
    m.def("argmax", &FunctionsInterface::argmaxArray<ComplexDouble>);
    m.def("argmin", &FunctionsInterface::argminArray<double>);
    m.def("argmin", &FunctionsInterface::argminArray<ComplexDouble>);
    m.def("argsort", &FunctionsInterface::argsortArray<double>);
    m.def("argsort", &FunctionsInterface::argsortArray<ComplexDouble>);
    m.def("argwhere", &FunctionsInterface::argwhere<double>);
    m.def("argwhere", &FunctionsInterface::argwhere<ComplexDouble>);
    m.def("aroundScaler", &FunctionsInterface::aroundScaler<double>);
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
    m.def("bitwise_and", &FunctionsInterface::bitwise_and<uint64>);
    m.def("bitwise_not", &FunctionsInterface::bitwise_not<uint64>);
    m.def("bitwise_or", &FunctionsInterface::bitwise_or<uint64>);
    m.def("bitwise_xor", &FunctionsInterface::bitwise_xor<uint64>);
    m.def("blackman", &FunctionsInterface::blackman);
    m.def("andOperatorArray", &FunctionsInterface::andOperatorArray<uint64>);
    m.def("andOperatorScaler", &FunctionsInterface::andOperatorScaler<uint64>);
    m.def("orOperatorArray", &FunctionsInterface::orOperatorArray<uint64>);
    m.def("orOperatorScaler", &FunctionsInterface::orOperatorScaler<uint64>);
    m.def("byteswap", &FunctionsInterface::byteswap<uint64>);

    m.def("cbrtScaler", &FunctionsInterface::cbrtScaler<double>);
    m.def("cbrtArray", &FunctionsInterface::cbrtArray<double>);
    m.def("ceilScaler", &FunctionsInterface::ceilScaler<double>);
    m.def("centerOfMass", &FunctionsInterface::centerOfMass<double>);
    m.def("ceilArray", &FunctionsInterface::ceilArray<double>);
    m.def("clipScaler", &FunctionsInterface::clipScaler<double>);
    m.def("clipScaler", &FunctionsInterface::clipScaler<ComplexDouble>);
    m.def("clipArray", &FunctionsInterface::clipArray<double>);
    m.def("clipArray", &FunctionsInterface::clipArray<ComplexDouble>);
    m.def("column_stack", &FunctionsInterface::column_stack<double>);
    m.def("complexScaler", &FunctionsInterface::complexScalerSingle<double>);
    m.def("complexScaler", &FunctionsInterface::complexScaler<double>);
    m.def("complexArray", &FunctionsInterface::complexArraySingle<double>);
    m.def("complexArray", &FunctionsInterface::complexArray<double>);
    m.def("conjScaler", &FunctionsInterface::conjScaler<double>);
    m.def("conjArray", &FunctionsInterface::conjArray<double>);
    m.def("concatenate", &FunctionsInterface::concatenate<double>);
    m.def("contains", &contains<double>);
    m.def("contains", &contains<ComplexDouble>);
    m.def("copy", &FunctionsInterface::copy<double>);
    m.def("copysign", &FunctionsInterface::copySign<double>);
    m.def("copyto", &FunctionsInterface::copyto<double>);
    m.def("cosScaler", &FunctionsInterface::cosScaler<double>);
    m.def("cosScaler", &FunctionsInterface::cosScaler<ComplexDouble>);
    m.def("cosArray", &FunctionsInterface::cosArray<double>);
    m.def("cosArray", &FunctionsInterface::cosArray<ComplexDouble>);
    m.def("coshScaler", &FunctionsInterface::coshScaler<double>);
    m.def("coshScaler", &FunctionsInterface::coshScaler<ComplexDouble>);
    m.def("coshArray", &FunctionsInterface::coshArray<double>);
    m.def("coshArray", &FunctionsInterface::coshArray<ComplexDouble>);
    m.def("count_nonzero", &FunctionsInterface::count_nonzero<double>);
    m.def("count_nonzero", &FunctionsInterface::count_nonzero<ComplexDouble>);
    m.def("cross", &cross<double>);
    m.def("cross", &cross<ComplexDouble>);
    m.def("cube", &FunctionsInterface::cubeArray<double>);
    m.def("cube", &FunctionsInterface::cubeArray<ComplexDouble>);
    m.def("cumprod", &FunctionsInterface::cumprodArray<double>);
    m.def("cumprod", &FunctionsInterface::cumprodArray<ComplexDouble>);
    m.def("cumsum", &FunctionsInterface::cumsumArray<double>);
    m.def("cumsum", &FunctionsInterface::cumsumArray<ComplexDouble>);

    m.def("deg2radScaler", &FunctionsInterface::deg2radScaler<double>);
    m.def("deg2radArray", &FunctionsInterface::deg2radArray<double>);
    m.def("degreesScaler", &FunctionsInterface::degreesScaler<double>);
    m.def("degreesArray", &FunctionsInterface::degreesArray<double>);
    m.def("deleteIndicesScaler", &FunctionsInterface::deleteIndicesScaler<double>);
    m.def("deleteIndicesSlice", &FunctionsInterface::deleteIndicesSlice<double>);
    m.def("diag", &FunctionsInterface::diag<double>);
    m.def("diag", &FunctionsInterface::diag<ComplexDouble>);
    m.def("diagflat", &FunctionsInterface::diagflat<double>);
    m.def("diagflat", &FunctionsInterface::diagflat<ComplexDouble>);
    m.def("diagonal", &FunctionsInterface::diagonal<double>);
    m.def("diagonal", &FunctionsInterface::diagonal<ComplexDouble>);
    m.def("diff", &FunctionsInterface::diff<double>);
    m.def("diff", &FunctionsInterface::diff<ComplexDouble>);
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
    m.def("expScaler", &FunctionsInterface::expScaler<double>);
    m.def("expScaler", &FunctionsInterface::expScaler<ComplexDouble>);
    m.def("expArray", &FunctionsInterface::expArray<double>);
    m.def("expArray", &FunctionsInterface::expArray<ComplexDouble>);
    m.def("exp2Scaler", &FunctionsInterface::exp2Scaler<double>);
    m.def("exp2Array", &FunctionsInterface::exp2Array<double>);
    m.def("expm1Scaler", &FunctionsInterface::expm1Scaler<double>);
    m.def("expm1Scaler", &FunctionsInterface::expm1Scaler<ComplexDouble>);
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
    m.def("fixScaler", &FunctionsInterface::fixScaler<double>);
    m.def("fixArray", &FunctionsInterface::fixArray<double>);
    m.def("flatten", &flatten<double>);
    m.def("flatnonzero", &flatnonzero<double>);
    m.def("flatnonzero", &flatnonzero<ComplexDouble>);
    m.def("flip", &flip<double>);
    m.def("fliplr", &fliplr<double>);
    m.def("flipud", &flipud<double>);
    m.def("floorScaler", &FunctionsInterface::floorScaler<double>);
    m.def("floorArray", &FunctionsInterface::floorArray<double>);
    m.def("floor_divideScaler", &FunctionsInterface::floor_divideScaler<double>);
    m.def("floor_divideArray", &FunctionsInterface::floor_divideArray<double>);
    m.def("fmaxScaler", &FunctionsInterface::fmaxScaler<double>);
    m.def("fmaxScaler", &FunctionsInterface::fmaxScaler<ComplexDouble>);
    m.def("fmaxArray", &FunctionsInterface::fmaxArray<double>);
    m.def("fmaxArray", &FunctionsInterface::fmaxArray<ComplexDouble>);
    m.def("fminScaler", &FunctionsInterface::fminScaler<double>);
    m.def("fminScaler", &FunctionsInterface::fminScaler<ComplexDouble>);
    m.def("fminArray", &FunctionsInterface::fminArray<double>);
    m.def("fminArray", &FunctionsInterface::fminArray<ComplexDouble>);
    m.def("fmodScalerInt", &FunctionsInterface::fmodScaler<uint32>);
    m.def("fmodArrayInt", &FunctionsInterface::fmodArray<uint32>);
    m.def("fmodScalerFloat", &FunctionsInterface::fmodScaler<double>);
    m.def("fmodArrayFloat", &FunctionsInterface::fmodArray<double>);
    m.def("frombuffer", &FunctionsInterface::frombuffer<double>);
    m.def("frombuffer", &FunctionsInterface::frombuffer<ComplexDouble>);
    m.def("fromfile", &FunctionsInterface::fromfileBinary<double>);
    m.def("fromfile", &FunctionsInterface::fromfileTxt<double>);
    m.def("fromiter", &FunctionsInterface::fromiter<double>);
    m.def("fromiter", &FunctionsInterface::fromiter<ComplexDouble>);
    m.def("fullSquare", &FunctionsInterface::fullSquare<double>);
    m.def("fullSquareComplex", &FunctionsInterface::fullSquare<ComplexDouble>);
    m.def("fullRowCol", &FunctionsInterface::fullRowCol<double>);
    m.def("fullRowColComplex", &FunctionsInterface::fullRowCol<ComplexDouble>);
    m.def("fullShape", &FunctionsInterface::fullShape<double>);
    m.def("fullShapeComplex", &FunctionsInterface::fullShape<ComplexDouble>);
    m.def("full_like", &full_like<double>);
    m.def("full_likeComplex", &full_like<ComplexDouble>);

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_gcd_lcm)
    m.def("gcdScaler", &FunctionsInterface::gcdScaler<uint32>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("gcdArray", &FunctionsInterface::gcdArray<uint32>);
#endif
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
    m.def("hstack", &FunctionsInterface::hstack<double>);
    m.def("hypotScaler", &FunctionsInterface::hypotScaler<double>);
    m.def("hypotScalerTriple", &FunctionsInterface::hypotScalerTriple<double>);
    m.def("hypotArray", &FunctionsInterface::hypotArray<double>);

    m.def("identity", &identity<double>);
    m.def("identityComplex", &identity<ComplexDouble>);
    m.def("imagScaler", &FunctionsInterface::imagScaler<double>);
    m.def("imagArray", &FunctionsInterface::imagArray<double>);
    m.def("inner", &FunctionsInterface::inner<double>);
    m.def("interp", &FunctionsInterface::interp<double>);
    m.def("intersect1d", &intersect1d<uint32>);
    m.def("invert", &invert<uint32>);
    m.def("isclose", &isclose<double>);
    m.def("isinfScaler", &FunctionsInterface::isinfScaler<double>);
    m.def("isinfArray", &FunctionsInterface::isinfArray<double>);
    m.def("isposinfScaler", &FunctionsInterface::isposinfScaler<double>);
    m.def("isposinfArray", &FunctionsInterface::isposinfArray<double>);
    m.def("isneginfScaler", &FunctionsInterface::isneginfScaler<double>);
    m.def("isneginfArray", &FunctionsInterface::isneginfArray<double>);
    m.def("isnanScaler", &FunctionsInterface::isnanScaler<double>);
    m.def("isnanArray", &FunctionsInterface::isnanArray<double>);

    m.def("kaiser", &FunctionsInterface::kaiser);

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_gcd_lcm)
    m.def("lcmScaler", &FunctionsInterface::lcmScaler<uint32>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("lcmArray", &FunctionsInterface::lcmArray<uint32>);
#endif
    m.def("ldexpScaler", &FunctionsInterface::ldexpScaler<double>);
    m.def("ldexpArray", &FunctionsInterface::ldexpArray<double>);
    m.def("left_shift", &left_shift<uint32>);
    m.def("less", &less<double>);
    m.def("less", &less<ComplexDouble>);
    m.def("less_equal", &less_equal<double>);
    m.def("less_equal", &less_equal<ComplexDouble>);
    m.def("linspace", &linspace<double>);
    m.def("load", &load<double>);
    m.def("logScaler", &FunctionsInterface::logScaler<double>);
    m.def("logArray", &FunctionsInterface::logArray<double>);
    m.def("logScaler", &FunctionsInterface::logScaler<ComplexDouble>);
    m.def("logArray", &FunctionsInterface::logArray<ComplexDouble>);
    m.def("logbScaler", &FunctionsInterface::logbScaler<double>);
    m.def("logbArray", &FunctionsInterface::logbArray<double>);
    m.def("logspace", &FunctionsInterface::logspace<double>);
    m.def("log10Scaler", &FunctionsInterface::log10Scaler<double>);
    m.def("log10Array", &FunctionsInterface::log10Array<ComplexDouble>);
    m.def("log10Scaler", &FunctionsInterface::log10Scaler<ComplexDouble>);
    m.def("log10Array", &FunctionsInterface::log10Array<double>);
    m.def("log1pScaler", &FunctionsInterface::log1pScaler<double>);
    m.def("log1pArray", &FunctionsInterface::log1pArray<double>);
    m.def("log2Scaler", &FunctionsInterface::log2Scaler<double>);
    m.def("log2Array", &FunctionsInterface::log2Array<double>);
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
    m.def("maximum", &FunctionsInterface::maximum<double>);
    m.def("maximum", &FunctionsInterface::maximum<ComplexDouble>);
    NdArray<double> (*meanDouble)(const NdArray<double>&, Axis) = &mean<double>; 
    m.def("mean", meanDouble);
    NdArray<ComplexDouble> (*meanComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &mean<double>; 
    m.def("mean", meanComplexDouble);
    m.def("median", &median<double>);
    m.def("meshgrid", &FunctionsInterface::meshgrid<double>);
    m.def("min", &FunctionsInterface::min<double>);
    m.def("min", &FunctionsInterface::min<ComplexDouble>);
    m.def("minimum", &FunctionsInterface::minimum<double>);
    m.def("minimum", &FunctionsInterface::minimum<ComplexDouble>);
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
    m.def("newbyteorderScaler", &FunctionsInterface::newbyteorderScaler<uint32>);
    m.def("newbyteorderArray", &FunctionsInterface::newbyteorderArray<uint32>);
    m.def("none", &FunctionsInterface::noneArray<double>);
    m.def("none", &FunctionsInterface::noneArray<ComplexDouble>);
    m.def("nonzero", &nonzero<double>);
    m.def("nonzero", &nonzero<ComplexDouble>);
    NdArray<double> (*normDouble)(const NdArray<double>&, Axis) = &norm<double>; 
    m.def("norm", normDouble);
    NdArray<ComplexDouble> (*normComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &norm<double>; 
    m.def("norm", normComplexDouble);
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

    m.def("pad", &pad<double>);
    m.def("pad", &pad<ComplexDouble>);
    m.def("partition", &partition<double>);
    m.def("partition", &partition<ComplexDouble>);
    m.def("percentile", &percentile<double>);
    m.def("place", &place<double>);
    m.def("polarScaler", &FunctionsInterface::polarScaler<double>);
    m.def("polarArray", &FunctionsInterface::polarArray<double>);
    m.def("powerArrayScaler", &FunctionsInterface::powerArrayScaler<double>);
    m.def("powerArrayArray", &FunctionsInterface::powerArrayArray<double>);
    m.def("powerArrayScaler", &FunctionsInterface::powerArrayScaler<ComplexDouble>);
    m.def("powerArrayArray", &FunctionsInterface::powerArrayArray<ComplexDouble>);
    m.def("powerfArrayScaler", &FunctionsInterface::powerfArrayScaler<double>);
    m.def("powerfArrayArray", &FunctionsInterface::powerfArrayArray<double>);
    m.def("powerfArrayScaler", &FunctionsInterface::powerfArrayScaler<ComplexDouble>);
    m.def("powerfArrayArray", &FunctionsInterface::powerfArrayArray<ComplexDouble>);
    m.def("prod", &prod<double>);
    m.def("prod", &prod<ComplexDouble>);
    m.def("projScaler", &FunctionsInterface::projScaler<double>);
    m.def("projArray", &FunctionsInterface::projArray<double>);
    m.def("ptp", &ptp<double>);
    m.def("ptp", &ptp<ComplexDouble>);
    NdArray<double>& (*putDoubleValue)(NdArray<double>&, const NdArray<uint32>&, double) = &put<double>;
    m.def("put", putDoubleValue, pb11::return_value_policy::reference);
    NdArray<double>& (*putDoubleArray)(NdArray<double>&, const NdArray<uint32>&, const NdArray<double>&) = &put<double>;
    m.def("put", putDoubleArray, pb11::return_value_policy::reference);
    m.def("putmask", &FunctionsInterface::putmask<double>);
    m.def("putmaskScaler", &FunctionsInterface::putmaskScaler<double>);

    m.def("rad2degScaler", &FunctionsInterface::rad2degScaler<double>);
    m.def("rad2degArray", &FunctionsInterface::rad2degArray<double>);
    m.def("radiansScaler", &FunctionsInterface::radiansScaler<double>);
    m.def("radiansArray", &FunctionsInterface::radiansArray<double>);
    m.def("ravel", &FunctionsInterface::ravel<double>, pb11::return_value_policy::reference);
    m.def("reciprocal", &FunctionsInterface::reciprocal<double>);
    m.def("reciprocal", &FunctionsInterface::reciprocal<ComplexDouble>);
    m.def("realScaler", &FunctionsInterface::realScaler<double>);
    m.def("realArray", &FunctionsInterface::realArray<double>);
    m.def("remainderScaler", &FunctionsInterface::remainderScaler<double>);
    m.def("remainderArray", &FunctionsInterface::remainderArray<double>);
    m.def("replace", &FunctionsInterface::replace<double>);
    m.def("replace", &FunctionsInterface::replace<ComplexDouble>);
    m.def("reshape", &FunctionsInterface::reshapeInt<double>, pb11::return_value_policy::reference);
    m.def("reshape", &FunctionsInterface::reshapeShape<double>, pb11::return_value_policy::reference);
    m.def("reshape", &FunctionsInterface::reshapeValues<double>, pb11::return_value_policy::reference);
    m.def("reshapeList", &FunctionsInterface::reshapeList<double>, pb11::return_value_policy::reference);
    m.def("resizeFast", &FunctionsInterface::resizeFast<double>, pb11::return_value_policy::reference);
    m.def("resizeFastList", &FunctionsInterface::resizeFastList<double>, pb11::return_value_policy::reference);
    m.def("resizeSlow", &FunctionsInterface::resizeSlow<double>, pb11::return_value_policy::reference);
    m.def("resizeSlowList", &FunctionsInterface::resizeSlowList<double>, pb11::return_value_policy::reference);
    m.def("right_shift", &right_shift<uint32>);
    m.def("rintScaler", &FunctionsInterface::rintScaler<double>);
    m.def("rintArray", &FunctionsInterface::rintArray<double>);
    NdArray<double> (*rmsDouble)(const NdArray<double>&, Axis) = &rms<double>; 
    m.def("rms", rmsDouble);
    NdArray<ComplexDouble> (*rmsComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &rms<double>; 
    m.def("rms", rmsComplexDouble);
    m.def("roll", &roll<double>);
    m.def("rot90", &rot90<double>);
    m.def("roundScaler", &FunctionsInterface::roundScaler<double>);
    m.def("roundArray", &FunctionsInterface::roundArray<double>);
    m.def("row_stack", &FunctionsInterface::row_stack<double>);

    m.def("select", &FunctionsInterface::select<double>);
    m.def("selectVector", &FunctionsInterface::selectVector<double>);
    m.def("select", &FunctionsInterface::selectInitializerList<double>);
    m.def("setdiff1d", &setdiff1d<uint32>);
    m.def("setdiff1d", &setdiff1d<std::complex<double>>);
    m.def("signScaler", &FunctionsInterface::signScaler<double>);
    m.def("signScaler", &FunctionsInterface::signScaler<ComplexDouble>);
    m.def("signArray", &FunctionsInterface::signArray<double>);
    m.def("signArray", &FunctionsInterface::signArray<ComplexDouble>);
    m.def("signbitScaler", &FunctionsInterface::signbitScaler<double>);
    m.def("signbitArray", &FunctionsInterface::signbitArray<double>);
    m.def("sinScaler", &FunctionsInterface::sinScaler<double>);
    m.def("sinScaler", &FunctionsInterface::sinScaler<ComplexDouble>);
    m.def("sinArray", &FunctionsInterface::sinArray<double>);
    m.def("sinArray", &FunctionsInterface::sinArray<ComplexDouble>);
    m.def("sincScaler", &FunctionsInterface::sincScaler<double>);
    m.def("sincArray", &FunctionsInterface::sincArray<double>);
    m.def("sinhScaler", &FunctionsInterface::sinhScaler<ComplexDouble>);
    m.def("sinhScaler", &FunctionsInterface::sinhScaler<double>);
    m.def("sinhArray", &FunctionsInterface::sinhArray<double>);
    m.def("sinhArray", &FunctionsInterface::sinhArray<ComplexDouble>);
    m.def("size", &size<double>);
    m.def("sort", &sort<double>);
    m.def("sort", &sort<ComplexDouble>);
    m.def("sqrtScaler", &FunctionsInterface::sqrtScaler<double>);
    m.def("sqrtScaler", &FunctionsInterface::sqrtScaler<ComplexDouble>);
    m.def("sqrtArray", &FunctionsInterface::sqrtArray<double>);
    m.def("sqrtArray", &FunctionsInterface::sqrtArray<ComplexDouble>);
    m.def("squareScaler", &FunctionsInterface::squareScaler<double>);
    m.def("squareScaler", &FunctionsInterface::squareScaler<ComplexDouble>);
    m.def("squareArray", &FunctionsInterface::squareArray<double>);
    m.def("squareArray", &FunctionsInterface::squareArray<ComplexDouble>);
    m.def("stack", &FunctionsInterface::stack<double>);
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
    m.def("swapaxes", &swapaxes<double>);
    m.def("swap", &nc::swap<double>);

    m.def("tanScaler", &FunctionsInterface::tanScaler<double>);
    m.def("tanScaler", &FunctionsInterface::tanScaler<ComplexDouble>);
    m.def("tanArray", &FunctionsInterface::tanArray<double>);
    m.def("tanArray", &FunctionsInterface::tanArray<ComplexDouble>);
    m.def("tanhScaler", &FunctionsInterface::tanhScaler<double>);
    m.def("tanhScaler", &FunctionsInterface::tanhScaler<ComplexDouble>);
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
    m.def("truncScaler", &FunctionsInterface::truncScaler<double>);
    m.def("truncArray", &FunctionsInterface::truncArray<double>);

    m.def("union1d", &union1d<uint32>);
    m.def("union1d", &union1d<std::complex<double>>);
    m.def("unique", &unique<uint32>);
    m.def("unique", &unique<std::complex<double>>);
    m.def("unwrapScaler", &FunctionsInterface::unwrapScaler<double>);
    m.def("unwrapArray", &FunctionsInterface::unwrapArray<double>);

    NdArray<double> (*varDouble)(const NdArray<double>&, Axis) = &var<double>;
    m.def("var", varDouble);
    NdArray<ComplexDouble> (*varComplexDouble)(const NdArray<ComplexDouble>&, Axis) = &var<double>; 
    m.def("var", varComplexDouble);
    m.def("vstack", &FunctionsInterface::vstack<double>);

    m.def("where", &FunctionsInterface::whereArrayArray<double>);
    m.def("where", &FunctionsInterface::whereArrayArray<ComplexDouble>);
    m.def("where", &FunctionsInterface::whereArrayScaler<double>);
    m.def("where", &FunctionsInterface::whereArrayScaler<ComplexDouble>);
    m.def("where", &FunctionsInterface::whereScalerArray<double>);
    m.def("where", &FunctionsInterface::whereScalerArray<ComplexDouble>);
    m.def("where", &FunctionsInterface::whereScalerScaler<double>);
    m.def("where", &FunctionsInterface::whereScalerScaler<ComplexDouble>);

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

    // Utils.hpp
    m.def("num2str", &utils::num2str<double>);
    m.def("sqr", &utils::sqr<double>);
    m.def("cube", &utils::cube<double>);
    m.def("power", &utils::power<double>);
    m.def("power", &utils::power<ComplexDouble>);
    decltype(utils::powerf<double, double>(double{ 0 }, double{ 0 }))(*powerf_double)(double, double) = &utils::powerf<double, double>;
    m.def("powerf", powerf_double);
    decltype(utils::powerf<ComplexDouble, ComplexDouble>(ComplexDouble{ 0 }, ComplexDouble{ 0 }))(*powerf_complexDouble)
        (ComplexDouble, ComplexDouble) = &utils::powerf<ComplexDouble, ComplexDouble>;
    m.def("powerf_complex", powerf_complexDouble);

    m.def("num2str", &utils::num2str<int64>);
    m.def("sqr", &utils::sqr<int64>);
    m.def("cube", &utils::cube<int64>);
    m.def("power", &utils::power<int64>);
    decltype(utils::powerf<int64, double>(int64{ 0 }, double{ 0 }))(*powerf_int64)(int64, double) = &utils::powerf<int64, double>;
    m.def("powerf", powerf_int64);

    // Random.hpp
    NdArray<bool> (*bernoulliArray)(const Shape&, double) = &random::bernoulli;
    bool (*bernoilliScalar)(double) = &random::bernoulli;
    m.def("bernoulli", bernoulliArray);
    m.def("bernoulli", bernoilliScalar);

#ifndef NUMCPP_NO_USE_BOOST
    NdArray<double> (*betaArray)(const Shape&, double, double) = &random::beta<double>;
    double (*betaScalar)(double, double) = &random::beta<double>;
    m.def("beta", betaArray);
    m.def("beta", betaScalar);
#endif

    NdArray<int32> (*binomialArray)(const Shape&, int32, double) = &random::binomial<int32>;
    int32 (*binomialScalar)(int32, double) = &random::binomial<int32>;
    m.def("binomial", binomialArray);
    m.def("binomial", binomialScalar);

    NdArray<double> (*cauchyArray)(const Shape&, double, double) = &random::cauchy<double>;
    double (*cauchyScalar)(double, double) = &random::cauchy<double>;
    m.def("cauchy", cauchyArray);
    m.def("cauchy", cauchyScalar);

    NdArray<double> (*chiSquareArray)(const Shape&, double) = &random::chiSquare<double>;
    double (*chiSquareScalar)(double) = &random::chiSquare<double>;
    m.def("chiSquare", chiSquareArray);
    m.def("chiSquare", chiSquareScalar);

    m.def("choiceSingle", &RandomInterface::choiceSingle<double>);
    m.def("choiceMultiple", &RandomInterface::choiceMultiple<double>);

    NdArray<int32> (*discreteArray)(const Shape&, const NdArray<double>&) = &random::discrete<int32>;
    int32 (*discreteScalar)(const NdArray<double>&) = &random::discrete<int32>;
    m.def("discrete", discreteArray);
    m.def("discrete", discreteScalar);

    NdArray<double> (*exponentialArray)(const Shape&, double) = &random::exponential<double>;
    double (*exponentialScalar)(double) = &random::exponential<double>;
    m.def("exponential", exponentialArray);
    m.def("exponential", exponentialScalar);

    NdArray<double> (*extremeValueArray)(const Shape&, double, double) = &random::extremeValue<double>;
    double (*extremeValueScalar)(double, double) = &random::extremeValue<double>;
    m.def("extremeValue", extremeValueArray);
    m.def("extremeValue", extremeValueScalar);

    NdArray<double> (*fArray)(const Shape&, double, double) = &random::f<double>;
    double (*fScalar)(double, double) = &random::f<double>;
    m.def("f", fArray);
    m.def("f", fScalar);

    NdArray<double> (*gammaArray)(const Shape&, double, double) = &random::gamma<double>;
    double (*gammaScalar)(double, double) = &random::gamma<double>;
    m.def("gamma", gammaArray);
    m.def("gamma", gammaScalar);

    NdArray<int32> (*geometricArray)(const Shape&, double) = &random::geometric<int32>;
    int32 (*geometricScalar)(double) = &random::geometric<int32>;
    m.def("geometric", geometricArray);
    m.def("geometric", geometricScalar);

#ifndef NUMCPP_NO_USE_BOOST
    NdArray<double> (*laplaceArray)(const Shape&, double, double) = &random::laplace<double>;
    double (*laplaceScalar)(double, double) = &random::laplace<double>;
    m.def("laplace", laplaceArray);
    m.def("laplace", laplaceScalar);
#endif

    NdArray<double> (*lognormalArray)(const Shape&, double, double) = &random::lognormal<double>;
    double (*lognormalScalar)(double, double) = &random::lognormal<double>;
    m.def("lognormal", lognormalArray);
    m.def("lognormal", lognormalScalar);

    NdArray<int32> (*negativeBinomialArray)(const Shape&, int32, double) = &random::negativeBinomial<int32>;
    int32 (*negativeBinomialScalar)(int32, double) = &random::negativeBinomial<int32>;
    m.def("negativeBinomial", negativeBinomialArray);
    m.def("negativeBinomial", negativeBinomialScalar);

#ifndef NUMCPP_NO_USE_BOOST
    NdArray<double> (*nonCentralChiSquaredArray)(const Shape&, double, double) = &random::nonCentralChiSquared<double>;
    double (*nonCentralChiSquaredScalar)(double, double) = &random::nonCentralChiSquared<double>;
    m.def("nonCentralChiSquared", nonCentralChiSquaredArray);
    m.def("nonCentralChiSquared", nonCentralChiSquaredScalar);
#endif

    NdArray<double> (*normalArray)(const Shape&, double, double) = &random::normal<double>;
    double (*normalScalar)(double, double) = &random::normal<double>;
    m.def("normal", normalArray);
    m.def("normal", normalScalar);

    m.def("permutationScaler", &RandomInterface::permutationScaler<double>);
    m.def("permutationArray", &RandomInterface::permutationArray<double>);

    NdArray<int32> (*poissonArray)(const Shape&, double) = &random::poisson<int32>;
    int32 (*poissonScalar)(double) = &random::poisson<int32>;
    m.def("poisson", poissonArray);
    m.def("poisson", poissonScalar);

    NdArray<double> (*randArray)(const Shape&) = &random::rand<double>;
    double (*randScalar)() = &random::rand<double>;
    m.def("rand", randArray);
    m.def("rand", randScalar);

    NdArray<double> (*randFloatArray)(const Shape&, double, double) = &random::randFloat<double>;
    double (*randFloatScalar)(double, double) = &random::randFloat<double>;
    m.def("randFloat", randFloatArray);
    m.def("randFloat", randFloatScalar);

    NdArray<int32> (*randIntArray)(const Shape&, int32, int32) = &random::randInt<int32>;
    int32 (*randIntScalar)(int32, int32) = &random::randInt<int32>;
    m.def("randInt", randIntArray);
    m.def("randInt", randIntScalar);

    NdArray<double> (*randNArray)(const Shape&) = &random::randN<double>;
    double (*randNScalar)() = &random::randN<double>;
    m.def("randN", randNArray);
    m.def("randN", randNScalar);

    m.def("seed", &random::seed);
    m.def("shuffle", &random::shuffle<double>);

    NdArray<double> (*standardNormalArray)(const Shape&) = &random::standardNormal<double>;
    double (*standardNormalScalar)() = &random::standardNormal<double>;
    m.def("standardNormal", standardNormalArray);
    m.def("standardNormal", standardNormalScalar);

    NdArray<double> (*studentTArray)(const Shape&, double) = &random::studentT<double>;
    double (*studentTScalar)(double) = &random::studentT<double>;
    m.def("studentT", studentTArray);
    m.def("studentT", studentTScalar);

#ifndef NUMCPP_NO_USE_BOOST
    NdArray<double> (*triangleArray)(const Shape&, double, double, double) = &random::triangle<double>;
    double (*triangleScalar)(double, double, double) = &random::triangle<double>;
    m.def("triangle", triangleArray);
    m.def("triangle", triangleScalar);
#endif

    NdArray<double> (*uniformArray)(const Shape&, double, double) = &random::uniform<double>;
    double (*uniformScalar)(double, double) = &random::uniform<double>;
    m.def("uniform", uniformArray);
    m.def("uniform", uniformScalar);

#ifndef NUMCPP_NO_USE_BOOST
    m.def("uniformOnSphere", &random::uniformOnSphere<double>);
#endif

    NdArray<double> (*weibullArray)(const Shape&, double, double) = &random::weibull<double>;
    double (*weibullScalar)(double, double) = &random::weibull<double>;
    m.def("weibull", weibullArray);
    m.def("weibull", weibullScalar);

    // Linalg.hpp
    m.def("cholesky", &linalg::cholesky<double>);
    m.def("det", &linalg::det<double>);
    m.def("hat", &LinalgInterface::hatArray<double>);
    m.def("inv", &linalg::inv<double>);
    m.def("lstsq", &linalg::lstsq<double>);
    m.def("lu_decomposition", &linalg::lu_decomposition<double>);
    m.def("matrix_power", &linalg::matrix_power<double>);
    m.def("multi_dot", &LinalgInterface::multi_dot<double>);
    m.def("multi_dot", &LinalgInterface::multi_dot<ComplexDouble>);
    m.def("pivotLU_decomposition", &LinalgInterface::pivotLU_decomposition<double>);
    m.def("solve", &LinalgInterface::solve<double>);
    m.def("svd", &linalg::svd<double>);

    // Rotations.hpp
    pb11::class_<rotations::Quaternion>
        (m, "Quaternion")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<double, double, double, double>())
        .def(pb11::init<Vec3, double>())
        .def(pb11::init<NdArray<double>, double>())
        .def(pb11::init<NdArray<double> >())
        .def_static("angleAxisRotationNdArray", &RotationsInterface::angleAxisRotationNdArray)
        .def_static("angleAxisRotationVec3", &RotationsInterface::angleAxisRotationVec3)
        .def("angularVelocity", &RotationsInterface::angularVelocity)
        .def("conjugate", &rotations::Quaternion::conjugate)
        .def("i", &rotations::Quaternion::i)
        .def_static("identity", &rotations::Quaternion::identity)
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
        .def_static("xRotation", &rotations::Quaternion::xRotation)
        .def("yaw", &rotations::Quaternion::yaw)
        .def_static("yRotation", &rotations::Quaternion::yRotation)
        .def_static("zRotation", &rotations::Quaternion::zRotation)
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

    pb11::class_<rotations::DCM>
        (m, "DCM")
        .def(pb11::init<>())
        .def_static("eulerAnglesValues", &RotationsInterface::eulerAnglesValues)
        .def_static("eulerAnglesArray", &RotationsInterface::eulerAnglesArray)
        .def_static("angleAxisRotationNdArray", &RotationsInterface::angleAxisRotationDcmNdArray)
        .def_static("angleAxisRotationVec3", &RotationsInterface::angleAxisRotationDcmVec3)
        .def_static("isValid", &rotations::DCM::isValid)
        .def_static("roll", &rotations::DCM::roll)
        .def_static("pitch", &rotations::DCM::pitch)
        .def_static("yaw", &rotations::DCM::yaw)
        .def_static("xRotation", &rotations::DCM::xRotation)
        .def_static("yRotation", &rotations::DCM::yRotation)
        .def_static("zRotation", &rotations::DCM::zRotation);

    m.def("rodriguesRotation", &RotationsInterface::rodriguesRotation<double>);
    m.def("wahbasProblem", &RotationsInterface::wahbasProblem<double>);
    m.def("wahbasProblemWeighted", &RotationsInterface::wahbasProblemWeighted<double>);

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

    // Image Processing
    using PixelDouble = imageProcessing::Pixel<double>;
    pb11::class_<PixelDouble>
        (m, "Pixel")
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
    pb11::class_<ClusterDouble>
        (m, "Cluster")
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
    pb11::class_<CentroidDouble>
        (m, "Centroid")
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

    // Coordinates.hpp
    pb11::class_<coordinates::RA>
        (m, "Ra")
        .def(pb11::init<>())
        .def(pb11::init<double>())
        .def(pb11::init<uint8, uint8, double>())
        .def(pb11::init<coordinates::RA>())
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

    pb11::enum_<coordinates::Sign>
        (m, "Sign")
        .value("POSITIVE", coordinates::Sign::POSITIVE)
        .value("NEGATIVE", coordinates::Sign::NEGATIVE);

    pb11::class_<coordinates::Dec>
        (m, "Dec")
        .def(pb11::init<>())
        .def(pb11::init<double>())
        .def(pb11::init<coordinates::Sign, uint8, uint8, double>())
        .def(pb11::init<coordinates::Dec>())
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

    pb11::class_<coordinates::Coordinate>
        (m, "Coordinate")
        .def(pb11::init<>())
        .def(pb11::init<double, double>())
        .def(pb11::init<uint8, uint8, double, coordinates::Sign, uint8, uint8, double>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<coordinates::RA, coordinates::Dec>())
        .def(pb11::init<NdArrayDouble>())
        .def(pb11::init<coordinates::Coordinate>())
        .def("dec", &coordinates::Coordinate::dec, pb11::return_value_policy::reference)
        .def("ra", &coordinates::Coordinate::ra, pb11::return_value_policy::reference)
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
    using DataCubeDouble = DataCube<double>;
    pb11::class_<DataCubeDouble>
        (m, "DataCube")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def("at", &DataCubeInterface::at<double>, pb11::return_value_policy::reference)
        .def("__getitem__", &DataCubeInterface::getItem<double>, pb11::return_value_policy::reference)
        .def("back", &DataCubeDouble::back, pb11::return_value_policy::reference)
        .def("dump", &DataCubeDouble::dump)
        .def("front", &DataCubeDouble::front, pb11::return_value_policy::reference)
        .def("isempty", &DataCubeDouble::isempty)
        .def("shape", &DataCubeDouble::shape, pb11::return_value_policy::reference)
        .def("sizeZ", &DataCubeDouble::sizeZ)
        .def("pop_back", &DataCubeDouble::pop_back)
        .def("push_back", &DataCubeDouble::push_back)
        .def("sliceZAll", &DataCubeInterface::sliceZIndexAll<double>)
        .def("sliceZ", &DataCubeInterface::sliceZIndex<double>)
        .def("sliceZAll", &DataCubeInterface::sliceZRowColAll<double>)
        .def("sliceZ", &DataCubeInterface::sliceZRowCol<double>)
        .def("sliceZAll", &DataCubeInterface::sliceZSliceScalerAll<double>)
        .def("sliceZ", &DataCubeInterface::sliceZSliceScaler<double>)
        .def("sliceZAll", &DataCubeInterface::sliceZScalerSliceAll<double>)
        .def("sliceZ", &DataCubeInterface::sliceZScalerSlice<double>)
        .def("sliceZAll", &DataCubeInterface::sliceZSliceSliceAll<double>)
        .def("sliceZ", &DataCubeInterface::sliceZSliceSlice<double>)
        .def("sliceZAllat", &DataCubeInterface::sliceZAtIndexAll<double>)
        .def("sliceZat", &DataCubeInterface::sliceZAtIndex<double>)
        .def("sliceZAllat", &DataCubeInterface::sliceZAtRowColAll<double>)
        .def("sliceZat", &DataCubeInterface::sliceZAtRowCol<double>)
        .def("sliceZAllat", &DataCubeInterface::sliceZAtSliceScalerAll<double>)
        .def("sliceZat", &DataCubeInterface::sliceZAtSliceScaler<double>)
        .def("sliceZAllat", &DataCubeInterface::sliceZAtScalerSliceAll<double>)
        .def("sliceZat", &DataCubeInterface::sliceZAtScalerSlice<double>)
        .def("sliceZAllat", &DataCubeInterface::sliceZAtSliceSliceAll<double>)
        .def("sliceZat", &DataCubeInterface::sliceZAtSliceSlice<double>);

    // Polynomial.hpp
    using Poly1d = polynomial::Poly1d<double>;

    Poly1d(*fit)(const NdArrayDouble&, const NdArrayDouble&, uint8) = &Poly1d::fit;
    Poly1d (*fitWeighted)(const NdArrayDouble&, const NdArrayDouble&, const NdArrayDouble&, uint8) = &Poly1d::fit;

    pb11::class_<Poly1d>
        (m, "Poly1d")
        .def(pb11::init<>())
        .def(pb11::init<NdArray<double>, bool>())
        .def("area", &Poly1d::area)
        .def("coefficients", &Poly1d::coefficients)
        .def("deriv", &Poly1d::deriv)
        .def_static("fit", fit)
        .def_static("fitWeighted", fitWeighted)
        .def("integ", &Poly1d::integ)
        .def("order", &Poly1d::order)
        .def("print", &Poly1d::print)
        .def("__str__", &Poly1d::str)
        .def("__repr__", &Poly1d::str)
        .def("__getitem__", &Poly1d::operator())
        .def("__add__", &Poly1d::operator+)
        .def("__iadd__", &Poly1d::operator+=, pb11::return_value_policy::reference)
        .def("__sub__", &Poly1d::operator-)
        .def("__isub__", &Poly1d::operator-=, pb11::return_value_policy::reference)
        .def("__mul__", &Poly1d::operator*)
        .def("__imul__", &Poly1d::operator*=, pb11::return_value_policy::reference)
        .def("__pow__", &Poly1d::operator^)
        .def("__ipow__", &Poly1d::operator^=, pb11::return_value_policy::reference);

#ifndef NUMCPP_NO_USE_BOOST
    m.def("chebyshev_t_Scaler", &PolynomialInterface::chebyshev_t_Scaler<double>);
    m.def("chebyshev_t_Array", &PolynomialInterface::chebyshev_t_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("chebyshev_u_Scaler", &PolynomialInterface::chebyshev_u_Scaler<double>);
    m.def("chebyshev_u_Array", &PolynomialInterface::chebyshev_u_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("hermite_Scaler", &PolynomialInterface::hermite_Scaler<double>);
    m.def("hermite_Array", &PolynomialInterface::hermite_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("laguerre_Scaler1", &PolynomialInterface::laguerre_Scaler1<double>);
    m.def("laguerre_Array1", &PolynomialInterface::laguerre_Array1<double>);
    m.def("laguerre_Scaler2", &PolynomialInterface::laguerre_Scaler2<double>);
    m.def("laguerre_Array2", &PolynomialInterface::laguerre_Array2<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("legendre_p_Scaler1", &PolynomialInterface::legendre_p_Scaler1<double>);
    m.def("legendre_p_Array1", &PolynomialInterface::legendre_p_Array1<double>);
    m.def("legendre_p_Scaler2", &PolynomialInterface::legendre_p_Scaler2<double>);
    m.def("legendre_p_Array2", &PolynomialInterface::legendre_p_Array2<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("legendre_q_Scaler", &PolynomialInterface::legendre_q_Scaler<double>);
    m.def("legendre_q_Array", &PolynomialInterface::legendre_q_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("spherical_harmonic", &PolynomialInterface::spherical_harmonic<double>);
    m.def("spherical_harmonic_r", &polynomial::spherical_harmonic_r<double, double>);
    m.def("spherical_harmonic_i", &polynomial::spherical_harmonic_i<double, double>);
#endif

    // Roots.hpp
    m.def("bisection_roots", &RootsInterface::bisection);
    m.def("brent_roots", &RootsInterface::brent);
    m.def("dekker_roots", &RootsInterface::dekker);
    m.def("newton_roots", &RootsInterface::newton);
    m.def("secant_roots", &RootsInterface::secant);

    // Integrate.hpp
    m.def("integrate_gauss_legendre", &IntegrateInterface::gauss_legendre);
    m.def("integrate_romberg", &IntegrateInterface::romberg);
    m.def("integrate_simpson", &IntegrateInterface::simpson);
    m.def("integrate_trapazoidal", &IntegrateInterface::trapazoidal);

    // Vec2.hpp
    pb11::class_<Vec2>
        (m, "Vec2")
        .def(pb11::init<>())
        .def(pb11::init<double, double>())
        .def(pb11::init<NdArray<double> >())
        .def_readwrite("x", &Vec2::x)
        .def_readwrite("y", &Vec2::y)
        .def("angle", &Vec2::angle)
        .def("clampMagnitude", &Vec2::clampMagnitude)
        .def("distance", &Vec2::distance)
        .def("dot", &Vec2::dot)
        .def_static("down", &Vec2::down)
        .def_static("left", &Vec2::left)
        .def("lerp", &Vec2::lerp)
        .def("norm", &Vec2::norm)
        .def("normalize", &Vec2::normalize)
        .def("project", &Vec2::project)
        .def_static("right", &Vec2::right)
        .def("__str__", &Vec2::toString)
        .def("toNdArray", &Vec2Interface::toNdArray)
        .def_static("up", &Vec2::up)
        .def("__eq__", &Vec2::operator==)
        .def("__ne__", &Vec2::operator!=)
        .def("__iadd__", &Vec2Interface::plusEqualScaler, pb11::return_value_policy::reference)
        .def("__iadd__", &Vec2Interface::plusEqualVec2, pb11::return_value_policy::reference)
        .def("__isub__", &Vec2Interface::minusEqualVec2, pb11::return_value_policy::reference)
        .def("__isub__", &Vec2Interface::minusEqualScaler, pb11::return_value_policy::reference)
        .def("__imul__", &Vec2::operator*=, pb11::return_value_policy::reference)
        .def("__itruediv__", &Vec2::operator/=, pb11::return_value_policy::reference);

    m.def("Vec2_addVec2", &Vec2Interface::addVec2);
    m.def("Vec2_addVec2Scaler", &Vec2Interface::addVec2Scaler);
    m.def("Vec2_addScalerVec2", &Vec2Interface::addScalerVec2);
    m.def("Vec2_minusVec2", &Vec2Interface::minusVec2);
    m.def("Vec2_minusVec2Scaler", &Vec2Interface::minusVec2Scaler);
    m.def("Vec2_minusScalerVec2", &Vec2Interface::minusScalerVec2);
    m.def("Vec2_multVec2Scaler", &Vec2Interface::multVec2Scaler);
    m.def("Vec2_multScalerVec2", &Vec2Interface::multScalerVec2);
    m.def("Vec2_divVec2Scaler", &Vec2Interface::divVec2Scaler);
    m.def("Vec2_print", &Vec2Interface::print);

    // Vec3.hpp
    pb11::class_<Vec3>
        (m, "Vec3")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<NdArray<double> >())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("angle", &Vec3::angle)
        .def_static("back", &Vec3::back)
        .def("clampMagnitude", &Vec3::clampMagnitude)
        .def("cross", &Vec3::cross)
        .def("distance", &Vec3::distance)
        .def("dot", &Vec3::dot)
        .def_static("down", &Vec3::down)
        .def_static("forward", &Vec3::forward)
        .def_static("left", &Vec3::left)
        .def("lerp", &Vec3::lerp)
        .def("norm", &Vec3::norm)
        .def("normalize", &Vec3::normalize)
        .def("project", &Vec3::project)
        .def_static("right", &Vec3::right)
        .def("__str__", &Vec3::toString)
        .def("toNdArray", &Vec3Interface::toNdArray)
        .def_static("up", &Vec3::up)
        .def("__eq__", &Vec3::operator==)
        .def("__ne__", &Vec3::operator!=)
        .def("__iadd__", &Vec3Interface::plusEqualScaler, pb11::return_value_policy::reference)
        .def("__iadd__", &Vec3Interface::plusEqualVec3, pb11::return_value_policy::reference)
        .def("__isub__", &Vec3Interface::minusEqualScaler, pb11::return_value_policy::reference)
        .def("__isub__", &Vec3Interface::minusEqualVec3, pb11::return_value_policy::reference)
        .def("__imul__", &Vec3::operator*=, pb11::return_value_policy::reference)
        .def("__itruediv__", &Vec3::operator/=, pb11::return_value_policy::reference);

    m.def("Vec3_addVec3", &Vec3Interface::addVec3);
    m.def("Vec3_addVec3Scaler", &Vec3Interface::addVec3Scaler);
    m.def("Vec3_addScalerVec3", &Vec3Interface::addScalerVec3);
    m.def("Vec3_minusVec3", &Vec3Interface::minusVec3);
    m.def("Vec3_minusVec3Scaler", &Vec3Interface::minusVec3Scaler);
    m.def("Vec3_minusScalerVec3", &Vec3Interface::minusScalerVec3);
    m.def("Vec3_multVec3Scaler", &Vec3Interface::multVec3Scaler);
    m.def("Vec3_multScalerVec3", &Vec3Interface::multScalerVec3);
    m.def("Vec3_divVec3Scaler", &Vec3Interface::divVec3Scaler);
    m.def("Vec3_print", &Vec3Interface::print);

    // Special.hpp
#ifndef NUMCPP_NO_USE_BOOST
    m.def("airy_ai_Scaler", &SpecialInterface::airy_ai_Scaler<double>);
    m.def("airy_ai_Array", &SpecialInterface::airy_ai_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("airy_ai_prime_Scaler", &SpecialInterface::airy_ai_prime_Scaler<double>);
    m.def("airy_ai_prime_Array", &SpecialInterface::airy_ai_prime_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("airy_bi_Scaler", &SpecialInterface::airy_bi_Scaler<double>);
    m.def("airy_bi_Array", &SpecialInterface::airy_bi_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("airy_bi_prime_Scaler", &SpecialInterface::airy_bi_prime_Scaler<double>);
    m.def("airy_bi_prime_Array", &SpecialInterface::airy_bi_prime_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("bernoulli_Scaler", &SpecialInterface::bernoulli_Scaler);
    m.def("bernoulli_Array", &SpecialInterface::bernoulli_Array);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("bessel_in_Scaler", &SpecialInterface::bessel_in_Scaler<double>);
    m.def("bessel_in_Array", &SpecialInterface::bessel_in_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("bessel_in_prime_Scaler", &SpecialInterface::bessel_in_prime_Scaler<double>);
    m.def("bessel_in_prime_Array", &SpecialInterface::bessel_in_prime_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("bessel_jn_Scaler", &SpecialInterface::bessel_jn_Scaler<double>);
    m.def("bessel_jn_Array", &SpecialInterface::bessel_jn_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("bessel_jn_prime_Scaler", &SpecialInterface::bessel_jn_prime_Scaler<double>);
    m.def("bessel_jn_prime_Array", &SpecialInterface::bessel_jn_prime_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("bessel_kn_Scaler", &SpecialInterface::bessel_kn_Scaler<double>);
    m.def("bessel_kn_Array", &SpecialInterface::bessel_kn_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("bessel_kn_prime_Scaler", &SpecialInterface::bessel_kn_prime_Scaler<double>);
    m.def("bessel_kn_prime_Array", &SpecialInterface::bessel_kn_prime_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("bessel_yn_Scaler", &SpecialInterface::bessel_yn_Scaler<double>);
    m.def("bessel_yn_Array", &SpecialInterface::bessel_yn_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("bessel_yn_prime_Scaler", &SpecialInterface::bessel_yn_prime_Scaler<double>);
    m.def("bessel_yn_prime_Array", &SpecialInterface::bessel_yn_prime_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("beta_Scaler", &SpecialInterface::beta_Scaler<double>);
    m.def("beta_Array", &SpecialInterface::beta_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("comp_ellint_1_Scaler", &SpecialInterface::comp_ellint_1_Scaler<double>);
    m.def("comp_ellint_1_Array", &SpecialInterface::comp_ellint_1_Array<double>);
#endif

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("comp_ellint_2_Scaler", &SpecialInterface::comp_ellint_2_Scaler<double>);
    m.def("comp_ellint_2_Array", &SpecialInterface::comp_ellint_2_Array<double>);
#endif

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("comp_ellint_3_Scaler", &SpecialInterface::comp_ellint_3_Scaler<double>);
    m.def("comp_ellint_3_Array", &SpecialInterface::comp_ellint_3_Array<double, double>);
#endif
    m.def("cnr", &special::cnr);
#ifndef NUMCPP_NO_USE_BOOST
    m.def("cyclic_hankel_1_Scaler", &SpecialInterface::cyclic_hankel_1_Scaler<double>);
    m.def("cyclic_hankel_1_Array", &SpecialInterface::cyclic_hankel_1_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("cyclic_hankel_2_Scaler", &SpecialInterface::cyclic_hankel_2_Scaler<double>);
    m.def("cyclic_hankel_2_Array", &SpecialInterface::cyclic_hankel_2_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("ellint_1_Scaler", &SpecialInterface::ellint_1_Scaler<double>);
    m.def("ellint_1_Array", &SpecialInterface::ellint_1_Array<double, double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("ellint_2_Scaler", &SpecialInterface::ellint_2_Scaler<double>);
    m.def("ellint_2_Array", &SpecialInterface::ellint_2_Array<double, double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("ellint_3_Scaler", &SpecialInterface::ellint_3_Scaler<double>);
    m.def("ellint_3_Array", &SpecialInterface::ellint_3_Array<double, double, double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("expint_Scaler", &SpecialInterface::expint_Scaler<double>);
    m.def("expint_Array", &SpecialInterface::expint_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("digamma_Scaler", &SpecialInterface::digamma_Scaler<double>);
    m.def("digamma_Array", &SpecialInterface::digamma_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("erf_Scaler", &SpecialInterface::erf_Scaler<double>);
    m.def("erf_Array", &SpecialInterface::erf_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("erf_inv_Scaler", &SpecialInterface::erf_inv_Scaler<double>);
    m.def("erf_inv_Array", &SpecialInterface::erf_inv_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("erfc_Scaler", &SpecialInterface::erfc_Scaler<double>);
    m.def("erfc_Array", &SpecialInterface::erfc_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("erfc_inv_Scaler", &SpecialInterface::erfc_inv_Scaler<double>);
    m.def("erfc_inv_Array", &SpecialInterface::erfc_inv_Array<double>);
#endif
    m.def("factorial_Scaler", &SpecialInterface::factorial_Scaler);
    m.def("factorial_Array", &SpecialInterface::factorial_Array);
#ifndef NUMCPP_NO_USE_BOOST
    m.def("gamma_Scaler", &SpecialInterface::gamma_Scaler<double>);
    m.def("gamma_Array", &SpecialInterface::gamma_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("gamma1pm1_Scaler", &SpecialInterface::gamma1pm1_Scaler<double>);
    m.def("gamma1pm1_Array", &SpecialInterface::gamma1pm1_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("log_gamma_Scaler", &SpecialInterface::log_gamma_Scaler<double>);
    m.def("log_gamma_Array", &SpecialInterface::log_gamma_Array<double>);
#endif
    m.def("pnr", &special::pnr);
#ifndef NUMCPP_NO_USE_BOOST
    m.def("polygamma_Scaler", &SpecialInterface::polygamma_Scaler<double>);
    m.def("polygamma_Array", &SpecialInterface::polygamma_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("prime_Scaler", &SpecialInterface::prime_Scaler);
    m.def("prime_Array", &SpecialInterface::prime_Array);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("riemann_zeta_Scaler", &SpecialInterface::riemann_zeta_Scaler<double>);
    m.def("riemann_zeta_Array", &SpecialInterface::riemann_zeta_Array<double>);
#endif
    m.def("softmax", &SpecialInterface::softmax<double>);
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("spherical_bessel_jn_Scaler", &SpecialInterface::spherical_bessel_jn_Scaler<double>);
    m.def("spherical_bessel_jn_Array", &SpecialInterface::spherical_bessel_jn_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("spherical_bessel_yn_Scaler", &SpecialInterface::spherical_bessel_yn_Scaler<double>);
    m.def("spherical_bessel_yn_Array", &SpecialInterface::spherical_bessel_yn_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("spherical_hankel_1_Scaler", &SpecialInterface::spherical_hankel_1_Scaler<double>);
    m.def("spherical_hankel_1_Array", &SpecialInterface::spherical_hankel_1_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("spherical_hankel_2_Scaler", &SpecialInterface::spherical_hankel_2_Scaler<double>);
    m.def("spherical_hankel_2_Array", &SpecialInterface::spherical_hankel_2_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("trigamma_Scaler", &SpecialInterface::trigamma_Scaler<double>);
    m.def("trigamma_Array", &SpecialInterface::trigamma_Array<double>);
#endif
}
