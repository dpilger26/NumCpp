#include "BindingsIncludes.hpp"

#include "NumCpp/Utils/essentiallyEqual.hpp"

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
} // namespace IteratorInterface

//================================================================================

namespace NdArrayInterface
{
    template<typename dtype>
    bool test1DListContructor()
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<dtype> test = { dtype{ 1 },   dtype{ 2 },   dtype{ 3 },     dtype{ 4 },
                                dtype{ 666 }, dtype{ 357 }, dtype{ 314159 } };
        if (test.size() != 7)
        {
            return false;
        }

        if (test.shape().rows != 1 || test.shape().cols != test.size())
        {
            return false;
        }

        return utils::essentiallyEqual(test[0], dtype{ 1 }) && utils::essentiallyEqual(test[1], dtype{ 2 }) &&
               utils::essentiallyEqual(test[2], dtype{ 3 }) && utils::essentiallyEqual(test[3], dtype{ 4 }) &&
               utils::essentiallyEqual(test[4], dtype{ 666 }) && utils::essentiallyEqual(test[5], dtype{ 357 }) &&
               utils::essentiallyEqual(test[6], dtype{ 314159 });
    }

    //================================================================================

    template<typename dtype>
    bool test2DListContructor()
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<dtype> test = { { dtype{ 1 }, dtype{ 2 } },
                                { dtype{ 4 }, dtype{ 666 } },
                                { dtype{ 314159 }, dtype{ 9 } },
                                { dtype{ 0 }, dtype{ 8 } } };
        if (test.size() != 8)
        {
            return false;
        }

        if (test.shape().rows != 4 || test.shape().cols != 2)
        {
            return false;
        }

        return utils::essentiallyEqual(test[0], dtype{ 1 }) && utils::essentiallyEqual(test[1], dtype{ 2 }) &&
               utils::essentiallyEqual(test[2], dtype{ 4 }) && utils::essentiallyEqual(test[3], dtype{ 666 }) &&
               utils::essentiallyEqual(test[4], dtype{ 314159 }) && utils::essentiallyEqual(test[5], dtype{ 9 }) &&
               utils::essentiallyEqual(test[6], dtype{ 0 }) && utils::essentiallyEqual(test[7], dtype{ 8 });
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test1dArrayConstructor(T value1, T value2)
    {
        std::array<T, 2> arr        = { value1, value2 };
        auto             newNcArray = NdArray<T>(arr);

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test2dArrayConstructor(T value1, T value2)
    {
        std::array<std::array<T, 2>, 2> arr2d{};
        arr2d[0][0]     = value1;
        arr2d[0][1]     = value2;
        arr2d[1][0]     = value1;
        arr2d[1][1]     = value2;
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
        const auto ncArray    = pybind2nc(inArray);
        auto       newNcArray = NdArray<T>(ncArray.cbegin(), ncArray.cend());

        return nc2pybind(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test1dPointerConstructor(pbArray<T> inArray)
    {
        const auto ncArray    = pybind2nc(inArray);
        auto       newNcArray = NdArray<T>(ncArray.data(), ncArray.size());

        return nc2pybind(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test2dPointerConstructor(pbArray<T> inArray)
    {
        const auto ncArray    = pybind2nc(inArray);
        auto       newNcArray = NdArray<T>(ncArray.data(), ncArray.numRows(), ncArray.numCols());

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test1dPointerShellConstructor(pbArray<T> inArray)
    {
        auto ncArray    = pybind2nc(inArray).copy();
        auto newNcArray = NdArray<T>(ncArray.data(), ncArray.size(), PointerPolicy::SHELL);

        return nc2pybind(newNcArray.reshape(ncArray.shape()));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric test2dPointerShellConstructor(pbArray<T> inArray)
    {
        auto ncArray    = pybind2nc(inArray).copy();
        auto newNcArray = NdArray<T>(ncArray.data(), ncArray.numRows(), ncArray.numCols(), PointerPolicy::SHELL);

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric testCopyConstructor(pbArray<T> inArray)
    {
        const auto ncArray    = pybind2nc(inArray);
        auto       newNcArray = NdArray<T>(ncArray);

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric testMoveConstructor(pbArray<T> inArray)
    {
        auto ncArray    = pybind2nc(inArray);
        auto newNcArray = NdArray<T>(std::move(ncArray));

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric testAssignementOperator(pbArray<T> inArray)
    {
        auto       ncArray = pybind2nc(inArray);
        NdArray<T> newNcArray;
        newNcArray = ncArray;

        return nc2pybind(newNcArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric testAssignementScalarOperator(pbArray<T> inArray, T value)
    {
        auto ncArray = pybind2nc(inArray);
        ncArray      = value;

        return nc2pybind(ncArray);
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric testMoveAssignementOperator(pbArray<T> inArray)
    {
        auto       ncArray = pybind2nc(inArray);
        NdArray<T> newNcArray;
        newNcArray = std::move(ncArray);

        return nc2pybind(newNcArray);
    }

    //================================================================================

    struct TestStruct
    {
        int    member1{ 0 };
        int    member2{ 0 };
        double member3{ 0.0 };
        bool   member4{ true };
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
        NdArray<TestStruct> test6 = { TestStruct{ 666, 357, 3.14519, true }, TestStruct{ 666, 357, 3.14519, true } };
        NdArray<TestStruct> test7 = { { TestStruct{ 666, 357, 3.14519, true }, TestStruct{ 667, 377, 3.7519, false } },
                                      { TestStruct{ 665, 357, 3.15519, false },
                                        TestStruct{ 69, 359, 3.19519, true } } };

        auto testStruct = TestStruct{ 666, 357, 3.14519, true };
        test6           = testStruct;
        test1.resizeFast({ 10, 10 });
        test1 = testStruct;

        [[maybe_unused]] const auto beg  = test7.begin();
        [[maybe_unused]] const auto beg0 = test5_1.begin(0);
        [[maybe_unused]] const auto end  = test5_2.end();
        [[maybe_unused]] const auto end0 = test2.end(0);

        test2.resizeFast({ 10, 10 });
        test2                              = TestStruct{ 666, 357, 3.14519, true };
        [[maybe_unused]] const auto slice1 = test2.rSlice();
        [[maybe_unused]] const auto slice2 = test2.cSlice();
        test2.back();
        [[maybe_unused]] const auto fr      = test2.column(0);
        [[maybe_unused]] const auto c       = test2.copy();
        [[maybe_unused]] const auto dataPtr = test2.data();
        [[maybe_unused]] const auto d       = test2.diagonal();
        test2.dump("test.bin");
        remove("test.bin");
        test2.fill(TestStruct{ 0, 1, 6.5, false });
        [[maybe_unused]] const auto f = test2.flatten();
        test2.front();
        test2[0];
        test2(0, 0);
        test2[0]    = TestStruct{ 0, 1, 6.5, false };
        test2(0, 0) = TestStruct{ 0, 1, 6.5, false };
        test2.isempty();
        test2.isflat();
        test2.issquare();
        [[maybe_unused]] const auto nb   = test2.nbytes();
        [[maybe_unused]] const auto rows = test2.numRows();
        [[maybe_unused]] const auto cols = test2.numCols();
        test2.put(0, TestStruct{ 0, 1, 6.5, false });
        test2.ravel();
        [[maybe_unused]] const auto r = test2.repeat({ 2, 2 });
        test2.reshape(test2.size(), 1);
        test2.resizeFast(1, 1);
        test2.resizeSlow(10, 10);
        [[maybe_unused]] const auto row0 = test2.row(0);
        [[maybe_unused]] const auto s1   = test2.shape();
        [[maybe_unused]] const auto s2   = test2.size();
        [[maybe_unused]] const auto s3   = test2.swapaxes();
        [[maybe_unused]] const auto t    = test2.transpose();

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
    pbArrayGeneric argpartition(NdArray<dtype>& self, uint32 inKth, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.argpartition(inKth, inAxis));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric argsort(const NdArray<dtype>& self, Axis inAxis = Axis::NONE)
    {
        return nc2pybind(self.argsort(inAxis));
    }

    //================================================================================

    template<typename dtype>
    dtype atValueFlat(NdArray<dtype>& self, int32 inIndex)
    {
        return self.at(inIndex);
    }

    //================================================================================

    template<typename dtype>
    dtype atValueFlatConst(const NdArray<dtype>& self, int32 inIndex)
    {
        return self.at(inIndex);
    }

    //================================================================================

    template<typename dtype>
    dtype atValueRowCol(NdArray<dtype>& self, int32 inRow, int32 inCol)
    {
        return self.at(inRow, inCol);
    }

    //================================================================================

    template<typename dtype>
    dtype atValueRowColConst(const NdArray<dtype>& self, int32 inRow, int32 inCol)
    {
        return self.at(inRow, inCol);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric atMask(const NdArray<dtype>& self, const NdArray<bool>& mask)
    {
        return nc2pybind(self.at(mask));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric atIndices(const NdArray<dtype>& self, const NdArray<int32>& indices)
    {
        return nc2pybind(self.at(indices));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric atSlice1D(const NdArray<dtype>& self, const Slice& inSlice)
    {
        return nc2pybind(self.at(inSlice));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric atSlice2D(const NdArray<dtype>& self, const Slice& inRowSlice, const Slice& inColSlice)
    {
        return nc2pybind(self.at(inRowSlice, inColSlice));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric atSlice2DCol(const NdArray<dtype>& self, const Slice& inRowSlice, int32 inColIndex)
    {
        return nc2pybind(self.at(inRowSlice, inColIndex));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric atSlice2DRow(const NdArray<dtype>& self, int32 inRowIndex, const Slice& inColSlice)
    {
        return nc2pybind(self.at(inRowIndex, inColSlice));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric atIndicesScalar(const NdArray<dtype>& self, const NdArray<int32>& rowIndices, int32 colIndex)
    {
        return nc2pybind(self.at(rowIndices, colIndex));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric atIndicesSlice(const NdArray<dtype>& self, const NdArray<int32>& rowIndices, Slice colSlice)
    {
        return nc2pybind(self.at(rowIndices, colSlice));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric atScalarIndices(const NdArray<dtype>& self, int32 rowIndex, const NdArray<int32>& colIndices)
    {
        return nc2pybind(self.at(rowIndex, colIndices));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric atSliceIndices(const NdArray<dtype>& self, Slice rowSlice, const NdArray<int32>& colIndices)
    {
        return nc2pybind(self.at(rowSlice, colIndices));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        atIndices2D(const NdArray<dtype>& self, const NdArray<int32>& rowIndices, const NdArray<int32>& colIndices)
    {
        return nc2pybind(self.at(rowIndices, colIndices));
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
    pbArrayGeneric getIndices(const NdArray<dtype>& self, const NdArray<int32>& indices)
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
    pbArrayGeneric getIndicesScalar(const NdArray<dtype>& self, const NdArray<int32>& rowIndices, int32 colIndex)
    {
        return nc2pybind(self(rowIndices, colIndex));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getIndicesSlice(const NdArray<dtype>& self, const NdArray<int32>& rowIndices, Slice colSlice)
    {
        return nc2pybind(self(rowIndices, colSlice));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getScalarIndices(const NdArray<dtype>& self, int32 rowIndex, const NdArray<int32>& colIndices)
    {
        return nc2pybind(self(rowIndex, colIndices));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric getSliceIndices(const NdArray<dtype>& self, Slice rowSlice, const NdArray<int32>& colIndices)
    {
        return nc2pybind(self(rowSlice, colIndices));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        getIndices2D(const NdArray<dtype>& self, const NdArray<int32>& rowIndices, const NdArray<int32>& colIndices)
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
    python_interface::tuple nonzero(const NdArray<dtype>& self)
    {
        auto rowCol = self.nonzero();
        return python_interface::make_tuple(nc2pybind(rowCol.first), nc2pybind(rowCol.second));
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
    pbArrayGeneric putIndices1DValue(NdArray<dtype>& self, const NdArray<int32>& inIndices, dtype inValue)
    {
        self.put(inIndices, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        putIndices1DValues(NdArray<dtype>& self, const NdArray<int32>& inIndices, pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inIndices, inValues);
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
    pbArrayGeneric putIndices2DValue(NdArray<dtype>&       self,
                                     const NdArray<int32>& inRowIndices,
                                     const NdArray<int32>& inColIndices,
                                     dtype                 inValue)
    {
        self.put(inRowIndices, inColIndices, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putRowIndicesColSliceValues(NdArray<dtype>&       self,
                                               const NdArray<int32>& inRowIndices,
                                               const Slice&          inColSlice,
                                               pbArray<dtype>&       inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inRowIndices, inColSlice, inValues);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putRowSliceColIndicesValues(NdArray<dtype>&       self,
                                               const Slice&          inRowSlice,
                                               const NdArray<int32>& inColIndices,
                                               pbArray<dtype>&       inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inRowSlice, inColIndices, inValues);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        putSlice2DValue(NdArray<dtype>& self, const Slice& inSliceRow, const Slice& inSliceCol, dtype inValue)
    {
        self.put(inSliceRow, inSliceCol, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric
        putIndices2DValueRow(NdArray<dtype>& self, int32 inRowIndex, const NdArray<int32>& inColIndices, dtype inValue)
    {
        self.put(inRowIndex, inColIndices, inValue);
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
    pbArrayGeneric
        putIndices2DValueCol(NdArray<dtype>& self, const NdArray<int32>& inRowIndices, int32 inColIndex, dtype inValue)
    {
        self.put(inRowIndices, inColIndex, inValue);
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
    pbArrayGeneric putIndices2DValues(NdArray<dtype>&       self,
                                      const NdArray<int32>& inRowIndices,
                                      const NdArray<int32>& inColIndices,
                                      pbArray<dtype>&       inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inRowIndices, inColIndices, inValues);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putRowIndicesColSliceValue(NdArray<dtype>&       self,
                                              const NdArray<int32>& inRowIndices,
                                              const Slice&          inColSlice,
                                              dtype                 inValue)
    {
        self.put(inRowIndices, inColSlice, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putRowSliceColIndicesValue(NdArray<dtype>&       self,
                                              const Slice&          inRowSlice,
                                              const NdArray<int32>& inColIndices,
                                              dtype                 inValue)
    {
        self.put(inRowSlice, inColIndices, inValue);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValues(NdArray<dtype>& self,
                                    const Slice&    inSliceRow,
                                    const Slice&    inSliceCol,
                                    pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inSliceRow, inSliceCol, inValues);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putIndices2DValuesRow(NdArray<dtype>&       self,
                                         int32                 inRowIndex,
                                         const NdArray<int32>& inColIndices,
                                         pbArray<dtype>&       inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inRowIndex, inColIndices, inValues);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValuesRow(NdArray<dtype>& self,
                                       int32           inRowIndex,
                                       const Slice&    inSliceCol,
                                       pbArray<dtype>& inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inRowIndex, inSliceCol, inValues);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putIndices2DValuesCol(NdArray<dtype>&       self,
                                         const NdArray<int32>& inRowIndices,
                                         int32                 inColIndex,
                                         pbArray<dtype>&       inArrayValues)
    {
        NdArray<dtype> inValues = pybind2nc(inArrayValues);
        self.put(inRowIndices, inColIndex, inValues);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric putSlice2DValuesCol(NdArray<dtype>& self,
                                       const Slice&    inSliceRow,
                                       int32           inColIndex,
                                       pbArray<dtype>& inArrayValues)
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
        auto mask     = pybind2nc(inMask);
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
    pbArrayGeneric swapRows(NdArray<dtype>& self, int32 rowIdx1, int32 rowIdx2)
    {
        self.swapRows(rowIdx1, rowIdx2);
        return nc2pybind(self);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric swapCols(NdArray<dtype>& self, int32 colIdx1, int32 colIdx2)
    {
        self.swapCols(colIdx1, colIdx2);
        return nc2pybind(self);
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
    pbArrayGeneric operatorPlusEqualScalar(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs += rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorPlusEqualComplexArrayArithScalar(NdArray<std::complex<dtype>>& lhs, dtype rhs)
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
    pbArrayGeneric operatorPlusArithArrayComplexArray(const NdArray<dtype>&               lhs,
                                                      const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorPlusComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs,
                                                      const NdArray<dtype>&               rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorPlusArrayScalar(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    pbArrayGeneric operatorPlusScalarArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    pbArrayGeneric operatorPlusArithArrayComplexScalar(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    pbArrayGeneric operatorPlusComplexScalarArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    pbArrayGeneric operatorPlusComplexArrayArithScalar(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs + rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    pbArrayGeneric operatorPlusArithScalarComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
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
    pbArrayGeneric operatorMinusEqualComplexArrayArithArray(NdArray<std::complex<dtype>>& lhs,
                                                            const NdArray<dtype>&         rhs)
    {
        return nc2pybind(lhs -= rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorMinusEqualScalar(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs -= rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorMinusEqualComplexArrayArithScalar(NdArray<std::complex<dtype>>& lhs, dtype rhs)
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
    pbArrayGeneric operatorMinusArithArrayComplexArray(const NdArray<dtype>&               lhs,
                                                       const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorMinusComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs,
                                                       const NdArray<dtype>&               rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorMinusArrayScalar(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    pbArrayGeneric operatorMinusScalarArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    pbArrayGeneric operatorMinusArithArrayComplexScalar(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    pbArrayGeneric operatorMinusComplexScalarArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    pbArrayGeneric operatorMinusComplexArrayArithScalar(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs - rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    pbArrayGeneric operatorMinusArithScalarComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
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
    pbArrayGeneric operatorMultiplyEqualComplexArrayArithArray(NdArray<std::complex<dtype>>& lhs,
                                                               const NdArray<dtype>&         rhs)
    {
        return nc2pybind(lhs *= rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorMultiplyEqualScalar(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs *= rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorMultiplyEqualComplexArrayArithScalar(NdArray<std::complex<dtype>>& lhs, dtype rhs)
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
    pbArrayGeneric operatorMultiplyArithArrayComplexArray(const NdArray<dtype>&               lhs,
                                                          const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorMultiplyComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs,
                                                          const NdArray<dtype>&               rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorMultiplyArrayScalar(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    pbArrayGeneric operatorMultiplyScalarArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    pbArrayGeneric operatorMultiplyArithArrayComplexScalar(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    pbArrayGeneric operatorMultiplyComplexScalarArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    pbArrayGeneric operatorMultiplyComplexArrayArithScalar(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs * rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    pbArrayGeneric operatorMultiplyArithScalarComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
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
    pbArrayGeneric operatorDivideEqualComplexArrayArithArray(NdArray<std::complex<dtype>>& lhs,
                                                             const NdArray<dtype>&         rhs)
    {
        return nc2pybind(lhs /= rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorDivideEqualScalar(NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs /= rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorDivideEqualComplexArrayArithScalar(NdArray<std::complex<dtype>>& lhs, dtype rhs)
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
    pbArrayGeneric operatorDivideArithArrayComplexArray(const NdArray<dtype>&               lhs,
                                                        const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (3)
    pbArrayGeneric operatorDivideComplexArrayArithArray(const NdArray<std::complex<dtype>>& lhs,
                                                        const NdArray<dtype>&               rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (4)
    pbArrayGeneric operatorDivideArrayScalar(const NdArray<dtype>& lhs, dtype rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (5)
    pbArrayGeneric operatorDivideScalarArray(dtype lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (6)
    pbArrayGeneric operatorDivideArithArrayComplexScalar(const NdArray<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (7)
    pbArrayGeneric operatorDivideComplexScalarArithArray(const std::complex<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (8)
    pbArrayGeneric operatorDivideComplexArrayArithScalar(const NdArray<std::complex<dtype>>& lhs, dtype rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype> // (9)
    pbArrayGeneric operatorDivideArithScalarComplexArray(dtype lhs, const NdArray<std::complex<dtype>>& rhs)
    {
        return nc2pybind(lhs / rhs);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorModulusEqualArray(NdArray<dtype> inArray1, const NdArray<dtype>& inArray2)
    {
        inArray1 %= inArray2;
        return nc2pybind(inArray1);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorModulusScalar(const NdArray<dtype>& inArray, dtype inScalar)
    {
        return nc2pybind(inArray % inScalar);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorModulusScalarReversed(dtype inScalar, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inScalar % inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorModulusArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 % inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseOrEqualArray(NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        inArray1 |= inArray2;
        return nc2pybind(inArray1);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseOrScalar(const NdArray<dtype>& inArray, dtype inScalar)
    {
        return nc2pybind(inArray | inScalar);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseOrScalarReversed(dtype inScalar, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inScalar | inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseOrArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 | inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseAndEqualArray(NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        inArray1 &= inArray2;
        return nc2pybind(inArray1);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseAndScalar(const NdArray<dtype>& inArray, dtype inScalar)
    {
        return nc2pybind(inArray & inScalar);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseAndScalarReversed(dtype inScalar, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inScalar & inArray);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseAndArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return nc2pybind(inArray1 & inArray2);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseXorEqualArray(NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        inArray1 ^= inArray2;
        return nc2pybind(inArray1);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseXorScalar(const NdArray<dtype>& inArray, dtype inScalar)
    {
        return nc2pybind(inArray ^ inScalar);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorBitwiseXorScalarReversed(dtype inScalar, const NdArray<dtype>& inArray)
    {
        return nc2pybind(inScalar ^ inArray);
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
    pbArrayGeneric operatorEqualityScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray == inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorEqualityScalarReversed(dtype inValue, const NdArray<dtype>& inArray)
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
    pbArrayGeneric operatorNotEqualityScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray != inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorNotEqualityScalarReversed(dtype inValue, const NdArray<dtype>& inArray)
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
    pbArrayGeneric operatorLessScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray < inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLessScalarReversed(dtype inValue, const NdArray<dtype>& inArray)
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
    pbArrayGeneric operatorGreaterScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray > inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorGreaterScalarReversed(dtype inValue, const NdArray<dtype>& inArray)
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
    pbArrayGeneric operatorLessEqualScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray <= inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorLessEqualScalarReversed(dtype inValue, const NdArray<dtype>& inArray)
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
    pbArrayGeneric operatorGreaterEqualScalar(const NdArray<dtype>& inArray, dtype inValue)
    {
        return nc2pybind(inArray >= inValue);
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric operatorGreaterEqualScalarReversed(dtype inValue, const NdArray<dtype>& inArray)
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
} // namespace NdArrayInterface

//================================================================================

void initNdArray(python_interface::module& m)
{
    // NdArray.hpp
    python_interface::class_<NdArrayDoubleIterator>(m, "NdArrayDoubleIterator")
        .def(python_interface::init<>())
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

    python_interface::class_<NdArrayComplexDoubleIterator>(m, "NdArrayComplexDoubleIterator")
        .def(python_interface::init<>())
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

    python_interface::class_<NdArrayDoubleConstIterator>(m, "NdArrayDoubleConstIterator")
        .def(python_interface::init<>())
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

    python_interface::class_<NdArrayComplexDoubleConstIterator>(m, "NdArrayComplexDoubleConstIterator")
        .def(python_interface::init<>())
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

    python_interface::class_<NdArrayDoubleReverseIterator>(m, "NdArrayDoubleReverseIterator")
        .def(python_interface::init<>())
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

    python_interface::class_<NdArrayComplexDoubleReverseIterator>(m, "NdArrayComplexDoubleReverseIterator")
        .def(python_interface::init<>())
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

    python_interface::class_<NdArrayDoubleConstReverseIterator>(m, "NdArrayDoubleConstReverseIterator")
        .def(python_interface::init<>())
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

    python_interface::class_<NdArrayComplexDoubleConstReverseIterator>(m, "NdArrayComplexDoubleConstReverseIterator")
        .def(python_interface::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleConstReverseIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleConstReverseIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleConstReverseIterator>)
        .def("operatorMinusMinusPre",
             IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleConstReverseIterator>)
        .def("operatorMinusMinusPost",
             IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleConstReverseIterator>)
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

    python_interface::class_<NdArrayDoubleColumnIterator>(m, "NdArrayDoubleColumnIterator")
        .def(python_interface::init<>())
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

    python_interface::class_<NdArrayComplexDoubleColumnIterator>(m, "NdArrayComplexDoubleColumnIterator")
        .def(python_interface::init<>())
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

    python_interface::class_<NdArrayDoubleConstColumnIterator>(m, "NdArrayDoubleConstColumnIterator")
        .def(python_interface::init<>())
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

    python_interface::class_<NdArrayComplexDoubleConstColumnIterator>(m, "NdArrayComplexDoubleConstColumnIterator")
        .def(python_interface::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleConstColumnIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleConstColumnIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleConstColumnIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleConstColumnIterator>)
        .def("operatorMinusMinusPost",
             IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleConstColumnIterator>)
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

    python_interface::class_<NdArrayDoubleReverseColumnIterator>(m, "NdArrayDoubleReverseColumnIterator")
        .def(python_interface::init<>())
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

    python_interface::class_<NdArrayComplexDoubleReverseColumnIterator>(m, "NdArrayComplexDoubleReverseColumnIterator")
        .def(python_interface::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleReverseColumnIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleReverseColumnIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleReverseColumnIterator>)
        .def("operatorMinusMinusPre",
             IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleReverseColumnIterator>)
        .def("operatorMinusMinusPost",
             IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleReverseColumnIterator>)
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

    python_interface::class_<NdArrayDoubleConstReverseColumnIterator>(m, "NdArrayDoubleConstReverseColumnIterator")
        .def(python_interface::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayDoubleConstReverseColumnIterator>)
        .def("operatorPlusPlusPre", IteratorInterface::operatorPlusPlusPre<NdArrayDoubleConstReverseColumnIterator>)
        .def("operatorPlusPlusPost", IteratorInterface::operatorPlusPlusPost<NdArrayDoubleConstReverseColumnIterator>)
        .def("operatorMinusMinusPre", IteratorInterface::operatorMinusMinusPre<NdArrayDoubleConstReverseColumnIterator>)
        .def("operatorMinusMinusPost",
             IteratorInterface::operatorMinusMinusPost<NdArrayDoubleConstReverseColumnIterator>)
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

    python_interface::class_<NdArrayComplexDoubleConstReverseColumnIterator>(
        m,
        "NdArrayComplexDoubleConstReverseColumnIterator")
        .def(python_interface::init<>())
        .def("operatorDereference", &IteratorInterface::dereference<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("operatorPlusPlusPre",
             IteratorInterface::operatorPlusPlusPre<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("operatorPlusPlusPost",
             IteratorInterface::operatorPlusPlusPost<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("operatorMinusMinusPre",
             IteratorInterface::operatorMinusMinusPre<NdArrayComplexDoubleConstReverseColumnIterator>)
        .def("operatorMinusMinusPost",
             IteratorInterface::operatorMinusMinusPost<NdArrayComplexDoubleConstReverseColumnIterator>)
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

    NdArrayDoubleIterator (NdArrayDouble::*begin)()                                            = &NdArrayDouble::begin;
    NdArrayDoubleIterator (NdArrayDouble::*beginRow)(NdArrayDouble::size_type)                 = &NdArrayDouble::begin;
    NdArrayDoubleConstIterator (NdArrayDouble::*beginConst)() const                            = &NdArrayDouble::cbegin;
    NdArrayDoubleConstIterator (NdArrayDouble::*beginRowConst)(NdArrayDouble::size_type) const = &NdArrayDouble::cbegin;

    NdArrayComplexDoubleIterator (NdArrayComplexDouble::*beginComplex)() = &NdArrayComplexDouble::begin;
    NdArrayComplexDoubleIterator (NdArrayComplexDouble::*beginRowComplex)(NdArrayComplexDouble::size_type) =
        &NdArrayComplexDouble::begin;
    NdArrayComplexDoubleConstIterator (NdArrayComplexDouble::*beginConstComplex)() const =
        &NdArrayComplexDouble::cbegin;
    NdArrayComplexDoubleConstIterator (NdArrayComplexDouble::*beginRowConstComplex)(NdArrayComplexDouble::size_type)
        const = &NdArrayComplexDouble::cbegin;

    NdArrayDoubleColumnIterator (NdArrayDouble::*colbegin)()                            = &NdArrayDouble::colbegin;
    NdArrayDoubleColumnIterator (NdArrayDouble::*colbeginCol)(NdArrayDouble::size_type) = &NdArrayDouble::colbegin;
    NdArrayDoubleConstColumnIterator (NdArrayDouble::*colbeginConst)() const            = &NdArrayDouble::ccolbegin;
    NdArrayDoubleConstColumnIterator (NdArrayDouble::*colbeginColConst)(NdArrayDouble::size_type) const =
        &NdArrayDouble::ccolbegin;

    NdArrayComplexDoubleColumnIterator (NdArrayComplexDouble::*colbeginComplex)() = &NdArrayComplexDouble::colbegin;
    NdArrayComplexDoubleColumnIterator (NdArrayComplexDouble::*colbeginColComplex)(NdArrayComplexDouble::size_type) =
        &NdArrayComplexDouble::colbegin;
    NdArrayComplexDoubleConstColumnIterator (NdArrayComplexDouble::*colbeginConstComplex)() const =
        &NdArrayComplexDouble::ccolbegin;
    NdArrayComplexDoubleConstColumnIterator (NdArrayComplexDouble::*colbeginColConstComplex)(
        NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::ccolbegin;

    NdArrayDoubleReverseIterator (NdArrayDouble::*rbegin)()                            = &NdArrayDouble::rbegin;
    NdArrayDoubleReverseIterator (NdArrayDouble::*rbeginRow)(NdArrayDouble::size_type) = &NdArrayDouble::rbegin;
    NdArrayDoubleConstReverseIterator (NdArrayDouble::*rbeginConst)() const            = &NdArrayDouble::crbegin;
    NdArrayDoubleConstReverseIterator (NdArrayDouble::*rbeginRowConst)(NdArrayDouble::size_type) const =
        &NdArrayDouble::crbegin;

    NdArrayComplexDoubleReverseIterator (NdArrayComplexDouble::*rbeginComplex)() = &NdArrayComplexDouble::rbegin;
    NdArrayComplexDoubleReverseIterator (NdArrayComplexDouble::*rbeginRowComplex)(NdArrayComplexDouble::size_type) =
        &NdArrayComplexDouble::rbegin;
    NdArrayComplexDoubleConstReverseIterator (NdArrayComplexDouble::*rbeginConstComplex)() const =
        &NdArrayComplexDouble::crbegin;
    NdArrayComplexDoubleConstReverseIterator (NdArrayComplexDouble::*rbeginRowConstComplex)(
        NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::crbegin;

    NdArrayDoubleReverseColumnIterator (NdArrayDouble::*rcolbegin)() = &NdArrayDouble::rcolbegin;
    NdArrayDoubleReverseColumnIterator (NdArrayDouble::*rcolbeginCol)(NdArrayDouble::size_type) =
        &NdArrayDouble::rcolbegin;
    NdArrayDoubleConstReverseColumnIterator (NdArrayDouble::*rcolbeginConst)() const = &NdArrayDouble::crcolbegin;
    NdArrayDoubleConstReverseColumnIterator (NdArrayDouble::*rcolbeginColConst)(NdArrayDouble::size_type) const =
        &NdArrayDouble::crcolbegin;

    NdArrayComplexDoubleReverseColumnIterator (NdArrayComplexDouble::*rcolbeginComplex)() =
        &NdArrayComplexDouble::rcolbegin;
    NdArrayComplexDoubleReverseColumnIterator (NdArrayComplexDouble::*rcolbeginColComplex)(
        NdArrayComplexDouble::size_type) = &NdArrayComplexDouble::rcolbegin;
    NdArrayComplexDoubleConstReverseColumnIterator (NdArrayComplexDouble::*rcolbeginConstComplex)() const =
        &NdArrayComplexDouble::crcolbegin;
    NdArrayComplexDoubleConstReverseColumnIterator (NdArrayComplexDouble::*rcolbeginColConstComplex)(
        NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::crcolbegin;

    NdArrayDoubleIterator (NdArrayDouble::*end)()                                            = &NdArrayDouble::end;
    NdArrayDoubleIterator (NdArrayDouble::*endRow)(NdArrayDouble::size_type)                 = &NdArrayDouble::end;
    NdArrayDoubleConstIterator (NdArrayDouble::*endConst)() const                            = &NdArrayDouble::cend;
    NdArrayDoubleConstIterator (NdArrayDouble::*endRowConst)(NdArrayDouble::size_type) const = &NdArrayDouble::cend;

    NdArrayComplexDoubleIterator (NdArrayComplexDouble::*endComplex)() = &NdArrayComplexDouble::end;
    NdArrayComplexDoubleIterator (NdArrayComplexDouble::*endRowComplex)(NdArrayComplexDouble::size_type) =
        &NdArrayComplexDouble::end;
    NdArrayComplexDoubleConstIterator (NdArrayComplexDouble::*endConstComplex)() const = &NdArrayComplexDouble::cend;
    NdArrayComplexDoubleConstIterator (NdArrayComplexDouble::*endRowConstComplex)(NdArrayComplexDouble::size_type)
        const = &NdArrayComplexDouble::cend;

    NdArrayDoubleColumnIterator (NdArrayDouble::*colend)()                            = &NdArrayDouble::colend;
    NdArrayDoubleColumnIterator (NdArrayDouble::*colendCol)(NdArrayDouble::size_type) = &NdArrayDouble::colend;
    NdArrayDoubleConstColumnIterator (NdArrayDouble::*colendConst)() const            = &NdArrayDouble::ccolend;
    NdArrayDoubleConstColumnIterator (NdArrayDouble::*colendColConst)(NdArrayDouble::size_type) const =
        &NdArrayDouble::ccolend;

    NdArrayComplexDoubleColumnIterator (NdArrayComplexDouble::*colendComplex)() = &NdArrayComplexDouble::colend;
    NdArrayComplexDoubleColumnIterator (NdArrayComplexDouble::*colendColComplex)(NdArrayComplexDouble::size_type) =
        &NdArrayComplexDouble::colend;
    NdArrayComplexDoubleConstColumnIterator (NdArrayComplexDouble::*colendConstComplex)() const =
        &NdArrayComplexDouble::ccolend;
    NdArrayComplexDoubleConstColumnIterator (NdArrayComplexDouble::*colendColConstComplex)(
        NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::ccolend;

    NdArrayDoubleReverseIterator (NdArrayDouble::*rend)()                            = &NdArrayDouble::rend;
    NdArrayDoubleReverseIterator (NdArrayDouble::*rendRow)(NdArrayDouble::size_type) = &NdArrayDouble::rend;
    NdArrayDoubleConstReverseIterator (NdArrayDouble::*rendConst)() const            = &NdArrayDouble::crend;
    NdArrayDoubleConstReverseIterator (NdArrayDouble::*rendRowConst)(NdArrayDouble::size_type) const =
        &NdArrayDouble::crend;

    NdArrayComplexDoubleReverseIterator (NdArrayComplexDouble::*rendComplex)() = &NdArrayComplexDouble::rend;
    NdArrayComplexDoubleReverseIterator (NdArrayComplexDouble::*rendRowComplex)(NdArrayComplexDouble::size_type) =
        &NdArrayComplexDouble::rend;
    NdArrayComplexDoubleConstReverseIterator (NdArrayComplexDouble::*rendConstComplex)() const =
        &NdArrayComplexDouble::crend;
    NdArrayComplexDoubleConstReverseIterator (NdArrayComplexDouble::*rendRowConstComplex)(
        NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::crend;

    NdArrayDoubleReverseColumnIterator (NdArrayDouble::*rcolend)()                            = &NdArrayDouble::rcolend;
    NdArrayDoubleReverseColumnIterator (NdArrayDouble::*rcolendCol)(NdArrayDouble::size_type) = &NdArrayDouble::rcolend;
    NdArrayDoubleConstReverseColumnIterator (NdArrayDouble::*rcolendConst)() const = &NdArrayDouble::crcolend;
    NdArrayDoubleConstReverseColumnIterator (NdArrayDouble::*rcolendColConst)(NdArrayDouble::size_type) const =
        &NdArrayDouble::crcolend;

    NdArrayComplexDoubleReverseColumnIterator (NdArrayComplexDouble::*rcolendComplex)() =
        &NdArrayComplexDouble::rcolend;
    NdArrayComplexDoubleReverseColumnIterator (NdArrayComplexDouble::*rcolendColComplex)(
        NdArrayComplexDouble::size_type) = &NdArrayComplexDouble::rcolend;
    NdArrayComplexDoubleConstReverseColumnIterator (NdArrayComplexDouble::*rcolendConstComplex)() const =
        &NdArrayComplexDouble::crcolend;
    NdArrayComplexDoubleConstReverseColumnIterator (NdArrayComplexDouble::*rcolendColConstComplex)(
        NdArrayComplexDouble::size_type) const = &NdArrayComplexDouble::crcolend;

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
    m.def("testAssignementScalarOperator", &NdArrayInterface::testAssignementScalarOperator<double>);
    m.def("testAssignementScalarOperator", &NdArrayInterface::testAssignementScalarOperator<ComplexDouble>);
    m.def("testMoveAssignementOperator", &NdArrayInterface::testMoveAssignementOperator<double>);
    m.def("testMoveAssignementOperator", &NdArrayInterface::testMoveAssignementOperator<ComplexDouble>);

    python_interface::class_<NdArrayDouble>(m, "NdArray")
        .def(python_interface::init<>())
        .def(python_interface::init<NdArrayDouble::size_type>())
        .def(python_interface::init<NdArrayDouble::size_type, NdArrayDouble::size_type>())
        .def(python_interface::init<Shape>())
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
        .def("get", &NdArrayInterface::getIndicesScalar<double>)
        .def("get", &NdArrayInterface::getIndicesSlice<double>)
        .def("get", &NdArrayInterface::getScalarIndices<double>)
        .def("get", &NdArrayInterface::getSliceIndices<double>)
        .def("get", &NdArrayInterface::getIndices2D<double>)
        .def("at", &NdArrayInterface::atValueFlat<double>)
        .def("atConst", &NdArrayInterface::atValueFlatConst<double>)
        .def("at", &NdArrayInterface::atValueRowCol<double>)
        .def("atConst", &NdArrayInterface::atValueRowColConst<double>)
        .def("at", &NdArrayInterface::atMask<double>)
        .def("at", &NdArrayInterface::atIndices<double>)
        .def("at", &NdArrayInterface::atSlice1D<double>)
        .def("at", &NdArrayInterface::atSlice2D<double>)
        .def("at", &NdArrayInterface::atSlice2DRow<double>)
        .def("at", &NdArrayInterface::atSlice2DCol<double>)
        .def("at", &NdArrayInterface::atIndicesScalar<double>)
        .def("at", &NdArrayInterface::atIndicesSlice<double>)
        .def("at", &NdArrayInterface::atScalarIndices<double>)
        .def("at", &NdArrayInterface::atSliceIndices<double>)
        .def("at", &NdArrayInterface::atIndices2D<double>)
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
        .def("argpartition", &NdArrayInterface::argpartition<double>)
        .def("argsort", &NdArrayInterface::argsort<double>)
        .def("astypeUint32", &NdArrayDouble::astype<uint32>)
        .def("astypeComplex", &NdArrayDouble::astype<ComplexDouble>)
        .def("back", &NdArrayInterface::back<double>)
        .def("backReference", &NdArrayInterface::backReference<double>)
        .def("back", &NdArrayInterface::backRow<double>)
        .def("backReference", &NdArrayInterface::backRowReference<double>)
        .def("clip", &NdArrayInterface::clip<double>)
        .def("column", &NdArrayDouble::column)
        .def("columns",
             [](const NdArray<double>& self, pbArray<uint32> rowIndices)
             { return nc2pybind(self.columns(pybind2nc(rowIndices))); })
        .def("contains", &NdArrayInterface::contains<double>)
        .def("copy", &NdArrayInterface::copy<double>)
        .def("cumprod", &NdArrayInterface::cumprod<double>)
        .def("cumsum", &NdArrayInterface::cumsum<double>)
        .def("diagonal", &NdArrayInterface::diagonal<double>)
        .def("dimSize", &NdArrayDouble::dimSize)
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
        .def("isscalar", &NdArrayDouble::isscalar)
        .def("issquare", &NdArrayDouble::issquare)
        .def("item", &NdArrayDouble::item)
        .def("max", &NdArrayInterface::max<double>)
        .def("min", &NdArrayInterface::min<double>)
        .def("median", &NdArrayInterface::median<double>)
        .def("nans", &NdArrayDouble::nans, python_interface::return_value_policy::reference)
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
        .def("put", &NdArrayInterface::putFlat<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putRowCol<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putIndices1DValue<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putIndices1DValues<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putSlice1DValue<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putSlice1DValues<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putIndices2DValue<double>, python_interface::return_value_policy::reference)
        .def("put",
             &NdArrayInterface::putRowIndicesColSliceValue<double>,
             python_interface::return_value_policy::reference)
        .def("put",
             &NdArrayInterface::putRowSliceColIndicesValue<double>,
             python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putSlice2DValue<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putIndices2DValueRow<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putSlice2DValueRow<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putIndices2DValueCol<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putSlice2DValueCol<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putIndices2DValues<double>, python_interface::return_value_policy::reference)
        .def("put",
             &NdArrayInterface::putRowIndicesColSliceValues<double>,
             python_interface::return_value_policy::reference)
        .def("put",
             &NdArrayInterface::putRowSliceColIndicesValues<double>,
             python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putSlice2DValues<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putIndices2DValuesRow<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putSlice2DValuesRow<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putIndices2DValuesCol<double>, python_interface::return_value_policy::reference)
        .def("put", &NdArrayInterface::putSlice2DValuesCol<double>, python_interface::return_value_policy::reference)
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
        .def("rows",
             [](const NdArray<double>& self, pbArray<uint32> rowIndices)
             { return nc2pybind(self.rows(pybind2nc(rowIndices))); })
        .def("shape", &NdArrayDouble::shape)
        .def("size", &NdArrayDouble::size)
        .def("sort", &NdArrayInterface::sort<double>)
        .def("sum", &NdArrayInterface::sum<double>)
        .def("swapaxes", &NdArrayInterface::swapaxes<double>)
        .def("swapRows", &NdArrayInterface::swapRows<double>)
        .def("swapCols", &NdArrayInterface::swapCols<double>)
        .def("tofile", &NdArrayInterface::tofileBinary<double>)
        .def("tofile", &NdArrayInterface::tofileTxt<double>)
        .def("toIndices", &NdArrayDouble::toIndices)
        .def("toStlVector", &NdArrayDouble::toStlVector)
        .def("trace", &NdArrayDouble::trace)
        .def("transpose", &NdArrayInterface::transpose<double>)
        .def("zeros", &NdArrayDouble::zeros, python_interface::return_value_policy::reference);

    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualArray<double>);                   // (1)
    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualArray<ComplexDouble>);            // (1)
    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualComplexArrayArithArray<double>);  // (2)
    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualScalar<double>);                  // (3)
    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualScalar<ComplexDouble>);           // (3)
    m.def("operatorPlusEqual", &NdArrayInterface::operatorPlusEqualComplexArrayArithScalar<double>); // (4)

    m.def("operatorPlus", &NdArrayInterface::operatorPlusArray<double>);                   // (1)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArray<ComplexDouble>);            // (1)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArithArrayComplexArray<double>);  // (2)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusComplexArrayArithArray<double>);  // (3)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArrayScalar<double>);             // (4)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArrayScalar<ComplexDouble>);      // (4)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusScalarArray<double>);             // (5)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusScalarArray<ComplexDouble>);      // (5)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArithArrayComplexScalar<double>); // (6)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusComplexScalarArithArray<double>); // (7)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusComplexArrayArithScalar<double>); // (8)
    m.def("operatorPlus", &NdArrayInterface::operatorPlusArithScalarComplexArray<double>); // (9)

    m.def("operatorNegative", &NdArrayInterface::operatorNegative<double>);
    m.def("operatorNegative", &NdArrayInterface::operatorNegative<ComplexDouble>);

    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualArray<double>);                   // (1)
    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualArray<ComplexDouble>);            // (1)
    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualComplexArrayArithArray<double>);  // (2)
    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualScalar<double>);                  // (3)
    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualScalar<ComplexDouble>);           // (3)
    m.def("operatorMinusEqual", &NdArrayInterface::operatorMinusEqualComplexArrayArithScalar<double>); // (4)

    m.def("operatorMinus", &NdArrayInterface::operatorMinusArray<double>);                   // (1)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArray<ComplexDouble>);            // (1)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArithArrayComplexArray<double>);  // (2)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusComplexArrayArithArray<double>);  // (3)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArrayScalar<double>);             // (4)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArrayScalar<ComplexDouble>);      // (4)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusScalarArray<double>);             // (5)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusScalarArray<ComplexDouble>);      // (5)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArithArrayComplexScalar<double>); // (6)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusComplexScalarArithArray<double>); // (7)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusComplexArrayArithScalar<double>); // (8)
    m.def("operatorMinus", &NdArrayInterface::operatorMinusArithScalarComplexArray<double>); // (9)

    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualArray<double>);                   // (1)
    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualArray<ComplexDouble>);            // (1)
    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualComplexArrayArithArray<double>);  // (2)
    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualScalar<double>);                  // (3)
    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualScalar<ComplexDouble>);           // (3)
    m.def("operatorMultiplyEqual", &NdArrayInterface::operatorMultiplyEqualComplexArrayArithScalar<double>); // (4)

    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArray<double>);                   // (1)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArray<ComplexDouble>);            // (1)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithArrayComplexArray<double>);  // (2)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexArrayArithArray<double>);  // (3)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArrayScalar<double>);             // (4)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArrayScalar<ComplexDouble>);      // (4)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyScalarArray<double>);             // (5)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyScalarArray<ComplexDouble>);      // (5)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithArrayComplexScalar<double>); // (6)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexScalarArithArray<double>); // (7)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyComplexArrayArithScalar<double>); // (8)
    m.def("operatorMultiply", &NdArrayInterface::operatorMultiplyArithScalarComplexArray<double>); // (9)

    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualArray<double>);                   // (1)
    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualArray<ComplexDouble>);            // (1)
    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualComplexArrayArithArray<double>);  // (2)
    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualScalar<double>);                  // (3)
    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualScalar<ComplexDouble>);           // (3)
    m.def("operatorDivideEqual", &NdArrayInterface::operatorDivideEqualComplexArrayArithScalar<double>); // (4)

    m.def("operatorDivide", &NdArrayInterface::operatorDivideArray<double>);                   // (1)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArray<ComplexDouble>);            // (1)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArithArrayComplexArray<double>);  // (2)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideComplexArrayArithArray<double>);  // (3)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArrayScalar<double>);             // (4)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArrayScalar<ComplexDouble>);      // (4)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideScalarArray<double>);             // (5)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideScalarArray<ComplexDouble>);      // (5)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArithArrayComplexScalar<double>); // (6)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideComplexScalarArithArray<double>); // (7)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideComplexArrayArithScalar<double>); // (8)
    m.def("operatorDivide", &NdArrayInterface::operatorDivideArithScalarComplexArray<double>); // (9)

    m.def("operatorEquality", &NdArrayInterface::operatorEqualityScalar<double>);
    m.def("operatorEquality", &NdArrayInterface::operatorEqualityScalar<ComplexDouble>);
    m.def("operatorEquality", &NdArrayInterface::operatorEqualityScalarReversed<double>);
    m.def("operatorEquality", &NdArrayInterface::operatorEqualityScalarReversed<ComplexDouble>);
    m.def("operatorEquality", &NdArrayInterface::operatorEqualityArray<double>);
    m.def("operatorEquality", &NdArrayInterface::operatorEqualityArray<ComplexDouble>);

    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScalar<double>);
    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScalar<ComplexDouble>);
    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScalarReversed<double>);
    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScalarReversed<ComplexDouble>);
    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityArray<double>);
    m.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityArray<ComplexDouble>);

    m.def("operatorLess", &NdArrayInterface::operatorLessScalar<double>);
    m.def("operatorLess", &NdArrayInterface::operatorLessScalar<ComplexDouble>);
    m.def("operatorLess", &NdArrayInterface::operatorLessScalarReversed<double>);
    m.def("operatorLess", &NdArrayInterface::operatorLessScalarReversed<ComplexDouble>);
    m.def("operatorLess", &NdArrayInterface::operatorLessArray<double>);
    m.def("operatorLess", &NdArrayInterface::operatorLessArray<ComplexDouble>);

    m.def("operatorGreater", &NdArrayInterface::operatorGreaterScalar<double>);
    m.def("operatorGreater", &NdArrayInterface::operatorGreaterScalar<ComplexDouble>);
    m.def("operatorGreater", &NdArrayInterface::operatorGreaterScalarReversed<double>);
    m.def("operatorGreater", &NdArrayInterface::operatorGreaterScalarReversed<ComplexDouble>);
    m.def("operatorGreater", &NdArrayInterface::operatorGreaterArray<double>);
    m.def("operatorGreater", &NdArrayInterface::operatorGreaterArray<ComplexDouble>);

    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScalar<double>);
    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScalar<ComplexDouble>);
    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScalarReversed<double>);
    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScalarReversed<ComplexDouble>);
    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualArray<double>);
    m.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualArray<ComplexDouble>);

    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScalar<double>);
    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScalar<ComplexDouble>);
    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScalarReversed<double>);
    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScalarReversed<ComplexDouble>);
    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualArray<double>);
    m.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualArray<ComplexDouble>);

    m.def("operatorPrePlusPlus", &NdArrayInterface::operatorPrePlusPlus<double>);
    m.def("operatorPostPlusPlus", &NdArrayInterface::operatorPostPlusPlus<double>);

    m.def("operatorPreMinusMinus", &NdArrayInterface::operatorPreMinusMinus<double>);
    m.def("operatorPostMinusMinus", &NdArrayInterface::operatorPostMinusMinus<double>);

    m.def("operatorModulusEqualArray", &NdArrayInterface::operatorModulusEqualArray<uint32>);
    m.def("operatorModulusScalar", &NdArrayInterface::operatorModulusScalar<uint32>);
    m.def("operatorModulusScalar", &NdArrayInterface::operatorModulusScalarReversed<uint32>);
    m.def("operatorModulusArray", &NdArrayInterface::operatorModulusArray<uint32>);

    m.def("operatorModulusEqualArray", &NdArrayInterface::operatorModulusEqualArray<double>);
    m.def("operatorModulusScalar", &NdArrayInterface::operatorModulusScalar<double>);
    m.def("operatorModulusScalar", &NdArrayInterface::operatorModulusScalarReversed<double>);
    m.def("operatorModulusArray", &NdArrayInterface::operatorModulusArray<double>);

    m.def("operatorBitwiseOrEqualArray", &NdArrayInterface::operatorBitwiseOrEqualArray<uint32>);
    m.def("operatorBitwiseOrScalar", &NdArrayInterface::operatorBitwiseOrScalar<uint32>);
    m.def("operatorBitwiseOrScalar", &NdArrayInterface::operatorBitwiseOrScalarReversed<uint32>);
    m.def("operatorBitwiseOrArray", &NdArrayInterface::operatorBitwiseOrArray<uint32>);

    m.def("operatorBitwiseAndEqualArray", &NdArrayInterface::operatorBitwiseAndEqualArray<uint32>);
    m.def("operatorBitwiseAndScalar", &NdArrayInterface::operatorBitwiseAndScalar<uint32>);
    m.def("operatorBitwiseAndScalar", &NdArrayInterface::operatorBitwiseAndScalarReversed<uint32>);
    m.def("operatorBitwiseAndArray", &NdArrayInterface::operatorBitwiseAndArray<uint32>);

    m.def("operatorBitwiseXorEqualArray", &NdArrayInterface::operatorBitwiseXorEqualArray<uint32>);
    m.def("operatorBitwiseXorScalar", &NdArrayInterface::operatorBitwiseXorScalar<uint32>);
    m.def("operatorBitwiseXorScalar", &NdArrayInterface::operatorBitwiseXorScalarReversed<uint32>);
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
    python_interface::class_<NdArrayUInt32>(m, "NdArrayUInt32")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayUInt32::item)
        .def("shape", &NdArrayUInt32::shape)
        .def("size", &NdArrayUInt32::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint32>)
        .def("endianess", &NdArrayUInt32::endianess)
        .def("setArray", &NdArrayInterface::setArray<uint32>)
        .def("byteswap", &NdArrayUInt32::byteswap, python_interface::return_value_policy::reference)
        .def("newbyteorder", &NdArrayInterface::newbyteorder<uint32>);

    using NdArrayUInt64 = NdArray<uint64>;
    python_interface::class_<NdArrayUInt64>(m, "NdArrayUInt64")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayUInt64::item)
        .def("shape", &NdArrayUInt64::shape)
        .def("size", &NdArrayUInt64::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint64>)
        .def("endianess", &NdArrayUInt64::endianess)
        .def("setArray", &NdArrayInterface::setArray<uint64>);

    using NdArrayUInt16 = NdArray<uint16>;
    python_interface::class_<NdArrayUInt16>(m, "NdArrayUInt16")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayUInt16::item)
        .def("shape", &NdArrayUInt16::shape)
        .def("size", &NdArrayUInt16::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint16>)
        .def("endianess", &NdArrayUInt16::endianess)
        .def("setArray", NdArrayInterface::setArray<uint16>);

    using NdArrayUInt8 = NdArray<uint8>;
    python_interface::class_<NdArrayUInt8>(m, "NdArrayUInt8")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayUInt8::item)
        .def("shape", &NdArrayUInt8::shape)
        .def("size", &NdArrayUInt8::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint8>)
        .def("endianess", &NdArrayUInt8::endianess)
        .def("setArray", NdArrayInterface::setArray<uint8>);

    using NdArrayInt64 = NdArray<int64>;
    python_interface::class_<NdArrayInt64>(m, "NdArrayInt64")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayInt64::item)
        .def("shape", &NdArrayInt64::shape)
        .def("size", &NdArrayInt64::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int64>)
        .def("endianess", &NdArrayInt64::endianess)
        .def("replace", &NdArrayInterface::replace<int64>)
        .def("setArray", &NdArrayInterface::setArray<int64>);

    using NdArrayInt32 = NdArray<int32>;
    python_interface::class_<NdArrayInt32>(m, "NdArrayInt32")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayInt32::item)
        .def("shape", &NdArrayInt32::shape)
        .def("size", &NdArrayInt32::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int32>)
        .def("endianess", &NdArrayInt32::endianess)
        .def("replace", &NdArrayInterface::replace<int32>)
        .def("setArray", &NdArrayInterface::setArray<int32>);

    using NdArrayInt16 = NdArray<int16>;
    python_interface::class_<NdArrayInt16>(m, "NdArrayInt16")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayInt16::item)
        .def("shape", &NdArrayInt16::shape)
        .def("size", &NdArrayInt16::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int16>)
        .def("endianess", &NdArrayInt16::endianess)
        .def("replace", &NdArrayInterface::replace<int16>)
        .def("setArray", &NdArrayInterface::setArray<int16>);

    using NdArrayInt8 = NdArray<int8>;
    python_interface::class_<NdArrayInt8>(m, "NdArrayInt8")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayInt8::item)
        .def("shape", &NdArrayInt8::shape)
        .def("size", &NdArrayInt8::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<int8>)
        .def("endianess", &NdArrayInt8::endianess)
        .def("replace", &NdArrayInterface::replace<int8>)
        .def("setArray", &NdArrayInterface::setArray<int8>);

    using NdArrayFloat = NdArray<float>;
    python_interface::class_<NdArrayFloat>(m, "NdArrayFloat")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayFloat::item)
        .def("shape", &NdArrayFloat::shape)
        .def("size", &NdArrayFloat::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<float>)
        .def("endianess", &NdArrayFloat::endianess)
        .def("setArray", &NdArrayInterface::setArray<float>);

    using NdArrayBool = NdArray<bool>;
    python_interface::class_<NdArrayBool>(m, "NdArrayBool")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayBool::item)
        .def("shape", &NdArrayBool::shape)
        .def("size", &NdArrayBool::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<bool>)
        .def("endianess", &NdArrayBool::endianess)
        .def("setArray", NdArrayInterface::setArray<bool>);

    using NdArrayComplexLongDouble = NdArray<std::complex<long double>>;
    python_interface::class_<NdArrayComplexLongDouble>(m, "NdArrayComplexLongDouble")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayComplexLongDouble::item)
        .def("shape", &NdArrayComplexLongDouble::shape)
        .def("size", &NdArrayComplexLongDouble::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<std::complex<long double>>)
        .def("setArray", NdArrayInterface::setArray<std::complex<long double>>);

    using NdArrayComplexFloat = NdArray<std::complex<float>>;
    python_interface::class_<NdArrayComplexFloat>(m, "NdArrayComplexFloat")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def("item", &NdArrayComplexFloat::item)
        .def("shape", &NdArrayComplexFloat::shape)
        .def("size", &NdArrayComplexFloat::size)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<std::complex<float>>)
        .def("setArray", NdArrayInterface::setArray<std::complex<float>>);

    python_interface::class_<NdArrayComplexDouble>(m, "NdArrayComplexDouble")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def(python_interface::init<NdArrayComplexDouble>())
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
        .def("get", &NdArrayInterface::getIndicesScalar<ComplexDouble>)
        .def("get", &NdArrayInterface::getIndicesSlice<ComplexDouble>)
        .def("get", &NdArrayInterface::getScalarIndices<ComplexDouble>)
        .def("get", &NdArrayInterface::getSliceIndices<ComplexDouble>)
        .def("get", &NdArrayInterface::getIndices2D<ComplexDouble>)
        .def("at", &NdArrayInterface::atValueFlat<ComplexDouble>)
        .def("atConst", &NdArrayInterface::atValueFlatConst<ComplexDouble>)
        .def("at", &NdArrayInterface::atValueRowCol<ComplexDouble>)
        .def("atConst", &NdArrayInterface::atValueRowColConst<ComplexDouble>)
        .def("at", &NdArrayInterface::atMask<ComplexDouble>)
        .def("at", &NdArrayInterface::atIndices<ComplexDouble>)
        .def("at", &NdArrayInterface::atSlice1D<ComplexDouble>)
        .def("at", &NdArrayInterface::atSlice2D<ComplexDouble>)
        .def("at", &NdArrayInterface::atSlice2DRow<ComplexDouble>)
        .def("at", &NdArrayInterface::atSlice2DCol<ComplexDouble>)
        .def("at", &NdArrayInterface::atIndicesScalar<ComplexDouble>)
        .def("at", &NdArrayInterface::atIndicesSlice<ComplexDouble>)
        .def("at", &NdArrayInterface::atScalarIndices<ComplexDouble>)
        .def("at", &NdArrayInterface::atSliceIndices<ComplexDouble>)
        .def("at", &NdArrayInterface::atIndices2D<ComplexDouble>)
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
        .def("argpartition", &NdArrayInterface::argpartition<ComplexDouble>)
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
        .def("put", &NdArrayInterface::putIndices1DValue<ComplexDouble>)
        .def("put", &NdArrayInterface::putIndices1DValues<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice1DValue<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice1DValues<ComplexDouble>)
        .def("put", &NdArrayInterface::putIndices2DValue<ComplexDouble>)
        .def("put", &NdArrayInterface::putRowIndicesColSliceValue<ComplexDouble>)
        .def("put", &NdArrayInterface::putRowSliceColIndicesValue<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice2DValue<ComplexDouble>)
        .def("put", &NdArrayInterface::putIndices2DValueRow<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice2DValueRow<ComplexDouble>)
        .def("put", &NdArrayInterface::putIndices2DValueCol<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice2DValueCol<ComplexDouble>)
        .def("put", &NdArrayInterface::putIndices2DValues<ComplexDouble>)
        .def("put", &NdArrayInterface::putRowIndicesColSliceValues<ComplexDouble>)
        .def("put", &NdArrayInterface::putRowSliceColIndicesValues<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice2DValues<ComplexDouble>)
        .def("put", &NdArrayInterface::putIndices2DValuesRow<ComplexDouble>)
        .def("put", &NdArrayInterface::putSlice2DValuesRow<ComplexDouble>)
        .def("put", &NdArrayInterface::putIndices2DValuesCol<ComplexDouble>)
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
        .def("zeros", &NdArrayComplexDouble::zeros, python_interface::return_value_policy::reference);

    m.def("testStructuredArray", &NdArrayInterface::testStructuredArray);
}