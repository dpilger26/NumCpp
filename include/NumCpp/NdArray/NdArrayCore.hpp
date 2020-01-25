/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.2
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
/// Holds 1D and 2D arrays, the main work horse of the NumCpp library
///
#pragma once

#include "NumCpp/Core/Constants.hpp"
#include "NumCpp/Core/DtypeInfo.hpp"
#include "NumCpp/Core/Error.hpp"
#include "NumCpp/Core/Filesystem.hpp"
#include "NumCpp/Core/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Utils/num2str.hpp"
#include "NumCpp/Utils/power.hpp"
#include "NumCpp/Utils/sqr.hpp"

#include <boost/algorithm/clamp.hpp>
#include <boost/predef/other/endian.h>
#include <boost/endian/conversion.hpp>

#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nc
{
    //================================================================================
    // Class Description:
    ///						Holds 1D and 2D arrays, the main work horse of the NumCpp library
    template<typename dtype = double>
    class NdArray
    {
    public:
        //====================================Typedefs================================
        typedef dtype*			iterator;
        typedef const dtype*	const_iterator;

    private:
        //====================================Attributes==============================
        Shape			shape_{ 0, 0 };
        uint32			size_{ 0 };
        Endian          endianess_{ Endian::NATIVE };
        dtype*			array_{ nullptr };
        bool            ownsPtr_{ false };

        //============================================================================
        // Method Description:
        ///						Deletes the internal array
        ///
        void deleteArray() noexcept
        {
            if (ownsPtr_ && array_ != nullptr)
            {
                delete[] array_;
            }

            array_ = nullptr;
            shape_ = Shape(0, 0);
            size_ = 0;
            ownsPtr_ = false;
        }

        //============================================================================
        // Method Description:
        ///						Creates a new internal array
        ///
        /// @param
        ///				inShape
        ///
        void newArray(const Shape& inShape) noexcept
        {
            deleteArray();

            shape_ = inShape;
            size_ = inShape.size();
            endianess_ = Endian::NATIVE;
            array_ = new dtype[size_];
            ownsPtr_ = true;
        }

    public:
        //============================================================================
        // Method Description:
        ///						Defualt Constructor, not very usefull...
        ///
        NdArray() noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param
        ///				inSquareSize: square number of rows and columns
        ///
        explicit NdArray(uint32 inSquareSize) noexcept :
            shape_(inSquareSize, inSquareSize),
            size_(inSquareSize * inSquareSize),
            array_(new dtype[size_]),
            ownsPtr_(true)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param				inNumRows
        /// @param				inNumCols
        ///
        NdArray(uint32 inNumRows, uint32 inNumCols) noexcept :
            shape_(inNumRows, inNumCols),
            size_(inNumRows * inNumCols),
            array_(new dtype[size_]),
            ownsPtr_(true)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param
        ///				inShape
        ///
        explicit NdArray(const Shape& inShape) noexcept :
            shape_(inShape),
            size_(shape_.size()),
            array_(new dtype[size_]),
            ownsPtr_(true)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param
        ///				inList
        ///
        NdArray(const std::initializer_list<dtype>& inList) noexcept :
            shape_(1, static_cast<uint32>(inList.size())),
            size_(shape_.size()),
            array_(new dtype[size_]),
            ownsPtr_(true)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
            stl_algorithms::copy(inList.begin(), inList.end(), array_);
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param
        ///				inList: 2D initializer list
        ///
        NdArray(const std::initializer_list<std::initializer_list<dtype> >& inList) :
            shape_(static_cast<uint32>(inList.size()), 0)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            for (auto& list : inList)
            {
                size_ += static_cast<uint32>(list.size());

                if (shape_.cols == 0)
                {
                    shape_.cols = static_cast<uint32>(list.size());
                }
                else if (list.size() != shape_.cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("All rows of the initializer list needs to have the same number of elements");
                }
            }

            array_ = new dtype[size_];
            uint32 row = 0;
            for (auto& list : inList)
            {
                auto ptr = array_ + row * shape_.cols;
                stl_algorithms::copy(list.begin(), list.end(), ptr);
                ++row;
            }

            ownsPtr_ = true;
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param
        ///				inArray
        ///
        template<size_t ArraySize>
        explicit NdArray(const std::array<dtype, ArraySize>& inArray) noexcept :
            shape_(1, static_cast<uint32>(inArray.size())),
            size_(shape_.size()),
            array_(new dtype[size_]),
            ownsPtr_(true)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
            stl_algorithms::copy(inArray.begin(), inArray.end(), array_);
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param
        ///				inVector
        ///
        explicit NdArray(const std::vector<dtype>& inVector) noexcept :
            shape_(1, static_cast<uint32>(inVector.size())),
            size_(shape_.size()),
            array_(new dtype[size_]),
            ownsPtr_(true)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
            stl_algorithms::copy(inVector.begin(), inVector.end(), array_);
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param
        ///				inDeque
        ///
        explicit NdArray(const std::deque<dtype>& inDeque) noexcept :
            shape_(1, static_cast<uint32>(inDeque.size())),
            size_(shape_.size()),
            array_(new dtype[size_]),
            ownsPtr_(true)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
            stl_algorithms::copy(inDeque.begin(), inDeque.end(), array_);
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param
        ///				inSet
        ///
        explicit NdArray(const std::set<dtype>& inSet) noexcept :
            shape_(1, static_cast<uint32>(inSet.size())),
            size_(shape_.size()),
            array_(new dtype[size_]),
            ownsPtr_(true)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
            stl_algorithms::copy(inSet.begin(), inSet.end(), array_);
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param				inFirst
        /// @param				inLast
        ///
        explicit NdArray(const_iterator inFirst, const_iterator inLast) noexcept :
            shape_(1, static_cast<uint32>(inLast - inFirst)),
            size_(shape_.size()),
            array_(new dtype[size_]),
            ownsPtr_(true)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
            stl_algorithms::copy(inFirst, inLast, array_);
        }

        //============================================================================
        // Method Description:
        ///						Constructor. Operates as a shell around an already existing
        ///                     array of data.
        ///
        /// @param				inPtr: dtype* to beginning of the array
        /// @param				numRows: the number of rows in the array
        /// @param              numCols: the nubmer of column in the array
        /// @param              takeOwnership: whether or not to take ownership of the data
        ///                     and call delete[] in the destructor.
        ///
        NdArray(dtype* inPtr, uint32 numRows, uint32 numCols, bool takeOwnership = false) noexcept :
            shape_(numRows, numCols),
            size_(numRows * numCols),
            array_(inPtr),
            ownsPtr_(takeOwnership)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
        }

        //============================================================================
        // Method Description:
        ///						Constructor.  Copies the contents of the buffer into 
        ///                     the array.
        ///
        /// @param				inPtr: dtype* to beginning of buffer
        /// @param				inSize: number of elements in buffer
        ///
        NdArray(const dtype* const inPtr, uint32 inSize) noexcept :
            shape_(1, inSize),
            size_(inSize),
            array_(new dtype[size_]),
            ownsPtr_(true)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);
            stl_algorithms::copy(inPtr, inPtr + size_, begin());
        }

        //============================================================================
        // Method Description:
        ///						Copy Constructor
        ///
        /// @param
        ///				inOtherArray
        ///
        NdArray(const NdArray<dtype>& inOtherArray) noexcept :
            shape_(inOtherArray.shape_),
            size_(inOtherArray.size_),
            endianess_(inOtherArray.endianess_),
            array_(new dtype[inOtherArray.size_]),
            ownsPtr_(true)
        {
            stl_algorithms::copy(inOtherArray.cbegin(), inOtherArray.cend(), begin());
        }

        //============================================================================
        // Method Description:
        ///						Move Constructor
        ///
        /// @param
        ///				inOtherArray
        ///
        NdArray(NdArray<dtype>&& inOtherArray) noexcept :
            shape_(inOtherArray.shape_),
            size_(inOtherArray.size_),
            endianess_(inOtherArray.endianess_),
            array_(inOtherArray.array_),
            ownsPtr_(true)
        {
            inOtherArray.shape_.rows = inOtherArray.shape_.cols = inOtherArray.size_ = 0;
            inOtherArray.array_ = nullptr;
        }

        //============================================================================
        // Method Description:
        ///						Destructor
        ///
        ~NdArray()
        {
            deleteArray();
        }

        //============================================================================
        // Method Description:
        ///						Assignment operator, performs a deep copy
        ///
        /// @param
        ///				rhs
        /// @return
        ///				NdArray<dtype>
        ///
        NdArray<dtype>& operator=(const NdArray<dtype>& rhs) noexcept
        {
            if (&rhs != this)
            {
                newArray(rhs.shape_);
                endianess_ = rhs.endianess_;

                stl_algorithms::copy(rhs.cbegin(), rhs.cend(), begin());
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Assignment operator, sets the entire array to a single
        ///                     scalar value.
        ///
        /// @param
        ///				inValue
        /// @return
        ///				NdArray<dtype>
        ///
        NdArray<dtype>& operator=(dtype inValue) noexcept
        {
            stl_algorithms::fill(begin(), end(), inValue);

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Move operator, performs a deep move
        ///
        /// @param
        ///				rhs
        /// @return
        ///				NdArray<dtype>
        ///
        NdArray<dtype>& operator=(NdArray<dtype>&& rhs) noexcept
        {
            if (&rhs != this)
            {
                deleteArray();
                shape_ = rhs.shape_;
                size_ = rhs.size_;
                endianess_ = rhs.endianess_;
                array_ = rhs.array_;
                ownsPtr_ = rhs.ownsPtr_;

                rhs.shape_.rows = rhs.shape_.cols = rhs.size_ = 0;
                rhs.array_ = nullptr;
                rhs.ownsPtr_ = false;
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						1D access operator with no bounds checking
        ///
        /// @param
        ///				inIndex
        /// @return
        ///				value
        ///
        dtype& operator[](int32 inIndex) noexcept
        {
            if (inIndex < 0)
            {
                inIndex += size_;
            }

            return array_[inIndex];
        }

        //============================================================================
        // Method Description:
        ///						const 1D access operator with no bounds checking
        ///
        /// @param
        ///				inIndex
        /// @return
        ///				value
        ///
        const dtype& operator[](int32 inIndex) const noexcept
        {
            if (inIndex < 0)
            {
                inIndex += size_;
            }

            return array_[inIndex];
        }

        //============================================================================
        // Method Description:
        ///						2D access operator with no bounds checking
        ///
        /// @param				inRowIndex
        /// @param				inColIndex
        /// @return
        ///				value
        ///
        dtype& operator()(int32 inRowIndex, int32 inColIndex) noexcept
        {
            if (inRowIndex < 0)
            {
                inRowIndex += shape_.rows;
            }

            if (inColIndex < 0)
            {
                inColIndex += shape_.cols;
            }

            return array_[inRowIndex * shape_.cols + inColIndex];
        }

        //============================================================================
        // Method Description:
        ///						const 2D access operator with no bounds checking
        ///
        /// @param				inRowIndex
        /// @param				inColIndex
        /// @return
        ///				value
        ///
        const dtype& operator()(int32 inRowIndex, int32 inColIndex) const noexcept
        {
            if (inRowIndex < 0)
            {
                inRowIndex += shape_.rows;
            }

            if (inColIndex < 0)
            {
                inColIndex += shape_.cols;
            }

            return array_[inRowIndex * shape_.cols + inColIndex];
        }

        //============================================================================
        // Method Description:
        ///						1D Slicing access operator with bounds checking.
        ///						returned array is of the range [start, stop).
        ///
        /// @param
        ///				inSlice
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator[](const Slice& inSlice) const
        {
            Slice inSliceCopy(inSlice);

            uint32 counter = 0;
            NdArray<dtype> returnArray(1, inSliceCopy.numElements(size_));
            for (int32 i = inSliceCopy.start; i < inSliceCopy.stop; i += inSliceCopy.step)
            {
                returnArray[counter++] = at(i);
            }

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns the values from the input mask
        ///
        /// @param
        ///				inMask
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator[](const NdArray<bool>& inMask) const
        {
            if (inMask.shape() != shape_)
            {
                THROW_INVALID_ARGUMENT_ERROR("input inMask must have the same shape as the NdArray it will be masking.");
            }

            auto indices = inMask.flatnonzero();
            auto outArray = NdArray<dtype>(1, indices.size());
            for (uint32 i = 0; i < indices.size(); ++i)
            {
                outArray[i] = operator[](indices[i]);
            }

            return outArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns the values from the input indices
        ///
        /// @param
        ///				inIndices
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator[](const NdArray<uint32>& inIndices) const
        {
            if (inIndices.max().item() > size_ - 1)
            {
                THROW_INVALID_ARGUMENT_ERROR("input indices must be less than the array size.");
            }

            auto outArray = NdArray<dtype>(1, static_cast<uint32>(inIndices.size()));
            uint32 i = 0;
            for (auto& index : inIndices)
            {
                outArray[i++] = operator[](index);
            }

            return outArray;
        }

        //============================================================================
        // Method Description:
        ///						2D Slicing access operator with bounds checking.
        ///						returned array is of the range [start, stop).
        ///
        /// @param				inRowSlice
        /// @param				inColSlice
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator()(const Slice& inRowSlice, const Slice& inColSlice) const
        {
            Slice inRowSliceCopy(inRowSlice);
            Slice inColSliceCopy(inColSlice);

            NdArray<dtype> returnArray(inRowSliceCopy.numElements(shape_.rows), inColSliceCopy.numElements(shape_.cols));

            uint32 rowCounter = 0;
            uint32 colCounter = 0;
            for (int32 row = inRowSliceCopy.start; row < inRowSliceCopy.stop; row += inRowSliceCopy.step)
            {
                for (int32 col = inColSliceCopy.start; col < inColSliceCopy.stop; col += inColSliceCopy.step)
                {
                    returnArray(rowCounter, colCounter++) = at(row, col);
                }
                colCounter = 0;
                ++rowCounter;
            }

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						2D Slicing access operator with bounds checking.
        ///						returned array is of the range [start, stop).
        ///
        /// @param				inRowSlice
        /// @param				inColIndex
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator()(const Slice& inRowSlice, int32 inColIndex) const
        {
            Slice inRowSliceCopy(inRowSlice);

            NdArray<dtype> returnArray(inRowSliceCopy.numElements(shape_.rows), 1);

            uint32 rowCounter = 0;
            for (int32 row = inRowSliceCopy.start; row < inRowSliceCopy.stop; row += inRowSliceCopy.step)
            {
                returnArray(rowCounter++, 0) = at(row, inColIndex);
            }

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						2D Slicing access operator with bounds checking.
        ///						returned array is of the range [start, stop).
        ///
        /// @param				inRowIndex
        /// @param				inColSlice
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator()(int32 inRowIndex, const Slice& inColSlice) const
        {
            Slice inColSliceCopy(inColSlice);

            NdArray<dtype> returnArray(1, inColSliceCopy.numElements(shape_.cols));

            uint32 colCounter = 0;
            for (int32 col = inColSliceCopy.start; col < inColSliceCopy.stop; col += inColSliceCopy.step)
            {
                returnArray(0, colCounter++) = at(inRowIndex, col);
            }

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns a Slice object for slicing a row to the end of
        ///                     array.
        ///
        /// @param      inStartIdx (default 0)
        /// @param      inStepSize (default 1)
        /// @return
        ///				Slice
        ///
        const Slice cSlice(int32 inStartIdx = 0, uint32 inStepSize = 1) const noexcept
        {
            return Slice(inStartIdx, shape_.cols, inStepSize);
        }

        //============================================================================
        // Method Description:
        ///						Returns a Slice object for slicing a column to the end
        ///                     of the array.
        ///
        /// @param      inStartIdx (default 0)
        /// @param      inStepSize (default 1)
        /// @return
        ///				Slice
        ///
        const Slice rSlice(int32 inStartIdx = 0, uint32 inStepSize = 1) const noexcept
        {
            return Slice(inStartIdx, shape_.rows, inStepSize);
        }

        //============================================================================
        // Method Description:
        ///						1D access method with bounds checking
        ///
        /// @param
        ///				inIndex
        /// @return
        ///				value
        ///
        dtype& at(int32 inIndex)
        {
            // this doesn't allow for calling the first element as -size_...
            // but why would you really want to do that anyway?
            if (std::abs(inIndex) > static_cast<int64>(size_ - 1))
            {
                std::string errStr = "Input index " + utils::num2str(inIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(size_) + ".";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            return operator[](inIndex);
        }

        //============================================================================
        // Method Description:
        ///						const 1D access method with bounds checking
        ///
        /// @param
        ///				inIndex
        /// @return
        ///				value
        ///
        const dtype& at(int32 inIndex) const
        {
            // this doesn't allow for calling the first element as -size_...
            // but why would you really want to do that anyway?
            if (std::abs(inIndex) > static_cast<int64>(size_ - 1))
            {
                std::string errStr = "Input index " + utils::num2str(inIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(size_) + ".";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            return operator[](inIndex);
        }

        //============================================================================
        // Method Description:
        ///						2D access method with bounds checking
        ///
        /// @param				inRowIndex
        /// @param				inColIndex
        /// @return
        ///				value
        ///
        dtype& at(int32 inRowIndex, int32 inColIndex)
        {
            // this doesn't allow for calling the first element as -size_...
            // but why would you really want to do that anyway?
            if (std::abs(inRowIndex) > static_cast<int32>(shape_.rows - 1))
            {
                std::string errStr = "Row index " + utils::num2str(inRowIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(shape_.rows) + ".";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            // this doesn't allow for calling the first element as -size_...
            // but why would you really want to that anyway?
            if (std::abs(inColIndex) > static_cast<int32>(shape_.cols - 1))
            {
                std::string errStr = "Column index " + utils::num2str(inColIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(shape_.cols) + ".";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            return operator()(inRowIndex, inColIndex);
        }

        //============================================================================
        // Method Description:
        ///						const 2D access method with bounds checking
        ///
        /// @param				inRowIndex
        /// @param				inColIndex
        /// @return
        ///				value
        ///
        const dtype& at(int32 inRowIndex, int32 inColIndex) const
        {
            // this doesn't allow for calling the first element as -size_...
            // but why would you really want to do that anyway?
            if (std::abs(inRowIndex) > static_cast<int32>(shape_.rows - 1))
            {
                std::string errStr = "Row index " + utils::num2str(inRowIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(shape_.rows) + ".";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            // this doesn't allow for calling the first element as -size_...
            // but why would you really want to do that anyway?
            if (std::abs(inColIndex) > static_cast<int32>(shape_.cols - 1))
            {
                std::string errStr = "Column index " + utils::num2str(inColIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(shape_.cols) + ".";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            return operator()(inRowIndex, inColIndex);
        }

        //============================================================================
        // Method Description:
        ///						const 1D access method with bounds checking
        ///
        /// @param
        ///				inSlice
        /// @return
        ///				Ndarray
        ///
        NdArray<dtype> at(const Slice& inSlice) const
        {
            // the slice operator already provides bounds checking. just including
            // the at method for completeness
            return operator[](inSlice);
        }

        //============================================================================
        // Method Description:
        ///						const 2D access method with bounds checking
        ///
        /// @param				inRowSlice
        /// @param				inColSlice
        /// @return
        ///				Ndarray
        ///
        NdArray<dtype> at(const Slice& inRowSlice, const Slice& inColSlice) const
        {
            // the slice operator already provides bounds checking. just including
            // the at method for completeness
            return operator()(inRowSlice, inColSlice);
        }

        //============================================================================
        // Method Description:
        ///						const 2D access method with bounds checking
        ///
        /// @param				inRowSlice
        /// @param				inColIndex
        /// @return
        ///				Ndarray
        ///
        NdArray<dtype> at(const Slice& inRowSlice, int32 inColIndex) const
        {
            // the slice operator already provides bounds checking. just including
            // the at method for completeness
            return operator()(inRowSlice, inColIndex);
        }

        //============================================================================
        // Method Description:
        ///						const 2D access method with bounds checking
        ///
        /// @param				inRowIndex
        /// @param				inColSlice
        /// @return
        ///				Ndarray
        ///
        NdArray<dtype> at(int32 inRowIndex, const Slice& inColSlice) const
        {
            // the slice operator already provides bounds checking. just including
            // the at method for completeness
            return operator()(inRowIndex, inColSlice);
        }

        //============================================================================
        // Method Description:
        ///						iterator to the beginning of the flattened array	None
        /// @return
        ///				iterator
        ///
        iterator begin() noexcept
        {
            return array_;
        }

        //============================================================================
        // Method Description:
        ///						iterator to the beginning of the input row
        ///
        /// @param
        ///				inRow
        /// @return
        ///				iterator
        ///
        iterator begin(uint32 inRow)
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return array_ + inRow * shape_.cols;
        }

        //============================================================================
        // Method Description:
        ///						const iterator to the beginning of the flattened array	None
        /// @return
        ///				const_iterator
        ///
        const_iterator begin() const noexcept
        {
            return cbegin();
        }

        //============================================================================
        // Method Description:
        ///						const iterator to the beginning of the input row
        ///
        /// @param
        ///				inRow
        /// @return
        ///				const_iterator
        ///
        const_iterator begin(uint32 inRow) const
        {
            return cbegin(inRow);
        }

        //============================================================================
        // Method Description:
        ///						iterator to 1 past the end of the flattened array
        /// @return
        ///				iterator
        ///
        iterator end() noexcept
        {
            return array_ + size_;
        }

        //============================================================================
        // Method Description:
        ///						iterator to the 1 past end of the row
        ///
        /// @param
        ///				inRow
        /// @return
        ///				iterator
        ///
        iterator end(uint32 inRow)
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return array_ + inRow * shape_.cols + shape_.cols;
        }

        //============================================================================
        // Method Description:
        ///						const iterator to 1 past the end of the flattened array
        /// @return
        ///				const_iterator
        ///
        const_iterator end() const noexcept
        {
            return cend();
        }

        //============================================================================
        // Method Description:
        ///						const iterator to the 1 past end of the row
        ///
        /// @param
        ///				inRow
        /// @return
        ///				const_iterator
        ///
        const_iterator end(uint32 inRow) const
        {
            return cend(inRow);
        }

        //============================================================================
        // Method Description:
        ///						const iterator to the beginning of the flattened array
        ///
        /// @return
        ///				const_iterator
        ///
        const_iterator cbegin() const noexcept
        {
            return array_;
        }

        //============================================================================
        // Method Description:
        ///						const iterator to the beginning of the input row
        ///
        /// @param
        ///				inRow
        /// @return
        ///				const_iterator
        ///
        const_iterator cbegin(uint32 inRow) const
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return array_ + inRow * shape_.cols;
        }

        //============================================================================
        // Method Description:
        ///						const iterator to 1 past the end of the flattened array
        ///
        /// @return
        ///				const_iterator
        ///
        const_iterator cend() const noexcept
        {
            return array_ + size_;
        }

        //============================================================================
        // Method Description:
        ///						const iterator to 1 past the end of the input row
        ///
        /// @param
        ///				inRow
        /// @return
        ///				const_iterator
        ///
        const_iterator cend(uint32 inRow) const
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }
            return array_ + inRow * shape_.cols + shape_.cols;
        }

        //============================================================================
        // Method Description:
        ///						Returns True if all elements evaluate to True or non zero
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.all.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<bool> all(Axis inAxis = Axis::NONE) const noexcept
        {
            auto function = [](dtype i) noexcept -> bool
            {
                return i != dtype{ 0 };
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<bool> returnArray = { stl_algorithms::all_of(cbegin(), cend(), function) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<bool> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::all_of(cbegin(row), cend(row), function);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> arrayTransposed = transpose();
                    NdArray<bool> returnArray(1, arrayTransposed.shape_.rows);
                    for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::all_of(arrayTransposed.cbegin(row), arrayTransposed.cend(row), function);
                    }

                    return returnArray;
                }
                default:
                {
                    // this isn't actually possible, just putting this here to get rid
                    // of the compiler warning.
                    return NdArray<bool>(0);
                }
            }
        }

        //============================================================================
        // Method Description:
        ///						Returns True if any elements evaluate to True or non zero
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.any.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<bool> any(Axis inAxis = Axis::NONE) const noexcept
        {
            auto function = [](dtype i) noexcept -> bool
            {
                return i != dtype{ 0 };
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<bool> returnArray = { stl_algorithms::any_of(cbegin(), cend(), function) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<bool> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::any_of(cbegin(row), cend(row), function);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> arrayTransposed = transpose();
                    NdArray<bool> returnArray(1, arrayTransposed.shape_.rows);
                    for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::any_of(arrayTransposed.cbegin(row), arrayTransposed.cend(row), function);
                    }

                    return returnArray;
                }
                default:
                {
                    // this isn't actually possible, just putting this here to get rid
                    // of the compiler warning.
                    return NdArray<bool>(0);
                }
            }
        }

        //============================================================================
        // Method Description:
        ///						Return indices of the maximum values along the given axis.
        ///						Only the first index is returned.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.argmax.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<uint32> argmax(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<uint32> returnArray = { static_cast<uint32>(stl_algorithms::max_element(cbegin(), cend()) - cbegin()) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<uint32> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = static_cast<uint32>(stl_algorithms::max_element(cbegin(row), cend(row)) - cbegin(row));
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> arrayTransposed = transpose();
                    NdArray<uint32> returnArray(1, arrayTransposed.shape_.rows);
                    for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
                    {
                        returnArray(0, row) = static_cast<uint32>(stl_algorithms::max_element(arrayTransposed.cbegin(row),
                            arrayTransposed.cend(row)) - arrayTransposed.cbegin(row));
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
        ///						Return indices of the minimum values along the given axis.
        ///						Only the first index is returned.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.argmin.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<uint32> argmin(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<uint32> returnArray = { static_cast<uint32>(stl_algorithms::min_element(cbegin(), cend()) - cbegin()) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<uint32> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = static_cast<uint32>(stl_algorithms::min_element(cbegin(row), cend(row)) - cbegin(row));
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> arrayTransposed = transpose();
                    NdArray<uint32> returnArray(1, arrayTransposed.shape_.rows);
                    for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
                    {
                        returnArray(0, row) = static_cast<uint32>(stl_algorithms::min_element(arrayTransposed.cbegin(row),
                            arrayTransposed.cend(row)) - arrayTransposed.cbegin(row));
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
        ///						Returns the indices that would sort this array.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.argsort.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<uint32> argsort(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    std::vector<uint32> idx(size_);
                    std::iota(idx.begin(), idx.end(), 0);

                    auto function = [this](uint32 i1, uint32 i2) noexcept -> bool
                    {
                        return operator[](i1) < operator[](i2);
                    };

                    stl_algorithms::stable_sort(idx.begin(), idx.end(), function);
                    return NdArray<uint32>(idx);
                }
                case Axis::COL:
                {
                    NdArray<uint32> returnArray(shape_);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        std::vector<uint32> idx(shape_.cols);
                        std::iota(idx.begin(), idx.end(), 0);

                        auto function = [this, row](uint32 i1, uint32 i2) noexcept -> bool
                        {
                            return operator()(row, i1) < operator()(row, i2);
                        };

                        stl_algorithms::stable_sort(idx.begin(), idx.end(), function);

                        for (uint32 col = 0; col < shape_.cols; ++col)
                        {
                            returnArray(row, col) = idx[col];
                        }
                    }
                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> arrayTransposed = transpose();
                    NdArray<uint32> returnArray(shape_.cols, shape_.rows);
                    for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
                    {
                        std::vector<uint32> idx(arrayTransposed.shape_.cols);
                        std::iota(idx.begin(), idx.end(), 0);

                        auto function = [&arrayTransposed, row](uint32 i1, uint32 i2) noexcept -> bool
                        {
                            return arrayTransposed(row, i1) < arrayTransposed(row, i2);
                        };

                        stl_algorithms::stable_sort(idx.begin(), idx.end(), function);

                        for (uint32 col = 0; col < arrayTransposed.shape_.cols; ++col)
                        {
                            returnArray(row, col) = idx[col];
                        }
                    }
                    return returnArray.transpose();
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
        ///						Returns a copy of the array, cast to a specified type.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html
        ///
        /// @return
        ///				NdArray
        ///
        template<typename dtypeOut>
        NdArray<dtypeOut> astype() const noexcept
        {
            NdArray<dtypeOut> outArray(shape_);

            if (std::is_same<dtypeOut, dtype>::value)
            {
                std::copy(cbegin(), cend(), outArray.begin());
            }
            else
            {
                auto function = [](dtype value) noexcept -> dtypeOut
                {
                    return static_cast<dtypeOut>(value);
                };

                stl_algorithms::transform(cbegin(), cend(), outArray.begin(), function);
            }

            return outArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns the last element of the flattened array.
        ///
        /// @return
        ///				dtype
        ///
        dtype back() const noexcept
        {
            return *(cend() - 1);
        }

        //============================================================================
        // Method Description:
        ///						Swap the bytes of the array elements in place
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.byteswap.html
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& byteswap() noexcept
        {
            switch (endianess_)
            {
                case Endian::BIG:
                {
                    *this = newbyteorder(Endian::LITTLE);
                    break;
                }
                case Endian::LITTLE:
                {
                    *this = newbyteorder(Endian::BIG);
                    break;
                }
                case Endian::NATIVE:
                {
#if BOOST_ENDIAN_BIG_BYTE
                    *this = newbyteorder(Endian::LITTLE);
#elif BOOST_ENDIAN_LITTLE_BYTE
                    *this = newbyteorder(Endian::BIG);
#endif
                    break;
                }
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array whose values are limited to [min, max].
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.clip.html
        ///
        /// @param				inMin: min value to clip to
        /// @param				inMax: max value to clip to
        /// @return
        ///				clipped value
        ///
        NdArray<dtype> clip(dtype inMin, dtype inMax) const noexcept
        {
            NdArray<dtype> outArray(shape_);
            boost::algorithm::clamp_range(cbegin(), cend(), outArray.begin(), inMin, inMax);
            return outArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns the full column of the array
        ///
        ///
        /// @return
        ///				Shape
        ///
        NdArray<dtype> column(uint32 inColumn)
        {
            return operator()(rSlice(), inColumn);
        }

        //============================================================================
        // Method Description:
        ///						returns whether or not a value is included the array
        ///
        /// @param				inValue
        /// @param				inAxis (Optional, default NONE)
        /// @return
        ///				bool
        ///
        NdArray<bool> contains(dtype inValue, Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<bool> returnArray = { stl_algorithms::find(cbegin(), cend(), inValue) != cend() };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<bool> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::find(cbegin(row), cend(row), inValue) != cend(row);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transArray = transpose();
                    NdArray<bool> returnArray(1, transArray.shape_.rows);
                    for (uint32 row = 0; row < transArray.shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::find(transArray.cbegin(row), transArray.cend(row), inValue) != transArray.cend(row);
                    }

                    return returnArray;
                }
                default:
                {
                    // this isn't actually possible, just putting this here to get rid
                    // of the compiler warning.
                    return NdArray<bool>(0);
                }
            }
        }

        //============================================================================
        // Method Description:
        ///						Return a copy of the array
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.copy.html
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> copy() const noexcept
        {
            return NdArray<dtype>(*this);
        }

        //============================================================================
        // Method Description:
        ///						Return the cumulative product of the elements along the given axis.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.cumprod.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> cumprod(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<dtype> returnArray(1, size_);
                    returnArray[0] = front();
                    for (uint32 i = 1; i < size_; ++i)
                    {
                        returnArray[i] = returnArray[i - 1] * array_[i];
                    }

                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(shape_);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(row, 0) = operator()(row, 0);
                        for (uint32 col = 1; col < shape_.cols; ++col)
                        {
                            returnArray(row, col) = returnArray(row, col - 1) * operator()(row, col);
                        }
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> returnArray(shape_);
                    for (uint32 col = 0; col < shape_.cols; ++col)
                    {
                        returnArray(0, col) = operator()(0, col);
                        for (uint32 row = 1; row < shape_.rows; ++row)
                        {
                            returnArray(row, col) = returnArray(row - 1, col) * operator()(row, col);
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
        ///						Return the cumulative sum of the elements along the given axis.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.cumsum.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> cumsum(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<dtype> returnArray(1, size_);
                    returnArray[0] = front();
                    for (uint32 i = 1; i < size_; ++i)
                    {
                        returnArray[i] = returnArray[i - 1] + array_[i];
                    }

                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(shape_);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(row, 0) = operator()(row, 0);
                        for (uint32 col = 1; col < shape_.cols; ++col)
                        {
                            returnArray(row, col) = returnArray(row, col - 1) + operator()(row, col);
                        }
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> returnArray(shape_);
                    for (uint32 col = 0; col < shape_.cols; ++col)
                    {
                        returnArray(0, col) = operator()(0, col);
                        for (uint32 row = 1; row < shape_.rows; ++row)
                        {
                            returnArray(row, col) = returnArray(row - 1, col) + operator()(row, col);
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
        ///						Returns the raw pointer to the underlying data
        /// @return dtype*
        ///
        dtype* data() const noexcept
        {
            return array_;
        }

        //============================================================================
        // Method Description:
        ///						Releases the internal data pointer so that the destructor
        ///                     will not call delete on it, and returns the raw pointer
        ///                     to the underlying data.
        /// @return dtype*
        ///
        dtype* dataRelease() noexcept
        {
            ownsPtr_ = false;
            return array_;
        }

        //============================================================================
        // Method Description:
        ///						Return specified diagonals.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.diagonal.html
        ///
        /// @param				inOffset: Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.
        /// @param				inAxis: (Optional, default ROW) axis the offset is applied to
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> diagonal(int32 inOffset = 0, Axis inAxis = Axis::ROW) const noexcept
        {
            switch (inAxis)
            {
                case Axis::COL:
                {
                    std::vector<dtype> diagnolValues;
                    int32 col = inOffset;
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        if (col < 0)
                        {
                            col++;
                            continue;
                        }
                        else if (col >= static_cast<int32>(shape_.cols))
                        {
                            break;
                        }

                        diagnolValues.push_back(operator()(row, static_cast<uint32>(col)));
                        ++col;
                    }

                    return NdArray<dtype>(diagnolValues);
                }
                case Axis::ROW:
                {
                    std::vector<dtype> diagnolValues;
                    uint32 col = 0;
                    for (int32 row = inOffset; row < static_cast<int32>(shape_.rows); ++row)
                    {
                        if (row < 0)
                        {
                            ++col;
                            continue;
                        }
                        else if (col >= shape_.cols)
                        {
                            break;
                        }

                        diagnolValues.push_back(operator()(static_cast<uint32>(row), col));
                        ++col;
                    }

                    return NdArray<dtype>(diagnolValues);
                }
                default:
                {
                    return NdArray<dtype>(0);
                }
            }
        }

        //============================================================================
        // Method Description:
        ///						Dot product of two arrays.
        ///
        ///						For 2-D arrays it is equivalent to matrix multiplication,
        ///						and for 1-D arrays to inner product of vectors.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.dot.html
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				dot product
        ///
        NdArray<dtype> dot(const NdArray<dtype>& inOtherArray) const
        {
            if (shape_ == inOtherArray.shape_ && (shape_.rows == 1 || shape_.cols == 1))
            {
                dtype dotProduct = std::inner_product(cbegin(), cend(), inOtherArray.cbegin(), dtype{ 0 });
                NdArray<dtype> returnArray = { dotProduct };
                return returnArray;
            }
            else if (shape_.cols == inOtherArray.shape_.rows)
            {
                // 2D array, use matrix multiplication
                NdArray<dtype> returnArray(shape_.rows, inOtherArray.shape_.cols);
                auto otherArrayT = inOtherArray.transpose();

                for (uint32 i = 0; i < shape_.rows; ++i)
                {
                    for (uint32 j = 0; j < otherArrayT.shape_.rows; ++j)
                    {
                        returnArray(i, j) = std::inner_product(otherArrayT.cbegin(j), otherArrayT.cend(j), cbegin(i), dtype{ 0 });
                    }
                }

                return returnArray;
            }
            else
            {
                std::string errStr = "shapes of [" + utils::num2str(shape_.rows) + ", " + utils::num2str(shape_.cols) + "]";
                errStr += " and [" + utils::num2str(inOtherArray.shape_.rows) + ", " + utils::num2str(inOtherArray.shape_.cols) + "]";
                errStr += " are not consistent.";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            return NdArray<dtype>();  // getting rid of compiler warning
        }

        //============================================================================
        // Method Description:
        ///						Dump a binary file of the array to the specified file.
        ///						The array can be read back with or NC::load.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.dump.html
        ///
        /// @param  inFilename
        ///
        void dump(const std::string& inFilename) const
        {
            filesystem::File f(inFilename);
            if (!f.hasExt())
            {
                f.withExt(".bin");
            }

            std::ofstream ofile(f.fullName().c_str(), std::ios::binary);
            if (!ofile.good())
            {
                THROW_RUNTIME_ERROR("Unable to open the input file:\n\t" + inFilename);
            }

            ofile.write(reinterpret_cast<const char*>(array_), size_ * sizeof(dtype));
            ofile.close();
        }

        //============================================================================
        // Method Description:
        ///						Return the indices of the flattened array of the
        ///						elements that are non-zero.
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<uint32> flatnonzero() const noexcept
        {
            std::vector<uint32> indices;
            uint32 idx = 0;
            for (auto value : *this)
            {
                if (value != dtype{ 0 })
                {
                    indices.push_back(idx);
                }
                ++idx;
            }

            return NdArray<uint32>(indices);
        }

        //============================================================================
        // Method Description:
        ///						Return if the NdArray is empty. ie the default construtor
        ///						was used.
        ///
        /// @return
        ///				boolean
        ///
        bool isempty() const noexcept
        {
            return size_ == 0;
        }

        //============================================================================
        // Method Description:
        ///						Return if the NdArray is empty. ie the default construtor
        ///						was used.
        ///
        /// @return
        ///				boolean
        ///
        bool isflat() const noexcept
        {
            return shape_.rows == 1 || shape_.cols == 1;
        }

        //============================================================================
        // Method Description:
        ///						Return if the NdArray is sorted.
        ///
        /// @param inAxis
        /// @return boolean
        ///
        NdArray<bool> issorted(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    return { stl_algorithms::is_sorted(cbegin(), cend()) };
                }
                case Axis::ROW:
                {
                    NdArray<bool> returnArray(shape_.cols, 1);
                    auto transposedArray = transpose();
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::is_sorted(transposedArray.cbegin(row), transposedArray.cend(row));
                    }

                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<bool> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::is_sorted(cbegin(row), cend(row));
                    }

                    return returnArray;
                }
                default:
                {
                    // not actually possible, just getting rid of compiler warning
                    return NdArray<bool>(0);
                }
            }
        }

        //============================================================================
        // Method Description:
        ///						Return if the NdArray is sorted.
        ///
        /// @param inAxis
        /// @return boolean
        ///
        bool issquare() const noexcept
        {
            return shape_.issquare();
        }

        //============================================================================
        // Method Description:
        ///						Return the NdArrays endianess
        ///
        /// @return
        ///				Endian
        ///
        Endian endianess() const noexcept
        {
            return endianess_;
        }

        //============================================================================
        // Method Description:
        ///						Fill the array with a scalar value.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.fill.html
        ///
        /// @param
        ///				inFillValue
        /// @return
        ///				None
        ///
        NdArray<dtype>& fill(dtype inFillValue) noexcept
        {
            stl_algorithms::fill(begin(), end(), inFillValue);
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Return a copy of the array collapsed into one dimension.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.flatten.html
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> flatten() const noexcept
        {
            NdArray<dtype> outArray(1, size_);
            stl_algorithms::copy(cbegin(), cend(), outArray.begin());
            return outArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns the first element of the flattened array.
        ///
        /// @return
        ///				dtype
        ///
        dtype front() const noexcept
        {
            return *cbegin();
        }

        //============================================================================
        // Method Description:
        ///                     Returns a new flat array with the givin flat input indices.
        ///
        /// @param
        ///				inIndices
        /// @return
        ///				values
        ///
        NdArray<dtype> getByIndices(const NdArray<uint32>& inIndices) const
        {
            return operator[](inIndices);
        }

        //============================================================================
        // Method Description:
        ///                     Takes in a boolean mask the same size as the array
        ///                     and returns a flattened array with the values cooresponding
        ///                     to the input mask.
        ///
        /// @param
        ///				inMask
        /// @return
        ///				values
        ///
        NdArray<dtype> getByMask(const NdArray<bool>& inMask) const
        {
            return operator[](inMask);
        }

        //============================================================================
        // Method Description:
        ///						Copy an element of an array to a standard C++ scalar and return it.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.item.html
        ///
        /// @return
        ///				array element
        ///
        dtype item() const
        {
            if (size_ == 1)
            {
                return array_[0];
            }
            else
            {
                THROW_INVALID_ARGUMENT_ERROR("Can only convert an array of size 1 to a C++ scalar");
            }

            return 0; // getting rid of compiler warning
        }

        //============================================================================
        // Method Description:
        ///						Return the maximum along a given axis.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.max.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> max(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<dtype> returnArray = { *stl_algorithms::max_element(cbegin(), cend()) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = *stl_algorithms::max_element(cbegin(row), cend(row));
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<dtype> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        returnArray(0, row) = *stl_algorithms::max_element(transposedArray.cbegin(row), transposedArray.cend(row));
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
        ///						Return the minimum along a given axis.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.min.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> min(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<dtype> returnArray = { *stl_algorithms::min_element(cbegin(), cend()) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = *stl_algorithms::min_element(cbegin(row), cend(row));
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<dtype> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        returnArray(0, row) = *stl_algorithms::min_element(transposedArray.cbegin(row), transposedArray.cend(row));
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
        ///						Return the mean along a given axis.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.mean.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<double> mean(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    double sum = static_cast<double>(std::accumulate(cbegin(), cend(), 0.0));
                    NdArray<double> returnArray = { sum /= static_cast<double>(size_) };

                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<double> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        double sum = static_cast<double>(std::accumulate(cbegin(row), cend(row), 0.0));
                        returnArray(0, row) = sum / static_cast<double>(shape_.cols);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<double> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        double sum = static_cast<double>(std::accumulate(transposedArray.cbegin(row), transposedArray.cend(row), 0.0));
                        returnArray(0, row) = sum / static_cast<double>(transposedArray.shape_.cols);
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
        ///						Return the median along a given axis. 
        ///                     If the dtype is floating point then the middle elements will be
        ///                     averaged for arrays of even number of elements. 
        ///                     If the dtype is integral then the middle elements will be intager
        ///                     averaged (rounded down to integer) for arrays of even number of elements. 
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> median(Axis inAxis = Axis::NONE) const noexcept
        {
            if (size_ == 0)
            {
                THROW_RUNTIME_ERROR("Median is undefined for an array of size = 0.");
            }

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<dtype> copyArray(*this);

                    const uint32 middleIdx = size_ / 2;  // integer division
                    stl_algorithms::nth_element(copyArray.begin(), copyArray.begin() + middleIdx, copyArray.end());

                    dtype medianValue = copyArray.array_[middleIdx];
                    if (size_ % 2 == 0)
                    {
                        const uint32 lhsIndex = middleIdx - 1;
                        stl_algorithms::nth_element(copyArray.begin(), copyArray.begin() + lhsIndex, copyArray.end());
                        medianValue = (medianValue + copyArray.array_[lhsIndex]) / static_cast<dtype>(2); // potentially integer division, ok
                    }

                    return { medianValue };
                }
                case Axis::COL:
                {
                    NdArray<dtype> copyArray(*this);
                    NdArray<dtype> returnArray(1, shape_.rows);

                    const bool isEven = shape_.cols % 2 == 0;
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        uint32 middleIdx = shape_.cols / 2;  // integer division
                        stl_algorithms::nth_element(copyArray.begin(row), copyArray.begin(row) + middleIdx, copyArray.end(row));

                        dtype medianValue = copyArray(row, middleIdx);
                        if (isEven)
                        {
                            const uint32 lhsIndex = middleIdx - 1;
                            stl_algorithms::nth_element(copyArray.begin(row), copyArray.begin(row) + lhsIndex, copyArray.end(row));
                            medianValue = (medianValue + copyArray(row, lhsIndex)) / static_cast<dtype>(2); // potentially integer division, ok
                        }

                        returnArray(0, row) = medianValue;
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<dtype> returnArray(1, transposedArray.shape_.rows);

                    const bool isEven = shape_.rows % 2 == 0;
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        const uint32 middleIdx = transposedArray.shape_.cols / 2;  // integer division
                        stl_algorithms::nth_element(transposedArray.begin(row), transposedArray.begin(row) + middleIdx, transposedArray.end(row));

                        dtype medianValue = transposedArray(row, middleIdx);
                        if (isEven)
                        {
                            const uint32 lhsIndex = middleIdx - 1;
                            stl_algorithms::nth_element(transposedArray.begin(row), transposedArray.begin(row) + lhsIndex, transposedArray.end(row));
                            medianValue = (medianValue + transposedArray(row, lhsIndex)) / static_cast<dtype>(2); // potentially integer division, ok
                        }

                        returnArray(0, row) = medianValue;
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
        ///						Fills the array with nans; only really works with.
        ///                     Only really works for dtype = float/double
        ///
        ///
        NdArray<dtype>& nans() noexcept
        {
            fill(constants::nan);

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Returns the number of bytes held by the array
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.nbytes.html
        ///
        /// @return
        ///				number of bytes
        ///
        uint64 nbytes() const noexcept
        {
            return static_cast<uint64>(sizeof(dtype) * size_);
        }

        //============================================================================
        // Method Description:
        ///						Return the array with the same data viewed with a
        ///						different byte order. only works for integer types,
        ///						floating point types will not compile and you will
        ///						be confused as to why...
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.newbyteorder.html
        ///
        /// @param
        ///				inEndianess
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> newbyteorder(Endian inEndianess) const noexcept
        {
            STATIC_ASSERT_INTEGER(dtype);

            switch (endianess_)
            {
                case Endian::NATIVE:
                {
                    switch (inEndianess)
                    {
                        case Endian::NATIVE:
                        {
                            return NdArray(*this);
                        }
                        case Endian::BIG:
                        {
                            NdArray<dtype> outArray(shape_);

                            stl_algorithms::transform(cbegin(), end(), outArray.begin(), boost::endian::native_to_big<dtype>);

                            outArray.endianess_ = Endian::BIG;
                            return outArray;
                        }
                        case Endian::LITTLE:
                        {
                            NdArray<dtype> outArray(shape_);

                            stl_algorithms::transform(cbegin(), cend(), outArray.begin(), boost::endian::native_to_little<dtype>);

                            outArray.endianess_ = Endian::LITTLE;
                            return outArray;
                        }
                        default:
                        {
                            // this isn't actually possible, just putting this here to get rid
                            // of the compiler warning.
                            return NdArray<dtype>(0);
                        }
                    }
                    break;
                }
                case Endian::BIG:
                {
                    switch (inEndianess)
                    {
                        case Endian::NATIVE:
                        {
                            NdArray<dtype> outArray(shape_);

                            stl_algorithms::transform(cbegin(), cend(), outArray.begin(), boost::endian::big_to_native<dtype>);

                            outArray.endianess_ = Endian::NATIVE;
                            return outArray;
                        }
                        case Endian::BIG:
                        {
                            return NdArray(*this);
                        }
                        case Endian::LITTLE:
                        {
                            NdArray<dtype> outArray(shape_);

                            stl_algorithms::transform(cbegin(), cend(), outArray.begin(),
                                [](dtype value) noexcept -> dtype
                                {
                                    return boost::endian::native_to_little<dtype>(boost::endian::big_to_native<dtype>(value));
                                });

                            outArray.endianess_ = Endian::LITTLE;
                            return outArray;
                        }
                        default:
                        {
                            // this isn't actually possible, just putting this here to get rid
                            // of the compiler warning.
                            return NdArray<dtype>(0);
                        }
                    }
                    break;
                }
                case Endian::LITTLE:
                {
                    switch (inEndianess)
                    {
                        case Endian::NATIVE:
                        {
                            NdArray<dtype> outArray(shape_);

                            stl_algorithms::transform(cbegin(), cend(), outArray.begin(), boost::endian::little_to_native<dtype>);

                            outArray.endianess_ = Endian::NATIVE;
                            return outArray;
                        }
                        case Endian::BIG:
                        {
                            NdArray<dtype> outArray(shape_);

                            auto function = [](dtype value) noexcept -> dtype
                            {
                                return boost::endian::native_to_big<dtype>(boost::endian::little_to_native<dtype>(value));
                            };

                            stl_algorithms::transform(cbegin(), cend(), outArray.begin(), function);

                            outArray.endianess_ = Endian::BIG;
                            return outArray;
                        }
                        case Endian::LITTLE:
                        {
                            return NdArray(*this);
                        }
                        default:
                        {
                            // this isn't actually possible, just putting this here to get rid
                            // of the compiler warning.
                            return NdArray<dtype>(0);
                        }
                    }
                    break;
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
        ///						Returns True if none elements evaluate to True or non zero
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.any.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<bool> none(Axis inAxis = Axis::NONE) const noexcept
        {
            auto function = [](dtype i) noexcept -> bool
            {
                return i != dtype{ 0 };
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<bool> returnArray = { stl_algorithms::none_of(cbegin(), cend(), function) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<bool> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::none_of(cbegin(row), cend(row), function);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> arrayTransposed = transpose();
                    NdArray<bool> returnArray(1, arrayTransposed.shape_.rows);
                    for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::none_of(arrayTransposed.cbegin(row), arrayTransposed.cend(row), function);
                    }

                    return returnArray;
                }
                default:
                {
                    // this isn't actually possible, just putting this here to get rid
                    // of the compiler warning.
                    return NdArray<bool>(0);
                }
            }
        }

        //============================================================================
        // Method Description:
        ///						Return the row/col indices of the array of the
        ///						elements that are non-zero.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.nonzero.html
        ///
        /// @return
        ///				std::pair<NdArray, NdArray> where first is the row indices and second is the
        ///             column indices
        ///
        std::pair<NdArray<uint32>, NdArray<uint32>> nonzero() const noexcept;

        //============================================================================
        // Method Description:
        ///						Returns the norm of the array
        ///
        ///                     Numpy Reference: http://www.numpy.org/devdocs/reference/generated/numpy.linalg.norm.html?highlight=norm#numpy.linalg.norm
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				norm
        ///
        NdArray<double> norm(Axis inAxis = Axis::NONE) const noexcept
        {
            double sumOfSquares = 0.0;
            auto function = [&sumOfSquares](dtype value) noexcept -> void
            {
                sumOfSquares += utils::sqr(static_cast<double>(value));
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    std::for_each(cbegin(), cend(), function);

                    NdArray<double> returnArray = { std::sqrt(sumOfSquares) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<double> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        sumOfSquares = 0.0;
                        std::for_each(cbegin(row), cend(row), function);
                        returnArray(0, row) = std::sqrt(sumOfSquares);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<double> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        sumOfSquares = 0.0;
                        std::for_each(transposedArray.cbegin(row), transposedArray.cend(row), function);
                        returnArray(0, row) = std::sqrt(sumOfSquares);
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
        ///						Returns the number of columns in the array
        ///
        ///
        /// @return
        ///				uint32
        ///
        uint32 numCols() noexcept
        {
            return shape_.cols;
        }

        //============================================================================
        // Method Description:
        ///						Returns the number of rows in the array
        ///
        ///
        /// @return
        ///				uint32
        ///
        uint32 numRows() noexcept
        {
            return shape_.rows;
        }

        //============================================================================
        // Method Description:
        ///						Fills the array with ones
        ///
        ///
        NdArray<dtype>& ones() noexcept
        {
            fill(1);

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Returns whether or not the array object owns the underlying data
        ///
        /// @return bool
        ///
        bool ownsInternalData() noexcept
        {
            return ownsPtr_;
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
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.partition.html
        ///
        /// @param				inKth: kth element
        /// @param				inAxis (Optional, default NONE)
        /// @return
        ///				None
        ///
        NdArray<dtype>& partition(uint32 inKth, Axis inAxis = Axis::NONE)
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    if (inKth >= size_)
                    {
                        std::string errStr = "kth(=" + utils::num2str(inKth);
                        errStr += ") out of bounds (" + utils::num2str(size_) + ")";
                        THROW_INVALID_ARGUMENT_ERROR(errStr);
                    }

                    stl_algorithms::nth_element(begin(), begin() + inKth, end());
                    break;
                }
                case Axis::COL:
                {
                    if (inKth >= shape_.cols)
                    {
                        std::string errStr = "kth(=" + utils::num2str(inKth);
                        errStr += ") out of bounds (" + utils::num2str(shape_.cols) + ")";
                        THROW_INVALID_ARGUMENT_ERROR(errStr);
                    }

                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        stl_algorithms::nth_element(begin(row), begin(row) + inKth, end(row));
                    }
                    break;
                }
                case Axis::ROW:
                {
                    if (inKth >= shape_.rows)
                    {
                        std::string errStr = "kth(=" + utils::num2str(inKth);
                        errStr += ") out of bounds (" + utils::num2str(shape_.rows) + ")";
                        THROW_INVALID_ARGUMENT_ERROR(errStr);
                    }

                    NdArray<dtype> transposedArray = transpose();
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        stl_algorithms::nth_element(transposedArray.begin(row), transposedArray.begin(row) + inKth, transposedArray.end(row));
                    }
                    *this = transposedArray.transpose();
                    break;
                }
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Prints the array to the console.
        ///
        ///
        void print() const noexcept
        {
            std::cout << *this;
        }

        //============================================================================
        // Method Description:
        ///						Return the product of the array elements over the given axis
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.prod.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> prod(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    dtype product = std::accumulate(cbegin(), cend(),
                        dtype{ 1 }, std::multiplies<dtype>());
                    NdArray<dtype> returnArray = { product };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = std::accumulate(cbegin(row), cend(row),
                            dtype{ 1 }, std::multiplies<dtype>());
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<dtype> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        returnArray(0, row) = std::accumulate(transposedArray.cbegin(row), transposedArray.cend(row),
                            dtype{ 1 }, std::multiplies<dtype>());
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
        ///						Peak to peak (maximum - minimum) value along a given axis.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.ptp.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> ptp(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    auto result = stl_algorithms::minmax_element(cbegin(), cend());
                    NdArray<dtype> returnArray = { *result.second - *result.first };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        auto result = stl_algorithms::minmax_element(cbegin(row), cend(row));
                        returnArray(0, row) = *result.second - *result.first;
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<dtype> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        auto result = stl_algorithms::minmax_element(transposedArray.cbegin(row), transposedArray.cend(row));
                        returnArray(0, row) = *result.second - *result.first;
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
        ///						set the flat index element to the value
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inIndex
        /// @param				inValue
        ///
        NdArray<dtype>& put(int32 inIndex, dtype inValue)
        {
            at(inIndex) = inValue;

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						set the 2D row/col index element to the value
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inRow
        /// @param				inCol
        /// @param				inValue
        ///
        NdArray<dtype>& put(int32 inRow, int32 inCol, dtype inValue)
        {
            at(inRow, inCol) = inValue;

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Set a.flat[n] = values for all n in indices.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inIndices
        /// @param				inValue
        ///
        NdArray<dtype>& put(const NdArray<uint32>& inIndices, dtype inValue)
        {
            for (auto index : inIndices)
            {
                put(index, inValue);
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Set a.flat[n] = values[n] for all n in indices.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inIndices
        /// @param				inValues
        ///
        NdArray<dtype>& put(const NdArray<uint32>& inIndices, const NdArray<dtype>& inValues)
        {
            if (inIndices.size() != inValues.size())
            {
                THROW_INVALID_ARGUMENT_ERROR("Input indices do not match values dimensions.");
            }

            uint32 counter = 0;
            for (auto index : inIndices)
            {
                put(index, inValues[counter++]);
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Set the slice indices to the input value.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inSlice
        /// @param				inValue
        ///
        NdArray<dtype>& put(const Slice& inSlice, dtype inValue)
        {
            Slice inSliceCopy(inSlice);
            inSliceCopy.makePositiveAndValidate(size_);

            for (int32 i = inSliceCopy.start; i < inSliceCopy.stop; i += inSliceCopy.step)
            {
                put(i, inValue);
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Set the slice indices to the input values.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inSlice
        /// @param  			inValues
        ///
        NdArray<dtype>& put(const Slice& inSlice, const NdArray<dtype>& inValues)
        {
            Slice inSliceCopy(inSlice);
            inSliceCopy.makePositiveAndValidate(size_);

            std::vector<uint32> indices;
            for (int32 i = inSliceCopy.start; i < inSliceCopy.stop; i += inSliceCopy.step)
            {
                indices.push_back(i);
            }

            return put(NdArray<uint32>(indices), inValues);
        }

        //============================================================================
        // Method Description:
        ///						Set the slice indices to the input value.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inRowSlice
        /// @param				inColSlice
        /// @param				inValue
        ///
        NdArray<dtype>& put(const Slice& inRowSlice, const Slice& inColSlice, dtype inValue)
        {
            Slice inRowSliceCopy(inRowSlice);
            Slice inColSliceCopy(inColSlice);

            inRowSliceCopy.makePositiveAndValidate(shape_.rows);
            inColSliceCopy.makePositiveAndValidate(shape_.cols);

            std::vector<uint32> indices;
            for (int32 row = inRowSliceCopy.start; row < inRowSliceCopy.stop; row += inRowSliceCopy.step)
            {
                for (int32 col = inColSliceCopy.start; col < inColSliceCopy.stop; col += inColSliceCopy.step)
                {
                    put(row, col, inValue);
                }
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Set the slice indices to the input value.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inRowSlice
        /// @param				inColIndex
        /// @param				inValue
        ///
        NdArray<dtype>& put(const Slice& inRowSlice, int32 inColIndex, dtype inValue)
        {
            Slice inRowSliceCopy(inRowSlice);
            inRowSliceCopy.makePositiveAndValidate(shape_.rows);

            std::vector<uint32> indices;
            for (int32 row = inRowSliceCopy.start; row < inRowSliceCopy.stop; row += inRowSliceCopy.step)
            {
                put(row, inColIndex, inValue);
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Set the slice indices to the input value.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inRowIndex
        /// @param				inColSlice
        /// @param				inValue
        ///
        NdArray<dtype>& put(int32 inRowIndex, const Slice& inColSlice, dtype inValue)
        {
            Slice inColSliceCopy(inColSlice);
            inColSliceCopy.makePositiveAndValidate(shape_.cols);

            std::vector<uint32> indices;
            for (int32 col = inColSliceCopy.start; col < inColSliceCopy.stop; col += inColSliceCopy.step)
            {
                put(inRowIndex, col, inValue);
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Set the slice indices to the input values.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inRowSlice
        /// @param				inColSlice
        /// @param				inValues
        ///
        NdArray<dtype>& put(const Slice& inRowSlice, const Slice& inColSlice, const NdArray<dtype>& inValues)
        {
            Slice inRowSliceCopy(inRowSlice);
            Slice inColSliceCopy(inColSlice);

            inRowSliceCopy.makePositiveAndValidate(shape_.rows);
            inColSliceCopy.makePositiveAndValidate(shape_.cols);

            std::vector<uint32> indices;
            for (int32 row = inRowSliceCopy.start; row < inRowSliceCopy.stop; row += inRowSliceCopy.step)
            {
                for (int32 col = inColSliceCopy.start; col < inColSliceCopy.stop; col += inColSliceCopy.step)
                {
                    const uint32 index = row * shape_.cols + col;
                    indices.push_back(index);
                }
            }

            return put(NdArray<uint32>(indices), inValues);
        }

        //============================================================================
        // Method Description:
        ///						Set the slice indices to the input values.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inRowSlice
        /// @param				inColIndex
        /// @param				inValues
        ///
        NdArray<dtype>& put(const Slice& inRowSlice, int32 inColIndex, const NdArray<dtype>& inValues)
        {
            Slice inRowSliceCopy(inRowSlice);
            inRowSliceCopy.makePositiveAndValidate(shape_.rows);

            std::vector<uint32> indices;
            for (int32 row = inRowSliceCopy.start; row < inRowSliceCopy.stop; row += inRowSliceCopy.step)
            {
                const uint32 index = row * shape_.cols + inColIndex;
                indices.push_back(index);
            }

            return put(NdArray<uint32>(indices), inValues);
        }

        //============================================================================
        // Method Description:
        ///						Set the slice indices to the input values.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
        ///
        /// @param				inRowIndex
        /// @param				inColSlice
        /// @param				inValues
        ///
        NdArray<dtype>& put(int32 inRowIndex, const Slice& inColSlice, const NdArray<dtype>& inValues)
        {
            Slice inColSliceCopy(inColSlice);
            inColSliceCopy.makePositiveAndValidate(shape_.cols);

            std::vector<uint32> indices;
            for (int32 col = inColSliceCopy.start; col < inColSliceCopy.stop; col += inColSliceCopy.step)
            {
                const uint32 index = inRowIndex * shape_.cols + col;
                indices.push_back(index);
            }

            return put(NdArray<uint32>(indices), inValues);
        }

        //============================================================================
        // Method Description:
        ///						Set the mask indices to the input value.
        ///
        /// @param				inMask
        /// @param				inValue
        ///
        NdArray<dtype>& putMask(const NdArray<bool>& inMask, dtype inValue)
        {
            if (inMask.shape() != shape_)
            {
                THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the array it is masking.");
            }

            return put(inMask.flatnonzero(), inValue);
        }

        //============================================================================
        // Method Description:
        ///						Set the mask indices to the input values.
        ///
        /// @param				inMask
        /// @param				inValues
        ///
        NdArray<dtype>& putMask(const NdArray<bool>& inMask, const NdArray<dtype>& inValues)
        {
            if (inMask.shape() != shape_)
            {
                THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the array it is masking.");
            }

            return put(inMask.flatnonzero(), inValues);
        }

        //============================================================================
        // Method Description:
        ///	Flattens the array but does not make a copy.
        ///
        /// Numpy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html
        ///
        /// @return NdArray
        ///
        NdArray<dtype>& ravel()
        {
            reshape(size_);
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Repeat elements of an array.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.repeat.html
        ///
        /// @param				inNumRows
        /// @param				inNumCols
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> repeat(uint32 inNumRows, uint32 inNumCols) const noexcept
        {
            NdArray<dtype> returnArray(shape_.rows * inNumRows, shape_.cols * inNumCols);

            for (uint32 row = 0; row < inNumRows; ++row)
            {
                for (uint32 col = 0; col < inNumCols; ++col)
                {
                    std::vector<uint32> indices(shape_.size());

                    const uint32 rowStart = row * shape_.rows;
                    const uint32 colStart = col * shape_.cols;

                    const uint32 rowEnd = (row + 1) * shape_.rows;
                    const uint32 colEnd = (col + 1) * shape_.cols;

                    uint32 counter = 0;
                    for (uint32 rowIdx = rowStart; rowIdx < rowEnd; ++rowIdx)
                    {
                        for (uint32 colIdx = colStart; colIdx < colEnd; ++colIdx)
                        {
                            indices[counter++] = rowIdx * returnArray.shape_.cols + colIdx;
                        }
                    }

                    returnArray.put(NdArray<uint32>(indices), *this);
                }
            }

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Repeat elements of an array.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.repeat.html
        ///
        /// @param
        ///				inRepeatShape
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> repeat(const Shape& inRepeatShape) const noexcept
        {
            return repeat(inRepeatShape.rows, inRepeatShape.cols);
        }

        //============================================================================
        // Method Description:
        ///						Replaces a value of the array with another value
        ///
        /// @param  oldValue: the value to replace
        /// @param  newValue: the value to replace with
        ///
        void replace(dtype oldValue, dtype newValue) noexcept
        {
            stl_algorithms::replace(begin(), end(), oldValue, newValue);
        }

        //============================================================================
        // Method Description:
        ///	The new shape should be compatible with the original shape. If an single integer,
        /// then the result will be a 1-D array of that length. One shape dimension 
        /// can be -1. In this case, the value is inferred from the length of the 
        /// array and remaining dimensions. 
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.reshape.html
        ///
        /// @param      inSize
        ///
        NdArray<dtype>& reshape(uint32 inSize)
        {
            if (inSize != size_)
            {
                std::string errStr = "Cannot reshape array of size " + utils::num2str(size_) + " into shape ";
                errStr += "[" + utils::num2str(1) + ", " + utils::num2str(inSize) + "]";
                THROW_RUNTIME_ERROR(errStr);
            }

            shape_.rows = 1;
            shape_.cols = inSize;

            return *this;
        }

        //============================================================================
        // Method Description:
        ///	The new shape should be compatible with the original shape. If an single integer,
        /// then the result will be a 1-D array of that length. One shape dimension 
        /// can be -1. In this case, the value is inferred from the length of the 
        /// array and remaining dimensions. 
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.reshape.html
        ///
        /// @param      inNumRows
        /// @param      inNumCols
        ///
        NdArray<dtype>& reshape(int32 inNumRows, int32 inNumCols)
        {
            if (inNumRows < 0)
            {
                if (size_ % inNumCols == 0)
                {
                    return reshape(size_ / inNumCols, inNumCols);
                }
                else
                {
                    std::string errStr = "Cannot reshape array of size " + utils::num2str(size_) + " into a shape ";
                    errStr += "with " + utils::num2str(inNumCols) + " columns";
                    THROW_INVALID_ARGUMENT_ERROR(errStr);
                }
            }

            if (inNumCols < 0)
            {
                if (size_ % inNumRows == 0)
                {
                    return reshape(inNumRows, size_ / inNumRows);
                }
                else
                {
                    std::string errStr = "Cannot reshape array of size " + utils::num2str(size_) + " into a shape ";
                    errStr += "with " + utils::num2str(inNumRows) + " rows";
                    THROW_INVALID_ARGUMENT_ERROR(errStr);
                }
            }

            if (static_cast<uint32>(inNumRows * inNumCols) != size_)
            {
                std::string errStr = "Cannot reshape array of size " + utils::num2str(size_) + " into shape ";
                errStr += "[" + utils::num2str(inNumRows) + ", " + utils::num2str(inNumCols) + "]";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            shape_.rows = static_cast<uint32>(inNumRows);
            shape_.cols = static_cast<uint32>(inNumCols);

            return *this;
        }

        //============================================================================
        // Method Description:
        ///	The new shape should be compatible with the original shape. If an single integer,
        /// then the result will be a 1-D array of that length. One shape dimension 
        /// can be -1. In this case, the value is inferred from the length of the 
        /// array and remaining dimensions. 
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.reshape.html
        ///
        /// @param
        ///				inShape
        ///
        NdArray<dtype>& reshape(const Shape& inShape)
        {
            return reshape(inShape.rows, inShape.cols);
        }

        //============================================================================
        // Method Description:
        ///						Change shape and size of array in-place. All previous
        ///						data of the array is lost.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.resize.html
        ///
        /// @param      inNumRows
        /// @param      inNumCols
        ///
        NdArray<dtype>& resizeFast(uint32 inNumRows, uint32 inNumCols) noexcept
        {
            newArray(Shape(inNumRows, inNumCols));
            zeros();

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Change shape and size of array in-place. All previous
        ///						data of the array is lost.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.resize.html
        ///
        /// @param
        ///				inShape
        ///
        NdArray<dtype>& resizeFast(const Shape& inShape) noexcept
        {
            return resizeFast(inShape.rows, inShape.cols);
        }

        //============================================================================
        // Method Description:
        ///						Return a new array with the specified shape. If new shape
        ///						is larger than old shape then array will be padded with zeros.
        ///						If new shape is smaller than the old shape then the data will
        ///						be discarded.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.resize.html
        ///
        /// @param				inNumRows
        /// @param				inNumCols
        ///
        NdArray<dtype>& resizeSlow(uint32 inNumRows, uint32 inNumCols) noexcept
        {
            std::vector<dtype> oldData(size_);
            stl_algorithms::copy(begin(), end(), oldData.begin());

            const Shape inShape(inNumRows, inNumCols);
            const Shape oldShape = shape_;

            newArray(inShape);

            for (uint32 row = 0; row < inShape.rows; ++row)
            {
                for (uint32 col = 0; col < inShape.cols; ++col)
                {
                    if (row >= oldShape.rows || col >= oldShape.cols)
                    {
                        operator()(row, col) = dtype{ 0 }; // zero fill
                    }
                    else
                    {
                        operator()(row, col) = oldData[row * oldShape.cols + col];
                    }
                }
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Return a new array with the specified shape. If new shape
        ///						is larger than old shape then array will be padded with zeros.
        ///						If new shape is smaller than the old shape then the data will
        ///						be discarded.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.resize.html
        ///
        /// @param
        ///				inShape
        ///
        NdArray<dtype>& resizeSlow(const Shape& inShape) noexcept
        {
            return resizeSlow(inShape.rows, inShape.cols);
        }

        //============================================================================
        // Method Description:
        ///						Return the root mean square (RMS) along a given axis.
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<double> rms(Axis inAxis = Axis::NONE) const noexcept
        {
            double squareSum = 0.0;
            auto function = [&squareSum](dtype value) noexcept -> void
            {
                squareSum += utils::sqr(static_cast<double>(value));
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    std::for_each(cbegin(), cend(), function);
                    NdArray<double> returnArray = { std::sqrt(squareSum / static_cast<double>(size_)) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<double> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        squareSum = 0.0;
                        std::for_each(cbegin(row), cend(row), function);
                        returnArray(0, row) = std::sqrt(squareSum / static_cast<double>(shape_.cols));
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<double> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        squareSum = 0.0;
                        std::for_each(transposedArray.cbegin(row), transposedArray.cend(row), function);
                        returnArray(0, row) = std::sqrt(squareSum / static_cast<double>(transposedArray.shape_.cols));
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
        ///						Return a with each element rounded to the given number
        ///						of decimals.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.round.html
        ///
        /// @param
        ///				inNumDecimals (default 0)
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> round(uint8 inNumDecimals = 0) const noexcept
        {
            if (DtypeInfo<dtype>::isInteger())
            {
                return NdArray<dtype>(*this);
            }
            else
            {
                NdArray<dtype> returnArray(shape_);
                double multFactor = utils::power(10.0, inNumDecimals);
                auto function = [multFactor](dtype value) -> dtype
                {
                    return static_cast<dtype>(std::nearbyint(static_cast<double>(value) * multFactor) / multFactor);
                };

                stl_algorithms::transform(cbegin(), cend(), returnArray.begin(), function);

                return returnArray;
            }
        }

        //============================================================================
        // Method Description:
        ///						Returns the full row of the array
        ///
        ///
        /// @return
        ///				Shape
        ///
        NdArray<dtype> row(uint32 inRow)
        {
            return NdArray<dtype>(cbegin(inRow), cend(inRow));
        }

        //============================================================================
        // Method Description:
        ///						Return the shape of the array
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.shape.html
        ///
        /// @return
        ///				Shape
        ///
        Shape shape() const noexcept
        {
            return shape_;
        }

        //============================================================================
        // Method Description:
        ///						Return the size of the array
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.size.html
        ///
        /// @return
        ///				size
        ///
        uint32 size() const noexcept
        {
            return size_;
        }

        //============================================================================
        // Method Description:
        ///						Sort an array, in-place.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.sort.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				size
        ///
        NdArray<dtype>& sort(Axis inAxis = Axis::NONE) noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    stl_algorithms::sort(begin(), end());
                    break;
                }
                case Axis::COL:
                {
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        stl_algorithms::sort(begin(row), end(row));
                    }
                    break;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        stl_algorithms::sort(transposedArray.begin(row), transposedArray.end(row));
                    }

                    *this = transposedArray.transpose();
                    break;
                }
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Return the std along a given axis.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.std.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<double> stdev(Axis inAxis = Axis::NONE) const noexcept
        {
            double meanValue = 0.0;
            double sum = 0.0;

            auto function = [&sum, &meanValue](dtype value) noexcept-> void
            {
                sum += utils::sqr(static_cast<double>(value) - meanValue);
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    meanValue = mean(inAxis).item();
                    std::for_each(cbegin(), cend(), function);

                    NdArray<double> returnArray = { std::sqrt(sum / size_) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<double> meanValueArray = mean(inAxis);
                    NdArray<double> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        meanValue = meanValueArray[row];
                        sum = 0.0;
                        std::for_each(cbegin(row), cend(row), function);

                        returnArray(0, row) = std::sqrt(sum / shape_.cols);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<double> meanValueArray = mean(inAxis);
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<double> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        meanValue = meanValueArray[row];
                        sum = 0.0;
                        std::for_each(transposedArray.cbegin(row), transposedArray.cend(row), function);

                        returnArray(0, row) = std::sqrt(sum / transposedArray.shape_.cols);
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
        ///						returns the NdArray as a string representation
        ///
        /// @return
        ///				string
        ///
        std::string str() const noexcept
        {
            std::string out;
            out += "[";
            for (uint32 row = 0; row < shape_.rows; ++row)
            {
                out += "[";
                for (uint32 col = 0; col < shape_.cols; ++col)
                {
                    out += utils::num2str(operator()(row, col)) + ", ";
                }

                if (row == shape_.rows - 1)
                {
                    out += "]";
                }
                else
                {
                    out += "]\n";
                }
            }
            out += "]\n";
            return out;
        }

        //============================================================================
        // Method Description:
        ///						Return the sum of the array elements over the given axis.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.sum.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> sum(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<dtype> returnArray = { std::accumulate(cbegin(), cend(), dtype{ 0 }) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = std::accumulate(cbegin(row), cend(row), dtype{ 0 });
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    const Shape transShape = transposedArray.shape();
                    NdArray<dtype> returnArray(1, transShape.rows);
                    for (uint32 row = 0; row < transShape.rows; ++row)
                    {
                        returnArray(0, row) = std::accumulate(transposedArray.cbegin(row), transposedArray.cend(row), dtype{ 0 });
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
        ///						Interchange two axes of an array. Equivalent to transpose...
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.swapaxes.html
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> swapaxes() const noexcept
        {
            return transpose();
        }

        //============================================================================
        // Method Description:
        ///						Write array to a file as text or binary (default)..
        ///						The data produced by this method can be recovered
        ///						using the function fromfile().
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.tofile.html
        ///
        /// @param				inFilename
        /// @param				inSep: Separator between array items for text output. If "" (empty), a binary file is written
        /// @return
        ///				None
        ///
        void tofile(const std::string& inFilename, const std::string& inSep = "") const
        {
            if (inSep.compare("") == 0)
            {
                dump(inFilename);
            }
            else
            {
                filesystem::File f(inFilename);
                if (!f.hasExt())
                {
                    f.withExt("txt");
                }

                std::ofstream ofile(f.fullName().c_str());
                if (!ofile.good())
                {
                    THROW_RUNTIME_ERROR("Input file could not be opened:\n\t" + inFilename);
                }

                uint32 counter = 0;
                for (auto value : *this)
                {
                    ofile << value;
                    if (counter++ != size_ - 1)
                    {
                        ofile << inSep;
                    }
                }
                ofile.close();
            }
        }

        //============================================================================
        // Method Description:
        ///						Write flattened array to an STL vector
        ///
        /// @return
        ///				std::vector
        ///
        std::vector<dtype> toStlVector() const noexcept
        {
            return std::vector<dtype>(cbegin(), cend());
        }

        //============================================================================
        // Method Description:
        ///						Return the sum along diagonals of the array.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.trace.html
        ///
        /// @param				inOffset: Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.
        /// @param				inAxis: (Optional, default ROW) Axis to offset from
        ///
        /// @return
        ///				value
        ///
        dtype trace(uint32 inOffset = 0, Axis inAxis = Axis::ROW) const noexcept
        {
            uint32 rowStart = 0;
            uint32 colStart = 0;
            switch (inAxis)
            {
                case Axis::ROW:
                {
                    rowStart += inOffset;
                    break;
                }
                case Axis::COL:
                {
                    colStart += inOffset;
                    break;
                }
                default:
                {
                    // if the user input NONE, override back to ROW
                    inAxis = Axis::ROW;
                    break;
                }
            }

            if (rowStart >= shape_.rows || colStart >= shape_.cols)
            {
                return dtype{ 0 };
            }

            uint32 col = colStart;
            dtype sum = 0;
            for (uint32 row = rowStart; row < shape_.rows; ++row)
            {
                if (col >= shape_.cols)
                {
                    break;
                }
                sum += operator()(row, col++);
            }

            return sum;
        }

        //============================================================================
        // Method Description:
        ///						Tranpose the rows and columns of an array
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.transpose.html
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> transpose() const noexcept
        {
            NdArray<dtype> transArray(shape_.cols, shape_.rows);
            for (uint32 row = 0; row < shape_.rows; ++row)
            {
                for (uint32 col = 0; col < shape_.cols; ++col)
                {
                    transArray(col, row) = operator()(row, col);
                }
            }
            return transArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns the variance of the array elements, along given axis.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.var.html
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<double> var(Axis inAxis = Axis::NONE) const noexcept
        {
            NdArray<double> stdValues = stdev(inAxis);
            auto function = [](double& value) noexcept -> void
            {
                value *= value;
            };

            stl_algorithms::for_each(stdValues.begin(), stdValues.end(), function);
            return stdValues;
        }

        //============================================================================
        // Method Description:
        ///						Fills the array with zeros
        ///
        ///
        NdArray<dtype>& zeros()
        {
            fill(0);

            return *this;
        }
    };

    // NOTE: this needs to be defined outside of the class to get rid of a compiler
    // error in Visual Studio
    template<typename dtype>
    std::pair<NdArray<uint32>, NdArray<uint32>> NdArray<dtype>::nonzero() const noexcept
    {
        std::vector<uint32> rowIndices;
        std::vector<uint32> colIndices;

        for (uint32 row = 0; row < shape_.rows; ++row)
        {
            for (uint32 col = 0; col < shape_.cols; ++col)
            {
                if (operator()(row, col) != dtype{ 0 })
                {
                    rowIndices.push_back(row);
                    colIndices.push_back(col);
                }
            }
        }

        return std::make_pair(NdArray<uint32>(rowIndices), NdArray<uint32>(colIndices));
    }

    //============================================================================
    // Method Description:
    ///						io operator for the NdArray class
    ///
    /// @param      inOStream
    /// @param      inArray
    /// @return
    ///				std::ostream
    ///
    template<typename dtype>
    std::ostream& operator<<(std::ostream& inOStream, const NdArray<dtype>& inArray) noexcept
    {
        inOStream << inArray.str();
        return inOStream;
    }
}
