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
/// Holds 1D and 2D arrays, the main work horse of the NumCpp library
///
#pragma once

#include"NumCpp/DtypeInfo.hpp"
#include"NumCpp/Shape.hpp"
#include"NumCpp/Slice.hpp"
#include"NumCpp/Types.hpp"
#include"NumCpp/Utils.hpp"
#include"NumCpp/Constants.hpp"

#include<boost/algorithm/clamp.hpp>
#include<boost/filesystem.hpp>
#include<boost/predef/other/endian.h>
#include<boost/endian/conversion.hpp>

#include<algorithm>
#include<cmath>
#include<deque>
#include<functional>
#include<fstream>
#include<initializer_list>
#include<iostream>
#include<numeric>
#include<set>
#include<stdexcept>
#include<string>
#include<utility>
#include<vector>

namespace nc
{
    //================================================================================
    // Class Description:
    ///						Holds 1D and 2D arrays, the main work horse of the NumCpp library
    template<typename dtype>
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
        NdArray() = default;

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
        {};

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
        {};

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
        {};

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
            std::copy(inList.begin(), inList.end(), array_);
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
            for (auto& list : inList)
            {
                size_ += static_cast<uint32>(list.size());

                if (shape_.cols == 0)
                {
                    shape_.cols = static_cast<uint32>(list.size());
                }
                else if (list.size() != shape_.cols)
                {
                    std::string errStr = "ERROR: NdArray::Constructor: All rows of the initializer list needs to have the same number of elements";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }
            }

            array_ = new dtype[size_];
            uint32 row = 0;
            for (auto& list : inList)
            {
                std::copy(list.begin(), list.end(), array_ + row * shape_.cols);
                ++row;
            }

            ownsPtr_ = true;
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
            std::copy(inVector.begin(), inVector.end(), array_);
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
            std::copy(inDeque.begin(), inDeque.end(), array_);
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
            std::copy(inSet.begin(), inSet.end(), array_);
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
            std::copy(inFirst, inLast, array_);
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
        {}

        //============================================================================
        // Method Description:
        ///						Constructor.  Copies the contents of the buffer into 
        ///                     the array.
        ///
        /// @param				inPtr: dtype* to beginning of buffer
        /// @param				inNumBytes: number of bytes
        ///
        NdArray(dtype* inPtr, uint32 inNumBytes) noexcept :
            shape_(1, inNumBytes / sizeof(dtype)),
            size_(shape_.size()),
            array_(new dtype[size_]),
            ownsPtr_(true)
        {
            for (uint32 i = 0; i < size_; ++i)
            {
                std::copy(inPtr, inPtr + size_, begin());
            }
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
            std::copy(inOtherArray.cbegin(), inOtherArray.cend(), begin());
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
        ///				inOtherArray
        /// @return
        ///				NdArray<dtype>
        ///
        NdArray<dtype>& operator=(const NdArray<dtype>& inOtherArray) noexcept
        {
            newArray(inOtherArray.shape_);
            endianess_ = inOtherArray.endianess_;

            std::copy(inOtherArray.cbegin(), inOtherArray.cend(), begin());

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
            std::fill(begin(), end(), inValue);

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Move operator, performs a deep move
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray<dtype>
        ///
        NdArray<dtype>& operator=(NdArray<dtype>&& inOtherArray) noexcept
        {
            if (&inOtherArray != this)
            {
                deleteArray();
                shape_ = inOtherArray.shape_;
                size_ = inOtherArray.size_;
                endianess_ = inOtherArray.endianess_;
                array_ = inOtherArray.array_;
                ownsPtr_ = inOtherArray.ownsPtr_;

                inOtherArray.shape_.rows = inOtherArray.shape_.cols = inOtherArray.size_ = 0;
                inOtherArray.array_ = nullptr;
                inOtherArray.ownsPtr_ = false;
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
                std::string errStr = "ERROR: operator[]: input inMask must have the same shape as the NdArray it will be masking.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            auto indices = inMask.nonzero();
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
                std::string errStr = "ERROR: operator[]: input indices must be less than the array size.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
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
                std::string errStr = "ERROR: NdArray::at: Input index " + utils::num2str(inIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(size_) + ".";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
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
                std::string errStr = "ERROR: NdArray::at: Input index " + utils::num2str(inIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(size_) + ".";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
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
                std::string errStr = "ERROR: NdArray::at: Row index " + utils::num2str(inRowIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(shape_.rows) + ".";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            // this doesn't allow for calling the first element as -size_...
            // but why would you really want to that anyway?
            if (std::abs(inColIndex) > static_cast<int32>(shape_.cols - 1))
            {
                std::string errStr = "ERROR: NdArray::at: Column index " + utils::num2str(inColIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(shape_.cols) + ".";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
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
                std::string errStr = "ERROR: NdArray::at: Row index " + utils::num2str(inRowIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(shape_.rows) + ".";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            // this doesn't allow for calling the first element as -size_...
            // but why would you really want to do that anyway?
            if (std::abs(inColIndex) > static_cast<int32>(shape_.cols - 1))
            {
                std::string errStr = "ERROR: NdArray::at: Column index " + utils::num2str(inColIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(shape_.cols) + ".";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
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
                std::string errStr = "ERROR: NdArray::begin: input row is greater than the number of rows in the array.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
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
                std::string errStr = "ERROR: NdArray::begin: input row is greater than the number of rows in the array.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
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
                std::string errStr = "ERROR: NdArray::begin: input row is greater than the number of rows in the array.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
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
                std::string errStr = "ERROR: NdArray::begin: input row is greater than the number of rows in the array.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
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
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<bool> returnArray = { std::all_of(cbegin(), cend(),
                        [](dtype i) noexcept -> bool {return i != static_cast<dtype>(0); }) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<bool> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = std::all_of(cbegin(row), cend(row),
                            [](dtype i) noexcept -> bool {return i != static_cast<dtype>(0); });
                    }
                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> arrayTransposed = transpose();
                    NdArray<bool> returnArray(1, arrayTransposed.shape_.rows);
                    for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
                    {
                        returnArray(0, row) = std::all_of(arrayTransposed.cbegin(row), arrayTransposed.cend(row),
                            [](dtype i) noexcept -> bool {return i != static_cast<dtype>(0); });
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
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<bool> returnArray = { std::any_of(cbegin(), cend(),
                        [](dtype i) noexcept -> bool {return i != static_cast<dtype>(0); }) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<bool> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = std::any_of(cbegin(row), cend(row),
                            [](dtype i) noexcept -> bool {return i != static_cast<dtype>(0); });
                    }
                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> arrayTransposed = transpose();
                    NdArray<bool> returnArray(1, arrayTransposed.shape_.rows);
                    for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
                    {
                        returnArray(0, row) = std::any_of(arrayTransposed.cbegin(row), arrayTransposed.cend(row),
                            [](dtype i) noexcept -> bool {return i != static_cast<dtype>(0); });
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
                    NdArray<uint32> returnArray = { static_cast<uint32>(std::max_element(cbegin(), cend()) - cbegin()) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<uint32> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = static_cast<uint32>(std::max_element(cbegin(row), cend(row)) - cbegin(row));
                    }
                    return returnArray;;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> arrayTransposed = transpose();
                    NdArray<uint32> returnArray(1, arrayTransposed.shape_.rows);
                    for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
                    {
                        returnArray(0, row) = static_cast<uint32>(std::max_element(arrayTransposed.cbegin(row),
                            arrayTransposed.cend(row)) - arrayTransposed.cbegin(row));
                    }
                    return returnArray;;
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
                    NdArray<uint32> returnArray = { static_cast<uint32>(std::min_element(cbegin(), cend()) - cbegin()) };
                    return returnArray;;
                }
                case Axis::COL:
                {
                    NdArray<uint32> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = static_cast<uint32>(std::min_element(cbegin(row), cend(row)) - cbegin(row));
                    }
                    return returnArray;;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> arrayTransposed = transpose();
                    NdArray<uint32> returnArray(1, arrayTransposed.shape_.rows);
                    for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
                    {
                        returnArray(0, row) = static_cast<uint32>(std::min_element(arrayTransposed.cbegin(row),
                            arrayTransposed.cend(row)) - arrayTransposed.cbegin(row));
                    }
                    return returnArray;;
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
                    std::stable_sort(idx.begin(), idx.end(),
                        [this](uint32 i1, uint32 i2) noexcept -> bool {return this->array_[i1] < this->array_[i2]; });
                    return NdArray<uint32>(idx);
                }
                case Axis::COL:
                {
                    NdArray<uint32> returnArray(shape_);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        std::vector<uint32> idx(shape_.cols);
                        std::iota(idx.begin(), idx.end(), 0);
                        std::stable_sort(idx.begin(), idx.end(),
                            [this, row](uint32 i1, uint32 i2) noexcept -> bool
                            {return operator()(row, i1) < operator()(row, i2); });

                        for (uint32 col = 0; col < shape_.cols; ++col)
                        {
                            returnArray(row, col) = idx[col];
                        }
                    }
                    return returnArray;;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> arrayTransposed = transpose();
                    NdArray<uint32> returnArray(shape_.cols, shape_.rows);
                    for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
                    {
                        std::vector<uint32> idx(arrayTransposed.shape_.cols);
                        std::iota(idx.begin(), idx.end(), 0);
                        std::stable_sort(idx.begin(), idx.end(),
                            [&arrayTransposed, row](uint32 i1, uint32 i2) noexcept -> bool
                            {return arrayTransposed(row, i1) < arrayTransposed(row, i2); });

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
            std::transform(cbegin(), cend(), outArray.begin(),
                [](dtype value) noexcept -> dtypeOut { return static_cast<dtypeOut>(value); });

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
        void byteswap() noexcept
        {
            switch (endianess_)
            {
                case Endian::BIG:
                {
                    *this = newbyteorder(Endian::LITTLE);
                    return;
                }
                case Endian::LITTLE:
                {
                    *this = newbyteorder(Endian::BIG);
                    return;
                }
                case Endian::NATIVE:
                {
#if BOOST_ENDIAN_BIG_BYTE
                    *this = newbyteorder(Endian::LITTLE);
#elif BOOST_ENDIAN_LITTLE_BYTE
                    *this = newbyteorder(Endian::BIG);
#endif
                    return;
                }
            }
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
                    NdArray<bool> returnArray = { std::find(cbegin(), cend(), inValue) != cend() };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<bool> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = std::find(cbegin(row), cend(row), inValue) != cend(row);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transArray = transpose();
                    NdArray<bool> returnArray(1, transArray.shape_.rows);
                    for (uint32 row = 0; row < transArray.shape_.rows; ++row)
                    {
                        returnArray(0, row) = std::find(transArray.cbegin(row), transArray.cend(row), inValue) != transArray.cend(row);
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
        dtype* data() noexcept
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
                dtype dotProduct = std::inner_product(cbegin(), cend(), inOtherArray.cbegin(), static_cast<dtype>(0));
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
                        returnArray(i, j) = std::inner_product(otherArrayT.cbegin(j), otherArrayT.cend(j), cbegin(i), static_cast<dtype>(0));
                    }
                }

                return returnArray;
            }
            else
            {
                std::string errStr = "ERROR: NdArray::Array shapes of [" + utils::num2str(shape_.rows) + ", " + utils::num2str(shape_.cols) + "]";
                errStr += " and [" + utils::num2str(inOtherArray.shape_.rows) + ", " + utils::num2str(inOtherArray.shape_.cols) + "]";
                errStr += " are not consistent.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }
        }

        //============================================================================
        // Method Description:
        ///						Dump a binary file of the array to the specified file.
        ///						The array can be read back with or NC::load.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.dump.html
        ///
        /// @param
        ///				inFilename
        ///
        void dump(const std::string& inFilename) const
        {
            boost::filesystem::path p(inFilename);
            if (!boost::filesystem::exists(p.parent_path()))
            {
                std::string errStr = "ERROR: NdArray::dump: Input path does not exist:\n\t" + p.parent_path().string();
                std::cerr << errStr << std::endl;
                throw std::runtime_error(errStr);
            }

            std::string ext = "";
            if (!p.has_extension())
            {
                ext += ".bin";
            }

            std::ofstream ofile((inFilename + ext).c_str(), std::ios::binary);
            ofile.write(reinterpret_cast<const char*>(array_), size_ * sizeof(dtype));
            ofile.close();
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
        void fill(dtype inFillValue) noexcept
        {
            std::fill(begin(), end(), inFillValue);
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
            std::copy(cbegin(), cend(), outArray.begin());
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
                std::string errStr = "ERROR: NdArray::item: Can only convert an array of size 1 to a C++ scalar";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }
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
                    NdArray<dtype> returnArray = { *std::max_element(cbegin(), cend()) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = *std::max_element(cbegin(row), cend(row));
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<dtype> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        returnArray(0, row) = *std::max_element(transposedArray.cbegin(row), transposedArray.cend(row));
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
                    NdArray<dtype> returnArray = { *std::min_element(cbegin(), cend()) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = *std::min_element(cbegin(row), cend(row));
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<dtype> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        returnArray(0, row) = *std::min_element(transposedArray.cbegin(row), transposedArray.cend(row));
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
        ///						Return the median along a given axis. Does NOT average
        ///						if array has even number of elements!
        ///
        /// @param
        ///				inAxis (Optional, default NONE)
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> median(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<dtype> copyArray(*this);

                    uint32 middle = size_ / 2;
                    std::nth_element(copyArray.begin(), copyArray.begin() + middle, copyArray.end());
                    NdArray<dtype> returnArray = { copyArray.array_[middle] };

                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> copyArray(*this);
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        uint32 middle = shape_.cols / 2;
                        std::nth_element(copyArray.begin(row), copyArray.begin(row) + middle, copyArray.end(row));
                        returnArray(0, row) = copyArray(row, middle);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<dtype> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        uint32 middle = transposedArray.shape_.cols / 2;
                        std::nth_element(transposedArray.begin(row), transposedArray.begin(row) + middle, transposedArray.end(row));
                        returnArray(0, row) = transposedArray(row, middle);
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
        void nans() noexcept
        {
            fill(constants::nan);
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
            // only works with integer types
            static_assert(DtypeInfo<dtype>::isInteger(), "Type Error in newbyteorder: Can only compile newbyteorder method of NdArray<T> with integer types.");

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
                            std::transform(cbegin(), end(), outArray.begin(), boost::endian::native_to_big<dtype>);
                            outArray.endianess_ = Endian::BIG;
                            return outArray;
                        }
                        case Endian::LITTLE:
                        {
                            NdArray<dtype> outArray(shape_);
                            std::transform(cbegin(), cend(), outArray.begin(), boost::endian::native_to_little<dtype>);
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
                            std::transform(cbegin(), cend(), outArray.begin(), boost::endian::big_to_native<dtype>);
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
                            std::transform(cbegin(), cend(), outArray.begin(),
                                [](dtype value) noexcept -> dtype
                                {return boost::endian::native_to_little<dtype>(boost::endian::big_to_native<dtype>(value)); });
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
                            std::transform(cbegin(), cend(), outArray.begin(), boost::endian::little_to_native<dtype>);
                            outArray.endianess_ = Endian::NATIVE;
                            return outArray;
                        }
                        case Endian::BIG:
                        {
                            NdArray<dtype> outArray(shape_);
                            std::transform(cbegin(), cend(), outArray.begin(),
                                [](dtype value) noexcept -> dtype
                                {return boost::endian::native_to_big<dtype>(boost::endian::little_to_native<dtype>(value)); });
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
        ///						Return the indices of the flattened array of the
        ///						elements that are non-zero.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.nonzero.html
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<uint32> nonzero() const noexcept
        {
            std::vector<uint32> indices;
            uint32 counter = 0;
            for (auto value : *this)
            {
                if (value != static_cast<dtype>(0))
                {
                    indices.push_back(counter);
                }
                ++counter;
            }

            return NdArray<uint32>(indices);
        }

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
        NdArray<dtype> norm(Axis inAxis = Axis::NONE) const noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    dtype sumOfSquares = 0;
                    for (auto value : *this)
                    {
                        sumOfSquares += utils::sqr(value);
                    }

                    NdArray<dtype> returnArray = { std::sqrt(sumOfSquares) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        dtype sumOfSquares = 0;
                        for (uint32 col = 0; col < shape_.cols; ++col)
                        {
                            sumOfSquares += utils::sqr(operator()(row, col));
                        }
                        returnArray(0, row) = std::sqrt(sumOfSquares);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<dtype> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        dtype sumOfSquares = 0;
                        for (uint32 col = 0; col < transposedArray.shape_.cols; ++col)
                        {
                            sumOfSquares += utils::sqr(transposedArray(row, col));
                        }
                        returnArray(0, row) = std::sqrt(sumOfSquares);
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
        ///						Fills the array with ones
        ///
        ///
        void ones() noexcept
        {
            fill(1);
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
        void partition(uint32 inKth, Axis inAxis = Axis::NONE)
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    if (inKth >= size_)
                    {
                        std::string errStr = "ERROR: NdArray::partition: kth(=" + utils::num2str(inKth);
                        errStr += ") out of bounds (" + utils::num2str(size_) + ")";
                        throw std::invalid_argument(errStr);
                    }
                    std::nth_element(begin(), begin() + inKth, end());
                    break;
                }
                case Axis::COL:
                {
                    if (inKth >= shape_.cols)
                    {
                        std::string errStr = "ERROR: NdArray::partition: kth(=" + utils::num2str(inKth);
                        errStr += ") out of bounds (" + utils::num2str(shape_.cols) + ")";
                        std::cerr << errStr << std::endl;
                        throw std::invalid_argument(errStr);
                    }

                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        std::nth_element(begin(row), begin(row) + inKth, end(row));
                    }
                    break;
                }
                case Axis::ROW:
                {
                    if (inKth >= shape_.rows)
                    {
                        std::string errStr = "ERROR: NdArray::partition: kth(=" + utils::num2str(inKth);
                        errStr += ") out of bounds (" + utils::num2str(shape_.rows) + ")";
                        std::cerr << errStr << std::endl;
                        throw std::invalid_argument(errStr);
                    }

                    NdArray<dtype> transposedArray = transpose();
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        std::nth_element(transposedArray.begin(row), transposedArray.begin(row) + inKth, transposedArray.end(row));
                    }
                    *this = transposedArray.transpose();
                    break;
                }
            }
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
                        static_cast<dtype>(1), std::multiplies<dtype>());
                    NdArray<dtype> returnArray = { product };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = std::accumulate(cbegin(row), cend(row),
                            static_cast<dtype>(1), std::multiplies<dtype>());
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
                            static_cast<dtype>(1), std::multiplies<dtype>());
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
                    const std::pair<const dtype*, const dtype*> result = std::minmax_element(cbegin(), cend());
                    NdArray<dtype> returnArray = { *result.second - *result.first };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        std::pair<const dtype*, const dtype*> result = std::minmax_element(cbegin(row), cend(row));
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
                        std::pair<const dtype*, const dtype*> result = std::minmax_element(transposedArray.cbegin(row), transposedArray.cend(row));
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
                std::string errStr = "ERROR: NdArray::put: Input indices do not match values dimensions.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
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
                std::string errStr = "ERROR: putMask: input inMask must be the same shape as the array it is masking.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            return put(inMask.nonzero(), inValue);
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
                std::string errStr = "ERROR: putMask: input inMask must be the same shape as the array it is masking.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            return put(inMask.nonzero(), inValues);
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
        ///						Resizes the array to a new shape.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.repeat.html
        ///
        /// @param      inNumRows
        /// @param      inNumCols
        ///
        void reshape(uint32 inNumRows, uint32 inNumCols)
        {
            if (inNumRows * inNumCols != size_)
            {
                std::string errStr = "ERROR: NdArray::reshape: Cannot reshape array of size " + utils::num2str(size_) + " into shape ";
                errStr += "[" + utils::num2str(inNumRows) + ", " + utils::num2str(inNumCols) + "]";
                std::cerr << errStr << std::endl;
                throw std::runtime_error(errStr);
            }

            shape_.rows = inNumRows;
            shape_.cols = inNumCols;
        }

        //============================================================================
        // Method Description:
        ///						Resizes the array to a new shape.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.reshape.html
        ///
        /// @param
        ///				inShape
        ///
        void reshape(const Shape& inShape)
        {
            reshape(inShape.rows, inShape.cols);
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
        void resizeFast(uint32 inNumRows, uint32 inNumCols) noexcept
        {
            newArray(Shape(inNumRows, inNumCols));
            zeros();
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
        void resizeFast(const Shape& inShape) noexcept
        {
            resizeFast(inShape.rows, inShape.cols);
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
        void resizeSlow(uint32 inNumRows, uint32 inNumCols) noexcept
        {
            std::vector<dtype> oldData(size_);
            std::copy(begin(), end(), oldData.begin());

            const Shape inShape(inNumRows, inNumCols);
            const Shape oldShape = shape_;

            newArray(inShape);

            for (uint32 row = 0; row < inShape.rows; ++row)
            {
                for (uint32 col = 0; col < inShape.cols; ++col)
                {
                    if (row >= oldShape.rows || col >= oldShape.cols)
                    {
                        operator()(row, col) = static_cast<dtype>(0); // zero fill
                    }
                    else
                    {
                        operator()(row, col) = oldData[row * oldShape.cols + col];
                    }
                }
            }
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
        void resizeSlow(const Shape& inShape) noexcept
        {
            resizeSlow(inShape.rows, inShape.cols);
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
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    double squareSum = 0;
                    std::for_each(cbegin(), cend(), [&squareSum](dtype value) noexcept -> void
                        { squareSum += utils::sqr(static_cast<double>(value)); });
                    NdArray<double> returnArray = { std::sqrt(squareSum / static_cast<double>(size_)) };

                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<double> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        double squareSum = 0;
                        std::for_each(cbegin(row), cend(row), [&squareSum](dtype value) noexcept -> void
                            { squareSum += utils::sqr(static_cast<double>(value)); });
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
                        double squareSum = 0;
                        std::for_each(transposedArray.cbegin(row), transposedArray.cend(row),
                            [&squareSum](dtype value) noexcept -> void
                            { squareSum += utils::sqr(static_cast<double>(value)); });
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
                for (uint32 i = 0; i < size_; ++i)
                {
                    returnArray[i] = static_cast<dtype>(std::nearbyint(static_cast<double>(array_[i]) * multFactor) / multFactor);
                }

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
        void sort(Axis inAxis = Axis::NONE) noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    std::sort(begin(), end());
                    break;
                }
                case Axis::COL:
                {
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        std::sort(begin(row), end(row));
                    }
                    break;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        std::sort(transposedArray.begin(row), transposedArray.end(row));
                    }
                    *this = transposedArray.transpose();
                    break;
                }
            }
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
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    double meanValue = mean(inAxis).item();
                    double sum = 0;
                    for (auto value : *this)
                    {
                        sum += utils::sqr(static_cast<double>(value) - meanValue);
                    }
                    NdArray<double> returnArray = { std::sqrt(sum / size_) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<double> meanValue = mean(inAxis);
                    NdArray<double> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        double sum = 0;
                        for (uint32 col = 0; col < shape_.cols; ++col)
                        {
                            sum += utils::sqr(static_cast<double>(operator()(row, col)) - meanValue[row]);
                        }
                        returnArray(0, row) = std::sqrt(sum / shape_.cols);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<double> meanValue = mean(inAxis);
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<double> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        double sum = 0;
                        for (uint32 col = 0; col < transposedArray.shape_.cols; ++col)
                        {
                            sum += utils::sqr(static_cast<double>(transposedArray(row, col)) - meanValue[row]);
                        }
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
                    NdArray<dtype> returnArray = { std::accumulate(cbegin(), cend(), static_cast<dtype>(0)) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = std::accumulate(cbegin(row), cend(row), static_cast<dtype>(0));
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
                        returnArray(0, row) = std::accumulate(transposedArray.cbegin(row), transposedArray.cend(row), static_cast<dtype>(0));
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
                boost::filesystem::path p(inFilename);
                if (!boost::filesystem::exists(p.parent_path()))
                {
                    std::string errStr = "ERROR: NdArray::tofile: Input path does not exist:\n\t" + p.parent_path().string();
                    std::cerr << errStr << std::endl;
                    throw std::runtime_error(errStr);
                }

                std::string ext = "";
                if (!p.has_extension())
                {
                    ext += ".txt";
                }

                std::ofstream ofile((inFilename + ext).c_str());
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
                return static_cast<dtype>(0);
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
            std::for_each(stdValues.begin(), stdValues.end(),
                [](double& value) noexcept -> void { value *= value; });
            return stdValues;
        }

        //============================================================================
        // Method Description:
        ///						Fills the array with zeros
        ///
        ///
        void zeros()
        {
            fill(0);
        }

        //============================================================================
        // Method Description:
        ///						Adds the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator+(const NdArray<dtype>& inOtherArray) const
        {
            return NdArray<dtype>(*this) += inOtherArray;
        }

        //============================================================================
        // Method Description:
        ///						Adds the scalar to the array
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator+(dtype inScalar) const noexcept
        {
            return NdArray<dtype>(*this) += inScalar;
        }

        //============================================================================
        // Method Description:
        ///						Adds the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator+=(const NdArray<dtype>& inOtherArray)
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator+=: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            std::transform(begin(), end(), inOtherArray.cbegin(), begin(), std::plus<dtype>());

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Adds the scalar to the array
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator+=(dtype inScalar) noexcept
        {
            std::for_each(begin(), end(),
                [=](dtype& value) noexcept -> dtype { return value += inScalar; });

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Subtracts the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator-(const NdArray<dtype>& inOtherArray) const
        {
            return NdArray<dtype>(*this) -= inOtherArray;
        }

        //============================================================================
        // Method Description:
        ///						Subtracts the scalar from the array
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator-(dtype inScalar) const noexcept
        {
            return NdArray<dtype>(*this) -= inScalar;
        }

        //============================================================================
        // Method Description:
        ///						Subtracts the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator-=(const NdArray<dtype>& inOtherArray)
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator-=: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            std::transform(begin(), end(), inOtherArray.cbegin(), begin(), std::minus<dtype>());

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Subtracts the scalar from the array
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator-=(dtype inScalar) noexcept
        {
            std::for_each(begin(), end(),
                [=](dtype& value) noexcept -> dtype { return value -= inScalar; });

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Multiplies the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator*(const NdArray<dtype>& inOtherArray) const
        {
            return NdArray<dtype>(*this) *= inOtherArray;
        }

        //============================================================================
        // Method Description:
        ///						Muliplies the scalar to the array
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator*(dtype inScalar) const noexcept
        {
            return NdArray<dtype>(*this) *= inScalar;
        }

        //============================================================================
        // Method Description:
        ///						Multiplies the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator*=(const NdArray<dtype>& inOtherArray)
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator*=: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            std::transform(begin(), end(), inOtherArray.cbegin(), begin(), std::multiplies<dtype>());

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Muliplies the scalar to the array
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator*=(dtype inScalar) noexcept
        {
            std::for_each(begin(), end(),
                [=](dtype& value) noexcept -> dtype { return value *= inScalar; });

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Divides the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator/(const NdArray<dtype>& inOtherArray) const
        {
            return NdArray<dtype>(*this) /= inOtherArray;
        }

        //============================================================================
        // Method Description:
        ///						Divides the array by the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator/(dtype inScalar) const noexcept
        {
            return NdArray<dtype>(*this) /= inScalar;
        }

        //============================================================================
        // Method Description:
        ///						Divides the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator/=(const NdArray<dtype>& inOtherArray)
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator/=: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            std::transform(begin(), end(), inOtherArray.cbegin(), begin(), std::divides<dtype>());

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Divides the array by the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator/=(dtype inScalar) noexcept
        {
            std::for_each(begin(), end(),
                [=](dtype& value) noexcept -> dtype { return value /= inScalar; });

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Takes the modulus of the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator%(const NdArray<dtype>& inOtherArray) const
        {
            return NdArray<dtype>(*this) %= inOtherArray;
        }

        //============================================================================
        // Method Description:
        ///						Modulus of the array and the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator%(dtype inScalar) const noexcept
        {
            return NdArray<dtype>(*this) %= inScalar;
        }

        //============================================================================
        // Method Description:
        ///						Takes the modulus of the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator%=(const NdArray<dtype>& inOtherArray)
        {
            // can only be called on integer types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: NdArray::% operator can only be compiled with integer types.");

            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator%=: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            std::transform(begin(), end(), inOtherArray.cbegin(), begin(), std::modulus<dtype>());

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Modulus of the array and the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator%=(dtype inScalar)
        {
            // can only be called on integer types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: NdArray::% operator can only be compiled with integer types.");

            if (inScalar == 0)
            {
                std::string errStr = "ERROR: NdArray::operator%=: modulus by zero.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            std::for_each(begin(), end(),
                [=](dtype& value) noexcept -> dtype { return value %= inScalar; });

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise or of the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator|(const NdArray<dtype>& inOtherArray) const
        {
            return NdArray<dtype>(*this) |= inOtherArray;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise or of the array and the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator|(dtype inScalar) const noexcept
        {
            return NdArray<dtype>(*this) |= inScalar;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise or of the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator|=(const NdArray<dtype>& inOtherArray)
        {
            // can only be called on integer types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: NdArray::| operator can only be compiled with integer types.");

            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator|=: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            std::transform(begin(), end(), inOtherArray.cbegin(), begin(), std::bit_or<dtype>());

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise or of the array and the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator|=(dtype inScalar) noexcept
        {
            // can only be called on integer types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: NdArray::| operator can only be compiled with integer types.");

            std::for_each(begin(), end(),
                [=](dtype& value) noexcept -> dtype { return value |= inScalar; });

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise and of the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator&(const NdArray<dtype>& inOtherArray) const
        {
            return NdArray<dtype>(*this) &= inOtherArray;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise and of the array and the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator&(dtype inScalar) const noexcept
        {
            return NdArray<dtype>(*this) &= inScalar;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise and of the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator&=(const NdArray<dtype>& inOtherArray)
        {
            // can only be called on integer types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: NdArray::& operator can only be compiled with integer types.");

            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator&=: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            std::transform(begin(), end(), inOtherArray.cbegin(), begin(), std::bit_and<dtype>());

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise and of the array and the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator&=(dtype inScalar) noexcept
        {
            // can only be called on integer types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: NdArray::& operator can only be compiled with integer types.");

            std::for_each(begin(), end(),
                [=](dtype& value) noexcept -> dtype { return value &= inScalar; });

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise xor of the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				None
        ///
        NdArray<dtype> operator^(const NdArray<dtype>& inOtherArray) const
        {
            return NdArray<dtype>(*this) ^= inOtherArray;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise xor of the array and the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator^(dtype inScalar) const noexcept
        {
            return NdArray<dtype>(*this) ^= inScalar;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise xor of the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				None
        ///
        NdArray<dtype>& operator^=(const NdArray<dtype>& inOtherArray)
        {
            // can only be called on integer types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: NdArray::^ operator can only be compiled with integer types.");

            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator^=: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            std::transform(begin(), end(), inOtherArray.cbegin(), begin(), std::bit_xor<dtype>());

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise xor of the array and the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator^=(dtype inScalar) noexcept
        {
            // can only be called on integer types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: NdArray::^ operator can only be compiled with integer types.");

            std::for_each(begin(), end(),
                [=](dtype& value) noexcept -> dtype { return value ^= inScalar; });

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Takes the and of the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				None
        ///
        NdArray<dtype> operator&&(const NdArray<dtype>& inOtherArray) const
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator&&: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            NdArray<dtype> returnArray(shape_);
            std::transform(cbegin(), cend(), inOtherArray.cbegin(), returnArray.begin(),
                [](dtype value1, dtype value2) noexcept -> dtype { return value1 && value2; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Takes the and of the array and the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator&&(dtype inScalar) const noexcept
        {
            NdArray<dtype> returnArray(shape_);
            std::transform(cbegin(), cend(), returnArray.begin(),
                [inScalar](dtype value) noexcept -> dtype { return value && inScalar; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Takes the or of the elements of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				None
        ///
        NdArray<dtype> operator||(const NdArray<dtype>& inOtherArray) const
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator||: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            NdArray<dtype> returnArray(shape_);
            std::transform(cbegin(), cend(), inOtherArray.cbegin(), returnArray.begin(),
                [](dtype value1, dtype value2) noexcept -> dtype { return value1 || value2; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Takes the or of the array and the scalar
        ///
        /// @param
        ///				inScalar
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator||(dtype inScalar) const noexcept
        {
            NdArray<dtype> returnArray(shape_);
            std::transform(cbegin(), cend(), returnArray.begin(),
                [inScalar](dtype value) noexcept -> dtype { return value || inScalar; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Takes the bitwise not of the array
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator~() const noexcept
        {
            // can only be called on integer types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: NdArray::~ operator can only be compiled with integer types.");

            NdArray<dtype> returnArray(shape_);
            std::transform(cbegin(), cend(), returnArray.begin(),
                [](dtype value) noexcept -> dtype { return ~value; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Takes the not of the array
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator!() const noexcept
        {
            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), returnArray.begin(),
                [](dtype value) noexcept -> bool { return !value; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						of two arrays
        ///
        /// @param
        ///				inValue
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator==(dtype inValue) const noexcept
        {
            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), returnArray.begin(),
                [inValue](dtype value) noexcept -> bool { return value == inValue; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator==(const NdArray<dtype>& inOtherArray) const
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator==: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), inOtherArray.cbegin(), returnArray.begin(), std::equal_to<dtype>());

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						of two arrays
        ///
        /// @param
        ///				inValue
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator!=(dtype inValue) const noexcept
        {
            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), returnArray.begin(),
                [inValue](dtype value) noexcept -> bool { return value != inValue; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator!=(const NdArray<dtype>& inOtherArray) const
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator!=: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), inOtherArray.cbegin(), returnArray.begin(), std::not_equal_to<dtype>());

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						the array and a scalar
        ///
        /// @param
        ///				inValue
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator<(dtype inValue) const noexcept
        {
            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), returnArray.begin(),
                [inValue](dtype value) noexcept -> bool { return value < inValue; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator<(const NdArray<dtype>& inOtherArray) const
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator<: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), inOtherArray.cbegin(), returnArray.begin(), std::less<dtype>());

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						the array and a scalar
        ///
        /// @param
        ///				inValue
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator>(dtype inValue) const noexcept
        {
            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), returnArray.begin(),
                [inValue](dtype value) noexcept -> bool { return value > inValue; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator>(const NdArray<dtype>& inOtherArray) const
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator>: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), inOtherArray.cbegin(), returnArray.begin(), std::greater<dtype>());

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						the array and a scalar
        ///
        /// @param
        ///				inValue
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator<=(dtype inValue) const noexcept
        {
            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), returnArray.begin(),
                [inValue](dtype value) noexcept -> bool { return value <= inValue; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator<=(const NdArray<dtype>& inOtherArray) const
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator<=: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), inOtherArray.cbegin(), returnArray.begin(), std::less_equal<dtype>());

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						the array and a scalar
        ///
        /// @param
        ///				inValue
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator>=(dtype inValue) const noexcept
        {
            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), returnArray.begin(),
                [inValue](dtype value) noexcept -> bool { return value >= inValue; });

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns an array of booleans of element wise comparison
        ///						of two arrays
        ///
        /// @param
        ///				inOtherArray
        /// @return
        ///				NdArray
        ///
        NdArray<bool> operator>=(const NdArray<dtype>& inOtherArray) const
        {
            if (shape_ != inOtherArray.shape_)
            {
                std::string errStr = "ERROR: NdArray::operator>=: Array dimensions do not match.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            NdArray<bool> returnArray(shape_);
            std::transform(cbegin(), cend(), inOtherArray.cbegin(), returnArray.begin(), std::greater_equal<dtype>());

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Bitshifts left the elements of the array
        ///
        /// @param      lhs
        /// @param      inNumBits
        /// @return
        ///				NdArray
        ///
        friend NdArray<dtype> operator<<(const NdArray<dtype>& lhs, uint8 inNumBits) noexcept
        {
            NdArray<dtype> returnArray(lhs);
            returnArray <<= inNumBits;
            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Bitshifts left the elements of the array
        ///
        /// @param      lhs
        /// @param      inNumBits
        /// @return
        ///				NdArray
        ///
        friend NdArray<dtype>& operator<<=(NdArray<dtype>& lhs, uint8 inNumBits) noexcept
        {
            std::for_each(lhs.begin(), lhs.end(),
                [inNumBits](dtype& value) noexcept -> void { value <<= inNumBits;  });

            return lhs;
        }

        //============================================================================
        // Method Description:
        ///						Bitshifts right the elements of the array
        ///
        /// @param      lhs
        /// @param      inNumBits
        /// @return
        ///				NdArray
        ///
        friend NdArray<dtype> operator>>(const NdArray<dtype>& lhs, uint8 inNumBits) noexcept
        {
            NdArray<dtype> returnArray(lhs);
            returnArray >>= inNumBits;
            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Bitshifts right the elements of the array
        ///
        /// @param      lhs
        /// @param      inNumBits
        /// @return
        ///				NdArray
        ///
        friend NdArray<dtype>& operator>>=(NdArray<dtype>& lhs, uint8 inNumBits) noexcept
        {
            std::for_each(lhs.begin(), lhs.end(),
                [inNumBits](dtype& value) noexcept -> void { value >>= inNumBits;  });

            return lhs;
        }

        //============================================================================
        // Method Description:
        ///						prefix incraments the elements of an array
        ///
        /// @return
        ///				NdArray
        ///

        NdArray<dtype>& operator++() noexcept
        {
            std::for_each(begin(), end(), [](dtype& value) noexcept -> void { ++value; });

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						prefix decrements the elements of an array
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<dtype>& operator--() noexcept
        {
            std::for_each(begin(), end(), [](dtype& value) noexcept -> void { --value; });

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						postfix increments the elements of an array
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator++(int) noexcept
        {
            NdArray<dtype> copy(*this);
            std::for_each(begin(), end(), [](dtype& value) noexcept -> void { ++value; });

            return copy;
        }

        //============================================================================
        // Method Description:
        ///						postfix decrements the elements of an array
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> operator--(int) noexcept
        {
            NdArray<dtype> copy(*this);
            std::for_each(begin(), end(), [](dtype& value) noexcept -> void { --value; });

            return copy;
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
        friend std::ostream& operator<<(std::ostream& inOStream, const NdArray<dtype>& inArray) noexcept
        {
            inOStream << inArray.str();
            return inOStream;
        }
    };
}
