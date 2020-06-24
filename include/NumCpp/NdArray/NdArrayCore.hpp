/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.0.0
///
/// @section License
/// Copyright 2020 David Pilger
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
#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/Filesystem.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Internal/TypeTraits.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray/NdArrayIterators.hpp"
#include "NumCpp/Utils/num2str.hpp"
#include "NumCpp/Utils/power.hpp"
#include "NumCpp/Utils/sqr.hpp"
#include "NumCpp/Utils/value2str.hpp"

#include <boost/algorithm/clamp.hpp>
#include <boost/endian/conversion.hpp>
#include <boost/predef/other/endian.h>

#include <array>
#include <cmath>
#include <deque>
#include <forward_list>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
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
    ///	Holds 1D and 2D arrays, the main work horse of the NumCpp library
    template<typename dtype, class Allocator = std::allocator<dtype>>
    class NdArray
    {
    private:
        STATIC_ASSERT_VALID_DTYPE(dtype);
        static_assert(is_same_v<dtype, typename Allocator::value_type>, "value_type and Allocator::value_type must match");

        using AllocType     = typename std::allocator_traits<Allocator>::template rebind_alloc<dtype>;
        using AllocTraits   = std::allocator_traits<AllocType>;

    public:
        using value_type                    = dtype;
        using allocator_type                = Allocator;
        using pointer                       = typename AllocTraits::pointer;
        using const_pointer                 = typename AllocTraits::const_pointer;
        using reference                     = dtype&;
        using const_reference               = const dtype&;
        using size_type                     = uint32;
        using difference_type               = typename AllocTraits::difference_type;

        using iterator                      = NdArrayIterator<dtype, pointer, difference_type>;
        using const_iterator                = NdArrayConstIterator<dtype, const_pointer, difference_type>;
        using reverse_iterator              = std::reverse_iterator<iterator>;
        using const_reverse_iterator        = std::reverse_iterator<const_iterator>;

        using column_iterator               = NdArrayColumnIterator<dtype, size_type, pointer, difference_type>;
        using const_column_iterator         = NdArrayConstColumnIterator<dtype, size_type, const_pointer, difference_type>;
        using reverse_column_iterator       = std::reverse_iterator<column_iterator>;
        using const_reverse_column_iterator = std::reverse_iterator<const_column_iterator>;

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
        explicit NdArray(size_type inSquareSize) :
            shape_(inSquareSize, inSquareSize),
            size_(inSquareSize * inSquareSize)
        {
            newArray();
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param				inNumRows
        /// @param				inNumCols
        ///
        NdArray(size_type inNumRows, size_type inNumCols) :
            shape_(inNumRows, inNumCols),
            size_(inNumRows * inNumCols)
        {
            newArray();
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param
        ///				inShape
        ///
        explicit NdArray(const Shape& inShape) :
            shape_(inShape),
            size_(shape_.size())
        {
            newArray();
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param
        ///				inList
        ///
        NdArray(const std::initializer_list<dtype>& inList) :
            shape_(1, static_cast<uint32>(inList.size())),
            size_(shape_.size())
        {
            newArray();
            if (size_ > 0)
            {
                stl_algorithms::copy(inList.begin(), inList.end(), begin());
            }
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
            for (const auto& list : inList)
            {
                if (shape_.cols == 0)
                {
                    shape_.cols = static_cast<uint32>(list.size());
                }
                else if (list.size() != shape_.cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("All rows of the initializer list needs to have the same number of elements");
                }
            }

            size_ = shape_.size();
            newArray();
            uint32 row = 0;
            for (const auto& list : inList)
            {
                const auto ptr = begin() += row * shape_.cols;
                stl_algorithms::copy(list.begin(), list.end(), ptr);
                ++row;
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param      inArray
        ///	@param      copy: (optional) boolean for whether to make a copy and own the data, or 
        ///                   act as a non-owning shell. Default true.
        ///
        template<size_t ArraySize, 
            std::enable_if_t<is_valid_dtype_v<dtype>, int> = 0>
        NdArray(std::array<dtype, ArraySize>& inArray, bool copy = true) :
            shape_(1, static_cast<uint32>(ArraySize)),
            size_(shape_.size())
        {
            if (copy)
            {
                newArray();
                if (size_ > 0)
                {
                    stl_algorithms::copy(inArray.begin(), inArray.end(), begin());
                }
            }
            else
            {
                array_ = inArray.data();
                ownsPtr_ = false;
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param      in2dArray
        ///	@param      copy: (optional) boolean for whether to make a copy and own the data, or 
        ///                   act as a non-owning shell. Default true.
        ///
        template<size_t Dim0Size, size_t Dim1Size>
        NdArray(std::array<std::array<dtype, Dim1Size>, Dim0Size>& in2dArray, bool copy = true) :
            shape_(static_cast<uint32>(Dim0Size), static_cast<uint32>(Dim1Size)),
            size_(shape_.size())
        {
            if (copy)
            {
                newArray();
                if (size_ > 0)
                {
                    const auto start = in2dArray.front().begin();
                    stl_algorithms::copy(start, start + size_, begin());
                }
            }
            else
            {
                array_ = in2dArray.front().data();
                ownsPtr_ = false;
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param      inVector
        ///	@param      copy: (optional) boolean for whether to make a copy and own the data, or 
        ///                   act as a non-owning shell. Default true.
        ///
        template<std::enable_if_t<is_valid_dtype_v<dtype>, int> = 0>
        NdArray(std::vector<dtype>& inVector, bool copy = true) :
            shape_(1, static_cast<uint32>(inVector.size())),
            size_(shape_.size())
        {
            if (copy)
            {
                newArray();
                if (size_ > 0)
                {
                    stl_algorithms::copy(inVector.begin(), inVector.end(), begin());
                }
            }
            else
            {
                array_ = inVector.data();
                ownsPtr_ = false;
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param      in2dVector
        ///
        explicit NdArray(const std::vector<std::vector<dtype>>& in2dVector) :
            shape_(static_cast<uint32>(in2dVector.size()), 0)
        {
            for (const auto& row : in2dVector)
            {
                if (shape_.cols == 0)
                {
                    shape_.cols = static_cast<uint32>(row.size());
                }
                else if (row.size() != shape_.cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("All rows of the 2d vector need to have the same number of elements");
                }
            }

            size_ = shape_.size();

            newArray();
            auto currentPosition = begin();
            for (const auto& row : in2dVector)
            {
                stl_algorithms::copy(row.begin(), row.end(), currentPosition);
                currentPosition += shape_.cols;
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param      in2dArray
        ///	@param      copy: (optional) boolean for whether to make a copy and own the data, or 
        ///                   act as a non-owning shell. Default true.
        ///
        template<size_t Dim1Size>
        NdArray(std::vector<std::array<dtype, Dim1Size>>& in2dArray, bool copy = true) :
            shape_(static_cast<uint32>(in2dArray.size()), static_cast<uint32>(Dim1Size)),
            size_(shape_.size())
        {
            if (copy)
            {
                newArray();
                if (size_ > 0)
                {
                    const auto start = in2dArray.front().begin();
                    stl_algorithms::copy(start, start + size_, begin());
                }
            }
            else
            {
                array_ = in2dArray.front().data();
                ownsPtr_ = false;
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param      inDeque
        ///
        template<std::enable_if_t<is_valid_dtype_v<dtype>, int> = 0>
        explicit NdArray(const std::deque<dtype>& inDeque) :
            shape_(1, static_cast<uint32>(inDeque.size())),
            size_(shape_.size())
        {
            newArray();
            if (size_ > 0)
            {
                stl_algorithms::copy(inDeque.begin(), inDeque.end(), begin());
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param      in2dDeque
        ///
        explicit NdArray(const std::deque<std::deque<dtype>>& in2dDeque) :
            shape_(static_cast<uint32>(in2dDeque.size()), 0)
        {
            for (const auto& row : in2dDeque)
            {
                if (shape_.cols == 0)
                {
                    shape_.cols = static_cast<uint32>(row.size());
                }
                else if (row.size() != shape_.cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("All rows of the 2d vector need to have the same number of elements");
                }
            }

            size_ = shape_.size();

            newArray();
            auto currentPosition = begin();
            for (const auto& row : in2dDeque)
            {
                stl_algorithms::copy(row.begin(), row.end(), currentPosition);
                currentPosition += shape_.cols;
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param
        ///				inList
        ///
        explicit NdArray(const std::list<dtype>& inList) :
            shape_(1, static_cast<uint32>(inList.size())),
            size_(shape_.size())
        {
            newArray();
            if (size_ > 0)
            {
                stl_algorithms::copy(inList.begin(), inList.end(), begin());
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param				inFirst
        /// @param				inLast
        ///
        template<typename Iterator,
            std::enable_if_t<std::is_same<typename std::iterator_traits<Iterator>::value_type, dtype>::value, int> = 0>
        NdArray(Iterator inFirst, Iterator inLast) :
            shape_(1, static_cast<uint32>(std::distance(inFirst, inLast))),
            size_(shape_.size())
        {
            newArray();
            if (size_ > 0)
            {
                stl_algorithms::copy(inFirst, inLast, begin());
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor.  Copies the contents of the buffer into 
        ///                     the array.
        ///
        /// @param				inPtr: const_pointer to beginning of buffer
        /// @param				size: number of elements in buffer
        ///
        NdArray(const_pointer inPtr, size_type size) :
            shape_(1, size),
            size_(size)
        {
            newArray();
            if (inPtr != nullptr && size_ > 0)
            {
                stl_algorithms::copy(inPtr, inPtr + size_, begin());
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor.  Copies the contents of the buffer into 
        ///                     the array.
        ///
        /// @param				inPtr: const_pointer to beginning of buffer
        /// @param				numRows: number of rows of the buffer
        /// @param				numCols: number of cols of the buffer
        ///
        NdArray(const_pointer inPtr, uint32 numRows, uint32 numCols) :
            shape_(numRows, numCols),
            size_(shape_.size())
        {
            newArray();
            if (inPtr != nullptr && size_ > 0)
            {
                stl_algorithms::copy(inPtr, inPtr + size_, begin());
            }
        }

        //============================================================================
        // Method Description:
        ///						Constructor. Operates as a shell around an already existing
        ///                     array of data.
        ///
        /// @param				inPtr: pointer to beginning of the array
        /// @param				size: the number of elements in the array
        /// @param              takeOwnership: whether or not to take ownership of the data
        ///                     and call delete[] in the destructor.
        ///
        template<typename Bool,
            std::enable_if_t<std::is_same<Bool, bool>::value, int> = 0>
        NdArray(pointer inPtr, size_type size, Bool takeOwnership) noexcept :
            shape_(1, size),
            size_(size),
            array_(inPtr),
            ownsPtr_(takeOwnership)
        {}

        //============================================================================
        // Method Description:
        ///						Constructor. Operates as a shell around an already existing
        ///                     array of data.
        ///
        /// @param				inPtr: pointer to beginning of the array
        /// @param				numRows: the number of rows in the array
        /// @param              numCols: the nubmer of column in the array
        /// @param              takeOwnership: whether or not to take ownership of the data
        ///                     and call delete[] in the destructor.
        ///
        template<typename Bool,
            std::enable_if_t<std::is_same<Bool, bool>::value, int> = 0>
        NdArray(pointer inPtr, uint32 numRows, uint32 numCols, Bool takeOwnership) noexcept :
            shape_(numRows, numCols),
            size_(numRows * numCols),
            array_(inPtr),
            ownsPtr_(takeOwnership)
        {}

        //============================================================================
        // Method Description:
        ///						Copy Constructor
        ///
        /// @param
        ///				inOtherArray
        ///
        NdArray(const NdArray<dtype>& inOtherArray) :
            shape_(inOtherArray.shape_),
            size_(inOtherArray.size_),
            endianess_(inOtherArray.endianess_)
        {
            newArray();
            if (size_ > 0)
            {
                stl_algorithms::copy(inOtherArray.cbegin(), inOtherArray.cend(), begin());
            }
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
            ownsPtr_(inOtherArray.ownsPtr_)
        {
            inOtherArray.shape_.rows = inOtherArray.shape_.cols = 0;
            inOtherArray.size_ = 0;
            inOtherArray.ownsPtr_ = false;
            inOtherArray.array_ = nullptr;
        }

        //============================================================================
        // Method Description:
        ///						Destructor
        ///
        ~NdArray() noexcept
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
        NdArray<dtype>& operator=(const NdArray<dtype>& rhs)
        {
            if (&rhs != this)
            {
                if (rhs.size_ > 0)
                {
                    newArray(rhs.shape_);
                    endianess_ = rhs.endianess_;

                    stl_algorithms::copy(rhs.cbegin(), rhs.cend(), begin());
                }
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Assignment operator, sets the entire array to a single
        ///                     scaler value.
        ///
        /// @param
        ///				inValue
        /// @return
        ///				NdArray<dtype>
        ///
        NdArray<dtype>& operator=(value_type inValue) noexcept
        {
            if (array_ != nullptr)
            {
                stl_algorithms::fill(begin(), end(), inValue);
            }

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
        reference operator[](int32 inIndex) noexcept 
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
        const_reference operator[](int32 inIndex) const noexcept 
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
        reference operator()(int32 inRowIndex, int32 inColIndex) noexcept 
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
        const_reference operator()(int32 inRowIndex, int32 inColIndex) const noexcept 
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
            for (size_type i = 0; i < indices.size(); ++i)
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
        NdArray<dtype> operator[](const NdArray<size_type>& inIndices) const
        {
            if (inIndices.max().item() > size_ - 1)
            {
                THROW_INVALID_ARGUMENT_ERROR("input indices must be less than the array size.");
            }

            auto outArray = NdArray<dtype>(1, static_cast<size_type>(inIndices.size()));
            size_type i = 0;
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
        Slice cSlice(int32 inStartIdx = 0, uint32 inStepSize = 1) const noexcept
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
        Slice rSlice(int32 inStartIdx = 0, uint32 inStepSize = 1) const noexcept
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
        reference at(int32 inIndex)
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
        const_reference at(int32 inIndex) const
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
        reference at(int32 inRowIndex, int32 inColIndex)
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
        const_reference at(int32 inRowIndex, int32 inColIndex) const
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
        ///						iterator to the beginning of the flattened array
        /// @return
        ///				iterator
        ///
        iterator begin() noexcept 
        {
            return iterator(array_);
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
        iterator begin(size_type inRow)
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return begin() += (inRow * shape_.cols);
        }

        //============================================================================
        // Method Description:
        ///						const iterator to the beginning of the flattened array
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
        const_iterator begin(size_type inRow) const
        {
            return cbegin(inRow);
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
            return const_iterator(array_);
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
        const_iterator cbegin(size_type inRow) const
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return cbegin() += (inRow * shape_.cols);
        }

        //============================================================================
        // Method Description:
        ///						column_iterator to the beginning of the flattened array
        /// @return
        ///				column_iterator
        ///
        column_iterator colbegin() noexcept 
        {
            return column_iterator(array_, shape_.rows, shape_.cols);
        }

        //============================================================================
        // Method Description:
        ///						column_iterator to the beginning of the input column
        ///
        /// @param
        ///				inCol
        /// @return
        ///				column_iterator
        ///
        column_iterator colbegin(size_type inCol)
        {
            if (inCol >= shape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return colbegin() += (inCol * shape_.rows);
        }

        //============================================================================
        // Method Description:
        ///						const column_iterator to the beginning of the flattened array
        /// @return
        ///				const_column_iterator
        ///
        const_column_iterator colbegin() const noexcept 
        {
            return ccolbegin();
        }

        //============================================================================
        // Method Description:
        ///						const column_iterator to the beginning of the input column
        ///
        /// @param
        ///				inCol
        /// @return
        ///				const_column_iterator
        ///
        const_column_iterator colbegin(size_type inCol) const
        {
            return ccolbegin(inCol);
        }

        //============================================================================
        // Method Description:
        ///						const_column_iterator to the beginning of the flattened array
        ///
        /// @return
        ///				const_column_iterator
        ///
        const_column_iterator ccolbegin() const noexcept 
        {
            return const_column_iterator(array_, shape_.rows, shape_.cols);
        }

        //============================================================================
        // Method Description:
        ///						const_column_iterator to the beginning of the input column
        ///
        /// @param
        ///				inCol
        /// @return
        ///				const_column_iterator
        ///
        const_column_iterator ccolbegin(size_type inCol) const
        {
            if (inCol >= shape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return ccolbegin() += (inCol * shape_.rows);
        }

        //============================================================================
        // Method Description:
        ///						reverse_iterator to the beginning of the flattened array
        /// @return
        ///				reverse_iterator
        ///
        reverse_iterator rbegin() noexcept
        {
            return reverse_iterator(end());
        }

        //============================================================================
        // Method Description:
        ///						reverse_iterator to the beginning of the input row
        ///
        /// @param
        ///				inRow
        /// @return
        ///				reverse_iterator
        ///
        reverse_iterator rbegin(size_type inRow)
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return rbegin() += (shape_.rows - inRow - 1) * shape_.cols;
        }

        //============================================================================
        // Method Description:
        ///						const iterator to the beginning of the flattened array
        /// @return
        ///				const_iterator
        ///
        const_reverse_iterator rbegin() const noexcept
        {
            return crbegin();
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
        const_reverse_iterator rbegin(size_type inRow) const
        {
            return crbegin(inRow);
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_iterator to the beginning of the flattened array
        ///
        /// @return
        ///				const_reverse_iterator
        ///
        const_reverse_iterator crbegin() const noexcept
        {
            return const_reverse_iterator(cend());
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_iterator to the beginning of the input row
        ///
        /// @param
        ///				inRow
        /// @return
        ///				const_reverse_iterator
        ///
        const_reverse_iterator crbegin(size_type inRow) const
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return crbegin() += (shape_.rows - inRow - 1) * shape_.cols;
        }

        //============================================================================
        // Method Description:
        ///						reverse_column_iterator to the beginning of the flattened array
        /// @return
        ///				reverse_column_iterator
        ///
        reverse_column_iterator rcolbegin() noexcept
        {
            return reverse_column_iterator(colend());
        }

        //============================================================================
        // Method Description:
        ///						reverse_column_iterator to the beginning of the input column
        ///
        /// @param
        ///				inCol
        /// @return
        ///				reverse_column_iterator
        ///
        reverse_column_iterator rcolbegin(size_type inCol)
        {
            if (inCol >= shape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return rcolbegin() += (shape_.cols - inCol - 1) * shape_.rows;
        }

        //============================================================================
        // Method Description:
        ///						const iterator to the beginning of the flattened array
        /// @return
        ///				const_iterator
        ///
        const_reverse_column_iterator rcolbegin() const noexcept
        {
            return crcolbegin();
        }

        //============================================================================
        // Method Description:
        ///						const iterator to the beginning of the input column
        ///
        /// @param
        ///				inCol
        /// @return
        ///				const_iterator
        ///
        const_reverse_column_iterator rcolbegin(size_type inCol) const
        {
            return crcolbegin(inCol);
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_column_iterator to the beginning of the flattened array
        ///
        /// @return
        ///				const_reverse_column_iterator
        ///
        const_reverse_column_iterator crcolbegin() const noexcept
        {
            return const_reverse_column_iterator(ccolend());
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_column_iterator to the beginning of the input column
        ///
        /// @param
        ///				inCol
        /// @return
        ///				const_reverse_column_iterator
        ///
        const_reverse_column_iterator crcolbegin(size_type inCol) const
        {
            if (inCol >= shape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return crcolbegin() += (shape_.cols - inCol - 1) * shape_.rows;
        }

        //============================================================================
        // Method Description:
        ///						iterator to 1 past the end of the flattened array
        /// @return
        ///				iterator
        ///
        iterator end() noexcept 
        {
            return begin() += size_;
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
        iterator end(size_type inRow)
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return begin(inRow) += shape_.cols;
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
        const_iterator end(size_type inRow) const
        {
            return cend(inRow);
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
            return cbegin() += size_;
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
        const_iterator cend(size_type inRow) const
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return cbegin(inRow) += shape_.cols;
        }

        //============================================================================
        // Method Description:
        ///						reverse_iterator to 1 past the end of the flattened array
        /// @return
        ///				reverse_iterator
        ///
        reverse_iterator rend() noexcept
        {
            return rbegin() += size_;
        }

        //============================================================================
        // Method Description:
        ///						reverse_iterator to the 1 past end of the row
        ///
        /// @param
        ///				inRow
        /// @return
        ///				reverse_iterator
        ///
        reverse_iterator rend(size_type inRow)
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return rbegin(inRow) += shape_.cols;
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_iterator to 1 past the end of the flattened array
        /// @return
        ///				const_reverse_iterator
        ///
        const_reverse_iterator rend() const noexcept
        {
            return crend();
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_iterator to the 1 past end of the row
        ///
        /// @param
        ///				inRow
        /// @return
        ///				const_reverse_iterator
        ///
        const_reverse_iterator rend(size_type inRow) const
        {
            return crend(inRow);
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_iterator to 1 past the end of the flattened array
        ///
        /// @return
        ///				const_reverse_iterator
        ///
        const_reverse_iterator crend() const noexcept
        {
            return crbegin() += size_;
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_iterator to 1 past the end of the input row
        ///
        /// @param
        ///				inRow
        /// @return
        ///				const_reverse_iterator
        ///
        const_reverse_iterator crend(size_type inRow) const
        {
            if (inRow >= shape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return crbegin(inRow) += shape_.cols;
        }

        //============================================================================
        // Method Description:
        ///						column_iterator to 1 past the end of the flattened array
        /// @return
        ///				column_iterator
        ///
        column_iterator colend() noexcept 
        {
            return colbegin() += size_;
        }

        //============================================================================
        // Method Description:
        ///						column_iterator to the 1 past end of the column
        ///
        /// @param
        ///				inCol
        /// @return
        ///				column_iterator
        ///
        column_iterator colend(size_type inCol)
        {
            if (inCol >= shape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return colbegin(inCol) += shape_.rows;
        }

        //============================================================================
        // Method Description:
        ///						const column_iterator to 1 past the end of the flattened array
        /// @return
        ///				const_column_iterator
        ///
        const_column_iterator colend() const noexcept 
        {
            return ccolend();
        }

        //============================================================================
        // Method Description:
        ///						const column_iterator to the 1 past end of the column
        ///
        /// @param
        ///				inCol
        /// @return
        ///				const_column_iterator
        ///
        const_column_iterator colend(size_type inCol) const
        {
            return ccolend(inCol);
        }

        //============================================================================
        // Method Description:
        ///						const_column_iterator to 1 past the end of the flattened array
        ///
        /// @return
        ///				const_column_iterator
        ///
        const_column_iterator ccolend() const noexcept 
        {
            return ccolbegin() += size_;
        }

        //============================================================================
        // Method Description:
        ///						const_column_iterator to 1 past the end of the input col
        ///
        /// @param
        ///				inCol
        /// @return
        ///				const_column_iterator
        ///
        const_column_iterator ccolend(size_type inCol) const
        {
            if (inCol >= shape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return ccolbegin(inCol) += shape_.rows;
        }

        //============================================================================
        // Method Description:
        ///						reverse_column_iterator to 1 past the end of the flattened array
        /// @return
        ///				reverse_column_iterator
        ///
        reverse_column_iterator rcolend() noexcept
        {
            return rcolbegin() += size_;
        }

        //============================================================================
        // Method Description:
        ///						reverse_column_iterator to the 1 past end of the column
        ///
        /// @param
        ///				inCol
        /// @return
        ///				reverse_column_iterator
        ///
        reverse_column_iterator rcolend(size_type inCol)
        {
            if (inCol >= shape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return rcolbegin(inCol) += shape_.rows;
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_column_iterator to 1 past the end of the flattened array
        /// @return
        ///				const_reverse_column_iterator
        ///
        const_reverse_column_iterator rcolend() const noexcept
        {
            return crcolend();
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_column_iterator to the 1 past end of the column
        ///
        /// @param
        ///				inCol
        /// @return
        ///				const_reverse_column_iterator
        ///
        const_reverse_column_iterator rcolend(size_type inCol) const
        {
            return crcolend(inCol);
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_column_iterator to 1 past the end of the flattened array
        ///
        /// @return
        ///				const_reverse_column_iterator
        ///
        const_reverse_column_iterator crcolend() const noexcept
        {
            return crcolbegin() += size_;
        }

        //============================================================================
        // Method Description:
        ///						const_reverse_column_iterator to 1 past the end of the input col
        ///
        /// @param
        ///				inCol
        /// @return
        ///				const_reverse_column_iterator
        ///
        const_reverse_column_iterator crcolend(size_type inCol) const
        {
            if (inCol >= shape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return crcolbegin(inCol) += shape_.rows;
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
        NdArray<bool> all(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto function = [](dtype i)  -> bool
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
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        NdArray<bool> any(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto function = [](dtype i)  -> bool
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
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        NdArray<uint32> argmax(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool
            {
                return lhs < rhs;
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<uint32> returnArray = { static_cast<uint32>(stl_algorithms::max_element(cbegin(), 
                        cend(), comparitor) - cbegin()) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<uint32> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = static_cast<uint32>(stl_algorithms::max_element(cbegin(row), 
                            cend(row), comparitor) - cbegin(row));
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
                            arrayTransposed.cend(row), comparitor) - arrayTransposed.cbegin(row));
                    }

                    return returnArray;
                }
                default:
                {
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        NdArray<uint32> argmin(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool
            {
                return lhs < rhs;
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<uint32> returnArray = { static_cast<uint32>(stl_algorithms::min_element(cbegin(), 
                        cend(), comparitor) - cbegin()) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<uint32> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = static_cast<uint32>(stl_algorithms::min_element(cbegin(row), 
                            cend(row), comparitor) - cbegin(row));
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
                            arrayTransposed.cend(row), comparitor) - arrayTransposed.cbegin(row));
                    }

                    return returnArray;
                }
                default:
                {
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        NdArray<uint32> argsort(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    std::vector<uint32> idx(size_);
                    std::iota(idx.begin(), idx.end(), 0);

                    const auto function = [this](uint32 i1, uint32 i2) noexcept -> bool
                    {
                        return (*this)[i1] < (*this)[i2];
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

                        const auto function = [this, row](uint32 i1, uint32 i2) noexcept -> bool
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

                        const auto function = [&arrayTransposed, row](uint32 i1, uint32 i2) noexcept -> bool
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
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
                }
            }
        }

        //============================================================================
        // Method Description:
        ///						Returns a copy of the array, cast to a specified type.
        ///                     Arithmetic to Arithmetic
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html
        ///
        /// @return
        ///				NdArray
        ///
        template<typename dtypeOut, typename dtype_ = dtype,
            enable_if_t<std::is_same<dtype_, dtype>::value, int> = 0,
            enable_if_t<std::is_arithmetic<dtype_>::value, int> = 0,
            enable_if_t<std::is_arithmetic<dtypeOut>::value, int> = 0>
        NdArray<dtypeOut> astype() const 
        {
            NdArray<dtypeOut> outArray(shape_);

            if (std::is_same<dtypeOut, dtype>::value)
            {
                std::copy(cbegin(), cend(), outArray.begin());
            }
            else
            {
                const auto function = [](dtype value)  -> dtypeOut
                {
                    return static_cast<dtypeOut>(value);
                };

                stl_algorithms::transform(cbegin(), cend(), outArray.begin(), function);
            }

            return outArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns a copy of the array, cast to a specified type.
        ///                     Arithmetic to Complex
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html
        ///
        /// @return
        ///				NdArray
        ///
        template<typename dtypeOut, typename dtype_ = dtype, 
            enable_if_t<std::is_same<dtype_, dtype>::value, int> = 0,
            enable_if_t<std::is_arithmetic<dtype_>::value, int> = 0,
            enable_if_t<is_complex_v<dtypeOut>, int> = 0>
            NdArray<dtypeOut> astype() const 
        {
            NdArray<dtypeOut> outArray(shape_);

            const auto function = [](const_reference value)  -> dtypeOut
            {
                return std::complex<typename dtypeOut::value_type>(value);
            };

            stl_algorithms::transform(cbegin(), cend(), outArray.begin(), function);

            return outArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns a copy of the array, cast to a specified type.
        ///                     Complex to Complex
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html
        ///
        /// @return
        ///				NdArray
        ///
        template<typename dtypeOut, typename dtype_ = dtype, 
            enable_if_t<std::is_same<dtype_, dtype>::value, int> = 0,
            enable_if_t<is_complex_v<dtype_>, int> = 0,
            enable_if_t<is_complex_v<dtypeOut>, int> = 0>
        NdArray<dtypeOut> astype() const 
        {
            NdArray<dtypeOut> outArray(shape_);

            if (std::is_same<dtypeOut, dtype>::value)
            {
                std::copy(cbegin(), cend(), outArray.begin());
            }
            else
            {
                const auto function = [](const_reference value) noexcept -> dtypeOut
                {
                    return complex_cast<typename dtypeOut::value_type>(value);
                };

                stl_algorithms::transform(cbegin(), cend(), outArray.begin(), function);
            }

            return outArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns a copy of the array, cast to a specified type.
        ///                     Complex to Arithmetic
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html
        ///
        /// @return
        ///				NdArray
        ///
        template<typename dtypeOut, typename dtype_ = dtype, 
            enable_if_t<std::is_same<dtype_, dtype>::value, int> = 0,
            enable_if_t<is_complex_v<dtype_>, int> = 0,
            enable_if_t<std::is_arithmetic<dtypeOut>::value, int> = 0>
            NdArray<dtypeOut> astype() const 
        {
            NdArray<dtypeOut> outArray(shape_);

            const auto function = [](const_reference value)  -> dtypeOut
            {
                return static_cast<dtypeOut>(value.real());
            };

            stl_algorithms::transform(cbegin(), cend(), outArray.begin(), function);

            return outArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns a copy of the last element of the flattened array.
        ///
        /// @return
        ///				dtype
        ///
        value_type back() const noexcept 
        {
            return *(cend() - 1);
        }

        //============================================================================
        // Method Description:
        ///						Returns a reference the last element of the flattened array.
        ///
        /// @return
        ///				dtype
        ///
        reference back() noexcept 
        {
            return *(end() - 1);
        }

        //============================================================================
        // Method Description:
        ///						Returns a copy of the last element of the input row.
        ///
        /// @return
        ///				dtype
        ///
        value_type back(size_type row) const 
        {
            return *(cend(row) - 1);
        }

        //============================================================================
        // Method Description:
        ///						Returns a reference the last element of the input row.
        ///
        /// @return
        ///				dtype
        ///
        reference back(size_type row) 
        {
            return *(end(row) - 1);
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
        NdArray<dtype>& byteswap() 
        {
            STATIC_ASSERT_INTEGER(dtype);

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
        NdArray<dtype> clip(value_type inMin, value_type inMax) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            NdArray<dtype> outArray(shape_);
            boost::algorithm::clamp_range(cbegin(), cend(), outArray.begin(), inMin, inMax, 
                [](dtype lhs, dtype rhs) noexcept -> bool
                {
                    return lhs < rhs;
                });

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
        NdArray<bool> contains(value_type inValue, Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

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
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        NdArray<dtype> copy() const 
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
        NdArray<dtype> cumprod(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

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
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        NdArray<dtype> cumsum(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

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
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
                }
            }
        }

        //============================================================================
        // Method Description:
        ///						Returns the raw pointer to the underlying data
        /// @return pointer
        ///
        pointer data() noexcept 
        {
            return array_;
        }

        //============================================================================
        // Method Description:
        ///						Returns the raw pointer to the underlying data
        /// @return const_pointer
        ///
        const_pointer data() const noexcept 
        {
            return array_;
        }

        //============================================================================
        // Method Description:
        ///						Releases the internal data pointer so that the destructor
        ///                     will not call delete on it, and returns the raw pointer
        ///                     to the underlying data.
        /// @return pointer
        ///
        pointer dataRelease() noexcept
        {
            ownsPtr_ = false;
            return data();
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
        NdArray<dtype> diagonal(int32 inOffset = 0, Axis inAxis = Axis::ROW) const 
        {
            switch (inAxis)
            {
                case Axis::ROW:
                {
                    std::vector<dtype> diagnolValues;
                    int32 col = inOffset;
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        if (col < 0)
                        {
                            ++col;
                            continue;
                        }
                        if (col >= static_cast<int32>(shape_.cols))
                        {
                            break;
                        }

                        diagnolValues.push_back(operator()(row, static_cast<uint32>(col)));
                        ++col;
                    }

                    return NdArray<dtype>(diagnolValues);
                }
                case Axis::COL:
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
                        if (col >= shape_.cols)
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
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            if (shape_ == inOtherArray.shape_ && (shape_.rows == 1 || shape_.cols == 1))
            {
                dtype dotProduct = std::inner_product(cbegin(), cend(), inOtherArray.cbegin(), dtype{ 0 });
                NdArray<dtype> returnArray = { dotProduct };
                return returnArray;
            }
            if (shape_.cols == inOtherArray.shape_.rows)
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
            
            std::string errStr = "shapes of [" + utils::num2str(shape_.rows) + ", " + utils::num2str(shape_.cols) + "]";
            errStr += " and [" + utils::num2str(inOtherArray.shape_.rows) + ", " + utils::num2str(inOtherArray.shape_.cols) + "]";
            errStr += " are not consistent.";
            THROW_INVALID_ARGUMENT_ERROR(errStr);
            
            return NdArray<dtype>(); // get rid of compiler warning
        }

        //============================================================================
        // Method Description:
        ///						Dump a binary file of the array to the specified file.
        ///						The array can be read back with nc::load.
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

            if (array_ != nullptr)
            {
                ofile.write(reinterpret_cast<const char*>(array_), size_ * sizeof(dtype));
            }
            ofile.close();
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
            STATIC_ASSERT_ARITHMETIC(dtype);

            return endianess_;
        }

        //============================================================================
        // Method Description:
        ///						Fill the array with a scaler value.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.fill.html
        ///
        /// @param
        ///				inFillValue
        /// @return
        ///				None
        ///
        NdArray<dtype>& fill(value_type inFillValue) noexcept
        {
            stl_algorithms::fill(begin(), end(), inFillValue);
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Return the indices of the flattened array of the
        ///						elements that are non-zero.
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<uint32> flatnonzero() const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

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
        ///						Return a copy of the array collapsed into one dimension.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.flatten.html
        ///
        /// @return
        ///				NdArray
        ///
        NdArray<dtype> flatten() const 
        {
            NdArray<dtype> outArray(1, size_);
            stl_algorithms::copy(cbegin(), cend(), outArray.begin());
            return outArray;
        }

        //============================================================================
        // Method Description:
        ///						Returns a copy of the first element of the flattened array.
        ///
        /// @return
        ///				dtype
        ///
        value_type front() const noexcept 
        {
            return *cbegin();
        }

        //============================================================================
        // Method Description:
        ///						Returns a reference to the first element of the flattened array.
        ///
        /// @return
        ///				dtype
        ///
        reference front() noexcept 
        {
            return *begin();
        }

        //============================================================================
        // Method Description:
        ///						Returns a copy of the first element of the input row.
        ///
        /// @return
        ///				dtype
        ///
        value_type front(size_type row) const 
        {
            return *cbegin(row);
        }

        //============================================================================
        // Method Description:
        ///						Returns a reference to the first element of the input row.
        ///
        /// @return
        ///				dtype
        ///
        reference front(size_type row) 
        {
            return *begin(row);
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
        NdArray<bool> issorted(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool
            {
                return lhs < rhs;
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    return { stl_algorithms::is_sorted(cbegin(), cend(), comparitor) };
                }
                case Axis::ROW:
                {
                    NdArray<bool> returnArray(shape_.cols, 1);
                    auto transposedArray = transpose();
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::is_sorted(transposedArray.cbegin(row), 
                            transposedArray.cend(row), comparitor);
                    }

                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<bool> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = stl_algorithms::is_sorted(cbegin(row), cend(row), comparitor);
                    }

                    return returnArray;
                }
                default:
                {
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
                }
            }
        }

        //============================================================================
        // Method Description:
        ///						Return if the NdArray is sorted.
        ///
        /// @return boolean
        ///
        bool issquare() const noexcept
        {
            return shape_.issquare();
        }

        //============================================================================
        // Method Description:
        ///						Copy an element of an array to a standard C++ scaler and return it.
        ///
        ///                     Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.item.html
        ///
        /// @return
        ///				array element
        ///
        value_type item() const
        {
            if (size_ != 1)
            {
                THROW_INVALID_ARGUMENT_ERROR("Can only convert an array of size 1 to a C++ scaler");
            }

            return front();
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
        NdArray<dtype> max(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool
            {
                return lhs < rhs;
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<dtype> returnArray = { *stl_algorithms::max_element(cbegin(), cend(), comparitor) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = *stl_algorithms::max_element(cbegin(row), cend(row), comparitor);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<dtype> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        returnArray(0, row) = *stl_algorithms::max_element(transposedArray.cbegin(row), 
                            transposedArray.cend(row), comparitor);
                    }

                    return returnArray;
                }
                default:
                {
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        NdArray<dtype> min(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool
            {
                return lhs < rhs;
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<dtype> returnArray = { *stl_algorithms::min_element(cbegin(), cend(), comparitor) };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        returnArray(0, row) = *stl_algorithms::min_element(cbegin(row), cend(row), comparitor);
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    NdArray<dtype> returnArray(1, transposedArray.shape_.rows);
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        returnArray(0, row) = *stl_algorithms::min_element(transposedArray.cbegin(row), 
                            transposedArray.cend(row), comparitor);
                    }

                    return returnArray;
                }
                default:
                {
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        NdArray<dtype> median(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool
            {
                return lhs < rhs;
            };

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
                    stl_algorithms::nth_element(copyArray.begin(), copyArray.begin() + middleIdx, copyArray.end(), comparitor);

                    dtype medianValue = copyArray.array_[middleIdx];
                    if (size_ % 2 == 0)
                    {
                        const uint32 lhsIndex = middleIdx - 1;
                        stl_algorithms::nth_element(copyArray.begin(), copyArray.begin() + lhsIndex, copyArray.end(), comparitor);
                        medianValue = (medianValue + copyArray.array_[lhsIndex]) / dtype{2}; // potentially integer division, ok
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
                        const uint32 middleIdx = shape_.cols / 2;  // integer division
                        stl_algorithms::nth_element(copyArray.begin(row), copyArray.begin(row) + middleIdx,
                            copyArray.end(row), comparitor);

                        dtype medianValue = copyArray(row, middleIdx);
                        if (isEven)
                        {
                            const uint32 lhsIndex = middleIdx - 1;
                            stl_algorithms::nth_element(copyArray.begin(row), copyArray.begin(row) + lhsIndex, 
                                copyArray.end(row), comparitor);
                            medianValue = (medianValue + copyArray(row, lhsIndex)) / dtype{2}; // potentially integer division, ok
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
                        stl_algorithms::nth_element(transposedArray.begin(row), transposedArray.begin(row) + middleIdx, 
                            transposedArray.end(row), comparitor);

                        dtype medianValue = transposedArray(row, middleIdx);
                        if (isEven)
                        {
                            const uint32 lhsIndex = middleIdx - 1;
                            stl_algorithms::nth_element(transposedArray.begin(row), transposedArray.begin(row) + lhsIndex,
                                transposedArray.end(row), comparitor);
                            medianValue = (medianValue + transposedArray(row, lhsIndex)) / dtype{2}; // potentially integer division, ok
                        }

                        returnArray(0, row) = medianValue;
                    }

                    return returnArray;
                }
                default:
                {
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
                }
            }
        }

        //============================================================================
        // Method Description:
        ///						Fills the array with nans.
        ///
        ///
        NdArray<dtype>& nans() noexcept
        {
            STATIC_ASSERT_FLOAT(dtype);

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
        NdArray<dtype> newbyteorder(Endian inEndianess) const 
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
                            THROW_INVALID_ARGUMENT_ERROR("Unimplemented endian type.");
                            return {}; // get rid of compiler warning
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
                            THROW_INVALID_ARGUMENT_ERROR("Unimplemented endian type.");
                            return {}; // get rid of compiler warning
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

                            const auto function = [](dtype value) noexcept -> dtype
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
                            THROW_INVALID_ARGUMENT_ERROR("Unimplemented endian type.");
                            return {}; // get rid of compiler warning
                        }
                    }
                    break;
                }
                default:
                {
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented endian type.");
                    return {}; // get rid of compiler warning
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
        NdArray<bool> none(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto function = [](dtype i)  -> bool
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
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        std::pair<NdArray<uint32>, NdArray<uint32>> nonzero() const;

        //============================================================================
        // Method Description:
        ///						Returns the number of columns in the array
        ///
        ///
        /// @return
        ///				uint32
        ///
        uint32 numCols() const noexcept 
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
        uint32 numRows() const noexcept 
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
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            fill(dtype{ 1 });
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
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool
            {
                return lhs < rhs;
            };

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

                    stl_algorithms::nth_element(begin(), begin() + inKth, end(), comparitor);
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
                        stl_algorithms::nth_element(begin(row), begin(row) + inKth, end(row), comparitor);
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
                        stl_algorithms::nth_element(transposedArray.begin(row), transposedArray.begin(row) + inKth,
                            transposedArray.end(row), comparitor);
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
        void print() const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

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
        NdArray<dtype> prod(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

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
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        NdArray<dtype> ptp(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool
            {
                return lhs < rhs;
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    const auto result = stl_algorithms::minmax_element(cbegin(), cend(), comparitor);
                    NdArray<dtype> returnArray = { *result.second - *result.first };
                    return returnArray;
                }
                case Axis::COL:
                {
                    NdArray<dtype> returnArray(1, shape_.rows);
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        const auto result = stl_algorithms::minmax_element(cbegin(row), cend(row), comparitor);
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
                        const auto result = stl_algorithms::minmax_element(transposedArray.cbegin(row), transposedArray.cend(row), comparitor);
                        returnArray(0, row) = *result.second - *result.first;
                    }

                    return returnArray;
                }
                default:
                {
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        NdArray<dtype>& put(int32 inIndex, value_type inValue)
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
        NdArray<dtype>& put(int32 inRow, int32 inCol, value_type inValue)
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
        NdArray<dtype>& put(const NdArray<uint32>& inIndices, value_type inValue)
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
        NdArray<dtype>& put(const Slice& inSlice, value_type inValue)
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
        NdArray<dtype>& put(const Slice& inRowSlice, const Slice& inColSlice, value_type inValue)
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
        NdArray<dtype>& put(const Slice& inRowSlice, int32 inColIndex, value_type inValue)
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
        NdArray<dtype>& put(int32 inRowIndex, const Slice& inColSlice, value_type inValue)
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
        NdArray<dtype>& putMask(const NdArray<bool>& inMask, value_type inValue)
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
        NdArray<dtype>& ravel() noexcept
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
        NdArray<dtype> repeat(uint32 inNumRows, uint32 inNumCols) const 
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
        NdArray<dtype> repeat(const Shape& inRepeatShape) const 
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
        void replace(value_type oldValue, value_type newValue) 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

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
        NdArray<dtype>& reshape(size_type inSize)
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
                
                std::string errStr = "Cannot reshape array of size " + utils::num2str(size_) + " into a shape ";
                errStr += "with " + utils::num2str(inNumCols) + " columns";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
                
            }

            if (inNumCols < 0)
            {
                if (size_ % inNumRows == 0)
                {
                    return reshape(inNumRows, size_ / inNumRows);
                }
                
                std::string errStr = "Cannot reshape array of size " + utils::num2str(size_) + " into a shape ";
                errStr += "with " + utils::num2str(inNumRows) + " rows";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
                
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
        NdArray<dtype>& resizeFast(uint32 inNumRows, uint32 inNumCols) 
        {
            newArray(Shape(inNumRows, inNumCols));
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
        NdArray<dtype>& resizeFast(const Shape& inShape) 
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
        NdArray<dtype>& resizeSlow(uint32 inNumRows, uint32 inNumCols) 
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
        NdArray<dtype>& resizeSlow(const Shape& inShape) 
        {
            return resizeSlow(inShape.rows, inShape.cols);
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
        NdArray<dtype> round(uint8 inNumDecimals = 0) const 
        {
            STATIC_ASSERT_FLOAT(dtype);

            NdArray<dtype> returnArray(shape_);
            const double multFactor = utils::power(10.0, inNumDecimals);
            const auto function = [multFactor](dtype value) noexcept -> dtype
            {
                return static_cast<dtype>(std::nearbyint(static_cast<double>(value) * multFactor) / multFactor);
            };

            stl_algorithms::transform(cbegin(), cend(), returnArray.begin(), function);

            return returnArray;
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
        size_type size() const noexcept 
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
        NdArray<dtype>& sort(Axis inAxis = Axis::NONE) 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool
            {
                return lhs < rhs;
            };

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    stl_algorithms::sort(begin(), end(), comparitor);
                    break;
                }
                case Axis::COL:
                {
                    for (uint32 row = 0; row < shape_.rows; ++row)
                    {
                        stl_algorithms::sort(begin(row), end(row), comparitor);
                    }
                    break;
                }
                case Axis::ROW:
                {
                    NdArray<dtype> transposedArray = transpose();
                    for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
                    {
                        stl_algorithms::sort(transposedArray.begin(row), transposedArray.end(row), comparitor);
                    }

                    *this = transposedArray.transpose();
                    break;
                }
            }

            return *this;
        }

        //============================================================================
        // Method Description:
        ///						returns the NdArray as a string representation
        ///
        /// @return
        ///				string
        ///
        std::string str() const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            std::string out;
            out += "[";
            for (uint32 row = 0; row < shape_.rows; ++row)
            {
                out += "[";
                for (uint32 col = 0; col < shape_.cols; ++col)
                {
                    out += utils::value2str(operator()(row, col)) + ", ";
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
        NdArray<dtype> sum(Axis inAxis = Axis::NONE) const 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

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
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
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
        NdArray<dtype> swapaxes() const 
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
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            if (inSep.empty())
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
        std::vector<dtype> toStlVector() const 
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
        value_type trace(uint32 inOffset = 0, Axis inAxis = Axis::ROW) const noexcept 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

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
        NdArray<dtype> transpose() const 
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
        ///						Fills the array with zeros
        ///
        ///
        NdArray<dtype>& zeros() noexcept
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);
            
            fill(dtype{ 0 });
            return *this;
        }

    private:
        //====================================Attributes==============================
        allocator_type  allocator_{};
        Shape           shape_{ 0, 0 };
        size_type       size_{ 0 };
        Endian          endianess_{ Endian::NATIVE };
        pointer         array_{ nullptr };
        bool            ownsPtr_{ false };

        //============================================================================
        // Method Description:
        ///						Deletes the internal array
        ///
        void deleteArray() noexcept
        {
            if (ownsPtr_ && array_ != nullptr)
            {
                allocator_.deallocate(array_, size_);
            }

            array_ = nullptr;
            shape_.rows = shape_.cols = 0;
            size_ = 0;
            ownsPtr_ = false;
            endianess_ = Endian::NATIVE;
        }

        //============================================================================
        // Method Description:
        ///						Creates a new internal array
        ///
        void newArray() 
        {
            if (size_ > 0)
            {
                array_ = allocator_.allocate(size_);
                ownsPtr_ = true;
            }
        }

        //============================================================================
        // Method Description:
        ///						Creates a new internal array
        ///
        /// @param
        ///				inShape
        ///
        void newArray(const Shape& inShape) 
        {
            deleteArray();

            shape_ = inShape;
            size_ = inShape.size();
            newArray();
        }
    };

    // NOTE: this needs to be defined outside of the class to get rid of a compiler
    // error in Visual Studio
    template<typename dtype, class _Alloc>
    std::pair<NdArray<uint32>, NdArray<uint32>> NdArray<dtype, _Alloc>::nonzero() const 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

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
} // namespace nc
