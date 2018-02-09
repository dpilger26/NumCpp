// Copyright 2018 David Pilger
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files(the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, 
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
// permit persons to whom the Software is furnished to do so, subject to the following 
// conditions :
//
// The above copyright notice and this permission notice shall be included in all copies 
// or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
// DEALINGS IN THE SOFTWARE.

#pragma once

#include"Types.hpp"
#include"Shape.hpp"
#include"Slice.hpp"

#include<initializer_list>
#include<stdexcept>
#include<iostream>
#include<string>
#include<vector>

namespace NumC
{
	//================================================================================
	// Class Description:
	//						holds upto 2D arrays
	//
	template<typename dtype>
	class NdArray
	{
	public:
		//====================================Typedefs================================
		typedef dtype*			iterator;
		typedef const dtype*	const_iterator;

	private:
		//====================================Attributes==============================
		Shape	shape_;
		uint32	size_;
		dtype*	array_;

		//============================================================================
		// Method Description: 
		//						Deletes the internal array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		void deleteArray()
		{
			if (array_ != nullptr)
			{
				delete[] array_;
			}

			shape_ = Shape(0, 0);
			size_ = 0;
		}

	public:
		//============================================================================
		// Method Description: 
		//						Defualt Constructor
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		NdArray() :
			shape_(0, 0),
			size_(0),
			array_(nullptr)
		{};

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				square number or rows and columns
		// Outputs:
		//				None
		//
		NdArray(uint16 inSquareSize) :
			shape_(inSquareSize, inSquareSize),
			size_(static_cast<uint32>(inSquareSize) * static_cast<uint32>(inSquareSize)),
			array_(new dtype[size_])
		{
			zeros();
		}

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				number of rows,
		//				number of columns
		// Outputs:
		//				None
		//
		NdArray(uint16 inNumRows, uint16 inNumCols) :
			shape_(inNumRows, inNumCols),
			size_(static_cast<uint32>(inNumRows) * static_cast<uint32>(inNumCols)),
			array_(new dtype[size_])
		{
			zeros();
		}

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				Shape object
		// Outputs:
		//				None
		//
		NdArray(const Shape& inShape) :
			shape_(inShape),
			size_(static_cast<uint32>(inShape.rows) * static_cast<uint32>(inShape.cols)),
			array_(new dtype[size_])
		{
			zeros();
		}

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				1D initializer list
		// Outputs:
		//				None
		//
		NdArray(const std::initializer_list<dtype>& inList) :
			shape_(static_cast<uint16>(inList.size()), 1),
			size_(static_cast<uint32>(inList.size())),
			array_(new dtype[size_])
		{
			std::copy(inList.begin(), inList.end(), array_);
		}

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				2D initializer list
		// Outputs:
		//				None
		//
		NdArray(const std::initializer_list<std::initializer_list<dtype> >& inList) :
			shape_(0, 0),
			size_(0),
			array_(new dtype[size_])
		{
			typename std::initializer_list<std::initializer_list<dtype> >::iterator iter;
			for (iter = inList.begin(); iter < inList.end(); ++iter)
			{
				std::copy(iter->begin(), iter->end(), data_ + size_);

				size_ += iter->size();
				++shape_.rows;

				if (shape_.cols == 0)
				{
					shape_.cols = iter->size();
				}
				else if (iter->size() != shape_.cols)
				{
					throw std::runtime_error("ERROR: All rows of the initializer list needs to have the same number of elements");
				}
			}
		}

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				initializer list
		// Outputs:
		//				None
		//
		NdArray(const std::vector<dtype>& inVector) :
			shape_(static_cast<uint16>(inList.size()), 1),
			size_(static_cast<uint32>(inList.size())),
			array_(new dtype[shape_])
		{
			std::copy(inVector.begin(), inVector.end(), array_);
		}

		//============================================================================
		// Method Description: 
		//						Copy Constructor
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				None
		//
		NdArray(const NdArray<dtype>& inOtherArray) :
			shape_(inOtherArray.shape_),
			size_(inOtherArray.size_),
			array_(new dtype[inOtherArray.size_])
		{
			for (uint32 i = 0; i < inOtherArray.size_; ++i)
			{
				array_[i] = inOtherArray.array_[i];
			}
		}

		//============================================================================
		// Method Description: 
		//						Destructor
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		~NdArray()
		{
			deleteArray();
		}

		//============================================================================
		// Method Description: 
		//						Assignment operator, performs a deep copy
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				None
		//
		NdArray<dtype>& operator=(const NdArray<dtype>& inOtherArray)
		{
			shape_ = Shape(inOtherArray.shape_);
			size_ = inOtherArray.size_;
			array_ = new dtype[size_];

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] = inOtherArray.data_[i];
			}
		}

		//============================================================================
		// Method Description: 
		//						1D access operator with no bounds checking
		//		
		// Inputs:
		//				array index
		// Outputs:
		//				value
		//
		dtype& operator()(int64 inIndex)
		{
			if (inIndex > -1)
			{
				return array_[inIndex];
			}
			else
			{
				return array_[size_ + inIndex];
			}
		}

		//============================================================================
		// Method Description: 
		//						const 1D access operator with no bounds checking
		//		
		// Inputs:
		//				array index
		// Outputs:
		//				value
		//
		const dtype& operator()(int64 inIndex) const
		{
			if (inIndex > -1)
			{
				return array_[inIndex];
			}
			else
			{
				return array_[size_ + inIndex];
			}
		}

		//============================================================================
		// Method Description: 
		//						2D access operator with no bounds checking
		//		
		// Inputs:
		//				row index
		//				col index
		// Outputs:
		//				value
		//
		dtype& operator()(int32 inRowIndex, int32 inColIndex)
		{
			if (inRowIndex > -1 && inColIndex > -1)
			{
				return array_[inRowIndex * shape_.cols + inColIndex];
			}
			else if (inRowIndex < 0 && inColIndex < 0)
			{
				uint16 rowIdx = size_ + inRowIndex;
				uint16 colIdx = size_ + inColIndex;
				return array_[rowIdx * shape_.cols + colIdx];
			}
			else if (inRowIndex > -1 && inColIndex < 0)
			{
				uint16 colIdx = size_ + inColIndex;
				return array_[inRowIndex * shape_.cols + colIdx];
			}
			else if (inRowIndex < 0 && inColIndex > -1)
			{
				uint16 rowIdx = size_ + inRowIndex;
				return array_[rowIdx * shape_.cols + inColIndex];
			}
			else
			{
				// I don't think it is possible to get in here...
				throw std::runtime_error("ERROR: Investigate this!");
			}
		}

		//============================================================================
		// Method Description: 
		//						const 2D access operator with no bounds checking
		//		
		// Inputs:
		//				row index
		//				col index
		// Outputs:
		//				value
		//
		const dtype& operator()(int32 inRowIndex, int32 inColIndex) const
		{
			if (inRowIndex > -1 && inColIndex > -1)
			{
				return array_[inRowIndex * shape_.cols + inColIndex];
			}
			else if (inRowIndex < 0 && inColIndex < 0)
			{
				uint16 rowIdx = size_ + inRowIndex;
				uint16 colIdx = size_ + inColIndex;
				return array_[rowIdx * shape_.cols + colIdx];
			}
			else if (inRowIndex > -1 && inColIndex < 0)
			{
				uint16 colIdx = size_ + inColIndex;
				return array_[inRowIndex * shape_.cols + colIdx];
			}
			else if (inRowIndex < 0 && inColIndex > -1)
			{
				uint16 rowIdx = size_ + inRowIndex;
				return array_[rowIdx * shape_.cols + inColIndex];
			}
			else
			{
				// I don't think it is possible to get in here...
				throw std::runtime_error("ERROR: Investigate this!");
			}
		}

		//============================================================================
		// Method Description: 
		//						1D Slicing access operator with no bounds checking. 
		//						returned array is of the range [start, stop).
		//		
		// Inputs:
		//				Slice
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator()(const Slice& inSlice) const
		{
			uint32 numElements = 0;

			if ((inSlice.start > -1 && inSlice.stop > -1) || 
				(inSlice.start < 0 && inSlice.stop < 0))
			{
				numElements = (inSlice.stop - inSlice.start + 1) / inSlice.step;
			}
			else if (inSlice.start > -1 && inSlice.stop < 0)
			{
				uint16 stop = size_ + inSlice.stop - 1;
				numElements = (stop - inSlice.start) / inSlice.step;
			}
			else if (inSlice.start < 0 && inSlice.stop > -1)
			{
				uint16 start = size_ + inSlice.start;
				numElements = (inSlice.stop - start) / inSlice.step;
			}
			else
			{
				// I don't think it is possible to get in here...
				throw std::runtime_error("ERROR: Investigate this!");
			}

			uint32 counter = 0;
			NdArray<dtype> returnArray(numElements, 1);
			for (uint32 i = 0; i < numElements; i += inSlice.step)
			{
				returnArray(counter++) = array_[i];
			}
		}

		//============================================================================
		// Method Description: 
		//						2D Slicing access operator with no bounds checking.
		//						returned array is of the range [start, stop).
		//		
		// Inputs:
		//				Row Slice
		//				Col Slice
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator()(const Slice& inRowSlice, const Slice& inColSlice) const
		{

		}

		//============================================================================
		// Method Description: 
		//						1D access operator with bounds checking
		//		
		// Inputs:
		//				array index
		// Outputs:
		//				value
		//
		dtype& at(int64 inIndex)
		{
			if (inIndex > size_ - 1 || std::abs(inIndex) > size_)
			{
				std::string errStr = "ERROR: Input index " + num2str(inIndex) + " is out of bounds for array of size " + num2str(size_) + ".";
				throw std::invalid_argument(errStr);
			}

			return this->operator()(inIndex);
		}

		//============================================================================
		// Method Description: 
		//						const 1D access operator with bounds checking
		//		
		// Inputs:
		//				array index
		// Outputs:
		//				value
		//
		const dtype& at(int64 inIndex) const
		{
			if (inIndex > size_ - 1 || std::abs(inIndex) > size_)
			{
				std::string errStr = "ERROR: Input index " + num2str(inIndex) + " is out of bounds for array of size " + num2str(size_) + ".";
				throw std::invalid_argument(errStr);
			}

			return this->operator()(inIndex);
		}

		//============================================================================
		// Method Description: 
		//						2D access operator with bounds checking
		//		
		// Inputs:
		//				row index
		//				col index
		// Outputs:
		//				value
		//
		dtype& at(int32 inRowIndex, int32 inColIndex)
		{
			if (inRowIndex > shape_.rows - 1 || std::abs(inRowIndex) > shape_.rows)
			{
				std::string errStr = "ERROR: Row index " + num2str(inRowIndex) + " is out of bounds for array of size " + num2str(shape_.rows) + ".";
				throw std::invalid_argument(errStr);
			}

			if (inColIndex > shape_.cols - 1 || std::abs(inColIndex) > shape_.cols)
			{
				std::string errStr = "ERROR: Column index " + num2str(inColIndex) + " is out of bounds for array of size " + num2str(shape_.cols) + ".";
				throw std::invalid_argument(errStr);
			}

			return this->operator()(inRowIndex, inColIndex);
		}

		//============================================================================
		// Method Description: 
		//						const 2D access operator with bounds checking
		//		
		// Inputs:
		//				row index
		//				col index
		// Outputs:
		//				value
		//
		const dtype& at(int32 inRowIndex, int32 inColIndex) const
		{
			if (inRowIndex > shape_.rows - 1 || std::abs(inRowIndex) > shape_.rows)
			{
				std::string errStr = "ERROR: Row index " + num2str(inRowIndex) + " is out of bounds for array of size " + num2str(shape_.rows) + ".";
				throw std::invalid_argument(errStr);
			}

			if (inColIndex > shape_.cols - 1 || std::abs(inColIndex) > shape_.cols)
			{
				std::string errStr = "ERROR: Column index " + num2str(inColIndex) + " is out of bounds for array of size " + num2str(shape_.cols) + ".";
				throw std::invalid_argument(errStr);
			}

			return this->operator()(inRowIndex, inColIndex);
		}

		//============================================================================
		// Method Description: 
		//						1D Slicing access operator with bounds checking.
		//						returned array is of the range [start, stop).
		//		
		// Inputs:
		//				Slice
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> at(const Slice& inSlice) const
		{

		}

		//============================================================================
		// Method Description: 
		//						2D Slicing access operator with bounds checking.
		//						returned array is of the range [start, stop).
		//		
		// Inputs:
		//				Row Slice
		//				Col Slice
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> at(const Slice& inRowSlice, const Slice& inColSlice) const
		{

		}

		//============================================================================
		// Method Description: 
		//						iterator to the beginning of the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				iterator
		//
		iterator begin()
		{
			return array_;
		}

		//============================================================================
		// Method Description: 
		//						iterator to the end of the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				iterator
		//
		iterator end()
		{
			return array_ + size_;
		}

		//============================================================================
		// Method Description: 
		//						const iterator to the beginning of the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				const_iterator
		//
		const_iterator cbegin() const
		{
			return array_;
		}

		//============================================================================
		// Method Description: 
		//						const iterator to the end of the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				const_iterator
		//
		const_iterator cend() const
		{
			return array_ + size_;
		}

		//============================================================================
		// Method Description: 
		//						Returns True if all elements evaluate to True or non zero
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				bool
		//
		bool all(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Returns True if any elements evaluate to True or non zero
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				Bool
		//
		bool any(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return indices of the maximum values along the given axis.
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				NdArray
		//
		NdArray<uint16> argmax(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return indices of the minimum values along the given axis.
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				NdArray
		//
		NdArray<uint16> argmin(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Returns the indices that would sort this array.
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				NdArray
		//
		NdArray<uint16> argsort(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Returns a copy of the array, cast to a specified type.
		//		
		// Inputs:
		//				None
		// Outputs:
		//				NdArray
		//
		template<typename dtypeOut>
		NdArray<dtypeOut> astype() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Returns an array whose values are limited to [min, max].
		//		
		// Inputs:
		//				min value to clip to
		//				max value to clip to
		// Outputs:
		//				clipped value
		//
		NdArray<dtype> clip(dtype inMin, dtype inMax) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the cumulative product of the elements along the given axis.
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				NdArray
		//
		template<typename dtypeOut>
		NdArray<dtypeOut> cumprod(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the cumulative sum of the elements along the given axis.
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				NdArray
		//
		template<typename dtypeOut>
		NdArray<dtypeOut> cumsum(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Returns the determinant as if the array was a matrix
		//		
		// Inputs:
		//				None
		// Outputs:
		//				determinant
		//
		template<typename dtypeOut>
		dtypeOut det() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return specified diagonals.
		//		
		// Inputs:
		//				Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0. 
		//				(Optional) axis the offset is applied to
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> diagonal(uint16 inOffset = 0, Axis::Type inAxis = Axis::ROW) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Dot product of two arrays.
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				dot product
		//
		template<typename dtype>
		dtype dot(const NdArray<dtype>& inOtherArray) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Dump a binary file of the array to the specified file. 
		//						The array can be read back with or numpy::load.
		//		
		// Inputs:
		//				filename
		// Outputs:
		//				None
		//
		void dump(const std::string& inFilename) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Fill the array with a scalar value.
		//		
		// Inputs:
		//				fill value
		// Outputs:
		//				None
		//
		void fill(dtype inFillValue)
		{

		}

		//============================================================================
		// Method Description: 
		//						Return a copy of the array collapsed into one dimension.
		//		
		// Inputs:
		//				None
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> flatten() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the inverse of the array as if it were a matrix
		//		
		// Inputs:
		//				None
		// Outputs:
		//				NdArray
		//
		NdArray<double> inverse() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Copy an element of an array to a standard C++ scalar and return it.
		//		
		// Inputs:
		//				None
		// Outputs:
		//				array element
		//
		dtype item() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the maximum along a given axis.
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> max(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the minimum along a given axis.
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> min(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the mean along a given axis.
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> mean(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the median along a given axis.
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> median(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Returns the number of bytes held by the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				number of bytes
		//
		uint64 nbytes() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the array with the same data viewed with a different byte order.
		//		
		// Inputs:
		//				string new order
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> newbyteorder(const std::string& inNewOrder) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the indices of the elements that are non-zero.
		//		
		// Inputs:
		//				None
		// Outputs:
		//				NdArray
		//
		NdArray<uint16> nonzero() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Returns the norm as if the array was a matrix
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				norm
		//
		template<typename dtypeOut>
		dtypeOut norm(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Fills the array with ones
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		void ones()
		{

		}

		//============================================================================
		// Method Description: 
		//						Rearranges the elements in the array in such a way that 
		//						value of the element in kth position is in the position it 
		//						would be in a sorted array. All elements smaller than the kth 
		//						element are moved before this element and all equal or greater 
		//						are moved behind it. The ordering of the elements in the two 
		//						partitions is undefined.
		//		
		// Inputs:
		//				kth element
		//				(Optional) Axis
		// Outputs:
		//				None
		//
		void partition(dtype inKth, Axis::Type inAxis = Axis::NONE)
		{

		}

		//============================================================================
		// Method Description: 
		//						Prints the array to the console.
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		void print() const
		{
			std::cout << *this;
		}

		//============================================================================
		// Method Description: 
		//						Return the product of the array elements over the given axis
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		template<typename dtypeOut>
		NdArray<dtypeOut> prod(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Peak to peak (maximum - minimum) value along a given axis.
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> ptp(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Set a.flat[n] = values[n] for all n in indices.
		//		
		// Inputs:
		//				NdArray of indices
		//				NdArray of values
		// Outputs:
		//				None
		//
		void put(const NdArray<uint16>& inIndices, const NdArray<dtype>& inValues)
		{

		}

		//============================================================================
		// Method Description: 
		//						Change shape and size of array in-place. All data outide
		//						of new size is lost.
		//		
		// Inputs:
		//				Shape
		// Outputs:
		//				None
		//
		void resize(const Shape& inShape)
		{

		}

		//============================================================================
		// Method Description: 
		//						Return a with each element rounded to the given number
		//						of decimals.
		//		
		// Inputs:
		//				number of decimals to round to
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> round(uint8 inNumDecimals) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the shape of the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				Shape
		//
		Shape shape() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the size of the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				size
		//
		uint32 size() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Sort an array, in-place.
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				size
		//
		void sort(Axis::Type inAxis = Axis::NONE)
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the std along a given axis.
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> std(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the sum of the array elements over the given axis.
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		template<typename dtypeOut>
		NdArray<dtypeOut> sum(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Interchange two axes of an array.
		//		
		// Inputs:
		//				None
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> swapaxes() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return an array formed from the elements of a at the 
		//						given indices.
		//		
		// Inputs:
		//				array indices
		//				(Optional) axis
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> take(const NdArray<uint16>& inIndices, Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Write array to a file as text or binary (default)..
		//						The data produced by this method can be recovered 
		//						using the function fromfile().
		//		
		// Inputs:
		//				filename
		//				Separator between array items for text output. If “” (empty), a binary file is written 
		// Outputs:
		//				None
		//
		void tofile(const std::string& inFilename, const std::string& inSep = "") const
		{

		}

		//============================================================================
		// Method Description: 
		//						Write flattened array to an STL vector
		//		
		// Inputs:
		//				None 
		// Outputs:
		//				None
		//
		std::vector<dtype> toVector() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Return the sum along diagonals of the array.
		//		
		// Inputs:
		//				Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.
		//				(Optional) Axis to offset from
		//				
		// Outputs:
		//				None
		//
		template<typename dtypeOut>
		NdArray<dtypeOut> trace(uint16 inOffset = 0, Axis::Type inAxis = Axis::ROW) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Tranpose the rows and columns of an array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> transpose() const
		{

		}

		//============================================================================
		// Method Description: 
		//						Tranpose the rows and columns of an array
		//		
		// Inputs:
		//				(Optional) Axes
		// Outputs:
		//				NdArray
		//
		template<typename dtypeOut>
		NdArray<dtypeOut> var(Axis::Type inAxis = Axis::NONE) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Fills the array with zeros
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		void zeros()
		{

		}

		//============================================================================
		// Method Description: 
		//						Adds the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator+(const NdArray<dtype>& inOtherArray) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Adds the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator+=(const NdArray<dtype>& inOtherArray)
		{

		}

		//============================================================================
		// Method Description: 
		//						Subtracts the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator-(const NdArray<dtype>& inOtherArray) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Subtracts the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator-=(const NdArray<dtype>& inOtherArray)
		{

		}

		//============================================================================
		// Method Description: 
		//						Multiplies the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator*(const NdArray<dtype>& inOtherArray) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Multiplies the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator*=(const NdArray<dtype>& inOtherArray)
		{

		}

		//============================================================================
		// Method Description: 
		//						Divides the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator/(const NdArray<dtype>& inOtherArray) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Divides the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator/=(const NdArray<dtype>& inOtherArray)
		{

		}

		//============================================================================
		// Method Description: 
		//						Takes the modulus of the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator%(const NdArray<dtype>& inOtherArray) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Takes the modulus of the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator%=(const NdArray<dtype>& inOtherArray)
		{

		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise or of the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator|(const NdArray<dtype>& inOtherArray) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise or of the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator|=(const NdArray<dtype>& inOtherArray)
		{

		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise and of the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator&(const NdArray<dtype>& inOtherArray) const
		{

		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise and of the elements of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator&=(const NdArray<dtype>& inOtherArray)
		{

		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise and of the elements of two arrays
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> operator^(const NdArray<dtype>& inOtherArray)
		{

		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise and of the elements of two arrays
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype>& operator^=(const NdArray<dtype>& inOtherArray)
		{

		}

		//============================================================================
		// Method Description: 
		//						Returns an array of booleans of element wise comparison
		//						of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<bool> operator==(const NdArray<dtype>& inOtherArray)
		{

		}

		//============================================================================
		// Method Description: 
		//						Returns an array of booleans of element wise comparison
		//						of two arrays
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<bool> operator!=(const NdArray<dtype>& inOtherArray)
		{

		}

		//============================================================================
		// Method Description: 
		//						Bitshifts left the elements of the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> operator<<(uint8 inNumBits)
		{

		}

		//============================================================================
		// Method Description: 
		//						Bitshifts left the elements of the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype>& operator<<=(uint8 inNumBits)
		{

		}

		//============================================================================
		// Method Description: 
		//						Bitshifts right the elements of the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> operator>>(uint8 inNumBits)
		{

		}

		//============================================================================
		// Method Description: 
		//						Bitshifts right the elements of the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype>& operator>>=(uint8 inNumBits)
		{

		}

		//============================================================================
		// Method Description: 
		//						prefix incraments the elements of an array
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//

		NdArray<dtype>& operator++()
		{

		}

		//============================================================================
		// Method Description: 
		//						prefix decrements the elements of an array
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator--()
		{

		}

		//============================================================================
		// Method Description: 
		//						postfix increments the elements of an array
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator++(int) const
		{

		}

		//============================================================================
		// Method Description: 
		//						postfix decrements the elements of an array
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator--(int) const
		{

		}

		//============================================================================
		// Method Description: 
		//						io operator for the NdArray class
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		friend std::ostream& operator<<(std::ostream& inOStream, const NdArray& inArray)
		{
			Shape arrayShape = inArray.shape();
			inOStream << "[";
			for (uint16 row = 0; row < arrayShape.rows; ++row)
			{
				inOStream << "[";
				for (uint16 col = 0; col < arrayShape.cols; ++col)
				{
					inOStream << inArray(row, col) << ", ";
				}

				if (row == arrayShape.rows - 1)
				{
					inOStream << "]";
				}
				else
				{
					inOStream << "]" << std::endl;
				}
			}
			inOStream << "]" << std::endl;
			return inOStream;
		}
	};
}