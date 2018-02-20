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

#include<boost/filesystem.hpp>
#include<boost/endian/conversion.hpp>

#include<initializer_list>
#include<stdexcept>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<algorithm>
#include<limits>
#include<numeric>

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
		NdArray(uint32 inSquareSize) :
			shape_(inSquareSize, inSquareSize),
			size_(inSquareSize * inSquareSize),
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
		NdArray(uint32 inNumRows, uint32 inNumCols) :
			shape_(inNumRows, inNumCols),
			size_(inNumRows * inNumCols),
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
			size_(inShape.rows * inShape.cols),
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
			shape_(static_cast<uint32>(inList.size()), 1),
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
			array_(nullptr)
		{
			typename std::initializer_list<std::initializer_list<dtype> >::iterator iter;
			for (iter = inList.begin(); iter < inList.end(); ++iter)
			{
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

			array_ = new dtype[size_];
			typename std::initializer_list<std::initializer_list<dtype> >::iterator iter;
			uint16 row = 0;
			for (iter = inList.begin(); iter < inList.end(); ++iter)
			{
				std::copy(iter->begin(), iter->end(), array_ + row * shape_.cols);
				++row;
			}
		}

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				vector
		// Outputs:
		//				None
		//
		NdArray(const std::vector<dtype>& inVector) :
			shape_(1, static_cast<uint32>(inVector.size())),
			size_(static_cast<uint32>(inVector.size())),
			array_(new dtype[size_])
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
			deleteArray();

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
		dtype& operator[](int32 inIndex)
		{
			if (inIndex > -1)
			{
				return array_[inIndex];
			}
			else
			{
				uint32 index = static_cast<uint32>(static_cast<int32>(size_) + inIndex);
				return array_[index];
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
		const dtype& operator[](int32 inIndex) const
		{
			if (inIndex > -1)
			{
				return array_[inIndex];
			}
			else
			{
				uint32 index = static_cast<uint32>(static_cast<int32>(size_) + inIndex);
				return array_[index];
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
				uint32 rowIdx = static_cast<uint32>(static_cast<int32>(shape_.rows) + inRowIndex);
				uint32 colIdx = static_cast<uint32>(static_cast<int32>(shape_.cols) + inColIndex);
				return array_[rowIdx * shape_.cols + colIdx];
			}
			else if (inRowIndex > -1 && inColIndex < 0)
			{
				uint32 colIdx = static_cast<uint32>(static_cast<int32>(shape_.cols) + inColIndex);
				return array_[inRowIndex * shape_.cols + colIdx];
			}
			else if (inRowIndex < 0 && inColIndex > -1)
			{
				uint32 rowIdx = static_cast<uint32>(static_cast<int32>(shape_.rows) + inRowIndex);
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
				uint32 rowIdx = static_cast<uint32>(static_cast<int32>(shape_.rows) + inRowIndex);
				uint32 colIdx = static_cast<uint32>(static_cast<int32>(shape_.cols) + inColIndex);
				return array_[rowIdx * shape_.cols + colIdx];
			}
			else if (inRowIndex > -1 && inColIndex < 0)
			{
				uint32 colIdx = static_cast<uint32>(static_cast<int32>(shape_.cols) + inColIndex);
				return array_[inRowIndex * shape_.cols + colIdx];
			}
			else if (inRowIndex < 0 && inColIndex > -1)
			{
				uint32 rowIdx = static_cast<uint32>(static_cast<int32>(shape_.rows) + inRowIndex);
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
		NdArray<dtype> operator[](const Slice& inSlice) const
		{
			uint32 start = 0;
			uint32 stop = 0;

			if (inSlice.start > -1 && inSlice.stop > -1)
			{
				start = inSlice.start;
				stop = inSlice.stop;
			}
			else if (inSlice.start < 0 && inSlice.stop < 0)
			{
				start = size_ + inSlice.start;
				stop = size_ + inSlice.stop - 1;
			}
			else if (inSlice.start > -1 && inSlice.stop < 0)
			{
				start = inSlice.start;
				stop = size_ + inSlice.stop - 1;
			}
			else if (inSlice.start < 0 && inSlice.stop > -1)
			{
				start = size_ + inSlice.start;
				stop = inSlice.stop;
			}
			else
			{
				// I don't think it is possible to get in here...
				throw std::runtime_error("ERROR: Investigate this!");
			}

			if (stop < start)
			{
				std::string errStr = "ERROR: Slice stop must be less than slice start.";
				throw std::invalid_argument(errStr);
			}

			if (start > size_ || stop > size_)
			{
				std::string errStr = "ERROR: Slice parameters are out of bounds for array of size " + num2str(size_);
				throw std::invalid_argument(errStr);
			}

			uint32 numElements = (stop - start + 1) / inSlice.step;
			uint32 counter = 0;
			NdArray<dtype> returnArray(numElements, 1);
			for (uint32 i = start; i < stop; i += inSlice.step)
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
			uint32 rowStart = 0;
			uint32 rowStop = 0;

			if (inRowSlice.start > -1 && inRowSlice.stop > -1)
			{
				rowStart = inRowSlice.start;
				rowStop = inRowSlice.stop;
			}
			else if (inRowSlice.start < 0 && inRowSlice.stop < 0)
			{
				rowStart = shape_.rows + inRowSlice.start;
				rowStop = shape_.rows + inRowSlice.stop - 1;
			}
			else if (inRowSlice.start > -1 && inRowSlice.stop < 0)
			{
				rowStart = inRowSlice.start;
				rowStop = shape_.rows + inRowSlice.stop - 1;
			}
			else if (inRowSlice.start < 0 && inRowSlice.stop > -1)
			{
				rowStart = shape_.rows + inRowSlice.start;
				rowStop = inRowSlice.stop;
			}
			else
			{
				// I don't think it is possible to get in here...
				throw std::runtime_error("ERROR: Investigate this!");
			}

			if (rowStop < rowStart)
			{
				std::string errStr = "ERROR: Slice stop must be less than slice start.";
				throw std::invalid_argument(errStr);
			}

			if (rowStart > shape_.rows || rowStop > shape_.rows)
			{
				std::string errStr = "ERROR: Slice parameters are out of bounds for array of size [";
				errStr += num2str(shape_.rows) + ", " + num2str(shape_.cols) + "]";
				throw std::invalid_argument(errStr);
			}

			uint32 colStart = 0;
			uint32 colStop = 0;

			if (inColSlice.start > -1 && inColSlice.stop > -1)
			{
				colStart = inColSlice.start;
				colStop = inColSlice.stop;
			}
			else if (inColSlice.start < 0 && inColSlice.stop < 0)
			{
				colStart = shape_.cols + inColSlice.start;
				colStop = shape_.cols + inColSlice.stop - 1;
			}
			else if (inColSlice.start > -1 && inColSlice.stop < 0)
			{
				colStart = inColSlice.start;
				colStop = shape_.cols + inColSlice.stop - 1;
			}
			else if (inColSlice.start < 0 && inColSlice.stop > -1)
			{
				colStart = shape_.cols + inColSlice.start;
				colStop = inColSlice.stop;
			}
			else
			{
				// I don't think it is possible to get in here...
				throw std::runtime_error("ERROR: Investigate this!");
			}

			if (colStop < colStart)
			{
				std::string errStr = "ERROR: Slice stop must be less than slice start.";
				throw std::invalid_argument(errStr);
			}

			if (colStart > shape_.cols || colStop > shape_.cols)
			{
				std::string errStr = "ERROR: Slice parameters are out of bounds for array of size [";
				errStr += num2str(shape_.rows) + ", " + num2str(shape_.cols) + "]";
				throw std::invalid_argument(errStr);
			}

			uint16 numRowElements = (rowStop - rowStart + 1) / inRowSlice.step;
			uint16 numColElements = (colStop - colStart + 1) / inColSlice.step;

			uint16 rowCounter = 0;
			uint16 colCounter = 0;

			NdArray<dtype> returnArray(numRowElements, numColElements);
			for (uint16 row = rowStart; row < rowStop; row += inRowSlice.step)
			{
				for (uint16 col = colStart; col < colStop; col += inColSlice.step)
				{
					returnArray(rowCounter, colCounter++) = this->operator()(row, col);
				}
				colCounter = 0;
				++rowCounter;
			}
		}

		//============================================================================
		// Method Description: 
		//						1D access method with bounds checking
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

			return array_[inIndex];
		}

		//============================================================================
		// Method Description: 
		//						const 1D access method with bounds checking
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

			return array_[inIndex];
		}

		//============================================================================
		// Method Description: 
		//						2D access method with bounds checking
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
		//						const 2D access method with bounds checking
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
		//						const 1D access method with bounds checking
		//		
		// Inputs:
		//				Slice
		// Outputs:
		//				Ndarray
		//
		NdArray<dtype> at(const Slice& inSlice) const
		{
			// the slice operator already provides bounds checking. just including
			// the at method for completeness
			return this->operator[](inSlice);
		}

		//============================================================================
		// Method Description: 
		//						const 2D access method with bounds checking
		//		
		// Inputs:
		//				Row Slice,
		//				Column Slice
		// Outputs:
		//				Ndarray
		//
		NdArray<dtype> at(const Slice& inRowSlice, const Slice& inColSlice) const
		{
			// the slice operator already provides bounds checking. just including
			// the at method for completeness
			return this->operator()(inRowSlice, inColSlice);
		}

		//============================================================================
		// Method Description: 
		//						iterator to the beginning of the flattened array
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
		//						iterator to the beginning of the input row
		//		
		// Inputs:
		//				row
		// Outputs:
		//				iterator
		//
		iterator begin(uint32 inRow)
		{
			return array_ + inRow * shape_.cols;
		}

		//============================================================================
		// Method Description: 
		//						iterator to 1 past the end of the flattened array
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
		//						iterator to the 1 past end of the row
		//		
		// Inputs:
		//				row
		// Outputs:
		//				iterator
		//
		iterator end(uint32 inRow)
		{
			return array_ + inRow * shape_.cols + shape_.cols;
		}

		//============================================================================
		// Method Description: 
		//						const iterator to the beginning of the flattened array
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
		//						const iterator to the beginning of the input row
		//		
		// Inputs:
		//				row
		// Outputs:
		//				const_iterator
		//
		const_iterator cbegin(uint32 inRow) const
		{
			return array_ + inRow * shape_.cols;
		}

		//============================================================================
		// Method Description: 
		//						const iterator to 1 past the end of the flattened array
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
		//						const iterator to 1 past the end of the input row
		//		
		// Inputs:
		//				row
		// Outputs:
		//				const_iterator
		//
		const_iterator cend(uint32 inRow) const
		{
			return array_ + inRow * shape_.cols + shape_.cols;
		}

		//============================================================================
		// Method Description: 
		//						Returns True if all elements evaluate to True or non zero
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				NdArray
		//
		NdArray<bool> all(Axis::Type inAxis = Axis::NONE) const
		{
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<bool> returnArray(1);
					returnArray[0] = std::all_of(cbegin(), cend(), [](dtype i) {return i != static_cast<dtype>(0); });
					return returnArray;
				}
				case Axis::COL:
				{
					NdArray<bool> returnArray(1, shape_.rows);
					for (uint32 row = 0; row < shape_.rows; ++row)
					{
						returnArray(0, row) = std::all_of(cbegin(row), cend(row), [](dtype i) {return i != static_cast<dtype>(0); });
					}
					return returnArray;
				}
				case Axis::ROW:
				{
					NdArray<dtype> arrayTransposed = transpose();
					NdArray<bool> returnArray(1, arrayTransposed.shape_.rows);
					for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
					{
						returnArray(0, row) = std::all_of(arrayTransposed.cbegin(row), arrayTransposed.cend(row), [](dtype i) {return i != static_cast<dtype>(0); });
					}
					return returnArray;
				}
				default:
				{
					// this isn't actually possible, just putting this here to get rid
					// of the compiler warning.
					return NdArray<bool>(1);
				}
			}
		}

		//============================================================================
		// Method Description: 
		//						Returns True if any elements evaluate to True or non zero
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				NdArray
		//
		NdArray<bool> any(Axis::Type inAxis = Axis::NONE) const
		{
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<bool> returnArray(1);
					returnArray[0] = std::any_of(cbegin(), cend(), [](dtype i) {return i != static_cast<dtype>(0); });
					return returnArray;
				}
				case Axis::COL:
				{
					NdArray<bool> returnArray(1, shape_.rows);
					for (uint32 row = 0; row < shape_.rows; ++row)
					{
						returnArray(0, row) = std::any_of(cbegin(row), cend(row), [](dtype i) {return i != static_cast<dtype>(0); });
					}
					return returnArray;
				}
				case Axis::ROW:
				{
					NdArray<dtype> arrayTransposed = transpose();
					NdArray<bool> returnArray(1, arrayTransposed.shape_.rows);
					for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
					{
						returnArray(0, row) = std::any_of(arrayTransposed.cbegin(row), arrayTransposed.cend(row), [](dtype i) {return i != static_cast<dtype>(0); });
					}
					return returnArray;
				}
				default:
				{
					// this isn't actually possible, just putting this here to get rid
					// of the compiler warning.
					return NdArray<bool>(1);
				}
			}
		}

		//============================================================================
		// Method Description: 
		//						Return indices of the maximum values along the given axis.
		//						Only the first index is returned.
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				NdArray
		//
		NdArray<uint32> argmax(Axis::Type inAxis = Axis::NONE) const
		{
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<uint32> returnArray(1);
					returnArray[0] = static_cast<uint32>(std::max_element(cbegin(), cend()) - cbegin());
					return returnArray;
				}
				case Axis::COL:
				{
					NdArray<uint32> returnArray(1, shape_.rows);
					for (uint32 row = 0; row < shape_.rows; ++row)
					{
						returnArray(0, row) = static_cast<uint32>(std::max_element(cbegin(row), cend(row)) - cbegin(row));
					}
					return returnArray;
				}
				case Axis::ROW:
				{
					NdArray<dtype> arrayTransposed = transpose();
					NdArray<uint32> returnArray(1, arrayTransposed.shape_.rows);
					for (uint16 row = 0; row < arrayTransposed.shape_.rows; ++row)
					{
						returnArray(0, row) = static_cast<uint32>(std::max_element(arrayTransposed.cbegin(row), arrayTransposed.cend(row)) - arrayTransposed.cbegin(row));
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
		//						Return indices of the minimum values along the given axis.
		//						Only the first index is returned.
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				NdArray
		//
		NdArray<uint32> argmin(Axis::Type inAxis = Axis::NONE) const
		{
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<uint32> returnArray(1);
					returnArray[0] = static_cast<uint32>(std::min_element(cbegin(), cend()) - cbegin());
					return returnArray;
				}
				case Axis::COL:
				{
					NdArray<uint32> returnArray(1, shape_.rows);
					for (uint32 row = 0; row < shape_.rows; ++row)
					{
						returnArray(0, row) = static_cast<uint32>(std::min_element(cbegin(row), cend(row)) - cbegin(row));
					}
					return returnArray;
				}
				case Axis::ROW:
				{
					NdArray<dtype> arrayTransposed = transpose();
					NdArray<uint32> returnArray(1, arrayTransposed.shape_.rows);
					for (uint32 row = 0; row < arrayTransposed.shape_.rows; ++row)
					{
						returnArray(0, row) = static_cast<uint32>(std::min_element(arrayTransposed.cbegin(row), arrayTransposed.cend(row)) - arrayTransposed.cbegin(row));
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
		//						Returns the indices that would sort this array.
		//		
		// Inputs:
		//				(Optional) axis
		// Outputs:
		//				NdArray
		//
		NdArray<uint32> argsort(Axis::Type inAxis = Axis::NONE) const
		{
			switch (inAxis)
			{
				case Axis::NONE:
				{
					std::vector<uint32> idx(size_);
					std::iota(idx.begin(), idx.end(), 0);
					std::stable_sort(idx.begin(), idx.end(), [this](uint32 i1, uint32 i2) {return this->array_[i1] < this->array_[i2]; });
					return NdArray<uint32>(idx);
				}
				case Axis::COL:
				{
					NdArray<uint32> returnArray(shape_);
					for (uint32 row = 0; row < shape_.rows; ++row)
					{
						std::vector<uint32> idx(shape_.cols);
						std::iota(idx.begin(), idx.end(), 0);
						std::stable_sort(idx.begin(), idx.end(), [this, row](uint32 i1, uint32 i2) {return this->operator()(row, i1) < this->operator()(row, i2); });

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
						std::stable_sort(idx.begin(), idx.end(), [&arrayTransposed, row](uint32 i1, uint32 i2) {return arrayTransposed(row, i1) < arrayTransposed(row, i2); });

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
			NdArray<dtypeOut> outArray(shape_);
			for (uint32 i = 0; i < size_; ++i)
			{
				outArray.array_[i] = static_cast<dtypeOut>(array_[i]);
			}
			return outArray;
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
			NdArray<dtype> outArray(shape_);
			for (uint32 i = 0; i < size_; ++i)
			{
				if (array_[i] < inMin)
				{
					outArray.array_[i] = inMin;
				}
				else if (array_[i] > inMax)
				{
					outArray.array_[i] = inMax;
				}
				else
				{
					outArray.array_[i] = array_[i];
				}
			}
			return outArray;
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
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<dtypeOut> returnArray(1, size_);
					returnArray.array_[0] = array_[0];
					for (uint32 i = 1; i < size_; ++i)
					{
						returnArray.array_[i] = returnArray.array_[i - 1] * static_cast<dtypeOut>(array_[i]);
					}

					return returnArray;
				}
				case Axis::COL:
				{
					NdArray<dtypeOut> returnArray(shape_);
					for (uint32 row = 0; row < shape_.rows; ++row)
					{
						returnArray(row, 0) = this->operator()(row, 0);
						for (uint32 col = 1; col < shape_.cols; ++col)
						{
							returnArray(row, col) = returnArray(row, col - 1) * static_cast<dtypeOut>(this->operator()(row, col));
						}
					}

					return returnArray;
				}
				case Axis::ROW:
				{
					NdArray<dtypeOut> returnArray(shape_);
					for (uint32 col = 0; col < shape_.cols; ++col)
					{
						returnArray(0, col) = this->operator()(0, col);
						for (uint32 row = 1; row < shape_.rows; ++row)
						{
							returnArray(row, col) = returnArray(row - 1, col) * static_cast<dtypeOut>(this->operator()(row, col));
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
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<dtypeOut> returnArray(1, size_);
					returnArray.array_[0] = array_[0];
					for (uint32 i = 1; i < size_; ++i)
					{
						returnArray.array_[i] = returnArray.array_[i - 1] + static_cast<dtypeOut>(array_[i]);
					}

					return returnArray;
				}
				case Axis::COL:
				{
					NdArray<dtypeOut> returnArray(shape_);
					for (uint32 row = 0; row < shape_.rows; ++row)
					{
						returnArray(row, 0) = this->operator()(row, 0);
						for (uint32 col = 1; col < shape_.cols; ++col)
						{
							returnArray(row, col) = returnArray(row, col - 1) + static_cast<dtypeOut>(this->operator()(row, col));
						}
					}

					return returnArray;
				}
				case Axis::ROW:
				{
					NdArray<dtypeOut> returnArray(shape_);
					for (uint32 col = 0; col < shape_.cols; ++col)
					{
						returnArray(0, col) = this->operator()(0, col);
						for (uint32 row = 1; row < shape_.rows; ++row)
						{
							returnArray(row, col) = returnArray(row - 1, col) + static_cast<dtypeOut>(this->operator()(row, col));
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
		//						Return specified diagonals.
		//		
		// Inputs:
		//				Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0. 
		//				(Optional) axis the offset is applied to
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> diagonal(uint32 inOffset = 0, Axis::Type inAxis = Axis::ROW) const
		{
			switch (inAxis)
			{
				case Axis::COL:
				{
					std::vector<dtype> diagnolValues;
					uint32 col = inOffset;
					for (uint32 row = 0; row < shape_.rows; ++row)
					{
						if (col >= shape_.cols)
						{
							break;
						}
							
						diagnolValues.push_back(this->operator()(row, col));
						++col;
					}

					return NdArray<dtype>(diagnolValues);
				}
				case Axis::ROW:
				{
					std::vector<dtype> diagnolValues;
					uint32 col = 0;
					for (uint32 row = inOffset; row < shape_.rows; ++row)
					{
						if (col >= shape_.cols)
						{
							break;
						}

						diagnolValues.push_back(this->operator()(row, col));
						++col;
					}

					return NdArray<dtype>(diagnolValues);
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
		//						Dot product of two arrays.
		//
		//						For 2-D arrays it is equivalent to matrix multiplication, 
		//						and for 1-D arrays to inner product of vectors. For N 
		//						dimensions it is a sum product over the last axis of a 
		//						and the second-to-last of b:
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
			boost::filesystem::path p(inFilename);
			if (!boost::filesystem::exists(p.parent_path()))
			{
				std::string errStr = "ERROR: Input path does not exist:\n\t" + p.parent_path().string();
				throw std::runtime_error(errStr);
			}

			std::ofstream ofile(inFilename.c_str(), std::ios::binary);
			ofile.write(reinterpret_cast<const char*>(array_) , size_ * sizeof(dtype));
			ofile.close();
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
			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] = inFillValue;
			}
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
			NdArray<dtype> outArray(1, size_);
			for (uint32 i = 0; i < size_; ++i)
			{
				outArray.array_[i] = array_[i];
			}

			return outArray;
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
			if (size_ == 1)
			{
				return array_[0];
			}
			else
			{
				throw std::runtime_error("ERROR: Can only convert an array of size 1 to a C++ scalar");
			}
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
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<dtype> returnArray(1, 1);
					returnArray[0] = *std::max_element(cbegin(), cend());

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
		//						Return the minimum along a given axis.
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> min(Axis::Type inAxis = Axis::NONE) const
		{
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<dtype> returnArray(1, 1);
					returnArray[0] = *std::min_element(cbegin(), cend());

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
		//						Return the mean along a given axis.
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		NdArray<double> mean(Axis::Type inAxis = Axis::NONE) const
		{
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<double> returnArray(1, 1);
					double sum = static_cast<double>(std::accumulate(cbegin(), cend(), 0.0));
					returnArray[0] = sum / static_cast<double>(size_);

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
		//						Return the median along a given axis. Does NOT average
		//						if array has even number of elements!
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> median(Axis::Type inAxis = Axis::NONE) const
		{
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<dtype> copyArray(*this);
					NdArray<dtype> returnArray(1, 1);
					
					uint32 middle = size_ / 2;
					std::nth_element(copyArray.begin(), copyArray.begin() + middle, copyArray.end());
					returnArray[0] = copyArray.array_[middle];

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
		//						Returns the number of bytes held by the array
		//		
		// Inputs:
		//				None
		// Outputs:
		//				number of bytes
		//
		uint64 nbytes() const
		{
			return static_cast<uint64>(sizeof(dtype) * size_);
		}

		//============================================================================
		// Method Description: 
		//						Return the array with the same data viewed with a different byte order.
		//		
		// Inputs:
		//				None
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> swapbyteorder() const
		{
			NdArray outArray(shape_);
			for (uint32 i = 0; i < size_; ++i)
			{
				outArray[i] = boost::endian::endian_reverse(array_[i]);
			}

			return outArray;
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
		//						Returns an array containing the same data with a new shape.
		//		
		// Inputs:
		//				Shape
		// Outputs:
		//				None
		//
		void reshape(const Shape& inShape)
		{
			if (inShape.rows * inShape.cols != size_)
			{
				std::string errStr = "ERROR: Cannot reshape array of size " + num2str(size_) + " into shape ";
				errStr += "[" + num2str(inShape.rows) + ", " + num2str(inShape.cols) + "]";
				throw std::runtime_error(errStr);
			}

			shape_.rows = inShape.rows;
			shape_.cols = inShape.cols;
		}

		//============================================================================
		// Method Description: 
		//						Change shape and size of array in-place. All previous
		//						data of the array is lost.
		//		
		// Inputs:
		//				Shape
		// Outputs:
		//				None
		//
		void resizeFast(const Shape& inShape)
		{
			shape_ = inShape;
			size_ = shape_.rows * shape_.cols;
			deleteArray();
			array_ = new dtype[size_];
			zeros();
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
		void resizeSlow(const Shape& inShape)
		{
			T* oldData = new dtype[size_];
			for (uint32 i = 0; i < size_; ++i)
			{
				oldData[i] = array_[i];
			}

			uint32 newSize = shape_.rows * shape_.cols;

			deleteArray();
			array_ = new dtype[size_];
			for (uint16 row = 0; row < shape_.rows; ++row)
			{
				for (uint16 col = 0; col < shape_.cols; ++col)
				{
					this->operator()(row, col) = oldData[row * shape_.cols + col];
				}
			}

			delete[] oldData;
			shape_ = inShape;
			size_ = newSize;
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
			return shape_;
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
			return size_;
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
			NdArray<dtype> transArray(shape_.cols, shape_.rows);
			for (uint16 row = 0; row < shape_.rows; ++row)
			{
				for (uint16 col = 0; col < shape_.cols; ++col)
				{
					transArray(col, row) = this->operator()(row, col);
				}
			}
			return transArray;
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
			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] = 0;
			}
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