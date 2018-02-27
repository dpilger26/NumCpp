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
#include"Utils.hpp"

#include<boost/filesystem.hpp>
#include<boost/endian/conversion.hpp>

#include<initializer_list>
#include<stdexcept>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<set>
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
		Shape			shape_;
		uint32			size_;
		Endian::Type	endianess_;
		dtype*			array_;

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
			endianess_(Endian::NATIVE),
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
			endianess_(Endian::NATIVE),
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
			endianess_(Endian::NATIVE),
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
			size_(shape_.size()),
			endianess_(Endian::NATIVE),
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
			shape_(1, static_cast<uint32>(inList.size())),
			size_(shape_.size()),
			endianess_(Endian::NATIVE),
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
			shape_(static_cast<uint32>(inList.size()), 0),
			size_(0),
			endianess_(Endian::NATIVE),
			array_(nullptr)
		{
			typename std::initializer_list<std::initializer_list<dtype> >::iterator iter;
			for (iter = inList.begin(); iter < inList.end(); ++iter)
			{
				size_ += static_cast<uint32>(iter->size());

				if (shape_.cols == 0)
				{
					shape_.cols = static_cast<uint32>(iter->size());
				}
				else if (iter->size() != shape_.cols)
				{
					throw std::runtime_error("ERROR: All rows of the initializer list needs to have the same number of elements");
				}
			}

			array_ = new dtype[size_];
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
			size_(shape_.size()),
			endianess_(Endian::NATIVE),
			array_(new dtype[size_])
		{
			std::copy(inVector.begin(), inVector.end(), array_);
		}

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				set
		// Outputs:
		//				None
		//
		NdArray(const std::set<dtype>& inSet) :
			shape_(1, static_cast<uint32>(inSet.size())),
			size_(shape_.size()),
			endianess_(Endian::NATIVE),
			array_(new dtype[size_])
		{
			std::copy(inSet.begin(), inSet.end(), array_);
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
			endianess_(inOtherArray.endianess_),
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
			endianess_ = inOtherArray.endianess_;
			array_ = new dtype[size_];

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] = inOtherArray.array_[i];
			}

			return *this;
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
			if (inIndex < 0)
			{
				inIndex += size_;
			}

			return array_[inIndex];
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
			if (inIndex < 0)
			{
				inIndex += size_;
			}

			return array_[inIndex];
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
			Slice inSliceCopy(inSlice);

			uint32 counter = 0;
			NdArray<dtype> returnArray(1, inSliceCopy.numElements(size_));
			for (int32 i = inSliceCopy.start; i < inSliceCopy.stop; i += inSliceCopy.step)
			{
				returnArray[counter++] = this->at(i);
			}

			return returnArray;
		}

		//============================================================================
		// Method Description: 
		//						1D Slicing access operator with no bounds checking. 
		//						returned array is of the range [start, stop).
		//		
		// Inputs:
		//				initializer list
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator[](const std::initializer_list<int32>& inList) const
		{
			Slice inSlice(inList);

			uint32 counter = 0;
			NdArray<dtype> returnArray(1, inSlice.numElements(size_));
			for (int32 i = inSlice.start; i < inSlice.stop; i += inSlice.step)
			{
				returnArray[counter++] = this->at(i);
			}

			return returnArray;
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
			Slice inRowSliceCopy(inRowSlice);
			Slice inColSliceCopy(inColSlice);

			NdArray<dtype> returnArray(inRowSliceCopy.numElements(shape_.rows), inColSliceCopy.numElements(shape_.cols));

			uint32 rowCounter = 0;
			uint32 colCounter = 0;
			for (int32 row = inRowSliceCopy.start; row < inRowSliceCopy.stop; row += inRowSliceCopy.step)
			{
				for (int32 col = inColSliceCopy.start; col < inColSliceCopy.stop; col += inColSliceCopy.step)
				{
					returnArray(rowCounter, colCounter++) = this->at(row, col);
				}
				colCounter = 0;
				++rowCounter;
			}

			return returnArray;
		}

		//============================================================================
		// Method Description: 
		//						2D Slicing access operator with no bounds checking.
		//						returned array is of the range [start, stop).
		//		
		// Inputs:
		//				Row initializer list
		//				Col initializer list
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator()(const std::initializer_list<int32>& inRowList, const std::initializer_list<int32>& inColList) const
		{
			Slice inRowSlice(inRowList);
			Slice inColSlice(inColList);

			NdArray<dtype> returnArray(inRowSlice.numElements(shape_.rows), inColSlice.numElements(shape_.cols));

			uint32 rowCounter = 0;
			uint32 colCounter = 0;
			for (int32 row = inRowSlice.start; row < inRowSlice.stop; row += inRowSlice.step)
			{
				for (int32 col = inColSlice.start; col < inColSlice.stop; col += inColSlice.step)
				{
					returnArray(rowCounter, colCounter++) = this->at(row, col);
				}
				colCounter = 0;
				++rowCounter;
			}

			return returnArray;
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
		dtype& at(int32 inIndex)
		{
			// this doesn't allow for calling the first element as -size_... 
			// but why would you really want to that anyway?
			if (std::abs(inIndex) > static_cast<int64>(size_ - 1))
			{
				std::string errStr = "ERROR: Input index " + num2str(inIndex) + " is out of bounds for array of size " + num2str(size_) + ".";
				throw std::invalid_argument(errStr);
			}

			return this->operator[](inIndex);
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
		const dtype& at(int32 inIndex) const
		{
			// this doesn't allow for calling the first element as -size_... 
			// but why would you really want to that anyway?
			if (std::abs(inIndex) > static_cast<int64>(size_ - 1))
			{
				std::string errStr = "ERROR: Input index " + num2str(inIndex) + " is out of bounds for array of size " + num2str(size_) + ".";
				throw std::invalid_argument(errStr);
			}

			return this->operator[](inIndex);
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
			// this doesn't allow for calling the first element as -size_... 
			// but why would you really want to that anyway?
			if (std::abs(inRowIndex) > static_cast<int32>(shape_.rows - 1))
			{
				std::string errStr = "ERROR: Row index " + num2str(inRowIndex) + " is out of bounds for array of size " + num2str(shape_.rows) + ".";
				throw std::invalid_argument(errStr);
			}

			// this doesn't allow for calling the first element as -size_... 
			// but why would you really want to that anyway?
			if (std::abs(inColIndex) > static_cast<int32>(shape_.cols - 1))
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
			// this doesn't allow for calling the first element as -size_... 
			// but why would you really want to that anyway?
			if (std::abs(inRowIndex) > static_cast<int32>(shape_.rows - 1))
			{
				std::string errStr = "ERROR: Row index " + num2str(inRowIndex) + " is out of bounds for array of size " + num2str(shape_.rows) + ".";
				throw std::invalid_argument(errStr);
			}

			// this doesn't allow for calling the first element as -size_... 
			// but why would you really want to that anyway?
			if (std::abs(inColIndex) > static_cast<int32>(shape_.cols - 1))
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
		//						const 1D access method with bounds checking
		//		
		// Inputs:
		//				initializer list
		// Outputs:
		//				Ndarray
		//
		NdArray<dtype> at(const std::initializer_list<int32>& inList) const
		{
			// the slice operator already provides bounds checking. just including
			// the at method for completeness
			return this->operator[](inList);
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
		//						const 2D access method with bounds checking
		//		
		// Inputs:
		//				Row Slice,
		//				Column Slice
		// Outputs:
		//				Ndarray
		//
		NdArray<dtype> at(const std::initializer_list<int32>& inRowList, const std::initializer_list<int32>& inColList) const
		{
			// the slice operator already provides bounds checking. just including
			// the at method for completeness
			return this->operator()(inRowList, inColList);
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
		//						and for 1-D arrays to inner product of vectors. 
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				dot product
		//
		template<typename dtypeOut>
		dtypeOut dot(const NdArray<dtype>& inOtherArray) const
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
		//						Return the array with the same data viewed with a 
		//						different byte order. only works for integer types, 
		//						floating point types will not compile and you will
		//						be confused as to why...
		//		
		// Inputs:
		//				Endian::Type
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> newbyteorder(Endian::Type inEndiness) const
		{
			// only works with integer types
			static_assert(std::numeric_limits<dtype>::is_integer, "Type Error: Can only compile newbyteorder method of NdArray<T> with integer types.");

			switch (endianess_)
			{
				case Endian::NATIVE:
				{
					switch (inEndiness)
					{
						case Endian::NATIVE:
						{
							return NdArray(*this);
						}
						case Endian::BIG:
						{
							NdArray<dtype> outArray(shape_);
							for (uint32 i = 0; i < size_; ++i)
							{
								outArray[i] = boost::endian::native_to_big<dtype>(array_[i]);
							}

							outArray.endianess_ = Endian::BIG;
							return outArray;
						}
						case Endian::LITTLE:
						{
							NdArray<dtype> outArray(shape_);
							for (uint32 i = 0; i < size_; ++i)
							{
								outArray[i] = boost::endian::native_to_little<dtype>(array_[i]);
							}

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
				}
				case Endian::BIG:
				{
					switch (inEndiness)
					{
						case Endian::NATIVE:
						{
							NdArray<dtype> outArray(shape_);
							for (uint32 i = 0; i < size_; ++i)
							{
								outArray[i] = boost::endian::big_to_native<dtype>(array_[i]);
							}

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
							for (uint32 i = 0; i < size_; ++i)
							{
								outArray[i] = boost::endian::native_to_little<dtype>(boost::endian::big_to_native<dtype>(array_[i]));
							}

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
				}
				case Endian::LITTLE:
				{
					switch (inEndiness)
					{
						case Endian::NATIVE:
						{
							NdArray<dtype> outArray(shape_);
							for (uint32 i = 0; i < size_; ++i)
							{
								outArray[i] = boost::endian::little_to_native<dtype>(array_[i]);
							}

							outArray.endianess_ = Endian::NATIVE;
							return outArray;
						}
						case Endian::BIG:
						{
							NdArray<dtype> outArray(shape_);
							for (uint32 i = 0; i < size_; ++i)
							{
								outArray[i] = boost::endian::native_to_big<dtype>(boost::endian::little_to_native<dtype>(array_[i]));
							}

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
		//						Return the indices of the flattened array of the 
		//						elements that are non-zero.
		//		
		// Inputs:
		//				None
		// Outputs:
		//				NdArray
		//
		NdArray<uint32> nonzero() const
		{
			std::vector<uint32> indices;
			for (uint32 i = 0; i < size_; ++i)
			{
				if (array_[i] != static_cast<dtype>(0))
				{
					indices.push_back(i);
				}
			}

			return NdArray<uint32>(indices);
		}

		//============================================================================
		// Method Description: 
		//						Returns the norm of the array
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				norm
		//
		template<typename dtypeOut>
		NdArray<dtypeOut> norm(Axis::Type inAxis = Axis::NONE) const
		{
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<dtypeOut> returnArray(1, 1);
					dtypeOut sumOfSquares = 0;
					for (uint32 i = 0; i < size_; ++i)
					{
						sumOfSquares += static_cast<dtypeOut>(sqr(array_[i]));
					}
					returnArray[0] = std::sqrt(sumOfSquares);
					return returnArray;
				}
				case Axis::COL:
				{
					NdArray<dtypeOut> returnArray(1, shape_.rows);
					for (uint32 row = 0; row < shape_.rows; ++row)
					{
						dtypeOut sumOfSquares = 0;
						for (uint32 col = 0; col < shape_.cols; ++col)
						{
							sumOfSquares += static_cast<dtypeOut>(sqr(this->operator()(row, col)));
						}
						returnArray(0, row) = std::sqrt(sumOfSquares);
					}

					return returnArray;
				}
				case Axis::ROW:
				{
					NdArray<dtype> transposedArray = transpose();
					NdArray<dtypeOut> returnArray(1, transposedArray.shape_.rows);
					for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
					{
						dtypeOut sumOfSquares = 0;
						for (uint32 col = 0; col < transposedArray.shape_.cols; ++col)
						{
							sumOfSquares += static_cast<dtypeOut>(sqr(transposedArray(row, col)));
						}
						returnArray(0, row) = std::sqrt(sumOfSquares);
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
		//						Fills the array with ones
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		void ones()
		{
			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] = static_cast<dtype>(1);
			}
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
		void partition(uint32 inKth, Axis::Type inAxis = Axis::NONE)
		{
			switch (inAxis)
			{
				case Axis::NONE:
				{
					if (inKth >= size_)
					{
						std::string errStr = "ERROR: kth(=" + num2str(inKth) + ") out of bounds (" + num2str(size_) + ")";
						throw std::invalid_argument(errStr);
					}
					std::nth_element(begin(), begin() + inKth, end());
					break;
				}
				case Axis::COL:
				{
					if (inKth >= shape_.cols)
					{
						std::string errStr = "ERROR: kth(=" + num2str(inKth) + ") out of bounds (" + num2str(shape_.cols) + ")";
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
						std::string errStr = "ERROR: kth(=" + num2str(inKth) + ") out of bounds (" + num2str(shape_.rows) + ")";
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
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<dtypeOut> returnArray(1, 1);
					dtypeOut product = 1;
					for (uint32 i = 0; i < size_; ++i)
					{
						product *= static_cast<dtypeOut>(array_[i]);
					}
					returnArray[0] = product;
					return returnArray;
				}
				case Axis::COL:
				{
					NdArray<dtypeOut> returnArray(1, shape_.rows);
					for (uint32 row = 0; row < shape_.rows; ++row)
					{
						dtypeOut product = 1;
						for (uint32 col = 0; col < shape_.cols; ++col)
						{
							product *= static_cast<dtypeOut>(this->operator()(row, col));
						}
						returnArray(0, row) = product;
					}

					return returnArray;
				}
				case Axis::ROW:
				{
					NdArray<dtype> transposedArray = transpose();
					NdArray<dtypeOut> returnArray(1, transposedArray.shape_.rows);
					for (uint32 row = 0; row < transposedArray.shape_.rows; ++row)
					{
						dtypeOut product = 1;
						for (uint32 col = 0; col < transposedArray.shape_.cols; ++col)
						{
							product *= static_cast<dtypeOut>(transposedArray(row, col));
						}
						returnArray(0, row) = product;
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
		//						Peak to peak (maximum - minimum) value along a given axis.
		//		
		// Inputs:
		//				(Optional) Axis
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> ptp(Axis::Type inAxis = Axis::NONE) const
		{
			switch (inAxis)
			{
				case Axis::NONE:
				{
					NdArray<dtype> returnArray(1, 1);
					std::pair<const dtype*, const dtype*> result = std::minmax_element(cbegin(), cend());
					returnArray[0] = *result.second - *result.first;
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
		//						set the flat index element to the value
		//		
		// Inputs:
		//				index
		//				value
		// Outputs:
		//				None
		//
		void put(int32 inIndex, dtype inValue)
		{
			at(inIndex) = inValue;
		}

		//============================================================================
		// Method Description: 
		//						set the 2D row/col index element to the value
		//		
		// Inputs:
		//				row index
		//				col index
		//				value
		// Outputs:
		//				None
		//
		void put(int32 inRow, int32 inCol, dtype inValue)
		{
			at(inRow, inCol) = inValue;
		}

		//============================================================================
		// Method Description: 
		//						Set a.flat[n] = values for all n in indices.
		//		
		// Inputs:
		//				NdArray of indices
		//				value
		// Outputs:
		//				None
		//
		void put(const NdArray<uint32>& inIndices, dtype inValue)
		{
			for (uint32 i = 0; i < inIndices.size(); ++i)
			{
				put(inIndices[i], value);
			}
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
		void put(const NdArray<uint32>& inIndices, const NdArray<dtype>& inValues)
		{
			if (inIndices.size() != inValues.size())
			{
				throw std::invalid_argument("Error: Input indices do not match values dimensions.");
			}

			for (uint32 i = 0; i < inIndices.size(); ++i)
			{
				put(inIndices[i], inValues[i]);
			}
		}

		//============================================================================
		// Method Description: 
		//						Set the slice indices to the input value.
		//		
		// Inputs:
		//				Slice 1D
		//				value
		// Outputs:
		//				None
		//
		void put(const Slice& inSlice, dtype inValue)
		{
			Slice inSliceCopy(inSlice);
			inSliceCopy.makePositiveAndValidate(size_);

			for (int32 i = inSliceCopy.start; i < inSliceCopy.stop; i += inSliceCopy.step)
			{
				put(i, inValue);
			}
		}

		//============================================================================
		// Method Description: 
		//						Set the slice indices to the input value.
		//		
		// Inputs:
		//				initializer_list<int32>
		//				value
		// Outputs:
		//				None
		//
		void put(const std::initializer_list<int32>& inSliceList, dtype inValue)
		{
			put(Slice(inSliceList), inValue);
		}

		//============================================================================
		// Method Description: 
		//						Set the slice indices to the input values.
		//		
		// Inputs:
		//				Slice 1D
		//				NdArray of values
		// Outputs:
		//				None
		//
		void put(const Slice& inSlice, const NdArray<dtype>& inValues)
		{
			Slice inSliceCopy(inSlice);
			inSliceCopy.makePositiveAndValidate(size_);

			std::vector<uint32> indices;
			for (int32 i = inSliceCopy.start; i < inSliceCopy.stop; i += inSliceCopy.step)
			{
				indices.push_back(i);
			}

			put(NdArray<uint32>(indices), inValues);
		}

		//============================================================================
		// Method Description: 
		//						Set the slice indices to the input values.
		//		
		// Inputs:
		//				initializer_list<int32>
		//				NdArray of values
		// Outputs:
		//				None
		//
		void put(const std::initializer_list<int32>& inSliceList, const NdArray<dtype>& inValues)
		{
			put(Slice(inSliceList), inValues);
		}

		//============================================================================
		// Method Description: 
		//						Set the slice indices to the input values.
		//		
		// Inputs:
		//				Slice rows
		//				Slice cols
		//				value
		// Outputs:
		//				None
		//
		void put(const Slice& inRowSlice, const Slice& inColSlice, dtype inValue)
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
		}

		//============================================================================
		// Method Description: 
		//						Set the slice indices to the input values.
		//		
		// Inputs:
		//				initializer_list<int32> row slice
		//				initializer_list<int32> col slice
		//				value
		// Outputs:
		//				None
		//
		void put(const std::initializer_list<int32>& inRowSliceList, const std::initializer_list<int32>& inColSliceList, dtype inValue)
		{
			put(Slice(inRowSliceList), Slice(inColSliceList), inValue);
		}

		//============================================================================
		// Method Description: 
		//						Set the slice indices to the input values.
		//		
		// Inputs:
		//				Slice rows
		//				Slice cols
		//				NdArray of values
		// Outputs:
		//				None
		//
		void put(const Slice& inRowSlice, const Slice& inColSlice, const NdArray<dtype>& inValues)
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
					uint32 index = row * shape_.cols + col;
					indices.push_back(index);
				}
			}

			put(NdArray<uint32>(indices), inValues);
		}

		//============================================================================
		// Method Description: 
		//						Set the slice indices to the input values.
		//		
		// Inputs:
		//				initializer_list<int32> row slice
		//				initializer_list<int32> col slice
		//				NdArray of values
		// Outputs:
		//				None
		//
		void put(const std::initializer_list<int32>& inRowSliceList, const std::initializer_list<int32>& inColSliceList, const NdArray<dtype>& inValues)
		{
			put(Slice(inRowSliceList), Slice(inColSliceList), inValues);
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
		void reshape(uint32 inNumRows, uint32 inNumCols)
		{
			if (inNumRows * inNumCols != size_)
			{
				std::string errStr = "ERROR: Cannot reshape array of size " + num2str(size_) + " into shape ";
				errStr += "[" + num2str(inNumRows) + ", " + num2str(inNumCols) + "]";
				throw std::runtime_error(errStr);
			}

			shape_.rows = inNumRows;
			shape_.cols = inNumCols;
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
			reshape(inShape.rows, inShape.cols);
		}

		//============================================================================
		// Method Description: 
		//						Returns an array containing the same data with a new shape.
		//		
		// Inputs:
		//				initializer_list<uint32>
		// Outputs:
		//				None
		//
		void reshape(const std::initializer_list<uint32>& inShapeList)
		{
			reshape(Shape(inShapeList));
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
			size_ = shape_.size();
			deleteArray();
			array_ = new dtype[size_];
			zeros();
		}

		//============================================================================
		// Method Description: 
		//						Change shape and size of array in-place. All previous
		//						data of the array is lost.
		//		
		// Inputs:
		//				initializer_list<uint32>
		// Outputs:
		//				None
		//
		void resizeFast(const std::initializer_list<uint32>& inShapeList)
		{
			resizeFast(Shape(inShapeList));
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
			dtype* oldData = new dtype[size_];
			std::copy(begin(), end(), oldData);

			deleteArray();
			array_ = new dtype[inShape.size()];
			for (uint16 row = 0; row < inShape.rows; ++row)
			{
				for (uint16 col = 0; col < inShape.cols; ++col)
				{
					if (row >= shape_.rows || col >= shape_.cols)
					{
						this->operator()(row, col) = static_cast<dtype>(0); // zero fill
					}
					else
					{
						this->operator()(row, col) = oldData[row * inShape.cols + col];
					}
				}
			}

			delete[] oldData;
			shape_ = inShape;
			size_ = inShape.size();
		}

		//============================================================================
		// Method Description: 
		//						Change shape and size of array in-place. All data outide
		//						of new size is lost.
		//		
		// Inputs:
		//				initializer_list<uint32>
		// Outputs:
		//				None
		//
		void resizeSlow(const std::initializer_list<uint32>& inShapeList)
		{
			resizeSlow(Shape(inShapeList));
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
		NdArray<dtype> round(uint8 inNumDecimals=0) const
		{
			if (std::numeric_limits<dtype>::is_integer())
			{
				return NdArray<dtype>(*this);
			}
			else
			{
				NdArray<dtype> returnArray(shape_);
				double multFactor = power(10.0, inNumDecimals);
				for (uint32 i = 0; i < size_; ++i)
				{
					returnArray[i] = static_cast<dtype>(std::round(static_cast<double>(array_[i]) * multFactor) / multFactor);
				}

				return returnArray;
			}
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
			// this should have all of the operator and equivalants to put method overloads
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
		//						Returns the variance of the array elements, along given axis.
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
			return NdArray<dtype>(*this) += inOtherArray;
		}

		//============================================================================
		// Method Description: 
		//						Adds the scalar to the array
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator+(dtype inScalar) const
		{
			return NdArray<dtype>(*this) += inScalar;
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
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] += inOtherArray.array_[i];
			}

			return *this;
		}

		//============================================================================
		// Method Description: 
		//						Adds the scalar to the array
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator+=(dtype inScalar)
		{
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] += inScalar;
			}

			return *this;
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
			return NdArray<dtype>(*this) -= inOtherArray;
		}

		//============================================================================
		// Method Description: 
		//						Subtracts the scalar from the array
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator-(dtype inScalar) const
		{
			return NdArray<dtype>(*this) -= inScalar;
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
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] -= inOtherArray.array_[i];
			}

			return *this;
		}

		//============================================================================
		// Method Description: 
		//						Subtracts the scalar from the array
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator-=(dtype inScalar)
		{
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] -= inScalar;
			}

			return *this;
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
			return NdArray<dtype>(*this) *= inOtherArray;
		}

		//============================================================================
		// Method Description: 
		//						Muliplies the scalar to the array
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator*(dtype inScalar) const
		{
			return NdArray<dtype>(*this) *= inScalar;
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
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] *= inOtherArray.array_[i];
			}

			return *this;
		}

		//============================================================================
		// Method Description: 
		//						Muliplies the scalar to the array
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator*=(dtype inScalar)
		{
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] *= inScalar;
			}

			return *this;
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
			return NdArray<dtype>(*this) /= inOtherArray;
		}

		//============================================================================
		// Method Description: 
		//						Divides the array by the scalar
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator/(dtype inScalar) const
		{
			return NdArray<dtype>(*this) /= inScalar;
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
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] /= inOtherArray.array_[i];
			}

			return *this;
		}

		//============================================================================
		// Method Description: 
		//						Divides the array by the scalar
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator/=(dtype inScalar)
		{
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] /= inScalar;
			}

			return *this;
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
			return NdArray<dtype>(*this) %= inOtherArray;
		}

		//============================================================================
		// Method Description: 
		//						Modulus of the array and the scalar
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator%(dtype inScalar) const
		{
			return NdArray<dtype>(*this) %= inScalar;
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
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] %= inOtherArray.array_[i];
			}

			return *this;
		}

		//============================================================================
		// Method Description: 
		//						Modulus of the array and the scalar
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator%=(dtype inScalar)
		{
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] %= inScalar;
			}

			return *this;
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
			return NdArray<dtype>(*this) |= inOtherArray;
		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise or of the array and the scalar
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator|(dtype inScalar) const
		{
			return NdArray<dtype>(*this) |= inScalar;
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
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] |= inOtherArray.array_[i];
			}

			return *this;
		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise or of the array and the scalar
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator|=(dtype inScalar)
		{
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] |= inScalar;
			}

			return *this;
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
			return NdArray<dtype>(*this) &= inOtherArray;
		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise and of the array and the scalar
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator&(dtype inScalar) const
		{
			return NdArray<dtype>(*this) &= inScalar;
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
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] &= inOtherArray.array_[i];
			}

			return *this;
		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise and of the array and the scalar
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator&=(dtype inScalar)
		{
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] &= inScalar;
			}

			return *this;
		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise xor of the elements of two arrays
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype> operator^(const NdArray<dtype>& inOtherArray)
		{
			return NdArray<dtype>(*this) ^= inOtherArray;
		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise xor of the array and the scalar
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype> operator^(dtype inScalar) const
		{
			return NdArray<dtype>(*this) ^= inScalar;
		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise xor of the elements of two arrays
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		template<typename dtype>
		NdArray<dtype>& operator^=(const NdArray<dtype>& inOtherArray)
		{
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] ^= inOtherArray.array_[i];
			}

			return *this;
		}

		//============================================================================
		// Method Description: 
		//						Takes the bitwise xor of the array and the scalar
		//		
		// Inputs:
		//				scalar
		// Outputs:
		//				NdArray
		//
		NdArray<dtype>& operator^=(dtype inScalar)
		{
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] ^= inScalar;
			}

			return *this;
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
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			NdArray<bool> returnArray(shape_);
			for (uint32 i = 0; i < size_; ++i)
			{
				returnArray.array_[i] = array_[i] == inOtherArray.array_[i];
			}

			return returnArray;
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
			if (shape_ != inOtherArray.shape_)
			{
				throw std::invalid_argument("ERROR: Array dimensions do not match.");
			}

			NdArray<bool> returnArray(shape_);
			for (uint32 i = 0; i < size_; ++i)
			{
				returnArray.array_[i] = array_[i] != inOtherArray.array_[i];
			}

			return returnArray;
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
			return NdArray<dtype>(*this) <<= inNumBits;
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
			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] << inNumBits;
			}

			return *this;
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
			return NdArray<dtype>(*this) >>= inNumBits;
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
			for (uint32 i = 0; i < size_; ++i)
			{
				array_[i] >> inNumBits;
			}

			return *this;
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
			for (uint32 i = 0; i < size_; ++i)
			{
				++array_[i];
			}

			return *this;
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
			for (uint32 i = 0; i < size_; ++i)
			{
				--array_[i];
			}

			return *this;
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
			NdArray<dtype> copy(*this);
			for (uint32 i = 0; i < size_; ++i)
			{
				++array_[i];
			}

			return copy;
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
			NdArray<dtype> copy(*this);
			for (uint32 i = 0; i < size_; ++i)
			{
				--array_[i];
			}

			return copy;
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
		friend std::ostream& operator<<(std::ostream& inOStream, const NdArray<dtype>& inArray)
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