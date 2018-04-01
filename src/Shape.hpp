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

#include<iostream>
#include<stdexcept>

namespace NumC
{
	//================================================================================
	// Class Description:
	//						An object for slicing into NdArrays
	//
	class Shape
	{
	public:
		//====================================Attributes==============================
		uint32	rows;
		uint32	cols;

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				number of rows,
		//				number of cols
		// Outputs:
		//				None
		//
		Shape() :
			rows(0),
			cols(0)
		{};

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				number of rows and cols
		// Outputs:
		//				None
		//
		explicit Shape(uint32 inSquareSize) :
			rows(inSquareSize),
			cols(inSquareSize)
		{};

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				number of rows,
		//				number of cols
		// Outputs:
		//				None
		//
		Shape(uint32 inRows, uint32 inCols) :
			rows(inRows),
			cols(inCols)
		{};

		//============================================================================
		// Method Description: 
		//						equality operator
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		bool operator==(const Shape& inOtherShape) const
		{
			return rows == inOtherShape.rows && cols == inOtherShape.cols;
		}

		//============================================================================
		// Method Description: 
		//						not equality operator
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		bool operator!=(const Shape& inOtherShape) const
		{
			return !(*this == inOtherShape);
		}

		//============================================================================
		// Method Description: 
		//						returns the size of the shape
		//		
		// Inputs:
		//				None
		// Outputs:
		//				size
		//
		uint32 size() const
		{
			return rows * cols;
		}

		//============================================================================
		// Method Description: 
		//						returns whether the shape is null (constructed with the 
		//						default constructor).
		//		
		// Inputs:
		//				None
		// Outputs:
		//				bool
		//
		bool isnull ()
		{
			return rows == 0 && cols == 0;
		}

		//============================================================================
		// Method Description: 
		//						prints the shape to the console
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		void print()
		{
			std::cout << *this;
		}

		//============================================================================
		// Method Description: 
		//						io operator for the Shape class
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		friend std::ostream& operator<<(std::ostream& inOStream, const Shape& inShape)
		{
			inOStream << "[" << inShape.rows << ", " << inShape.cols << "]" << std::endl;
			return inOStream;
		}
	};
}
