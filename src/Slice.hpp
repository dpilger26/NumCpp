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
#include"Utils.hpp"

#include<iostream>
#include<stdexcept>

namespace NumC
{
	//================================================================================
	// Class Description:
	//						An object for slicing into NdArrays
	//
	class Slice
	{
	public:
		//====================================Attributes==============================
		int32	start;
		int32	stop;
		int32	step;

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		Slice() :
			start(0),
			stop(1),
			step(1)
		{};

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				stop index (not included)
		// Outputs:
		//				None
		//
		explicit Slice(int32 inStop) :
			start(0),
			stop(inStop),
			step(1)
		{};

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				start index,
		//				stop index (not included)
		// Outputs:
		//				None
		//
		Slice(int32 inStart, int32 inStop) :
			start(inStart),
			stop(inStop),
			step(1)
		{};

		//============================================================================
		// Method Description:
		//						Constructor
		//			
		// Inputs:
		//				start index,
		//				stop index (not included)
		//				step value
		// Outputs:
		//				None
		//
		Slice(int32 inStart, int32 inStop, int32 inStep) :
			start(inStart),
			stop(inStop),
			step(inStep)
		{};

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
		//						io operator for the Slice class
		//		
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		friend std::ostream& operator<<(std::ostream& inOStream, const Slice& inSlice)
		{
			inOStream << "[" << inSlice.start << ":" << inSlice.stop << ":" << inSlice.step << "]" << std::endl;
			return inOStream;
		}

		//============================================================================
		// Method Description:
		//						Make the slice all positive and does some error checking
		//			
		// Inputs:
		//				The calling array size
		// Outputs:
		//				None
		//
		void makePositiveAndValidate(uint32 inArraySize)
		{
			// convert the start value
			if (start < 0)
			{
				start += inArraySize;
			}
			if (start > static_cast<int32>(inArraySize - 1))
			{
				throw std::invalid_argument("ERROR: Invalid start value for array of size " + num2str(inArraySize) + ".");
			}

			// convert the stop value
			if (stop < 0)
			{
				stop += inArraySize;
			}
			if (stop > static_cast<int32>(inArraySize))
			{
				throw std::invalid_argument("ERROR: Invalid stop value for array of size " + num2str(inArraySize) + ".");
			}

			// do some error checking
			if (start < stop)
			{
				if (step < 0)
				{
					throw std::invalid_argument("ERROR: Invalid slice values.");
				}
			}

			if (stop < start)
			{
				if (step > 0)
				{
					throw std::invalid_argument("ERROR: Invalid slice values.");
				}

				// otherwise flip things around for my own sanity
				std::swap(start, stop);
				step *= -1;
			}
		}

		//============================================================================
		// Method Description:
		//						returns the number of elements that the slice contains.
		//						be aware that this method will also make the slice all 
		//						positive!
		//			
		// Inputs:
		//				The calling array size
		// Outputs:
		//				None
		//
		uint32 numElements(uint32 inArraySize)
		{
			makePositiveAndValidate(inArraySize);

			uint32 num = 0;
			for (int32 i = start; i < stop; i += step)
			{
				++num;
			}
			return num;
		}
	};
}