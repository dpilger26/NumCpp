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

#include<initializer_list>
#include<stdexcept>
#include<iostream>

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
		uint32	step;

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
		Slice(int32 inStop) :
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
		{
			isValid();
		}

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
		Slice(int32 inStart, int32 inStop, uint32 inStep) :
			start(inStart),
			stop(inStop),
			step(inStep)
		{
			isValid();
		}

		//============================================================================
		// Method Description: 
		//						Constructor
		//		
		// Inputs:
		//				initializer list. ie. {3, 4}
		// Outputs:
		//				None
		//
		Slice(const std::initializer_list<int32>& inList) :
			start(0),
			stop(0),
			step(0)
		{
			if (inList.size() > 3)
			{
				throw std::invalid_argument("ERROR: Slice initializer list must contain no more than 3 elements.");
			}

			switch (inList.size())
			{
				case 1:
				{
					start = 0;
					stop = *inList.begin();
					step = 1;
					break;
				}
				case 2:
				{
					start = *inList.begin();
					stop = *(inList.begin() + 1);
					step = 1;
					break;
				}
				case 3:
				{
					start = *inList.begin();
					stop = *(inList.begin() + 1);
					step = *(inList.begin() + 2);
					break;
				}
			}

			isValid();
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

	private:
		//============================================================================
		// Method Description:
		//						Checks that the slice is valid. Start must come before end
		//			
		// Inputs:
		//				None
		// Outputs:
		//				None
		//
		void isValid() const
		{
			// check that stop is after start for the two cases where start and stop are the same sign.
			// for the other cases... can't really check anything.
			if (start > -1 && stop > -1 && start >= stop)
			{
				throw std::invalid_argument("ERROR: Invalid start stop combination.");
			}
			else if (start < 0 && stop < 0 && start >= stop)
			{
				throw std::invalid_argument("ERROR: Invalid start stop combination.");
			}
		}
	};
}