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

namespace NumC
{
	//============================================================================
	// Method Description: 
	//						Squares in input value
	//		
	// Inputs:
	//				T
	// Outputs:
	//				T
	//
	template<typename T>
	T sqr(T inValue)
	{
		return inValue * inValue;
	}

	//============================================================================
	// Method Description: 
	//						Cubes in input value
	//		
	// Inputs:
	//				T
	// Outputs:
	//				T
	//
	template<typename T>
	T cube(T inValue)
	{
		return inValue * inValue * inValue;
	}

	//============================================================================
	// Method Description: 
	//						raises the input value to a power
	//		
	// Inputs:
	//				T
	// Outputs:
	//				T
	//
	template<typename T>
	T power(T inValue, uint8 inPower)
	{
		T returnVal = 1;
		for (uint8 i = 0; i < inPower; ++i)
		{
			returnVal *= inValue;
		}
		return returnVal;
	}
}
