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

#include<limits>

namespace NumC
{
    //================================================================================
    // Class Description:
    //						holds info about the dtype
    //
    template<typename dtype>
    class DtypeInfo
    {
    public:
        //============================================================================
        // Method Description: 
        //						For integer types: number of non-sign bits in the representation.
        //						For floating types : number of digits(in radix base) in the mantissa
        //		
        // Inputs:
        //				None
        // Outputs:
        //				number of bits
        //
        static constexpr dtype bits()
        {
            return std::numeric_limits<dtype>::digits;
        }

        //============================================================================
        // Method Description: 
        //						Machine epsilon (the difference between 1 and the least 
        //						value greater than 1 that is representable).
        //		
        // Inputs:
        //				None
        // Outputs:
        //				dtype
        //
        static constexpr dtype epsilon()
        {
            return std::numeric_limits<dtype>::epsilon();
        }

        //============================================================================
        // Method Description: 
        //						true if type is integer.
        //		
        // Inputs:
        //				None
        // Outputs:
        //				bool
        //
        static constexpr bool isInteger()
        {
            return std::numeric_limits<dtype>::is_integer;
        }

        //============================================================================
        // Method Description: 
        //						true if type is signed.
        //		
        // Inputs:
        //				None
        // Outputs:
        //				bool
        //
        static constexpr bool isSigned()
        {
            return std::numeric_limits<dtype>::is_signed;
        }

        //============================================================================
        // Method Description: 
        //						Returns the minimum value of the dtype
        //		
        // Inputs:
        //				None
        // Outputs:
        //				min value
        //
        static constexpr dtype min()
        {
            return std::numeric_limits<dtype>::min();
        }

        //============================================================================
        // Method Description: 
        //						Returns the maximum value of the dtype
        //		
        // Inputs:
        //				None
        // Outputs:
        //				max value
        //
        static constexpr dtype max()
        {
            return std::numeric_limits<dtype>::max();
        }
    };
}
