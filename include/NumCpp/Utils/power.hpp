/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.2.0
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
/// Raises the input value to an integer power
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Types.hpp"

#include<complex>

namespace nc
{
    namespace utils
    {
        //============================================================================
        ///						Raises the input value to an integer power
        ///
        /// @param      inValue
        /// @param      inPower
        ///
        /// @return     inValue raised to inPower
        ///
        template<typename dtype>
        dtype power(dtype inValue, uint8 inPower) noexcept 
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            if (inPower == 0)
            {
                return static_cast<dtype>(1);
            }

            dtype returnVal = inValue;
            for (uint8 exponent = 1; exponent < inPower; ++exponent)
            {
                returnVal *= inValue;
            }
            return returnVal;
        }
    }  // namespace utils
}  // namespace nc
