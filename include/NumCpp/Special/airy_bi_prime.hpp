/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
///
/// @section License
/// Copyright 2019 David Pilger
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
/// Special Functions
///
#pragma once

#include "NumCpp/NdArray.hpp"

#include "boost/math/special_functions/airy.hpp"

#include <algorithm>

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        ///	The derivative of the second linearly independent solution to the differential equation y'' - yz = 0.
        /// http://mathworld.wolfram.com/AiryFunctions.html
        ///
        /// @param
        ///				inValue
        /// @return
        ///				double
        ///
        template<typename dtype>
        double airy_bi_prime(dtype inValue) noexcept
        {
            return boost::math::airy_bi_prime(static_cast<double>(inValue));
        }

        //============================================================================
        // Method Description:
        ///	The derivative of the second linearly independent solution to the differential equation y'' - yz = 0.
        /// http://mathworld.wolfram.com/AiryFunctions.html
        ///
        /// @param
        ///				inArray
        /// @return
        ///				NdArray<double>
        ///
        template<typename dtype>
        NdArray<double> airy_bi_prime(const NdArray<dtype>& inArray) noexcept
        {
            NdArray<double> returnArray(inArray.shape());

            std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
                [](dtype inValue) -> double
                { return airy_bi_prime(inValue); });

            return returnArray;
        }
    }
}
