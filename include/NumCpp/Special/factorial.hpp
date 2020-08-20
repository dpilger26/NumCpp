/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
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
/// Description
/// Special Functions
///
#pragma once

#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include "boost/math/special_functions/factorials.hpp"

#include <limits>

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        /// Returns the factorial of the input value
        ///
        /// @param
        ///				inValue
        /// @return
        ///				double
        ///
        inline double factorial(uint32 inValue)
        {
            if (inValue <= boost::math::max_factorial<double>::value)
            {
                return boost::math::factorial<double>(inValue);   
            }
            
            return std::numeric_limits<double>::infinity();
        }

        //============================================================================
        // Method Description:
        /// Returns the factorial of the input value
        ///
        /// @param
        ///				inArray
        /// @return
        ///				NdArray<double>
        ///
        inline NdArray<double> factorial(const NdArray<uint32>& inArray)
        {
            NdArray<double> returnArray(inArray.shape());

            stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
                [](uint32 inValue) -> double
                { 
                    return factorial(inValue); 
                });

            return returnArray;
        }
    }  // namespace special
}  // namespace nc
