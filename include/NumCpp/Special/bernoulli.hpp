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
/// Special Functions
///
#pragma once

#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include "boost/math/special_functions/bernoulli.hpp"

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        ///						Both return the nth Bernoulli number B2n.
        ///
        /// @param
        ///				n
        /// @return
        ///				double
        ///
        inline double bernoilli(uint32 n)
        {
            if (n == 1)
            {
                return 0.5;
            }
            if (n % 2 != 0)
            {
                return 0.0;
            }

            return boost::math::bernoulli_b2n<double>(n / 2);
        }

        //============================================================================
        // Method Description:
        ///						Both return the nth Bernoulli number B2n.
        ///
        /// @param
        ///				inArray
        /// @return
        ///				NdArray<double>
        ///
        inline NdArray<double> bernoilli(const NdArray<uint32>& inArray)
        {
            NdArray<double> returnArray(inArray.shape());

            stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
                [](uint32 inValue) -> double
                { 
                    return bernoilli(inValue);
                });

            return returnArray;
        }
    } // namespace special
} // namespace nc
