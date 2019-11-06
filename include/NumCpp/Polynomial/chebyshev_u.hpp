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

#include "boost/math/special_functions/chebyshev.hpp"

#include <algorithm>

namespace nc
{
    namespace polynomial
    {
        //============================================================================
        // Method Description:
        ///	Chebyshev Polynomial of the second kind
        ///
        /// @param      n: the order of the chebyshev polynomial
        /// @param      x: the input value
        /// @return
        ///				double
        ///
        template<typename dtype>
        double chebyshev_u(uint32 n, dtype x) noexcept
        {
            return boost::math::chebyshev_u(n, static_cast<double>(x));
        }

        //============================================================================
        // Method Description:
        ///	Chebyshev Polynomial of the second kind
        ///
        /// @param      n: the order of the chebyshev polynomial
        /// @param      inArrayX: the input value
        /// @return
        ///				NdArray<double>
        ///
        template<typename dtype>
        NdArray<double> chebyshev_u(uint32 n, const NdArray<dtype>& inArrayX) noexcept
        {
            NdArray<double> returnArray(inArrayX.shape());

            auto function = [n](dtype x) -> double
            {
                return chebyshev_u(n, x);
            };

            stl_algorithms::transform(inArrayX.cbegin(), inArrayX.cend(), returnArray.begin(), function);

            return returnArray;
        }
    }
}
