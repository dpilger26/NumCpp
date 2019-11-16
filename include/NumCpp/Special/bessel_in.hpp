/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.2
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
#include "NumCpp/Core/StlAlgorithms.hpp"

#include "boost/math/special_functions/bessel.hpp"

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        ///	Modified Cylindrical Bessel function of the first kind
        ///
        /// @param      inV: the order of the bessel function
        /// @param      inX: the input value
        /// @return
        ///				double
        ///
        template<typename dtype1, typename dtype2>
        double bessel_in(dtype1 inV, dtype2 inX) noexcept
        {
            return boost::math::cyl_bessel_i(static_cast<double>(inV), static_cast<double>(inX));
        }

        //============================================================================
        // Method Description:
        ///	Modified Cylindrical Bessel function of the first kind
        ///
        /// @param      inV: the order of the bessel function
        /// @param      inArrayX: the input values
        /// @return
        ///				NdArray<double>
        ///
        template<typename dtype1, typename dtype2>
        NdArray<double> bessel_in(dtype1 inV, const NdArray<dtype2>& inArrayX) noexcept
        {
            NdArray<double> returnArray(inArrayX.shape());

            stl_algorithms::transform(inArrayX.cbegin(), inArrayX.cend(), returnArray.begin(),
                [inV](dtype2 inX) -> double
                {
                    return bessel_in(inV, inX); 
                });

            return returnArray;
        }
    }
}
