/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.0.0
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

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"

#include "boost/math/special_functions/bessel.hpp"

#include <type_traits>

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        ///	Cylindrical Bessel function of the first kind
        ///
        /// @param      inV: the order of the bessel function
        /// @param      inX: the input value
        /// @return
        ///				calculated-result-type 
        ///
        template<typename dtype1, typename dtype2>
        auto bessel_jn(dtype1 inV, dtype2 inX)
        {
            STATIC_ASSERT_ARITHMETIC(dtype1);
            STATIC_ASSERT_ARITHMETIC(dtype2);

            return boost::math::cyl_bessel_j(inV, inX);
        }

        //============================================================================
        // Method Description:
        ///	Cylindrical Bessel function of the first kind
        ///
        /// @param      inV: the order of the bessel function
        /// @param      inArrayX: the input values
        /// @return
        ///				NdArray
        ///
        template<typename dtype1, typename dtype2>
        auto bessel_jn(dtype1 inV, const NdArray<dtype2>& inArrayX)
        {
            NdArray<decltype(bessel_in(dtype1{ 0 }, dtype2{ 0 }))> returnArray(inArrayX.shape());

            stl_algorithms::transform(inArrayX.cbegin(), inArrayX.cend(), returnArray.begin(),
                [inV](dtype2 inX) -> auto
                { 
                    return bessel_jn(inV, inX);
                });

            return returnArray;
        }
    }
}
