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

#include "boost/math/special_functions/beta.hpp"

#include <type_traits>

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        ///	The beta function
        ///
        /// @param      a
        /// @param      b
        /// @return
        ///				calculated-result-type 
        ///
        template<typename dtype1, typename dtype2>
        auto beta(dtype1 a, dtype2 b)
        {
            STATIC_ASSERT_ARITHMETIC(dtype1);
            STATIC_ASSERT_ARITHMETIC(dtype2);

            return boost::math::beta(a, b);
        }

        //============================================================================
        // Method Description:
        ///	The beta function
        ///
        /// @param      inArrayA
        /// @param      inArrayB
        /// @return
        ///				NdArray
        ///
        template<typename dtype1, typename dtype2>
        auto beta(const NdArray<dtype1, Alloc>& inArrayA, const NdArray<dtype2, Alloc>& inArrayB)
        {
            NdArray<decltype(bessel_in(dtype1{ 0 }, dtype2{ 0 })), Alloc> returnArray(inArrayB.shape());

            stl_algorithms::transform(inArrayA.cbegin(), inArrayA.cend(), inArrayB.cbegin(), returnArray.begin(),
                [](dtype1 a, dtype2 b) -> auto
                { 
                    return beta(a, b); 
                });

            return returnArray;
        }
    }
}
