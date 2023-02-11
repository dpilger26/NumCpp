/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
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

#include <cmath>

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"

#ifndef __cpp_lib_math_special_functions
#include "boost/math/special_functions/bessel.hpp"
#endif

#include <type_traits>

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        /// Modified Cylindrical Bessel function of the second kind.
        /// NOTE: Use of this function requires either using the Boost
        /// includes or a C++17 compliant compiler.
        ///
        /// @param inV: the order of the bessel function
        /// @param inX: the input value
        /// @return calculated-result-type
        ///
        template<typename dtype1, typename dtype2>
        auto bessel_kn(dtype1 inV, dtype2 inX)
        {
            STATIC_ASSERT_ARITHMETIC(dtype1);
            STATIC_ASSERT_ARITHMETIC(dtype2);

#ifdef __cpp_lib_math_special_functions
            return std::cyl_bessel_k(static_cast<double>(inV), static_cast<double>(inX));
#else
            return boost::math::cyl_bessel_k(static_cast<double>(inV), static_cast<double>(inX));
#endif
        }

        //============================================================================
        // Method Description:
        /// Modified Cylindrical Bessel function of the second kind.
        /// NOTE: Use of this function requires either using the Boost
        /// includes or a C++17 compliant compiler.
        ///
        /// @param inV: the order of the bessel function
        /// @param inArrayX: the input values
        /// @return NdArray
        ///
        template<typename dtype1, typename dtype2>
        auto bessel_kn(dtype1 inV, const NdArray<dtype2>& inArrayX)
        {
            NdArray<decltype(bessel_kn(dtype1{ 0 }, dtype2{ 0 }))> returnArray(inArrayX.shape());

            stl_algorithms::transform(
                inArrayX.cbegin(),
                inArrayX.cend(),
                returnArray.begin(),
                [inV](dtype2 inX) -> auto{ return bessel_kn(inV, inX); });

            return returnArray;
        }
    } // namespace special
} // namespace nc

#endif // #if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
