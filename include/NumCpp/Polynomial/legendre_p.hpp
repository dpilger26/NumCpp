/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2022 David Pilger
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

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"

#ifndef __cpp_lib_math_special_functions
#include "boost/math/special_functions/legendre.hpp"
#endif

namespace nc
{
    namespace polynomial
    {
        //============================================================================
        // Method Description:
        /// Legendre Polynomial of the first kind.
        /// NOTE: Use of this function requires either using the Boost
        /// includes or a C++17 compliant compiler.
        ///
        /// @param n: the degree of the legendre polynomial
        /// @param x: the input value. Requires -1 <= x <= 1
        /// @return double
        ///
        template<typename dtype>
        double legendre_p(uint32 n, dtype x)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (x < -1.0 || x > 1.0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input x must be of the range [-1, 1].");
            }

#ifdef __cpp_lib_math_special_functions
            return std::legendre(n, static_cast<double>(x));
#else
            return boost::math::legendre_p(n, static_cast<double>(x));
#endif
        }

        //============================================================================
        // Method Description:
        /// Associated Legendre Polynomial of the first kind.
        /// NOTE: Use of this function requires either using the Boost
        /// includes or a C++17 compliant compiler.
        ///
        /// @param m: the order of the legendre polynomial
        /// @param n: the degree of the legendre polynomial
        /// @param x: the input value. Requires -1 <= x <= 1
        /// @return double
        ///
        template<typename dtype>
        double legendre_p(uint32 m, uint32 n, dtype x)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (x < -1.0 || x > 1.0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input x must be of the range [-1, 1].");
            }

#ifdef __cpp_lib_math_special_functions

            auto value = std::assoc_legendre(n, m, static_cast<double>(x));

#ifdef __GNUC__
#if __GNUC__ != 7 && __GNUC__ != 8

            // gcc std::assoc_legendre is not standard compliant
            value *= n % 2 == 0 ? 1 : -1;

#endif // #if __GNUC__ != 7 && __GNUC__ != 8
#endif // #ifdef __GNUC__

#ifdef __clang__
#if __clang_major__ != 7 && __clang_major__ != 8

            // clang uses gcc headers where std::assoc_legendre is not standard compliant
            value *= n % 2 == 0 ? 1 : -1;

#endif // #if __clang_major__ != 6 && __clang_major__ != 7 && __clang_major__ != 8
#endif // #ifdef __clang__

#ifdef _MSC_VER

            value *= n % 2 == 0 ? 1 : -1;

#endif // #ifdef _MSC_VER

            return value;

#else // #ifdef __cpp_lib_math_special_functions

            return boost::math::legendre_p(n, m, static_cast<double>(x));

#endif // #ifdef __cpp_lib_math_special_functions
        }

        //============================================================================
        // Method Description:
        /// Legendre Polynomial of the first kind.
        /// NOTE: Use of this function requires either using the Boost
        /// includes or a C++17 compliant compiler.
        ///
        /// @param n: the degree of the legendre polynomial
        /// @param inArrayX: the input value. Requires -1 <= x <= 1
        /// @return NdArray<double>
        ///
        template<typename dtype>
        NdArray<double> legendre_p(uint32 n, const NdArray<dtype>& inArrayX)
        {
            NdArray<double> returnArray(inArrayX.shape());

            const auto function = [n](dtype x) -> double { return legendre_p(n, x); };

            stl_algorithms::transform(inArrayX.cbegin(), inArrayX.cend(), returnArray.begin(), function);

            return returnArray;
        }

        //============================================================================
        // Method Description:
        /// Associated Legendre Polynomial of the first kind.
        /// NOTE: Use of this function requires either using the Boost
        /// includes or a C++17 compliant compiler.
        ///
        /// @param m: the order of the legendre polynomial
        /// @param n: the degree of the legendre polynomial
        /// @param inArrayX: the input value. Requires -1 <= x <= 1
        /// @return NdArray<double>
        ///
        template<typename dtype>
        NdArray<double> legendre_p(uint32 m, uint32 n, const NdArray<dtype>& inArrayX)
        {
            NdArray<double> returnArray(inArrayX.shape());

            const auto function = [m, n](dtype x) -> double { return legendre_p(m, n, x); };

            stl_algorithms::transform(inArrayX.cbegin(), inArrayX.cend(), returnArray.begin(), function);

            return returnArray;
        }
    } // namespace polynomial
} // namespace nc

#endif // #if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
