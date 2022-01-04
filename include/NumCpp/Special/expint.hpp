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

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"

#ifdef __cpp_lib_math_special_functions
#include <cmath>
#else
#include "boost/math/special_functions/expint.hpp"
#endif

#include <type_traits>

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        /// Exponential integral Ei.
        /// NOTE: Use of this function requires either using the Boost
        /// includes or a C++17 compliant compiler.
        ///
        /// @param      inX: value
        /// @return
        /// calculated-result-type 
        ///
        template<typename dtype>
        auto expint(dtype inX)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

#ifdef __cpp_lib_math_special_functions
            return std::expint(inX);
#else
            return boost::math::expint(inX);
#endif
        }

        //============================================================================
        // Method Description:
        /// Exponential integral Ei.
        /// NOTE: Use of this function requires either using the Boost
        /// includes or a C++17 compliant compiler.
        ///
        /// @param      inArrayX: value
        /// @return
        /// NdArray
        ///
        template<typename dtype>
        auto expint(const NdArray<dtype>& inArrayX)
        {
            NdArray<decltype(expint(dtype{ 0 }))> returnArray(inArrayX.shape());

            stl_algorithms::transform(inArrayX.cbegin(), inArrayX.cend(), returnArray.begin(),
                [](dtype inX) -> auto
                {
                    return expint(inX);
                });

            return returnArray;
        }
    }  // namespace special
}  // namespace nc

#endif // #if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
