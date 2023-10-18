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

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"

#ifndef __cpp_lib_math_special_functions
#include "boost/math/special_functions/ellint_1.hpp"
#endif

#include <type_traits>

namespace nc::special
{
    //============================================================================
    // Method Description:
    /// Computes the incomplete elliptic integral of the first kind of k and p.
    /// NOTE: Use of this function requires either using the Boost
    /// includes or a C++17 compliant compiler.
    ///
    /// @param inK: elliptic modulus or eccentricity
    /// @param inP: Jacobi amplitude (measured in radians)
    /// @return calculated-result-type
    ///
    template<typename dtype1, typename dtype2>
    auto ellint_1(dtype1 inK, dtype2 inP)
    {
        STATIC_ASSERT_ARITHMETIC(dtype1);
        STATIC_ASSERT_ARITHMETIC(dtype2);

#ifdef __cpp_lib_math_special_functions
        return std::ellint_1(inK, inP);
#else
        return boost::math::ellint_1(inK, inP);
#endif
    }

    //============================================================================
    // Method Description:
    /// Computes the incomplete elliptic integral of the first kind of k and p.
    /// NOTE: Use of this function requires either using the Boost
    /// includes or a C++17 compliant compiler.
    ///
    /// @param inArrayK: elliptic modulus or eccentricity
    /// @param inArrayP: Jacobi amplitude (measured in radians)
    /// @return NdArray
    ///
    template<typename dtype1, typename dtype2>
    auto ellint_1(const NdArray<dtype1>& inArrayK, const NdArray<dtype2>& inArrayP)
    {
        if (inArrayK.size() != inArrayP.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("Shapes of inArrayK and inArrayP must match.");
        }

        NdArray<decltype(ellint_1(dtype1{ 0 }, dtype2{ 0 }))> returnArray(inArrayK.shape());

        stl_algorithms::transform(inArrayK.cbegin(),
                                  inArrayK.cend(),
                                  inArrayP.cbegin(),
                                  returnArray.begin(),
                                  [](dtype1 inK, dtype2 inP) -> auto { return ellint_1(inK, inP); });

        return returnArray;
    }
} // namespace nc::special

#endif // #if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
