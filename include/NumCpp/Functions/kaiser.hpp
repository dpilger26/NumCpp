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
/// Functions for working with NdArrays
///
#pragma once

#include <cmath>

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Special/bessel_in.hpp"
#include "NumCpp/Utils/sqr.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// The Kaiser window is a taper formed by using a Bessel function.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html
    ///
    /// @param m: Number of points in the output window. If zero or less, an empty array is returned.
    /// @param beta: shape parameter for the window
    /// @return NdArray
    ///
    inline NdArray<double> kaiser(int32 m, double beta)
    {
        if (m < 1)
        {
            return {};
        }

        const auto mDouble        = static_cast<double>(m);
        const auto mMinus1        = mDouble - 1.;
        const auto mMinus1Over2   = mMinus1 / 2.;
        const auto mMinus1Squared = utils::sqr(mMinus1);
        const auto i0Beta         = special::bessel_in(0, beta);

        NdArray<double> result(1, m);
        int32           i = 0;
        for (auto n : linspace(-mMinus1Over2, mMinus1Over2, m, true))
        {
            auto value  = beta * std::sqrt(1. - (4. * utils::sqr(n)) / mMinus1Squared);
            result[i++] = special::bessel_in(0, value) / i0Beta;
        }

        return result;
    }
} // namespace nc

#endif // #if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
