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

#include "NumCpp/Functions/linspace.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// The Bartlett window is very similar to a triangular window, except that the end
    /// points are at zero. It is often used in signal processing for tapering a signal,
    /// without generating too much ripple in the frequency domain.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.bartlett.html
    ///
    /// @param m: Number of points in the output window. If zero or less, an empty array is returned.
    /// @return NdArray
    ///
    inline NdArray<double> bartlett(int32 m)
    {
        if (m < 1)
        {
            return {};
        }

        const auto mDouble         = static_cast<double>(m);
        const auto mMinus1Over2    = (mDouble - 1.) / 2.;
        const auto mMinus1Over2Inv = 1. / mMinus1Over2;

        NdArray<double> result(1, m);
        int32           i = 0;
        for (auto n : linspace(0., mDouble - 1., m, true))
        {
            result[i++] = mMinus1Over2Inv * (mMinus1Over2 - std::abs(n - mMinus1Over2));
        }

        return result;
    }
} // namespace nc
