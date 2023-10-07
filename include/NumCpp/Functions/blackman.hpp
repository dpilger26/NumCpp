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
/// Functions for working with NdArrays
///
#pragma once

#include <cmath>

#include "NumCpp/Core/Constants.hpp"
#include "NumCpp/Functions/linspace.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// The Blackman window is a taper formed by using the first three terms of a summation of
    /// cosines. It was designed to have close to the minimal leakage possible. It is close to
    /// optimal, only slightly worse than a Kaiser window.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.blackman.html
    ///
    /// @param m: Number of points in the output window. If zero or less, an empty array is returned.
    /// @return NdArray
    ///
    inline NdArray<double> blackman(int32 m)
    {
        if (m < 1)
        {
            return {};
        }

        const auto mDouble = static_cast<double>(m);

        NdArray<double> result(1, m);
        int32           i = 0;
        for (auto n : linspace(0., mDouble, m, EndPoint::TRUE))
        {
            const auto nOverM = n / mDouble;
            result[i++] =
                0.42 - 0.5 * std::cos(2. * constants::pi * nOverM) + 0.08 * std::cos(4. * constants::pi * nOverM);
        }

        return result;
    }
} // namespace nc
