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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Functions/stdev.hpp"

#include <algorithm>
#include <complex>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Compute the variance along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.var.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype, class Alloc>
    NdArray<double, Alloc> var(const NdArray<dtype, Alloc>& inArray, Axis inAxis = Axis::NONE) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        NdArray<double, Alloc> stdValues = stdev(inArray, inAxis);
        const auto function = [](double& value) noexcept -> void
        {
            value *= value;
        };

        stl_algorithms::for_each(stdValues.begin(), stdValues.end(), function);
        return stdValues;
    }

    //============================================================================
    // Method Description:
    ///						Compute the variance along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.var.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype, class Alloc>
    NdArray<std::complex<double>, Alloc> var(const NdArray<std::complex<dtype>, Alloc>& inArray,
        Axis inAxis = Axis::NONE) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        NdArray<std::complex<double>, Alloc> stdValues = stdev(inArray, inAxis);
        const auto function = [](std::complex<double>& value) noexcept -> void
        {
            value *= value;
        };

        stl_algorithms::for_each(stdValues.begin(), stdValues.end(), function);
        return stdValues;
    }
}
