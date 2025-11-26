/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2026 David Pilger
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

#include "NumCpp/Core/Enums.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Functions/linspace.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/powerf.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Return numbers spaced evenly on a log scale.
    ///
    /// This is similar to logspace, but with endpoints specified directly.
    /// Each output sample is a constant multiple of the previous.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.logspace.html
    ///
    /// @param start: the starting value of a sequence
    /// @param stop: The final value of the sequence, unless endpoint is False.
    /// In that case, num + 1 values are spaced over the interval
    /// in log-space, of which all but the last (a sequence of length num) are returned.
    /// @param num: Number of samples to generate. Default 50.
    /// @param endPoint: If true, stop is the last sample. Otherwise,it is not included. Default is true.
    /// @param base: The base of the log space. The step size between the elements in ln(samples) / ln(base)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double>
        logspace(dtype start, dtype stop, uint32 num = 50, EndPoint endPoint = EndPoint::YES, double base = 10.)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        auto spacedValues = linspace(static_cast<double>(start), static_cast<double>(stop), num, endPoint);
        stl_algorithms::for_each(spacedValues.begin(),
                                 spacedValues.end(),
                                 [base](auto& value) -> void { value = utils::powerf(base, value); });

        return spacedValues;
    }
} // namespace nc
