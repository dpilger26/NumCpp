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

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/roll.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::fft
{
    //===========================================================================
    // Method Description:
    /// Shift the zero-frequency component to the center of the spectrum.
    /// This function swaps half-spaces for all axes listed (defaults to all). Note that y[0] is the Nyquist component
    /// only if len(x) is even.
    ///
    /// NumPy Reference: <https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html>
    ///
    /// @param inX input array
    /// @param inAxis (Optional) Axes over which to shift. Default is None, which shifts all axes.
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fftshift(const NdArray<dtype>& inX, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                return roll(inX, inX.size() / 2, inAxis);
            }
            case Axis::COL:
            {
                return roll(inX, inX.numCols() / 2, inAxis);
            }
            case Axis::ROW:
            {
                return roll(inX, inX.numRows() / 2, inAxis);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {};
            }
        }
    }
} // namespace nc::fft
