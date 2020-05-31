/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.4.0
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
#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"

#include <complex>

namespace nc
{
    //============================================================================
    // Method Description:
    ///	Returns a complex number with magnitude r and phase angle theta.
    ///
    /// @param  magnitude
    /// @param  phaseAngle
    ///			
    /// @return
    ///				std::complex
    ///
    template<typename dtype>
    auto polar(dtype magnitude, dtype phaseAngle) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::polar(magnitude, phaseAngle);
    }

    //============================================================================
    // Method Description:
    ///	Returns a complex number with magnitude r and phase angle theta.
    ///
    /// @param  magnitude
    /// @param  phaseAngle
    /// @return
    ///				NdArray<std::complex>
    ///
    template<typename dtype>
    auto polar(const NdArray<dtype>& magnitude, const NdArray<dtype>& phaseAngle)
    {
        if (magnitude.shape() != phaseAngle.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Input magnitude and phaseAngle arrays must be the same shape");
        }

        NdArray<decltype(nc::polar(dtype{0}, dtype{0}))> returnArray(magnitude.shape());
        stl_algorithms::transform(magnitude.cbegin(), magnitude.cend(), phaseAngle.begin(), returnArray.begin(),
            [](dtype mag, dtype angle) noexcept -> auto
            {
                return nc::polar(mag, angle);
            });

        return returnArray;
    }
}
