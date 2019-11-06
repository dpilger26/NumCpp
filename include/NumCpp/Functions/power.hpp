/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
///
/// @section License
/// Copyright 2019 David Pilger
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

#include "NumCpp/Core/Error.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/power.hpp"

#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Raises the elements of the array to the input integer power
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param				inValue
    /// @param				inExponent
    /// @return
    ///				value raised to the power
    ///
    template<typename dtype>
    constexpr dtype power(dtype inValue, uint8 inExponent) noexcept
    {
        return utils::power(inValue, inExponent);
    }

    //============================================================================
    // Method Description:
    ///						Raises the elements of the array to the input integer power
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param				inArray
    /// @param				inExponent
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> power(const NdArray<dtype>& inArray, uint8 inExponent) noexcept
    {
        NdArray<dtype> returnArray(inArray.shape());
        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [inExponent](dtype inValue) noexcept -> dtype
            {
                return power(inValue, inExponent);
            });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Raises the elements of the array to the input integer powers
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param				inArray
    /// @param				inExponents
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> power(const NdArray<dtype>& inArray, const NdArray<uint8>& inExponents)
    {
        if (inArray.shape() != inExponents.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("input array shapes are not consistant.");
        }

        NdArray<dtype> returnArray(inArray.shape());
        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), inExponents.cbegin(), returnArray.begin(),
            [](dtype inValue, uint8 inExponent) noexcept -> dtype
            {
                return power(inValue, inExponent);
            });

        return returnArray;
    }
}
