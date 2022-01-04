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

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/powerf.hpp"

#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    /// Raises the elements of the array to the input floating point power
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param inValue
    /// @param inExponent
    /// @return value raised to the power
    ///
    template<typename dtype1, typename dtype2>
    auto powerf(dtype1 inValue, dtype2 inExponent) noexcept 
    {
        return utils::powerf(inValue, inExponent);
    }

    //============================================================================
    // Method Description:
    /// Raises the elements of the array to the input floating point power
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param inArray
    /// @param inExponent
    /// @return NdArray
    ///
    template<typename dtype1, typename dtype2>
    auto powerf(const NdArray<dtype1>& inArray, dtype2 inExponent) 
    {
        NdArray<decltype(powerf(dtype1{0}, dtype2{0}))> returnArray(inArray.shape());
        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [inExponent](dtype1 inValue) noexcept -> auto
            {
                return nc::powerf(inValue, inExponent); 
            });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Raises the elements of the array to the input floating point powers
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.power.html
    ///
    /// @param inArray
    /// @param inExponents
    /// @return NdArray
    ///
    template<typename dtype1, typename dtype2>
    auto powerf(const NdArray<dtype1>& inArray, const NdArray<dtype2>& inExponents)
    {
        if (inArray.shape() != inExponents.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("input array shapes are not consistant.");
        }

        NdArray<decltype(powerf(dtype1{0}, dtype2{0}))> returnArray(inArray.shape());
        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), inExponents.cbegin(), returnArray.begin(),
            [](dtype1 inValue, dtype2 inExponent) noexcept -> auto
            {
                return nc::powerf(inValue, inExponent);
            });

        return returnArray;
    }
}  // namespace nc
