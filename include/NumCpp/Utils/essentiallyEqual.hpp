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
/// tests that 2 floating point values are "essentially equal"
///
#pragma once

#include <cmath>
#include <complex>
#include <string>

#include "NumCpp/Core/DtypeInfo.hpp"
#include "NumCpp/Core/Internal/TypeTraits.hpp"

namespace nc::utils
{
    //============================================================================
    // Method Description:
    /// tests that 2 integer values are "essentially equal"
    ///
    /// @param inValue1
    /// @param inValue2
    ///
    /// @return bool
    ///
    template<typename dtype, std::enable_if_t<std::is_integral<dtype>::value, int> = 0>
    bool essentiallyEqual(dtype inValue1, dtype inValue2) noexcept
    {
        return inValue1 == inValue2;
    }

    //============================================================================
    // Method Description:
    /// tests that 2 floating point values are "essentially equal"
    ///
    /// @param inValue1
    /// @param inValue2
    /// @param inEpsilon
    ///
    /// @return bool
    ///
    template<typename dtype, std::enable_if_t<std::is_floating_point<dtype>::value, int> = 0>
    bool essentiallyEqual(dtype inValue1, dtype inValue2, dtype inEpsilon) noexcept
    {
        const auto absValue1 = std::abs(inValue1);
        const auto absValue2 = std::abs(inValue2);
        return std::abs(inValue1 - inValue2) <= ((absValue1 > absValue2 ? absValue2 : absValue1) * std::abs(inEpsilon));
    }

    //============================================================================
    // Method Description:
    /// tests that 2 floating point values are "essentially equal"
    ///
    /// @param inValue1
    /// @param inValue2
    ///
    /// @return bool
    ///
    template<typename dtype, std::enable_if_t<std::is_floating_point<dtype>::value, int> = 0>
    bool essentiallyEqual(dtype inValue1, dtype inValue2) noexcept
    {
        return essentiallyEqual(inValue1, inValue2, DtypeInfo<dtype>::epsilon());
    }
} // namespace nc::utils
