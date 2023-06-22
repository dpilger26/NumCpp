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
/// NdArray Functions
///
#pragma once

#include <cmath>

#include "NumCpp/Core/Constants.hpp"

namespace nc
{
    /**
     * @brief Wrap the input angle to [0, 2*pi]
     *
     * @param inAngle: in radians
     * @returns Wrapped angle
     */
    template<typename dtype>
    double wrap2Pi(dtype inAngle) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        auto angle = std::fmod(static_cast<double>(inAngle), constants::twoPi);
        if (angle < 0.)
        {
            angle += constants::twoPi;
        }

        return angle;
    }

    /**
     * @brief Wrap the input angle to [0, 2*pi]
     *
     * @param inAngles: in radians
     * @returns Wrapped angles
     */
    template<typename dtype>
    NdArray<double> wrap2Pi(const NdArray<dtype>& inAngles) noexcept
    {
        NdArray<double> returnArray(inAngles.size());
        stl_algorithms::transform(inAngles.begin(),
                                  inAngles.end(),
                                  returnArray.begin(),
                                  [](const auto angle) noexcept -> double { return wrap2Pi(angle); });
        return returnArray;
    }
} // namespace nc
