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
/// Coordinate Transforms
///
#pragma once

#include <cmath>

#include "NumCpp/Coordinates/ReferenceFrames/AzEl.hpp"
#include "NumCpp/Coordinates/ReferenceFrames/NED.hpp"
#include "NumCpp/Functions/wrap2Pi.hpp"

namespace nc::coordinates::transforms
{
    /**
     * @brief Converts the Cartesian XYZ (NED) coordinates to 2d speherical inertial coordinates.
     *        Range is not used.
     *        NOTE: positive elevation is defined as the negative z (up) direction
     *
     * @param cartesian: coordinates to convert
     * @returns AzEl
     */
    [[nodiscard]] inline reference_frames::AzEl NEDtoAzEl(const reference_frames::NED& target) noexcept
    {
        const auto hypotXy = std::hypot(target.x, target.y);
        const auto el      = -std::atan2(target.z, hypotXy);
        const auto az      = wrap2Pi(std::atan2(target.y, target.x));
        return { az, el };
    }
} // namespace nc::coordinates::transforms
