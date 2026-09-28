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
/// Coordinate Transforms
///
#pragma once

#include <cmath>

#include "NumCpp/Coordinates/Cartesian.hpp"
#include "NumCpp/Coordinates/ReferenceFrames/AER.hpp"

namespace nc::coordinates::transforms
{
    /**
     * @brief Converts cartiesian coordinates to spherical coordinates
     *
     * @param cartesian the point of interest
     * @return AER
     */
    [[nodiscard]] inline reference_frames::AER cart2sph(const Cartesian& cartesian) noexcept
    {
        const auto hxy       = std::hypot(cartesian.x, cartesian.y);
        const auto range     = std::hypot(hxy, cartesian.z);
        const auto elevation = std::atan2(cartesian.z, hxy);
        const auto azimuth   = std::atan2(cartesian.y, cartesian.x);
        return reference_frames::AER{ range, elevation, azimuth };
    }
} // namespace nc::coordinates::transforms
