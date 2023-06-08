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

#include "NumCpp/Coordinates/ReferenceFrames/ECEF.hpp"
#include "NumCpp/Coordinates/Transforms/ECEFtoLLA.hpp"
#include "NumCpp/Vector/Vec3.hpp"

namespace nc::coordinates::transforms
{
    /**
     * @brief get the local NED unit vectors wrt the ECEF coordinate system
     *        https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
     *
     * @param location: the ECEF location
     * @return std::array<Vec3, 3>
     */
    [[nodiscard]] inline std::array<Vec3, 3> NEDUnitVecsInECEF(const reference_frames::ECEF& location) noexcept
    {
        const auto lla = ECEFtoLLA(location);

        const auto sinLat = std::sin(lla.latitude);
        const auto cosLat = std::cos(lla.latitude);
        const auto sinLon = std::sin(lla.longitude);
        const auto cosLon = std::cos(lla.longitude);

        const auto xHat = Vec3{ -cosLon * sinLat, -sinLon * sinLat, cosLat };
        const auto yHat = Vec3{ -sinLon, cosLon, 0. };
        const auto zHat = Vec3{ -cosLon * cosLat, -sinLon * cosLat, -sinLat };

        return { xHat, yHat, zHat };
    }
} // namespace nc::coordinates::transforms
