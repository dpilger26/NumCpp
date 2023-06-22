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

#include "NumCpp/Coordinates/Euler.hpp"
#include "NumCpp/Coordinates/Orientation.hpp"
#include "NumCpp/Coordinates/ReferenceFrames/ECEF.hpp"
#include "NumCpp/Coordinates/Transforms/NEDUnitVecsInECEF.hpp"
#include "NumCpp/Functions/wrap.hpp"
#include "NumCpp/Rotations/Quaternion.hpp"
#include "NumCpp/Vector/Vec3.hpp"

namespace nc::coordinates::transforms
{
    /**
     * @brief Converts NED body roll/pitch/yaw to ECEF euler angles
     *
     * @param location: the ecef location
     * @param orientation: ned euler angles
     * @return Orientation
     */
    [[nodiscard]] inline Euler NEDRollPitchYawToECEFEuler(const reference_frames::ECEF& location,
                                                          const Orientation&            orientation) noexcept
    {
        // get the local NED unit vectors wrt the ECEF coordinate system
        const auto& [x0, y0, z0] = NEDUnitVecsInECEF(location);

        // first rotation array, z0 by yaw
        const auto quatYaw = rotations::Quaternion{ z0, orientation.yaw };

        // rotate
        const auto x1 = quatYaw * x0;
        const auto y1 = quatYaw * y0;

        // second rotation array, y1 by pitch
        const auto quatPitch = rotations::Quaternion{ y1, orientation.pitch };

        // rotate
        const auto x2 = quatPitch * x1;
        const auto y2 = quatPitch * y1;

        // third rotation array, x2 by roll
        const auto quatRoll = rotations::Quaternion{ x2, orientation.roll };

        // rotate
        const auto x3 = quatRoll * x2;
        const auto y3 = quatRoll * y2;

        // calculate phi and theta
        const auto xHat0 = Vec3::right();
        const auto yHat0 = Vec3::up();
        const auto zHat0 = Vec3::forward();

        const auto psi   = std::atan2(x3.dot(yHat0), x3.dot(xHat0));
        const auto theta = std::atan(-x3.dot(zHat0) / std::hypot(x3.dot(xHat0), x3.dot(yHat0)));

        // calculate phi
        const auto yHat2 = (rotations::Quaternion{ zHat0, psi } * yHat0);
        const auto zHat2 = (rotations::Quaternion{ yHat2, theta } * zHat0);
        const auto phi   = std::atan2(y3.dot(zHat2), y3.dot(yHat2));

        return { wrap(psi), theta, wrap(phi) };
    }
} // namespace nc::coordinates::transforms
