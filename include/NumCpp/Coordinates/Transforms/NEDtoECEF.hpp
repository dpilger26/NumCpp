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
#include <iostream>

#include "NumCpp/Coordinates/Euler.h"
#include "NumCpp/Coordinates/Orientation.h"
#include "NumCpp/Coordinates/ReferenceFrames/AzEl.hpp"
#include "NumCpp/Coordinates/ReferenceFrames/Cartesian.hpp"
#include "NumCpp/Core/Constants.hpp"
#include "NumCpp/Functions/sign.hpp"
#include "NumCpp/Functions/wrap.hpp"
#include "NumCpp/Functions/wrap2pi.hpp"
#include "NumCpp/Rotations/Quaternion.hpp"
#include "NumCpp/Utils/sqr.hpp"
#include "NumCpp/Vector/Vec3.hpp"

namespace nc::coordinates::transforms
{
    /**
     * @brief Converts the NED coordinates to ECEF
     *        https://apps.dtic.mil/sti/pdfs/AD1170763.pdf
     *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
     *        https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
     *
     * @param target: the target of interest
     * @param referencePoint: the referencePoint location
     * @returns NED
     */
    [[nodiscard]] inline reference_frames::ECEF NEDtoECEF(const reference_frames::NED&  target,
                                                          const reference_frames::ECEF& referencePoint) noexcept
    {
        const auto referencePointLLA = ECEFtoLLA(referencePoint);

        const auto sinLat = std::sin(referencePointLLA.latitude());
        const auto cosLat = std::cos(referencePointLLA.latitude());
        const auto sinLon = std::sin(referencePointLLA.longitude());
        const auto cosLon = std::cos(referencePointLLA.longitude());

        auto rotationMatrix  = Matrix<double, 3, 3>{};
        rotationMatrix(0, 0) = -sinLat * cosLon;
        rotationMatrix(1, 0) = -sinLat * sinLon;
        rotationMatrix(2, 0) = cosLat;
        rotationMatrix(0, 1) = -sinLon;
        rotationMatrix(1, 1) = cosLon;
        rotationMatrix(2, 1) = 0.;
        rotationMatrix(0, 2) = -cosLat * cosLon;
        rotationMatrix(1, 2) = -cosLat * sinLon;
        rotationMatrix(2, 2) = -sinLat;

        auto targetVec = Vector<double, 3>{};
        targetVec[0]   = target.north();
        targetVec[1]   = target.east();
        targetVec[2]   = target.down();

        auto referencePointVec = Vector<double, 3>{};
        referencePointVec[0]   = referencePoint.x();
        referencePointVec[1]   = referencePoint.y();
        referencePointVec[2]   = referencePoint.z();

        const auto targetECEFVec = (rotationMatrix * targetVec) + referencePointVec;
        return { targetECEFVec[0], targetECEFVec[1], targetECEFVec[2] };
    }

    /**
     * @brief Converts the NED coordinates to ECEF
     *        https://apps.dtic.mil/sti/pdfs/AD1170763.pdf
     *        Figure 11 https://apps.dtic.mil/sti/pdfs/AD1170763.pdf for a helpful diagram
     *        https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
     *
     * @param target: the target of interest
     * @param referencePoint: the referencePoint location
     * @returns NED
     */
    [[nodiscard]] inline reference_frames::ECEF NEDtoECEF(const reference_frames::NED& target,
                                                          const reference_frames::LLA& referencePoint) noexcept
    {
        return NEDtoECEF(target, LLAtoECEF(referencePoint));
    }
} // namespace nc::coordinates::transforms
