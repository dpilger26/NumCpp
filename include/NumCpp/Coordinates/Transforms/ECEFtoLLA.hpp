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
     * @brief Converts ECEF coordinates to LLA
     * 		  https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_geodetic_coordinates
     *
     * @param ecef the point of interest
     * @param tol Tolerance for the convergence of altitude (overriden if 10 iterations are processed)
     * @return reference_frames::LLA
     */
    [[nodiscard]] inline reference_frames::LLA ECEFtoLLA(const reference_frames::ECEF& ecef, double tol = 1e-8) noexcept
    {
        constexpr int  MAX_ITER = 10;
        constexpr auto E_SQR    = 1 - Sqr(constants::EARTH_POLAR_RADIUS / constants::EARTH_EQUATORIAL_RADIUS);

        const auto p   = std::hypot(ecef.x(), ecef.y());
        const auto lon = std::atan2(ecef.y(), ecef.x());

        double alt = 0.0;
        double lat = 0.0;

        if (p < tol)
        {
            lat = boost::math::sign(ecef.z()) * boost::math::constants::half_pi<double>();
            alt = std::abs(ecef.z()) - constants::EARTH_POLAR_RADIUS;
        }
        else
        {
            // Iteratively update latitude and altitude.
            // This is expected to converge in ~4 iterations, but apply a maximum number of iterations incase tol is
            // too small
            double err  = 1.0;
            int    iter = 0;
            while (err > tol && iter < MAX_ITER)
            {
                double N      = constants::EARTH_EQUATORIAL_RADIUS / std::sqrt(1 - E_SQR * Sqr(std::sin(lat)));
                lat           = std::atan((ecef.z() / p) / (1 - (N * E_SQR / (N + alt))));
                double newAlt = (p / std::cos(lat)) - N;
                err           = std::abs(alt - newAlt);
                alt           = newAlt;
                iter++;
            }
        }
        return reference_frames::LLA{ lat, lon, alt };
    }
} // namespace nc::coordinates::transforms
