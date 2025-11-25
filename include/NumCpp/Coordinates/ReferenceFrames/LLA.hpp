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
/// LLA Object
///
#pragma once

#include <iostream>

#include "NumCpp/Utils/essentiallyEqual.hpp"

namespace nc::coordinates::reference_frames
{
    /**
     * @brief Geodetic coordinates
     */
    class LLA
    {
    public:
        double latitude{ 0. };  // radians
        double longitude{ 0. }; // radians
        double altitude{ 0. };  // meters

        /**
         * @brief Default Constructor
         */
        LLA() = default;

        /**
         * @brief Constructor
         * @param inLatitude: latitude value in radians
         * @param inLongitude: longitude value in radians
         * @param inAltitude: altitude value in meters
         */
        // NOTLINTNEXTLINE(bugprone-easily-swappable-parameters)
        constexpr LLA(double inLatitude, double inLongitude, double inAltitude = 0.) noexcept :
            latitude(inLatitude),
            longitude(inLongitude),
            altitude(inAltitude)
        {
        }

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator==(const LLA& other) const noexcept
        {
            return utils::essentiallyEqual(latitude, other.latitude) &&
                   utils::essentiallyEqual(longitude, other.longitude) &&
                   utils::essentiallyEqual(altitude, other.altitude);
        }

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator!=(const LLA& other) const noexcept
        {
            return !(*this == other);
        }
    };

    /**
     * @brief Stream operator
     *
     * @param os: the output stream
     * @param point: the LLA point
     */
    inline std::ostream& operator<<(std::ostream& os, const LLA& point)
    {
        os << "LLA(latitude=" << point.latitude << ", longitude=" << point.longitude << ", altitude=" << point.altitude
           << ")\n";
        return os;
    }
} // namespace nc::coordinates::reference_frames
