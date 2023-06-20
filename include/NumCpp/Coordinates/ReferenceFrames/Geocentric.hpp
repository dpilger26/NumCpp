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
/// Geocentric Object
///
#pragma once

#include <iostream>

#include "NumCpp/Utils/essentiallyEqual.hpp"

namespace nc::coordinates::reference_frames
{
    /**
     * @brief Geocentric coordinates
     */
    class Geocentric
    {
    public:
        double latitude{ 0. };  // radians
        double longitude{ 0. }; // radians
        double radius{ 0. };    // meters

        /**
         * @brief Default Constructor
         */
        Geocentric() = default;

        /**
         * @brief Constructor
         * @param inLatitude: latitude value in radians
         * @param inLongitude: longitude value in radians
         * @param inRadius: radius value in meters
         */
        // NOTLINTNEXTLINE(bugprone-easily-swappable-parameters)
        constexpr Geocentric(double inLatitude, double inLongitude, double inRadius = 0.) noexcept :
            latitude(inLatitude),
            longitude(inLongitude),
            radius(inRadius)
        {
        }

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator==(const Geocentric& other) const noexcept
        {
            return utils::essentiallyEqual(latitude, other.latitude) &&
                   utils::essentiallyEqual(longitude, other.longitude) && utils::essentiallyEqual(radius, other.radius);
        }

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator!=(const Geocentric& other) const noexcept
        {
            return !(*this == other);
        }
    };

    /**
     * @brief Stream operator
     *
     * @param: os: the output stream
     * @param: point: the Geocentric point
     */
    inline std::ostream& operator<<(std::ostream& os, const Geocentric& point)
    {
        os << "Geocentric(latitude=" << point.latitude << ", longitude=" << point.longitude
           << ", radius=" << point.radius << ")\n";
        return os;
    }
} // namespace nc::coordinates::reference_frames
