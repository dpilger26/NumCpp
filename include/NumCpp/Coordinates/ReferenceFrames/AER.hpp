
/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2025 David Pilger
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
/// AER Object
///
#pragma once

#include <iostream>

#include "NumCpp/Utils/essentiallyEqual.hpp"

namespace nc::coordinates::reference_frames
{
    /**
     * @brief Az, El, Range coordinates
     */

    class AER
    {
    public:
        double az{ 0. };    // radians
        double el{ 0. };    // radians
        double range{ 0. }; // meters

        /**
         * @brief Default Constructor
         */
        AER() = default;

        /**
         * @brief Constructor
         * @param inAz: az value in radians
         * @param inEl: el value in radians
         * @param inRange: the range in meters
         */
        // NOTLINTNEXTLINE(bugprone-easily-swappable-parameters)
        constexpr AER(double inAz, double inEl, double inRange = 1.) noexcept :
            az(inAz),
            el(inEl),
            range(inRange)
        {
        }

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator==(const AER& other) const noexcept
        {
            return utils::essentiallyEqual(az, other.az) && utils::essentiallyEqual(el, other.el) &&
                   utils::essentiallyEqual(range, other.range);
        }

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator!=(const AER& other) const noexcept
        {
            return !(*this == other);
        }
    };

    /**
     * @brief Stream operator
     *
     * @param os: the output stream
     * @param point: the AER point
     */
    inline std::ostream& operator<<(std::ostream& os, const AER& point)
    {
        os << "AER(az=" << point.az << ", el=" << point.el << ", range=" << point.range << ")\n";
        return os;
    }

} // namespace nc::coordinates::reference_frames
