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
/// Orientation
///
#pragma once

#include <iostream>

#include "NumCpp/Utils/essentiallyEqual.hpp"

namespace nc::coordinates
{
    /**
     * @brief Orientation
     */
    class Orientation
    {
    public:
        double roll{ 0. };
        double pitch{ 0. };
        double yaw{ 0. };

        /**
         * @brief Default Constructor
         */
        Orientation() noexcept = default;

        /**
         * @brief Constructor
         *
         * @param: inRoll: the roll component
         * @param: inPitch: the pitch component
         * @param: inYaw: the yaw component
         */
        constexpr Orientation(double inRoll, double inPitch, double inYaw) noexcept :
            roll(inRoll),
            pitch(inPitch),
            yaw(inYaw)
        {
        }

        /**
         * @brief Copy Constructor
         *
         * @param: other: the other Orientation instance
         */
        Orientation(const Orientation& other) noexcept = default;

        /**
         * @brief Move Orientation
         *
         * @param: other: the other Orientation instance
         */
        Orientation(Orientation&& other) noexcept = default;

        /**
         * @brief Destructor
         */
        virtual ~Orientation() = default;

        /**
         * @brief Copy Assignement Operator
         *
         * @param: other: the other Orientation instance
         */
        Orientation& operator=(const Orientation& other) noexcept = default;

        /**
         * @brief Move Assignement Operator
         *
         * @param: other: the other Orientation instance
         */
        Orientation& operator=(Orientation&& other) noexcept = default;

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator==(const Orientation& other) const noexcept
        {
            return utils::essentiallyEqual(roll, other.roll) && utils::essentiallyEqual(pitch, other.pitch) &&
                   utils::essentiallyEqual(yaw, other.yaw);
        }

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator!=(const Orientation& other) const noexcept
        {
            return !(*this == other);
        }
    };

    /**
     * @brief Stream operator
     *
     * @param: os: the output stream
     * @param: vec: the cartesian vector
     */
    [[nodiscard]] inline std::ostream& operator<<(std::ostream& os, const Orientation& orientation)
    {
        os << "Orientation(roll=" << orientation.roll << ", pitch=" << orientation.pitch << ", yaw=" << orientation.yaw
           << ")\n";
        return os;
    }
} // namespace nc::coordinates
