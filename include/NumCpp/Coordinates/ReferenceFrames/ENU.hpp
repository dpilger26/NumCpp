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
/// ENU Object
///
#pragma once

#include <iostream>

#include "NumCpp/Coordinates/Cartesian.hpp"

namespace nc::coordinates::reference_frames
{
    /**
     * @brief East North Up coordinates
     */
    class ENU final : public Cartesian
    {
    public:
        using Cartesian::Cartesian;

        /**
         * @brief Constructor
         * @param cartesian: cartesian vector
         */
        constexpr ENU(const Cartesian& cartesian) noexcept :
            Cartesian(cartesian)
        {
        }

        /**
         * @brief Constructor
         * @param east: east value
         * @param north: north value
         * @param up: up value
         */
        // NOTLINTNEXTLINE(bugprone-easily-swappable-parameters)
        constexpr ENU(double east, double north, double up) noexcept :
            Cartesian(east, north, up)
        {
        }

        /**
         * @brief east getter
         *
         * @return east
         */
        [[nodiscard]] double east() const noexcept
        {
            return x;
        }

        /**
         * @brief east setter
         *
         * @param east: east value
         */
        void setEast(double east) noexcept
        {
            x = east;
        }

        /**
         * @brief north getter
         *
         * @return double
         */
        [[nodiscard]] double north() const noexcept
        {
            return y;
        }

        /**
         * @brief north setter
         *
         * @param north: north value
         */
        void setNorth(double north) noexcept
        {
            y = north;
        }

        /**
         * @brief up getter
         *
         * @return up
         */
        [[nodiscard]] double up() const noexcept
        {
            return z;
        }

        /**
         * @brief up setter
         *
         * @param up: up value
         */
        void setUp(double up) noexcept
        {
            z = up;
        }
    };

    /**
     * @brief Stream operator
     *
     * @param os: the output stream
     * @param point: the ENU point
     */
    inline std::ostream& operator<<(std::ostream& os, const ENU& point)
    {
        os << "ENU(east=" << point.east() << ", north=" << point.north() << ", up=" << point.up() << ")\n";
        return os;
    }
} // namespace nc::coordinates::reference_frames
