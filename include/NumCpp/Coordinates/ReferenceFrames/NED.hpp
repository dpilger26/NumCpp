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
/// NED Object
///
#pragma once

#include <iostream>

#include "NumCpp/Coordinates/Cartesian.hpp"

namespace nc::coordinates::reference_frames
{
    /**
     * @brief North east down coordinates
     */
    class NED final : public Cartesian
    {
    public:
        using Cartesian::Cartesian;

        /**
         * @brief Constructor
         * @param cartesian: cartesian vector
         */
        constexpr NED(const Cartesian& cartesian) noexcept :
            Cartesian(cartesian)
        {
        }

        /**
         * @brief Constructor
         * @param north: north value
         * @param east: east value
         * @param down: down value
         */
        // NOTLINTNEXTLINE(bugprone-easily-swappable-parameters)
        constexpr NED(double north, double east, double down) noexcept :
            Cartesian(north, east, down)
        {
        }

        /**
         * @brief north getter
         *
         * @return north
         */
        [[nodiscard]] double north() const noexcept
        {
            return x;
        }

        /**
         * @brief north setter
         *
         * @param north: north value
         */
        void setNorth(double north) noexcept
        {
            x = north;
        }

        /**
         * @brief east getter
         *
         * @return double
         */
        [[nodiscard]] double east() const noexcept
        {
            return y;
        }

        /**
         * @brief east setter
         *
         * @param east: east value
         */
        void setEast(double east) noexcept
        {
            y = east;
        }

        /**
         * @brief down getter
         *
         * @return down
         */
        [[nodiscard]] double down() const noexcept
        {
            return z;
        }

        /**
         * @brief down setter
         *
         * @param down: down value
         */
        void setDown(double down) noexcept
        {
            z = down;
        }
    };
} // namespace nc::coordinates::reference_frames
