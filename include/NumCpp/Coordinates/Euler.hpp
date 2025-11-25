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
/// Euler
///
#pragma once

#include <iostream>

#include "NumCpp/Utils/essentiallyEqual.hpp"

namespace nc::coordinates
{
    /**
     * @brief Euler
     */
    class Euler
    {
    public:
        double psi{ 0. };
        double theta{ 0. };
        double phi{ 0. };

        /**
         * @brief Default Constructor
         */
        Euler() noexcept = default;

        /**
         * @brief Constructor
         *
         * @param inPsi: the psi component
         * @param inTheta: the theta component
         * @param inPhi: the phi component
         */
        constexpr Euler(double inPsi, double inTheta, double inPhi) noexcept :
            psi(inPsi),
            theta(inTheta),
            phi(inPhi)
        {
        }

        /**
         * @brief Copy Constructor
         *
         * @param other: the other Euler instance
         */
        Euler(const Euler& other) noexcept = default;

        /**
         * @brief Move Euler
         *
         * @param other: the other Euler instance
         */
        Euler(Euler&& other) noexcept = default;

        /**
         * @brief Destructor
         */
        virtual ~Euler() = default;

        /**
         * @brief Copy Assignement Operator
         *
         * @param other: the other Euler instance
         */
        Euler& operator=(const Euler& other) noexcept = default;

        /**
         * @brief Move Assignement Operator
         *
         * @param other: the other Euler instance
         */
        Euler& operator=(Euler&& other) noexcept = default;

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator==(const Euler& other) const noexcept
        {
            return utils::essentiallyEqual(psi, other.psi) && utils::essentiallyEqual(theta, other.theta) &&
                   utils::essentiallyEqual(phi, other.phi);
        }

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator!=(const Euler& other) const noexcept
        {
            return !(*this == other);
        }
    };

    /**
     * @brief Stream operator
     *
     * @param os: the output stream
     * @param Euler: the euler angles
     */
    inline std::ostream& operator<<(std::ostream& os, const Euler& Euler)
    {
        os << "Euler(psi=" << Euler.psi << ", theta=" << Euler.theta << ", phi=" << Euler.phi << ")\n";
        return os;
    }
} // namespace nc::coordinates
