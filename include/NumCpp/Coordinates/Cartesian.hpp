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
/// Cartesian Object
///
#pragma once

#include <cmath>
#include <iostream>

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"
#include "NumCpp/Utils/sqr.hpp"
#include "NumCpp/Vector/Vec2.hpp"
#include "NumCpp/Vector/Vec3.hpp"

namespace nc::coordinates
{
    /**
     * @brief Cartensian coordinates
     */
    class Cartesian
    {
    public:
        double x{ 0. };
        double y{ 0. };
        double z{ 0. };

        /**
         * @brief Default Constructor
         */
        Cartesian() noexcept = default;

        /**
         * @brief Constructor
         *
         * @param inX: the x component
         * @param inY: the y component
         * @param inZ: the z component
         */
        constexpr Cartesian(double inX, double inY, double inZ = 0.) noexcept :
            x(inX),
            y(inY),
            z(inZ)
        {
        }

        /**
         * @brief Default Constructor
         *
         * @param inCartesianVector
         */
        Cartesian(const Vec2& inCartesianVector) :
            x(inCartesianVector.x),
            y(inCartesianVector.y)
        {
        }

        /**
         * @brief Default Constructor
         *
         * @param inCartesianVector
         */
        Cartesian(const Vec3& inCartesianVector) :
            x(inCartesianVector.x),
            y(inCartesianVector.y),
            z(inCartesianVector.z)
        {
        }

        //============================================================================
        /// Constructor
        ///
        /// @param inCartesianVector
        ///
        Cartesian(const NdArray<double>& inCartesianVector)
        {
            if (inCartesianVector.size() != 3)
            {
                THROW_INVALID_ARGUMENT_ERROR("NdArray input must be of length 3.");
            }

            x = inCartesianVector[0];
            y = inCartesianVector[1];
            z = inCartesianVector[2];
        }

        /**
         * @brief Copy Constructor
         *
         * @param other: the other Cartesian instance
         */
        Cartesian(const Cartesian& other) noexcept = default;

        /**
         * @brief Move Constructor
         *
         * @param other: the other Cartesian instance
         */
        Cartesian(Cartesian&& other) noexcept = default;

        /**
         * @brief Destructor
         */
        virtual ~Cartesian() = default;

        /**
         * @brief Copy Assignement Operator
         *
         * @param other: the other Cartesian instance
         */
        Cartesian& operator=(const Cartesian& other) noexcept = default;

        /**
         * @brief Move Assignement Operator
         *
         * @param other: the other Cartesian instance
         */
        Cartesian& operator=(Cartesian&& other) noexcept = default;

        /**
         * @brief x Unit Vector
         *
         * @return unit vector in x direction
         */
        [[nodiscard]] static Cartesian xHat() noexcept
        {
            return { 1., 0., 0. };
        }

        /**
         * @brief y Unit Vector
         *
         * @return unit vector in y direction
         */
        [[nodiscard]] static Cartesian yHat() noexcept
        {
            return { 0., 1., 0. };
        }

        /**
         * @brief z Unit Vector
         *
         * @return unit vector in z direction
         */
        [[nodiscard]] static Cartesian zHat() noexcept
        {
            return { 0., 0., 1. };
        }

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator==(const Cartesian& other) const noexcept
        {
            return utils::essentiallyEqual(x, other.x) && utils::essentiallyEqual(y, other.y) &&
                   utils::essentiallyEqual(z, other.z);
        }

        /**
         * @brief Non-Equality Operator
         *
         * @param other: other object
         * @return bool true if not equal equal
         */
        bool operator!=(const Cartesian& other) const noexcept
        {
            return !(*this == other);
        }
    };

    /**
     * @brief Addition of two cartesian points
     *
     * @param lhs: the left hand side object
     * @param rhs: the right hand side object
     */
    [[nodiscard]] inline Cartesian operator+(const Cartesian& lhs, const Cartesian& rhs) noexcept
    {
        return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
    }

    /**
     * @brief Subtraction of two cartesian points
     *
     * @param lhs: the left hand side object
     * @param rhs: the right hand side object
     */
    [[nodiscard]] inline Cartesian operator-(const Cartesian& lhs, const Cartesian& rhs) noexcept
    {
        return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
    }

    /**
     * @brief Dot product of two cartesian points
     *
     * @param lhs: the left hand side object
     * @param rhs: the right hand side object
     */
    [[nodiscard]] inline double operator*(const Cartesian& lhs, const Cartesian& rhs) noexcept
    {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

    /**
     * @brief Vector scalar multiplication
     *
     * @param scalar: the the scalar value
     * @param vec: the cartesian vector
     */
    [[nodiscard]] inline Cartesian operator*(double scalar, const Cartesian& vec) noexcept
    {
        return { vec.x * scalar, vec.y * scalar, vec.z * scalar };
    }

    /**
     * @brief Vector scalar multiplication
     *
     * @param vec: the cartesian vector
     * @param scalar: the the scalar value
     */
    [[nodiscard]] inline Cartesian operator*(const Cartesian& vec, double scalar) noexcept
    {
        return scalar * vec;
    }

    /**
     * @brief Scalar Division a cartesian point
     *
     * @param vec: the cartesian vector
     * @param denominator: the the scalar value
     */
    [[nodiscard]] inline Cartesian operator/(const Cartesian& vec, double denominator) noexcept
    {
        return vec * (1.0 / denominator);
    }

    /**
     * @brief Stream operator
     *
     * @param os: the output stream
     * @param vec: the cartesian vector
     */
    inline std::ostream& operator<<(std::ostream& os, const Cartesian& vec)
    {
        os << "Cartesian(x=" << vec.x << ", y=" << vec.y << ", z=" << vec.z << ")\n";
        return os;
    }

    /**
     * @brief Vector cross product
     *
     * @param vec1: cartesian vector
     * @param vec2: cartesian vector
     * @return: the vector cross product
     */
    [[nodiscard]] inline Cartesian cross(const Cartesian& vec1, const Cartesian& vec2) noexcept
    {
        return { vec1.y * vec2.z - vec1.z * vec2.y,
                 -(vec1.x * vec2.z - vec1.z * vec2.x),
                 vec1.x * vec2.y - vec1.y * vec2.x };
    }

    /**
     * @brief Vector norm
     *
     * @param vec: the cartesian vector
     * @return: the vector norm
     */
    [[nodiscard]] inline double norm(const Cartesian& vec) noexcept
    {
        return std::hypot(vec.x, vec.y, vec.z);
    }

    /**
     * @brief normalize the input vector
     *
     * @param vec: the cartesian vector
     * @return: normalized vector
     */
    [[nodiscard]] inline Cartesian normalize(const Cartesian& vec) noexcept
    {
        return vec / norm(vec);
    }

    /**
     * @brief angle between the two vectors
     *
     * @param vec1: cartesian vector
     * @param vec2: cartesian vector
     * @return unit vector in x direction
     */
    [[nodiscard]] inline double angle(const Cartesian& vec1, const Cartesian& vec2) noexcept
    {
        return std::acos(normalize(vec1) * normalize(vec2));
    }
} // namespace nc::coordinates
