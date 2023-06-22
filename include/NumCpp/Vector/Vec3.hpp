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
/// Simple 3D Vector class
///
#pragma once

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Functions/hypot.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"
#include "NumCpp/Utils/interp.hpp"
#include "NumCpp/Vector/Vec2.hpp"

//====================================================================================

namespace nc
{
    //================================================================================
    // Class Description:
    /// Holds a 3D vector
    class Vec3
    {
    public:
        //====================================Attributes==============================
        double x{ 0. };
        double y{ 0. };
        double z{ 0. };

        //============================================================================
        // Method Description:
        /// Default Constructor
        ///
        constexpr Vec3() = default;

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param inX: the x component
        /// @param inY: the y component
        /// @param inZ: the y component
        ///
        constexpr Vec3(double inX, double inY, double inZ) noexcept :
            x(inX),
            y(inY),
            z(inZ)
        {
        }

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param inList
        ///
        Vec3(const std::initializer_list<double>& inList)
        {
            if (inList.size() != 3)
            {
                THROW_INVALID_ARGUMENT_ERROR("input initializer list must have a size = 3");
            }

            x = *inList.begin();
            y = *(inList.begin() + 1);
            z = *(inList.begin() + 2);
        }

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param vec2: 2d vector
        ///
        constexpr Vec3(const Vec2& vec2) noexcept :
            x(vec2.x),
            y(vec2.y)
        {
        }

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param ndArray
        ///
        Vec3(const NdArray<double>& ndArray)
        {
            if (ndArray.size() != 3)
            {
                THROW_INVALID_ARGUMENT_ERROR("input NdArray must have a size = 3");
            }

            x = ndArray[0];
            y = ndArray[1];
            z = ndArray[2];
        }

        //============================================================================
        // Method Description:
        /// Returns the angle between the two vectors
        ///
        /// @param otherVec
        /// @return the angle in radians
        ///
        [[nodiscard]] double angle(const Vec3& otherVec) const noexcept
        {
            double dotProduct = dot(otherVec);
            dotProduct /= norm();
            dotProduct /= otherVec.norm();

            // clamp the value to the acos range just to be safe
            dotProduct = std::max(std::min(dotProduct, 1.), -1.);

            return std::acos(dotProduct);
        }

        //============================================================================
        // Method Description:
        /// Returns the unit vector [0, 0, -1]
        ///
        /// @return Vec3
        ///
        static constexpr Vec3 back() noexcept
        {
            return Vec3(0., 0., -1.); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// Returns a copy of the vector with its magnitude clamped
        /// to maxLength
        ///
        /// @param maxLength
        /// @return Vec3
        ///
        [[nodiscard]] Vec3 clampMagnitude(double maxLength) const noexcept
        {
            const double magnitude = norm();
            if (magnitude <= maxLength)
            {
                return *this;
            }

            Vec3 returnVec = Vec3(*this).normalize();
            returnVec *= maxLength;
            return returnVec;
        }

        //============================================================================
        // Method Description:
        /// Returns the cross product of the two vectors
        ///
        /// @param otherVec
        /// @return the dot product
        ///
        [[nodiscard]] Vec3 cross(const Vec3& otherVec) const noexcept
        {
            const double crossX = y * otherVec.z - z * otherVec.y;
            const double crossY = -(x * otherVec.z - z * otherVec.x);
            const double crossZ = x * otherVec.y - y * otherVec.x;

            return Vec3(crossX, crossY, crossZ); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// Returns the distance between the two vectors
        ///
        /// @param otherVec
        /// @return the distance (equivalent to (a - b).norm()
        ///
        [[nodiscard]] double distance(const Vec3& otherVec) const noexcept
        {
            return (Vec3(*this) -= otherVec).norm();
        }

        //============================================================================
        // Method Description:
        /// Returns the dot product of the two vectors
        ///
        /// @param otherVec
        /// @return the dot product
        ///
        [[nodiscard]] double dot(const Vec3& otherVec) const noexcept
        {
            return x * otherVec.x + y * otherVec.y + z * otherVec.z;
        }

        //============================================================================
        // Method Description:
        /// Returns the unit vector [0, -1, 0]
        ///
        /// @return Vec3
        ///
        static constexpr Vec3 down() noexcept
        {
            return Vec3(0., -1., 0.); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// Returns the unit vector [0, 0, 1]
        ///
        /// @return Vec3
        ///
        static constexpr Vec3 forward() noexcept
        {
            return Vec3(0., 0., 1.); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// Returns the unit vector [-1, 0, 0]
        ///
        /// @return Vec3
        ///
        static constexpr Vec3 left() noexcept
        {
            return Vec3(-1., 0., 0.); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// Linearly interpolates between two vectors
        ///
        /// @param otherVec
        /// @param t the amount to interpolate by (clamped from [0, 1]);
        /// @return Vec3
        ///
        [[nodiscard]] Vec3 lerp(const Vec3& otherVec, double t) const noexcept
        {
            t = std::max(std::min(t, 1.), 0.);

            Vec3 trajectory = otherVec;
            trajectory -= *this;
            const double xInterp = utils::interp(0., trajectory.x, t);
            const double yInterp = utils::interp(0., trajectory.y, t);
            const double zInterp = utils::interp(0., trajectory.z, t);

            return Vec3(*this) += Vec3(xInterp, yInterp, zInterp);
        }

        //============================================================================
        // Method Description:
        /// Returns the magnitude of the vector
        ///
        /// @return magnitude of the vector
        ///
        [[nodiscard]] double norm() const noexcept
        {
            return hypot(x, y, z);
        }

        //============================================================================
        // Method Description:
        /// Returns a new normalized Vec3
        ///
        /// @return Vec3
        ///
        [[nodiscard]] Vec3 normalize() const noexcept
        {
            return Vec3(*this) /= norm();
        }

        //============================================================================
        // Method Description:
        /// Projects the vector onto the input vector
        ///
        /// @param otherVec
        /// @return Vec3
        ///
        [[nodiscard]] Vec3 project(const Vec3& otherVec) const noexcept
        {
            const double projectedMagnitude = norm() * std::cos(angle(otherVec));
            return otherVec.normalize() *= projectedMagnitude;
        }

        //============================================================================
        // Method Description:
        /// Returns the unit vector [1, 0, 0]
        ///
        /// @return Vec3
        ///
        static constexpr Vec3 right() noexcept
        {
            return Vec3(1., 0., 0.); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// Returns the Vec3 as a string
        ///
        /// @return std::string
        ///
        [[nodiscard]] std::string toString() const
        {
            std::stringstream stream;
            stream << "Vec3[" << x << ", " << y << ", " << z << "]";
            return stream.str();
        }

        //============================================================================
        // Method Description:
        /// Returns the Vec2 as an NdArray
        ///
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<double> toNdArray() const
        {
            NdArray<double> returnArray = { x, y, z };
            return returnArray.transpose();
        }

        //============================================================================
        // Method Description:
        /// Returns the unit vector [0, 1, 0]
        ///
        /// @return Vec3
        ///
        static constexpr Vec3 up() noexcept
        {
            return Vec3(0., 1., 0.); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// Equality operator
        ///
        /// @param rhs
        /// @return bool
        ///
        bool operator==(const Vec3& rhs) const noexcept
        {
            return utils::essentiallyEqual(x, rhs.x) && utils::essentiallyEqual(y, rhs.y) &&
                   utils::essentiallyEqual(z, rhs.z);
        }

        //============================================================================
        // Method Description:
        /// Not Equality operator
        ///
        /// @param rhs
        /// @return bool
        ///
        bool operator!=(const Vec3& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        //============================================================================
        // Method Description:
        /// Adds the scalar to the vector
        ///
        /// @param scalar
        /// @return Vec3
        ///
        Vec3& operator+=(double scalar) noexcept
        {
            x += scalar;
            y += scalar;
            z += scalar;
            return *this;
        }

        //============================================================================
        // Method Description:
        /// Adds the two vectors
        ///
        /// @param rhs
        /// @return Vec3
        ///
        Vec3& operator+=(const Vec3& rhs) noexcept
        {
            x += rhs.x;
            y += rhs.y;
            z += rhs.z;
            return *this;
        }

        //============================================================================
        // Method Description:
        /// Subtracts the scalar from the vector
        ///
        /// @param scalar
        /// @return Vec3
        ///
        Vec3& operator-=(double scalar) noexcept
        {
            x -= scalar;
            y -= scalar;
            z -= scalar;
            return *this;
        }

        //============================================================================
        // Method Description:
        /// Subtracts the two vectors
        ///
        /// @param rhs
        /// @return Vec3
        ///
        Vec3& operator-=(const Vec3& rhs) noexcept
        {
            x -= rhs.x;
            y -= rhs.y;
            z -= rhs.z;
            return *this;
        }

        //============================================================================
        // Method Description:
        /// Scalar mulitplication
        ///
        /// @param scalar
        /// @return Vec3
        ///
        Vec3& operator*=(double scalar) noexcept
        {
            x *= scalar;
            y *= scalar;
            z *= scalar;
            return *this;
        }

        //============================================================================
        // Method Description:
        /// Scalar division
        ///
        /// @param scalar
        /// @return Vec3
        ///
        Vec3& operator/=(double scalar) noexcept
        {
            x /= scalar;
            y /= scalar;
            z /= scalar;
            return *this;
        }
    };

    //============================================================================
    // Method Description:
    /// Adds the scalar to the vector
    ///
    /// @param lhs
    /// @param rhs
    /// @return Vec3
    ///
    inline Vec3 operator+(const Vec3& lhs, double rhs) noexcept
    {
        return Vec3(lhs) += rhs;
    }

    //============================================================================
    // Method Description:
    /// Adds the scalar to the vector
    ///
    /// @param lhs
    /// @param rhs
    /// @return Vec3
    ///
    inline Vec3 operator+(double lhs, const Vec3& rhs) noexcept
    {
        return Vec3(rhs) += lhs;
    }

    //============================================================================
    // Method Description:
    /// Adds the two vectors
    ///
    /// @param lhs
    /// @param rhs
    /// @return Vec3
    ///
    inline Vec3 operator+(const Vec3& lhs, const Vec3& rhs) noexcept
    {
        return Vec3(lhs) += rhs;
    }

    //============================================================================
    // Method Description:
    /// Returns the negative vector
    ///
    /// @return Vec3
    ///
    inline Vec3 operator-(const Vec3& vec) noexcept
    {
        return Vec3(-vec.x, -vec.y, -vec.z); // NOLINT(modernize-return-braced-init-list)
    }

    //============================================================================
    // Method Description:
    /// Subtracts the scalar from the vector
    ///
    /// @param lhs
    /// @param rhs
    /// @return Vec3
    ///
    inline Vec3 operator-(const Vec3& lhs, double rhs) noexcept
    {
        return Vec3(lhs) -= rhs;
    }

    //============================================================================
    // Method Description:
    /// Subtracts the scalar from the vector
    ///
    /// @param lhs
    /// @param rhs
    /// @return Vec3
    ///
    inline Vec3 operator-(double lhs, const Vec3& rhs) noexcept
    {
        return -Vec3(rhs) += lhs;
    }

    //============================================================================
    // Method Description:
    /// Subtracts the two vectors
    ///
    /// @param lhs
    /// @param rhs
    /// @return Vec3
    ///
    inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs) noexcept
    {
        return Vec3(lhs) -= rhs;
    }

    //============================================================================
    // Method Description:
    /// Scalar mulitplication
    ///
    /// @param lhs
    /// @param rhs
    /// @return Vec3
    ///
    inline Vec3 operator*(const Vec3& lhs, double rhs) noexcept
    {
        return Vec3(lhs) *= rhs;
    }

    //============================================================================
    // Method Description:
    /// Scalar mulitplication
    ///
    /// @param lhs
    /// @param rhs
    /// @return Vec3
    ///
    inline Vec3 operator*(double lhs, const Vec3& rhs) noexcept
    {
        return Vec3(rhs) *= lhs;
    }

    //============================================================================
    // Method Description:
    /// Vector mulitplication (dot product)
    ///
    /// @param lhs
    /// @param rhs
    /// @return dot product
    ///
    ///
    inline double operator*(const Vec3& lhs, const Vec3& rhs) noexcept
    {
        return lhs.dot(rhs);
    }

    //============================================================================
    // Method Description:
    /// Scalar division
    ///
    /// @param lhs
    /// @param rhs
    /// @return Vec3
    ///
    inline Vec3 operator/(const Vec3& lhs, double rhs) noexcept
    {
        return Vec3(lhs) /= rhs;
    }

    //============================================================================
    // Method Description:
    /// stream output operator
    ///
    /// @param stream
    /// @param vec
    /// @return std::ostream
    ///
    inline std::ostream& operator<<(std::ostream& stream, const Vec3& vec)
    {
        stream << vec.toString() << std::endl;
        return stream;
    }
} // namespace nc
