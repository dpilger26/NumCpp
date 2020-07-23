/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.1.0
///
/// @section License
/// Copyright 2020 David Pilger
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
/// @section Description
/// Simple 2D Vector class
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"
#include "NumCpp/Utils/interp.hpp"

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>

//====================================================================================

namespace nc
{
    //================================================================================
    // Class Description:
    ///						Holds a 2D vector
    class Vec2
    {
    public:
        //====================================Attributes==============================
        double x{ 0.0 };
        double y{ 0.0 };

        //============================================================================
        // Method Description:
        ///						Default Constructor
        ///
        constexpr Vec2() = default;

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param  inX: the x component
        /// @param  inY: the y component
        ///
        constexpr Vec2(double inX, double inY) noexcept :
            x(inX),
            y(inY)
        {}

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param  inList
        ///
        Vec2(const std::initializer_list<double>& inList)
        {
            if (inList.size() != 2)
            {
                THROW_INVALID_ARGUMENT_ERROR("input initializer list must have a size = 2");
            }

            x = *inList.begin();
            y = *(inList.begin() + 1);
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param  ndArray
        ///
        Vec2(const NdArray<double>& ndArray)
        {
            if (ndArray.size() != 2)
            {
                THROW_INVALID_ARGUMENT_ERROR("input NdArray must have a size = 2");
            }

            x = ndArray[0];
            y = ndArray[1];
        }

        //============================================================================
        // Method Description:
        ///						Returns the angle between the two vectors
        ///
        /// @param      otherVec
        /// @return     the angle in radians
        ///
        double angle(const Vec2& otherVec) const noexcept 
        {
            double dotProduct = dot(otherVec);
            dotProduct /= norm();
            dotProduct /= otherVec.norm();

            // clamp the value to the acos range just to be safe
            dotProduct = std::max(std::min(dotProduct, 1.0), -1.0);

            return std::acos(dotProduct);
        }

        //============================================================================
        // Method Description:
        ///						Returns a copy of the vector with its magnitude clamped
        ///                     to maxLength
        ///
        /// @param      maxLength
        /// @return     Vec2
        ///
        Vec2 clampMagnitude(double maxLength) const noexcept 
        {
            const double magnitude = norm();
            if (magnitude <= maxLength)
            {
                return *this;
            }
            
            Vec2 returnVec = Vec2(*this).normalize();
            returnVec *= maxLength;
            return returnVec;
        }

        //============================================================================
        // Method Description:
        ///						Returns the distance between the two vectors
        ///
        /// @param      otherVec
        /// @return     the distance (equivalent to (a - b).norm()
        ///
        double distance(const Vec2& otherVec) const noexcept 
        {
            return (Vec2(*this) -= otherVec).norm();
        }

        //============================================================================
        // Method Description:
        ///						Returns the dot product of the two vectors
        ///
        /// @param      otherVec
        /// @return     the dot product
        ///
        double dot(const Vec2& otherVec) const noexcept 
        {
            return x * otherVec.x + y * otherVec.y;
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [0, -1]
        ///
        /// @return Vec2
        ///
        static constexpr Vec2 down() noexcept
        {
            return Vec2(0.0, -1.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [-1, 0]
        ///
        /// @return Vec2
        ///
        static constexpr Vec2 left() noexcept
        {
            return Vec2(-1.0, 0.0);
        }

        //============================================================================
        // Method Description:
        ///						Linearly interpolates between two vectors
        ///
        /// @param otherVec
        /// @param t the amount to interpolate by (clamped from [0, 1]);
        /// @return Vec2
        ///
        Vec2 lerp(const Vec2& otherVec, double t) const noexcept
        {
            t = std::max(std::min(t, 1.0), 0.0);

            Vec2 trajectory = otherVec;
            trajectory -= *this;
            const double xInterp = utils::interp(0.0, trajectory.x, t);
            const double yInterp = utils::interp(0.0, trajectory.y, t);

            return Vec2(*this) += Vec2(xInterp, yInterp);
        }

        //============================================================================
        // Method Description:
        ///						Returns the magnitude of the vector
        ///
        /// @return     magnitude of the vector
        ///
        double norm() const noexcept 
        {
            return std::hypot(x, y);
        }

        //============================================================================
        // Method Description:
        ///						Returns a new normalized Vec2
        ///
        /// @return     Vec2
        ///
        Vec2 normalize() const noexcept 
        {
            return Vec2(*this) /= norm();
        }

        //============================================================================
        // Method Description:
        ///						Projects the vector onto the input vector
        ///
        /// @param      otherVec
        /// @return     Vec2
        ///
        Vec2 project(const Vec2& otherVec) const noexcept 
        {
            const double projectedMagnitude = norm() * std::cos(angle(otherVec));
            return otherVec.normalize() *= projectedMagnitude;
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [1, 0]
        ///
        /// @return Vec2
        ///
        static constexpr Vec2 right() noexcept
        {
            return Vec2(1.0, 0.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the Vec2 as a string
        ///
        /// @return     std::string
        ///
        std::string toString() const 
        {
            std::stringstream stream;
            stream << "Vec2[" << x << ", " << y << "]";
            return stream.str();
        }

        //============================================================================
        // Method Description:
        ///						Returns the Vec2 as an NdArray
        ///
        /// @return     NdArray
        ///
        NdArray<double> toNdArray() const 
        {
            NdArray<double> returnArray = { x, y };
            return returnArray.transpose();
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [0, 1]
        ///
        /// @return Vec2
        ///
        static constexpr Vec2 up() noexcept
        {
            return Vec2(0.0, 1.0);
        }

        //============================================================================
        // Method Description:
        ///						Equality operator
        ///
        /// @param  rhs
        /// @return bool
        ///
        bool operator==(const Vec2& rhs) const noexcept 
        {
            return utils::essentiallyEqual(x, rhs.x) && utils::essentiallyEqual(y, rhs.y);
        }

        //============================================================================
        // Method Description:
        ///						Not Equality operator
        ///
        /// @param  rhs
        /// @return bool
        ///
        bool operator!=(const Vec2& rhs) const noexcept 
        {
            return !(*this == rhs);
        }

        //============================================================================
        // Method Description:
        ///						Adds the scaler to the vector
        ///
        /// @param  scaler
        /// @return Vec2
        ///
        Vec2& operator+=(double scaler) noexcept 
        {
            x += scaler;
            y += scaler;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Adds the two vectors
        ///
        /// @param  rhs
        /// @return Vec2
        ///
        Vec2& operator+=(const Vec2& rhs) noexcept 
        {
            x += rhs.x;
            y += rhs.y;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Subtracts the scaler from the vector
        ///
        /// @param  scaler
        /// @return Vec2
        ///
        Vec2& operator-=(double scaler) noexcept 
        {
            x -= scaler;
            y -= scaler;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Subtracts the two vectors
        ///
        /// @param  rhs
        /// @return Vec2
        ///
        Vec2& operator-=(const Vec2& rhs) noexcept 
        {
            x -= rhs.x;
            y -= rhs.y;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Scalar mulitplication
        ///
        /// @param  scaler
        /// @return Vec2
        ///
        Vec2& operator*=(double scaler) noexcept 
        {
            x *= scaler;
            y *= scaler;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Scalar division
        ///
        /// @param  scaler
        /// @return Vec2
        ///
        Vec2& operator/=(double scaler) noexcept 
        {
            x /= scaler;
            y /= scaler;
            return *this;
        }
    };

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the vector
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     Vec2
    ///
    inline Vec2 operator+(const Vec2& lhs, double rhs) noexcept 
    {
        return Vec2(lhs) += rhs;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the vector
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     Vec2
    ///
    inline Vec2 operator+(double lhs, const Vec2& rhs) noexcept 
    {
        return Vec2(rhs) += lhs;
    }

    //============================================================================
    // Method Description:
    ///						Adds the two vectors
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     Vec2
    ///
    inline Vec2 operator+(const Vec2& lhs, const Vec2& rhs) noexcept 
    {
        return Vec2(lhs) += rhs;
    }

    //============================================================================
    // Method Description:
    ///						Returns the negative vector
    ///
    /// @return     Vec2
    ///
    inline Vec2 operator-(const Vec2& vec) noexcept 
    {
        return {-vec.x, -vec.y};
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the vector
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     Vec2
    ///
    inline Vec2 operator-(const Vec2& lhs, double rhs) noexcept 
    {
        return Vec2(lhs) -= rhs;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the vector
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     Vec2
    ///
    inline Vec2 operator-(double lhs, const Vec2& rhs) noexcept 
    {
        return -Vec2(rhs) += lhs;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the two vectors
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     Vec2
    ///
    inline Vec2 operator-(const Vec2& lhs, const Vec2& rhs) noexcept 
    {
        return Vec2(lhs) -= rhs;
    }

    //============================================================================
    // Method Description:
    ///						Scalar mulitplication
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     Vec2
    ///
    inline Vec2 operator*(const Vec2& lhs, double rhs) noexcept 
    {
        return Vec2(lhs) *= rhs;
    }

    //============================================================================
    // Method Description:
    ///						Scalar mulitplication
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     Vec2
    ///
    inline Vec2 operator*(double lhs, const Vec2& rhs) noexcept 
    {
        return Vec2(rhs) *= lhs;
    }

    //============================================================================
    // Method Description:
    ///						Vector mulitplication (dot product)
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     dot product
    ///				
    ///
    inline double operator*(const Vec2& lhs, const Vec2& rhs) noexcept 
    {
        return lhs.dot(rhs);
    }

    //============================================================================
    // Method Description:
    ///						Scalar division
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     Vec2
    ///
    inline Vec2 operator/(const Vec2& lhs, double rhs) noexcept 
    {
        return Vec2(lhs) /= rhs;
    }

    //============================================================================
    // Method Description:
    ///						stream output operator
    ///
    /// @param      stream
    /// @param      vec
    /// @return     std::ostream
    ///
    inline std::ostream& operator<<(std::ostream& stream, const Vec2& vec) 
    {
        stream << vec.toString() << std::endl;
        return stream;
    }
}  // namespace nc
