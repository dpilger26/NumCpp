/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
/// Copyright 2019 David Pilger
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
/// Simple Vector classes
///
#pragma once

#include"NumCpp/NdArray.hpp"

#include<cmath>
#include<initializer_list>
#include<stdexcept>

namespace nc
{
    //================================================================================
    // Class Description:
    ///						Holds a 2D vector
    class Vec2
    {
    public:
        //====================================Attributes==============================
        double x;
        double y;

        //============================================================================
        // Method Description:
        ///						Default Constructor
        ///
        Vec2() noexcept :
            x(0),
            y(0)
        {}

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param  inX: the x component
        /// @param  inY: the y component
        ///
        Vec2(double inX, double inY) noexcept :
            x(inX),
            y(inY)
        {}

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param  inList
        ///
        Vec2(const std::initializer_list<double>& inList) noexcept
        {
            if (inList.size() != 2)
            {
                throw std::invalid_argument("ERROR: Vec2 constructor: input initializer list must have a size = 2");
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
                throw std::invalid_argument("ERROR: Vec2 constructor: input NdArray must have a size = 2");
            }

            x = ndArray[0];
            y = ndArray[1];
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [1, 0]
        ///
        /// @return Vec2
        ///
        static Vec2 right() noexcept
        {
            return Vec2(1.0, 0.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [-1, 0]
        ///
        /// @return Vec2
        ///
        static Vec2 left() noexcept
        {
            return Vec2(-1.0, 0.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [0, 1]
        ///
        /// @return Vec2
        ///
        static Vec2 up() noexcept
        {
            return Vec2(0.0, 1.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [0, -1]
        ///
        /// @return Vec2
        ///
        static Vec2 down() noexcept
        {
            return Vec2(0.0, -1.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the Vec2 as an NdArray
        ///
        /// @return     NdArray
        ///
        NdArray<double> toNdArray() const noexcept
        {
            return { x, y };
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
    /// @param      vec
    /// @param      scaler
    /// @return     Vec2
    ///
    inline Vec2 operator+(const Vec2& vec, double scaler) noexcept
    {
        return Vec2(vec) += scaler;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the vector
    ///
    /// @param      scaler
    /// @param      vec
    /// @return     Vec2
    ///
    inline Vec2 operator+(double scaler, const Vec2& vec) noexcept
    {
        return Vec2(vec) += scaler;
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
        return Vec2(-vec.x, -vec.y);
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the vector
    ///
    /// @param      vec
    /// @param      scaler
    /// @return     Vec2
    ///
    inline Vec2 operator-(const Vec2& vec, double scaler) noexcept
    {
        return Vec2(vec) -= scaler;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the vector
    ///
    /// @param      scaler
    /// @param      vec
    /// @return     Vec2
    ///
    inline Vec2 operator-(double scaler, const Vec2& vec) noexcept
    {
        return -Vec2(vec) += scaler;
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
    /// @param      vec
    /// @param      scaler
    /// @return     Vec2
    ///
    inline Vec2 operator*(const Vec2& vec, double scaler) noexcept
    {
        return Vec2(vec) *= scaler;
    }

    //============================================================================
    // Method Description:
    ///						Scalar mulitplication
    ///
    /// @param      vec
    /// @param      scaler
    /// @return     Vec2
    ///
    inline Vec2 operator*(double scaler, const Vec2& vec) noexcept
    {
        return Vec2(vec) *= scaler;
    }

    //============================================================================
    // Method Description:
    ///						Vector mulitplication (dot product)
    ///
    /// @param      rhs
    /// @param      lhs
    /// @return     dot product
    ///				
    ///
    inline double operator*(const Vec2& vec1, const Vec2& vec2) noexcept
    {
        return vec1.dot(vec2);
    }

    //============================================================================
    // Method Description:
    ///						Matrix mulitplication
    ///
    /// @param      vec
    /// @param      ndArray
    /// @return     NdArray
    ///				
    inline NdArray<double> operator*(const Vec2& vec, const NdArray<double>& ndArray)
    {
        return vec.toNdArray().dot(ndArray);
    }

    //============================================================================
    // Method Description:
    ///						Matrix mulitplication
    ///
    /// @param      ndArray
    /// @param      vec
    /// @return     NdArray
    ///				
    inline NdArray<double> operator*(const NdArray<double>& ndArray, const Vec2& vec)
    {
        return ndArray.dot(vec.toNdArray());
    }

    //============================================================================
    // Method Description:
    ///						Scalar division
    ///
    /// @param      vec
    /// @param      scaler
    /// @return     Vec2
    ///
    inline Vec2 operator/(const Vec2& vec, double scaler) noexcept
    {
        return Vec2(vec) /= scaler;
    }

    //================================================================================
    // Class Description:
    ///						Holds a 3D vector
    class Vec3
    {
    public:
        //====================================Attributes==============================
        double x;
        double y;
        double z;

        //============================================================================
        // Method Description:
        ///						Default Constructor
        ///
        Vec3() noexcept :
            x(0),
            y(0),
            z(0)
        {}

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param  inX: the x component
        /// @param  inY: the y component
        /// @param  inZ: the y component
        ///
        Vec3(double inX, double inY, double inZ) noexcept :
            x(inX),
            y(inY),
            z(inZ)
        {}

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param  inList
        ///
        Vec3(const std::initializer_list<double>& inList) noexcept
        {
            if (inList.size() != 3)
            {
                throw std::invalid_argument("ERROR: Vec3 constructor: input initializer list must have a size = 3");
            }

            x = *inList.begin();
            y = *(inList.begin() + 1);
            z = *(inList.begin() + 2);
        }

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param  ndArray
        ///
        Vec3(const NdArray<double>& ndArray)
        {
            if (ndArray.size() != 3)
            {
                throw std::invalid_argument("ERROR: Vec3 constructor: input NdArray must have a size = 3");
            }

            x = ndArray[0];
            y = ndArray[1];
            z = ndArray[2];
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [1, 0, 0]
        ///
        /// @return Vec3
        ///
        static Vec3 right() noexcept
        {
            return Vec3(1.0, 0.0, 0.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [-1, 0, 0]
        ///
        /// @return Vec3
        ///
        static Vec3 left() noexcept
        {
            return Vec3(-1.0, 0.0, 0.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [0, 1, 0]
        ///
        /// @return Vec3
        ///
        static Vec3 up() noexcept
        {
            return Vec3(0.0, 1.0, 0.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [0, -1, 0]
        ///
        /// @return Vec3
        ///
        static Vec3 down() noexcept
        {
            return Vec3(0.0, -1.0, 0.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [0, 0, 1]
        ///
        /// @return Vec3
        ///
        static Vec3 forward() noexcept
        {
            return Vec3(0.0, 0.0, 1.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [0, 0, -1]
        ///
        /// @return Vec3
        ///
        static Vec3 back() noexcept
        {
            return Vec3(0.0, 0.0, -1.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the Vec2 as an NdArray
        ///
        /// @return     NdArray
        ///
        NdArray<double> toNdArray() const noexcept
        {
            return { x, y, z };
        }

        //============================================================================
        // Method Description:
        ///						Returns the dot product of the two vectors
        ///
        /// @param      otherVec
        /// @return     the dot product
        ///
        double dot(const Vec3& otherVec) const noexcept
        {
            return x * otherVec.x + y * otherVec.y + z * otherVec.z;
        }

        //============================================================================
        // Method Description:
        ///						Returns the cross product of the two vectors
        ///
        /// @param      otherVec
        /// @return     the dot product
        ///
        Vec3 cross(const Vec3& otherVec) const noexcept
        {
            double crossX = y * otherVec.z - z * otherVec.y;
            double crossY = -(x * otherVec.z - z * otherVec.x);
            double crossZ = x * otherVec.y - y * otherVec.x;

            return Vec3(crossX, crossY, crossZ);
        }

        //============================================================================
        // Method Description:
        ///						Returns the magnitude of the vector
        ///
        /// @return     magnitude of the vector
        ///
        double norm() const noexcept
        {
            return std::hypot(x, y, z);
        }

        //============================================================================
        // Method Description:
        ///						Returns a new normalized Vec3
        ///
        /// @return     Vec3
        ///
        Vec3 normalize() const noexcept
        {
            return Vec3(*this) /= norm();
        }

        //============================================================================
        // Method Description:
        ///						Adds the scaler to the vector
        ///
        /// @param  scaler
        /// @return Vec3
        ///
        Vec3& operator+=(double scaler) noexcept
        {
            x += scaler;
            y += scaler;
            z += scaler;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Adds the two vectors
        ///
        /// @param  rhs
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
        ///						Subtracts the scaler from the vector
        ///
        /// @param  scaler
        /// @return Vec3
        ///
        Vec3& operator-=(double scaler) noexcept
        {
            x -= scaler;
            y -= scaler;
            z -= scaler;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Subtracts the two vectors
        ///
        /// @param  rhs
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
        ///						Scalar mulitplication
        ///
        /// @param  scaler
        /// @return Vec3
        ///
        Vec3& operator*=(double scaler) noexcept
        {
            x *= scaler;
            y *= scaler;
            z *= scaler;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Scalar division
        ///
        /// @param  scaler
        /// @return Vec3
        ///
        Vec3& operator/=(double scaler) noexcept
        {
            x /= scaler;
            y /= scaler;
            z /= scaler;
            return *this;
        }
    };

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the vector
    ///
    /// @param      vec
    /// @param      scaler
    /// @return     Vec3
    ///
    inline Vec3 operator+(const Vec3& vec, double scaler) noexcept
    {
        return Vec3(vec) += scaler;
    }

    //============================================================================
    // Method Description:
    ///						Adds the scaler to the vector
    ///
    /// @param      scaler
    /// @param      vec
    /// @return     Vec3
    ///
    inline Vec3 operator+(double scaler, const Vec3& vec) noexcept
    {
        return Vec3(vec) += scaler;
    }

    //============================================================================
    // Method Description:
    ///						Adds the two vectors
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     Vec3
    ///
    inline Vec3 operator+(const Vec3& lhs, const Vec3& rhs) noexcept
    {
        return Vec3(lhs) += rhs;
    }

    //============================================================================
    // Method Description:
    ///						Returns the negative vector
    ///
    /// @return     Vec3
    ///
    inline Vec3 operator-(const Vec3& vec) noexcept
    {
        return Vec3(-vec.x, -vec.y, -vec.z);
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the vector
    ///
    /// @param      vec
    /// @param      scaler
    /// @return     Vec3
    ///
    inline Vec3 operator-(const Vec3& vec, double scaler) noexcept
    {
        return Vec3(vec) -= scaler;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the scaler from the vector
    ///
    /// @param      scaler
    /// @param      vec
    /// @return     Vec3
    ///
    inline Vec3 operator-(double scaler, const Vec3& vec) noexcept
    {
        return -Vec3(vec) += scaler;
    }

    //============================================================================
    // Method Description:
    ///						Subtracts the two vectors
    ///
    /// @param      lhs
    /// @param      rhs
    /// @return     Vec3
    ///
    inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs) noexcept
    {
        return Vec3(lhs) -= rhs;
    }

    //============================================================================
    // Method Description:
    ///						Scalar mulitplication
    ///
    /// @param      vec
    /// @param      scaler
    /// @return     Vec3
    ///
    inline Vec3 operator*(const Vec3& vec, double scaler) noexcept
    {
        return Vec3(vec) *= scaler;
    }

    //============================================================================
    // Method Description:
    ///						Scalar mulitplication
    ///
    /// @param      vec
    /// @param      scaler
    /// @return     Vec3
    ///
    inline Vec3 operator*(double scaler, const Vec3& vec) noexcept
    {
        return Vec3(vec) *= scaler;
    }

    //============================================================================
    // Method Description:
    ///						Vector mulitplication (dot product)
    ///
    /// @param      rhs
    /// @param      lhs
    /// @return     dot product
    ///				
    ///
    inline double operator*(const Vec3& vec1, const Vec3& vec2) noexcept
    {
        return vec1.dot(vec2);
    }

    //============================================================================
    // Method Description:
    ///						Matrix mulitplication
    ///
    /// @param      vec
    /// @param      ndArray
    /// @return     NdArray
    ///				
    inline NdArray<double> operator*(const Vec3& vec, const NdArray<double>& ndArray)
    {
        return vec.toNdArray().dot(ndArray);
    }

    //============================================================================
    // Method Description:
    ///						Matrix mulitplication
    ///
    /// @param      ndArray
    /// @param      vec
    /// @return     NdArray
    ///				
    inline NdArray<double> operator*(const NdArray<double>& ndArray, const Vec3& vec)
    {
        return ndArray.dot(vec.toNdArray());
    }

    //============================================================================
    // Method Description:
    ///						Scalar division
    ///
    /// @param      vec
    /// @param      scaler
    /// @return     Vec3
    ///
    inline Vec3 operator/(const Vec3& vec, double scaler) noexcept
    {
        return Vec3(vec) /= scaler;
    }
}
