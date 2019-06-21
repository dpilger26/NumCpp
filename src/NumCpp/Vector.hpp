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
#include<stdexcept>

namespace nc
{
    //================================================================================
    // Class Description:
    ///						Holds a 2D vector
    class Vec2
    {
    public:
        //============================================================================
        // Method Description:
        ///						Default Constructor
        ///
        Vec2() noexcept = default;

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param  x: the x component
        /// @param  y: the y component
        ///
        Vec2(double x, double y) noexcept :
            x_(x),
            y_(y)
        {}

        //============================================================================
        // Method Description:
        ///						Constructor
        ///
        /// @param  array
        ///
        template<typename dtype>
        Vec2(const NdArray<dtype> ndArray)
        {
            if (ndArray.size() != 2)
            {
                throw std::invalid_argument("ERROR: Vec2 constructor: input NdArray must have a size = 2");
            }

            x_ = static_cast<double>(ndArray[0]);
            y_ = static_cast<double>(ndArray[1]);
        }

        //============================================================================
        // Method Description:
        ///						Returns the x component of the vector
        ///
        /// @return
        ///				x component
        ///
        double x() const noexcept
        {
            return x_;
        }

        //============================================================================
        // Method Description:
        ///						Returns the y component of the vector
        ///
        /// @return
        ///				y component
        ///
        double y() const noexcept
        {
            return y_;
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [1, 0]
        ///
        /// @return
        ///				Vec2
        ///
        static Vec2 right() noexcept
        {
            return Vec2(1.0, 0.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [-1, 0]
        ///
        /// @return
        ///				Vec2
        ///
        static Vec2 left() noexcept
        {
            return Vec2(-1.0, 0.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [0, 1]
        ///
        /// @return
        ///				Vec2
        ///
        static Vec2 up() noexcept
        {
            return Vec2(0.0, 1.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the unit vector [0, -1]
        ///
        /// @return
        ///				Vec2
        ///
        static Vec2 down() noexcept
        {
            return Vec2(0.0, -1.0);
        }

        //============================================================================
        // Method Description:
        ///						Returns the Vec2 as an NdArray<double>
        ///
        /// @return     NdArray
        ///
        NdArray<double> toNdArray() const noexcept
        {
            return {x_, y_};
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
            return std::hypot(x_, y_);
        }

        //============================================================================
        // Method Description:
        ///						Returns the new normalized Vec2
        ///
        /// @return     Vec2
        ///
        Vec2 normalize() const noexcept
        {
            return Vec2(*this) /= norm();
        }

        //============================================================================
        // Method Description:
        ///						Adds the scalar to the vector
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2 operator+(double scalar) const noexcept
        {
            return Vec2(*this) += scalar;
        }

        //============================================================================
        // Method Description:
        ///						Adds the two vectors
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2 operator+(const Vec2& rhs) const noexcept
        {
            return Vec2(*this) += rhs;
        }

        //============================================================================
        // Method Description:
        ///						Adds the scalar to the vector
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2& operator+=(double scalar) noexcept
        {
            x_ += scalar;
            y_ += scalar;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Adds the two vectors
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2& operator+=(const Vec2& rhs) noexcept
        {
            x_ += rhs.x_;
            y_ += rhs.y_;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Subtracts the scalar from the vector
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2 operator-(double scalar) const noexcept
        {
            return Vec2(*this) -= scalar;
        }

        //============================================================================
        // Method Description:
        ///						Subtracts the two vectors
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2 operator-(const Vec2& rhs) const noexcept
        {
            return Vec2(*this) -= rhs;
        }

        //============================================================================
        // Method Description:
        ///						Subtracts the scalar from the vector
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2& operator-=(double scalar) noexcept
        {
            x_ -= scalar;
            y_ -= scalar;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Subtracts the two vectors
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2& operator-=(const Vec2& rhs) noexcept
        {
            x_ -= rhs.x_;
            y_ -= rhs.y_;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Scalar mulitplication
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2 operator*(double scalar) const noexcept
        {
            return Vec2(*this) *= scalar;
        }

        //============================================================================
        // Method Description:
        ///						Matrix mulitplication
        ///
        /// @param      vec
        /// @param      ndArray
        /// @return
        ///				NdArray<double>
        ///
        template<typename dtype>
        friend NdArray<double> operator*(const Vec2& vec, const NdArray<dtype>& ndArray)
        {
            return vec.toNdArray().dot(ndArray.astype<double>());
        }

        //============================================================================
        // Method Description:
        ///						Matrix mulitplication
        ///
        /// @param      ndArray
        /// @param      vec
        /// @return
        ///				NdArray<double>
        ///
        template<typename dtype>
        friend NdArray<double> operator*(const NdArray<dtype>& ndArray, const Vec2& vec)
        {
            return vec * ndArray;
        }

        //============================================================================
        // Method Description:
        ///						Scalar mulitplication
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2& operator*=(double scalar) noexcept
        {
            x_ *= scalar;
            y_ *= scalar;
            return *this;
        }

        //============================================================================
        // Method Description:
        ///						Scalar division
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2 operator/(double scalar) const noexcept
        {
            return Vec2(*this) /= scalar;
        }

        //============================================================================
        // Method Description:
        ///						Scalar division
        ///
        /// @param
        ///				rhs
        /// @return
        ///				Vec2
        ///
        Vec2& operator/=(double scalar) noexcept
        {
            x_ /= scalar;
            y_ /= scalar;
            return *this;
        }

    private:
        //====================================Attributes==============================
        double x_;
        double y_;
    };

    //================================================================================
    // Class Description:
    ///						Holds a 3D vector
    class Vec3
    {

    };

    //================================================================================
    // Class Description:
    ///						Holds a 2D unit vector
    class Vec2Unit : public Vec2
    {

    };

    //================================================================================
    // Class Description:
    ///						Holds a 3D unit vector
    class Vec3Unit : public Vec3
    {

    };
}
