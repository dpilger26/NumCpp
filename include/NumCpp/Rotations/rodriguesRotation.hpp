/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2021 David Pilger
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
/// Performs Rodriques' rotation formula
/// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Vector/Vec3.hpp"

#include <cmath>

namespace nc
{
    namespace rotations
    {
        //============================================================================
        // Method Description:
        ///	Performs Rodriques' rotation formula
        /// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        ///
        /// @param  k: the axis to rotate around
        /// @param  theta: the angle in radians to rotate
        /// @param  v: the vector to rotate
        ///
        /// @return Vec3
        ///
        inline Vec3 rodriguesRotation(const Vec3& k, double theta, const Vec3& v) noexcept
        {
            const auto kUnit = k.normalize();

            const auto vCosTheta = v * std::cos(theta);

            auto kCrossV = kUnit.cross(v);
            kCrossV *= std::sin(theta);

            const auto kDotV = kUnit.dot(v);
            auto kkDotV = kUnit * kDotV;
            kkDotV *= 1 - std::cos(theta);

            auto vec = vCosTheta + kCrossV;
            vec += kkDotV;

            return vec;
        }

        //============================================================================
        // Method Description:
        ///	Performs Rodriques' rotation formula
        /// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        ///
        /// @param  k: the axis to rotate around
        /// @param  theta: the angle in radians to rotate
        /// @param  v: the vector to rotate
        ///
        /// @return NdArray<double>
        ///
        template<typename dtype>
        NdArray<double> rodriguesRotation(const NdArray<dtype>& k, double theta, const NdArray<dtype>& v)
        {
            return rodriguesRotation(Vec3(k), theta, Vec3(v)).toNdArray();
        }
    }  // namespace rotations
}  // namespace nc