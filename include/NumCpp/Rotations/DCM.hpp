/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2022 David Pilger
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
/// Factory methods for generating direction cosine matrices and vectors
///
#pragma once

#include "NumCpp/Functions/dot.hpp"
#include "NumCpp/Functions/round.hpp"
#include "NumCpp/Linalg/det.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Rotations/Quaternion.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"
#include "NumCpp/Vector/Vec3.hpp"

namespace nc
{
    namespace rotations
    {
        //================================================================================
        /// Factory methods for generating direction cosine matrices and vectors
        class DCM
        {
        public:
            //============================================================================
            // Method Description:
            /// returns a direction cosine matrix that rotates according
            /// to the input euler angles
            ///
            /// @param roll: euler roll angle in radians
            /// @param pitch: euler pitch angle in radians
            /// @param yaw: euler yaw angle in radians
            /// @return NdArray
            ///
            static NdArray<double> eulerAngles(double roll, double pitch, double yaw)
            {
                return Quaternion(roll, pitch, yaw).toDCM();
            }

            //============================================================================
            // Method Description:
            /// returns a direction cosine matrix that rotates according
            /// to the input euler angles
            ///
            /// @param angles: euler roll, pitch, angles
            /// @return NdArray
            ///
            static NdArray<double> eulerAngles(const NdArray<double>& angles)
            {
                return Quaternion(angles).toDCM();
            }

            //============================================================================
            // Method Description:
            /// returns a direction cosine matrix that rotates about
            /// the input axis by the input angle
            ///
            /// @param inAxis: euler axis cartesian vector with x,y,z components
            /// @param inAngle: euler angle in radians
            /// @return NdArray
            ///
            static NdArray<double> eulerAxisAngle(const NdArray<double>& inAxis, double inAngle)
            {
                return Quaternion(inAxis, inAngle).toDCM();
            }

            //============================================================================
            // Method Description:
            /// returns a direction cosine matrix that rotates about
            /// the input axis by the input angle
            ///
            /// @param inAxis: euler axis cartesian vector with x,y,z components
            /// @param inAngle: euler angle in radians
            /// @return NdArray
            ///
            static NdArray<double> eulerAxisAngle(const Vec3& inAxis, double inAngle)
            {
                return Quaternion(inAxis, inAngle).toDCM();
            }

            //============================================================================
            // Method Description:
            /// returns whether the input array is a direction cosine
            /// matrix
            ///
            /// @param inArray
            /// @return bool
            ///
            static bool isValid(const NdArray<double>& inArray)
            {
                const Shape inShape = inArray.shape();
                return inShape.rows == inShape.cols &&
                       utils::essentiallyEqual(round(linalg::det<double>(inArray), 2), 1.) &&
                       utils::essentiallyEqual(round(linalg::det<double>(inArray.transpose()), 2), 1.);
            }

            //============================================================================
            // Method Description:
            /// The euler roll angle in radians
            ///
            /// @param dcm: a valid direction cosine matrix
            /// @return euler roll angle in radians
            ///
            static double roll(const NdArray<double>& dcm)
            {
                return Quaternion(dcm).roll();
            }

            //============================================================================
            // Method Description:
            /// The euler pitch angle in radians
            ///
            /// @param dcm: a valid direction cosine matrix
            /// @return euler pitch angle in radians
            ///
            static double pitch(const NdArray<double>& dcm)
            {
                return Quaternion(dcm).pitch();
            }

            //============================================================================
            // Method Description:
            /// The euler yaw angle in radians
            ///
            /// @param dcm: a valid direction cosine matrix
            /// @return euler yaw angle in radians
            ///
            static double yaw(const NdArray<double>& dcm)
            {
                return Quaternion(dcm).yaw();
            }

            //============================================================================
            // Method Description:
            /// returns a direction cosine matrix that rotates about
            /// the x axis by the input angle
            ///
            /// @param inAngle (in radians)
            /// @return NdArray<double>
            ///
            static NdArray<double> xRotation(double inAngle)
            {
                return DCM::eulerAxisAngle(Vec3{ 1., 0., 0. }, inAngle);
            }

            //============================================================================
            // Method Description:
            /// returns a direction cosine matrix that rotates about
            /// the x axis by the input angle
            ///
            /// @param inAngle (in radians)
            /// @return NdArray<double>
            ///
            static NdArray<double> yRotation(double inAngle)
            {
                return DCM::eulerAxisAngle(Vec3{ 0., 1., 0. }, inAngle);
            }

            //============================================================================
            // Method Description:
            /// returns a direction cosine matrix that rotates about
            /// the x axis by the input angle
            ///
            /// @param inAngle (in radians)
            /// @return NdArray<double>
            ///
            static NdArray<double> zRotation(double inAngle)
            {
                return DCM::eulerAxisAngle(Vec3{ 0., 0., 1. }, inAngle);
            }
        };
    } // namespace rotations
} // namespace nc
