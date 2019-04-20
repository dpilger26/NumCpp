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
/// Module for dealing with rotations
///
#pragma once

#include"NumCpp/DtypeInfo.hpp"
#include"NumCpp/Methods.hpp"
#include"NumCpp/Linalg.hpp"
#include"NumCpp/NdArray.hpp"
#include"NumCpp/Types.hpp"
#include"NumCpp/Utils.hpp"

#include<cmath>
#include<iostream>
#include<stdexcept>
#include<string>

namespace nc
{
    //================================Rotations Namespace=============================
    /// Module for dealing with rotations
    namespace rotations
    {
        //================================================================================
        // Class Description:
        ///						Holds a unit quaternion
        class Quaternion
        {
        private:
            //====================================Attributes==============================
            double		data_[4] = { 0.0, 0.0, 0.0, 1.0 };

            //============================================================================
            // Method Description:
            ///						renormalizes the quaternion
            ///
            void normalize() noexcept
            {
                const double norm = std::sqrt(utils::sqr(data_[0]) + utils::sqr(data_[1]) + utils::sqr(data_[2]) + utils::sqr(data_[3]));
                data_[0] /= norm;
                data_[1] /= norm;
                data_[2] /= norm;
                data_[3] /= norm;
            }

        public:
            //============================================================================
            // Method Description:
            ///						Default Constructor, not super usefull on its own
            ///
            Quaternion() = default;

            //============================================================================
            // Method Description:
            ///						Constructor
            ///
            /// @param				inI
            /// @param				inJ
            /// @param				inK
            /// @param				inS
            ///
            Quaternion(double inI, double inJ, double inK, double inS) noexcept
            {
                const double norm = std::sqrt(utils::sqr(inI) + utils::sqr(inJ) + utils::sqr(inK) + utils::sqr(inS));
                data_[0] = inI / norm;
                data_[1] = inJ / norm;
                data_[2] = inK / norm;
                data_[3] = inS / norm;
            }

            //============================================================================
            // Method Description:
            ///						Constructor
            ///
            /// @param
            ///				inArray (size = 4)
            ///
            Quaternion(const NdArray<double>& inArray)
            {
                if (inArray.size() != 4)
                {
                    std::string errStr = "ERROR: rotations:::Quaternion::Quaternion(NdArray): input array must be of size = 4.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                double norm = std::sqrt(square(inArray).template sum<double>().item());
                data_[0] = inArray[0] / norm;
                data_[1] = inArray[1] / norm;
                data_[2] = inArray[2] / norm;
                data_[3] = inArray[3] / norm;
            }

            //============================================================================
            // Method Description:
            ///						returns a quaternion to rotate about the input axis by the input angle
            ///
            /// @param			inAxis (x,y,z vector components)
            /// @param			inAngle (radians)
            /// @return
            ///				Quaternion
            ///
            template<typename dtype>
            static Quaternion angleAxisRotation(const NdArray<dtype>& inAxis, double inAngle)
            {
                if (inAxis.size() != 3)
                {
                    std::string errStr = "ERROR: rotations::Quaternion::angleAxisRotation: input axis must be a cartesion vector of length = 3.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                // normalize the input vector
                NdArray<double> normAxis = inAxis.template astype<double>() / inAxis.template norm<double>().item();

                const double i = static_cast<double>(normAxis[0]) * std::sin(inAngle / 2.0);
                const double j = static_cast<double>(normAxis[1]) * std::sin(inAngle / 2.0);
                const double k = static_cast<double>(normAxis[2]) * std::sin(inAngle / 2.0);
                const double s = std::cos(inAngle / 2.0);

                return Quaternion(i, j, k, s);
            }

            //============================================================================
            // Method Description:
            ///						angular velocity vector between the two quaternions. The norm
            ///						of the array is the magnitude
            ///
            /// @param				inQuat1
            /// @param				inQuat2
            /// @param				inTime (seperation time)
            /// @return
            ///				NdArray<double>
            ///
            static NdArray<double> angularVelocity(const Quaternion& inQuat1, const Quaternion& inQuat2, double inTime)
            {
                NdArray<double> q0 = inQuat1.toNdArray();
                NdArray<double> q1 = inQuat2.toNdArray();

                NdArray<double> qDot = q1 - q0;
                qDot /= inTime;

                NdArray<double> eyeTimesScalar(3);
                eyeTimesScalar.zeros();
                eyeTimesScalar(0, 0) = inQuat2.s();
                eyeTimesScalar(1, 1) = inQuat2.s();
                eyeTimesScalar(2, 2) = inQuat2.s();

                NdArray<double> epsilonHat = linalg::hat<double>(inQuat2.i(), inQuat2.j(), inQuat2.k());
                NdArray<double> q(4, 3);
                q.put(Slice(0, 3), Slice(0, 3), eyeTimesScalar + epsilonHat);
                q(3, 0) = -inQuat2.i();
                q(3, 1) = -inQuat2.j();
                q(3, 2) = -inQuat2.k();

                NdArray<double> omega = q.transpose().template dot<double>(qDot.transpose());
                return omega *= 2;
            }

            //============================================================================
            // Method Description:
            ///						angular velocity vector between the two quaternions. The norm
            ///						of the array is the magnitude
            ///
            /// @param				inQuat2
            /// @param				inTime (seperation time)
            /// @return
            ///				NdArray<double>
            ///
            NdArray<double> angularVelocity(const Quaternion& inQuat2, double inTime) const
            {
                return angularVelocity(*this, inQuat2, inTime);
            }

            //============================================================================
            // Method Description:
            ///						quaternion conjugate
            ///
            /// @return
            ///				Quaternion
            ///
            Quaternion conjugate() const noexcept
            {
                return Quaternion(-i(), -j(), -k(), s());
            }

            //============================================================================
            // Method Description:
            ///						returns the i component
            ///
            /// @return
            ///				double
            ///
            double i() const noexcept
            {
                return data_[0];
            }

            //============================================================================
            // Method Description:
            ///						quaternion identity (0,0,0,1)
            ///
            /// @return
            ///				Quaternion
            ///
            static Quaternion identity() noexcept
            {
                return Quaternion(0.0, 0.0, 0.0, 1.0);
            }

            //============================================================================
            // Method Description:
            ///						quaternion inverse
            ///
            /// @return
            ///				Quaterion
            ///
            Quaternion inverse() const noexcept
            {
                /// for unit quaternions the inverse is equal to the conjugate
                return conjugate();
            }

            //============================================================================
            // Method Description:
            ///						returns the j component
            ///
            /// @return
            ///				double
            ///
            double j() const noexcept
            {
                return data_[1];
            }

            //============================================================================
            // Method Description:
            ///						returns the k component
            ///
            /// @return
            ///				double
            ///
            double k() const noexcept
            {
                return data_[2];
            }

            //============================================================================
            // Method Description:
            ///						converts from a direction cosine matrix to a quaternion
            ///
            /// @param
            ///				inDcm
            /// @return
            ///				Quaternion
            ///
            template<typename dtype>
            static Quaternion fromDCM(const NdArray<dtype>& inDcm)
            {
                const Shape inShape = inDcm.shape();
                if (!(inShape.rows == 3 && inShape.cols == 3))
                {
                    std::string errStr = "ERROR: rotations::Quaternion::fromDcm: input direction cosine matrix must have shape = (3,3).";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                NdArray<double> dcm = inDcm.template astype<double>();

                NdArray<double> checks(1, 4);
                checks[0] = dcm(0, 0) + dcm(1, 1) + dcm(2, 2);
                checks[1] = dcm(0, 0) - dcm(1, 1) + dcm(2, 2);
                checks[2] = dcm(0, 0) - dcm(1, 1) - dcm(2, 2);
                checks[3] = dcm(0, 0) + dcm(1, 1) - dcm(2, 2);

                uint32 maxIdx = argmax(checks).item();

                double q0 = 0;
                double q1 = 0;
                double q2 = 0;
                double q3 = 0;

                switch (maxIdx)
                {
                    case 0:
                    {
                        q3 = 0.5 * std::sqrt(1 + dcm(0, 0) + dcm(1, 1) + dcm(2, 2));
                        q0 = (dcm(1, 2) - dcm(2, 1)) / (4 * q3);
                        q1 = (dcm(2, 0) - dcm(0, 2)) / (4 * q3);
                        q2 = (dcm(0, 1) - dcm(1, 0)) / (4 * q3);

                        break;
                    }
                    case 1:
                    {
                        q2 = 0.5 * std::sqrt(1 - dcm(0, 0) - dcm(1, 1) + dcm(2, 2));
                        q0 = (dcm(0, 2) + dcm(2, 0)) / (4 * q2);
                        q1 = (dcm(1, 2) + dcm(2, 1)) / (4 * q2);
                        q3 = (dcm(0, 1) - dcm(1, 0)) / (4 * q2);

                        break;
                    }
                    case 2:
                    {
                        q0 = 0.5 * std::sqrt(1 + dcm(0, 0) - dcm(1, 1) - dcm(2, 2));
                        q1 = (dcm(0, 1) + dcm(1, 0)) / (4 * q0);
                        q2 = (dcm(0, 2) + dcm(2, 0)) / (4 * q0);
                        q3 = (dcm(1, 2) - dcm(2, 1)) / (4 * q0);

                        break;
                    }
                    case 3:
                    {
                        q1 = 0.5 * std::sqrt(1 - dcm(0, 0) + dcm(1, 1) - dcm(2, 2));
                        q0 = (dcm(0, 1) + dcm(1, 0)) / (4 * q1);
                        q2 = (dcm(1, 2) + dcm(2, 1)) / (4 * q1);
                        q3 = (dcm(2, 0) - dcm(0, 2)) / (4 * q1);

                        break;
                    }
                }

                return Quaternion(q0, q1, q2, q3);
            }

            //============================================================================
            // Method Description:
            ///						linearly interpolates between the two quaternions
            ///
            /// @param				inQuat1
            /// @param				inQuat2
            /// @param				inPercent [0, 1]
            /// @return
            ///				Quaternion
            ///
            static Quaternion nlerp(const Quaternion& inQuat1, const Quaternion& inQuat2, double inPercent)
            {
                if (inPercent < 0.0 || inPercent > 1.0)
                {
                    std::string errStr = "ERROR: rotations::Quaternion::nlerp: input percent must be of the range [0,1].";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                if (utils::essentiallyEqual(inPercent, 0.0))
                {
                    return inQuat1;
                }
                else if (utils::essentiallyEqual(inPercent, 1.0))
                {
                    return inQuat2;
                }

                const double oneMinus = 1.0 - inPercent;
                const double i = oneMinus * inQuat1.data_[0] + inPercent * inQuat2.data_[0];
                const double j = oneMinus * inQuat1.data_[1] + inPercent * inQuat2.data_[1];
                const double k = oneMinus * inQuat1.data_[2] + inPercent * inQuat2.data_[2];
                const double s = oneMinus * inQuat1.data_[3] + inPercent * inQuat2.data_[3];

                return Quaternion(i, j, k, s);
            }

            //============================================================================
            // Method Description:
            ///						linearly interpolates between the two quaternions
            ///
            /// @param				inQuat2
            /// @param				inPercent (0, 1)
            /// @return
            ///				Quaternion
            ///
            Quaternion nlerp(const Quaternion& inQuat2, double inPercent) const
            {
                return nlerp(*this, inQuat2, inPercent);
            }

            //============================================================================
            // Method Description:
            ///						prints the Quaternion to the console
            ///
            void print() const
            {
                std::cout << *this;
            }

            //============================================================================
            // Method Description:
            ///						rotate a vector using the quaternion
            ///
            /// @param
            ///				inVector (cartesian vector with x,y,z components)
            /// @return
            ///				NdArray<double> (cartesian vector with x,y,z components)
            ///
            template<typename dtype>
            NdArray<double> rotate(const NdArray<dtype>& inVector) const
            {
                if (inVector.size() != 3)
                {
                    std::string errStr = "ERROR: rotations::Quaternion::rotate: input inVector must be a cartesion vector of length = 3.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                return *this * inVector;
            }

            //============================================================================
            // Method Description:
            ///						returns the s component
            ///
            /// @return
            ///				double
            ///
            double s() const noexcept
            {
                return data_[3];
            }

            //============================================================================
            // Method Description:
            ///						spherical linear interpolates between the two quaternions
            ///
            /// @param				inQuat1
            /// @param				inQuat2
            /// @param				inPercent (0, 1)
            /// @return
            ///				Quaternion
            ///
            static Quaternion slerp(const Quaternion& inQuat1, const Quaternion& inQuat2, double inPercent)
            {
                if (inPercent < 0 || inPercent > 1)
                {
                    std::string errStr = "ERROR: rotations::Quaternion::slerp: input percent must be of the range [0, 1]";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                if (inPercent == 0)
                {
                    return inQuat1;
                }
                else if (inPercent == 1)
                {
                    return inQuat2;
                }

                double dotProduct = dot<double>(inQuat1.toNdArray(), inQuat2.toNdArray()).item();

                // If the dot product is negative, the quaternions
                // have opposite handed-ness and slerp won't take
                // the shorter path. Fix by reversing one quaternion.
                Quaternion quat1Copy(inQuat1);
                if (dotProduct < 0.0)
                {
                    quat1Copy *= -1;
                    dotProduct *= -1;
                }

                const double DOT_THRESHOLD = 0.9995;
                if (dotProduct > DOT_THRESHOLD) {
                    // If the inputs are too close for comfort, linearly interpolate
                    // and normalize the result.
                    return nlerp(inQuat1, inQuat2, inPercent);
                }

                dotProduct = clip(dotProduct, -1.0, 1.0);	// Robustness: Stay within domain of acos()
                const double theta0 = std::acos(dotProduct);		// angle between input vectors
                const double theta = theta0 * inPercent;			// angle between v0 and result

                const double s0 = std::cos(theta) - dotProduct * std::sin(theta) / std::sin(theta0);  // == sin(theta_0 - theta) / sin(theta_0)
                const double s1 = std::sin(theta) / std::sin(theta0);

                NdArray<double> interpQuat = (inQuat1.toNdArray() * s0) + (inQuat2.toNdArray() * s1);
                return Quaternion(interpQuat);
            }

            //============================================================================
            // Method Description:
            ///						spherical linear interpolates between the two quaternions
            ///
            /// @param				inQuat2
            /// @param				inPercent (0, 1)
            /// @return
            ///				Quaternion
            ///
            Quaternion slerp(const Quaternion& inQuat2, double inPercent) const
            {
                return slerp(*this, inQuat2, inPercent);
            }

            //============================================================================
            // Method Description:
            ///						returns the quaternion as a string representation
            ///
            /// @return
            ///				std::string
            ///
            std::string str() const
            {
                std::string output = "[" + utils::num2str(i()) + ", " + utils::num2str(j()) +
                    ", " + utils::num2str(k()) + ", " + utils::num2str(s()) + "]\n";

                return output;
            }

            //============================================================================
            // Method Description:
            ///						returns the direction cosine matrix
            ///
            /// @return
            ///				NdArray<double>
            ///
            NdArray<double> toDCM() const
            {
                NdArray<double> dcm(3);

                double q0 = i();
                double q1 = j();
                double q2 = k();
                double q3 = s();

                dcm(0, 0) = utils::sqr(q3) + utils::sqr(q0) - utils::sqr(q1) - utils::sqr(q2);
                dcm(0, 1) = 2 * (q0 * q1 + q3 * q2);
                dcm(0, 2) = 2 * (q0 * q2 - q3 * q1);
                dcm(1, 0) = 2 * (q0 * q1 - q3 * q2);
                dcm(1, 1) = utils::sqr(q3) - utils::sqr(q0) + utils::sqr(q1) - utils::sqr(q2);;
                dcm(1, 2) = 2 * (q1 * q2 + q3 * q0);
                dcm(2, 0) = 2 * (q0 * q2 + q3 * q1);
                dcm(2, 1) = 2 * (q1 * q2 - q3 * q0);
                dcm(2, 2) = utils::sqr(q3) - utils::sqr(q0) - utils::sqr(q1) + utils::sqr(q2);;

                return dcm;
            }

            //============================================================================
            // Method Description:
            ///						returns the quaternion as an NdArray
            ///
            /// @return
            ///				NdArray<double>
            ///
            NdArray<double> toNdArray() const
            {
                NdArray<double> returnArray = { data_[0], data_[1], data_[2], data_[3] };
                return returnArray;
            }

            //============================================================================
            // Method Description:
            ///						returns a quaternion to rotate about the x-axis by the input angle
            ///
            /// @param
            ///				inAngle (radians)
            /// @return
            ///				Quaternion
            ///
            static Quaternion xRotation(double inAngle)
            {
                return angleAxisRotation<double>({ 1.0, 0.0, 0.0 }, inAngle);
            }

            //============================================================================
            // Method Description:
            ///						returns a quaternion to rotate about the y-axis by the input angle
            ///
            /// @param
            ///				inAngle (radians)
            /// @return
            ///				Quaternion
            ///
            static Quaternion yRotation(double inAngle)
            {
                return angleAxisRotation<double>({ 0.0, 1.0, 0.0 }, inAngle);
            }

            //============================================================================
            // Method Description:
            ///						returns a quaternion to rotate about the y-axis by the input angle
            ///
            /// @param
            ///				inAngle (radians)
            /// @return
            ///				Quaternion
            ///
            static Quaternion zRotation(double inAngle)
            {
                return angleAxisRotation<double>({ 0.0, 0.0, 1.0 }, inAngle);
            }

            //============================================================================
            // Method Description:
            ///						equality operator
            ///
            /// @param
            ///				inRhs
            /// @return
            ///				bool
            ///
            bool operator==(const Quaternion& inRhs) const noexcept
            {
                return utils::essentiallyEqual(data_[0], inRhs.data_[0]) &&
                    utils::essentiallyEqual(data_[1], inRhs.data_[1]) &&
                    utils::essentiallyEqual(data_[2], inRhs.data_[2]) &&
                    utils::essentiallyEqual(data_[3], inRhs.data_[3]);
            }

            //============================================================================
            // Method Description:
            ///						equality operator
            ///
            /// @param
            ///				inRhs
            /// @return
            ///				bool
            ///
            bool operator!=(const Quaternion& inRhs) const noexcept
            {
                return !(*this == inRhs);
            }

            //============================================================================
            // Method Description:
            ///						addition operator
            ///
            /// @param
            ///				inRhs
            /// @return
            ///				Quaternion
            ///
            Quaternion operator+(const Quaternion& inRhs) const noexcept
            {
                return Quaternion(*this) += inRhs;
            }

            //============================================================================
            // Method Description:
            ///						addition assignment operator
            ///
            /// @param
            ///				inRhs
            /// @return
            ///				Quaternion
            ///
            Quaternion& operator+=(const Quaternion& inRhs) noexcept
            {
                data_[0] += inRhs.data_[0];
                data_[1] += inRhs.data_[1];
                data_[2] += inRhs.data_[2];
                data_[3] += inRhs.data_[3];
                normalize();

                return *this;
            }

            //============================================================================
            // Method Description:
            ///						subtraction operator
            ///
            /// @param
            ///				inRhs
            /// @return
            ///				Quaternion
            ///
            Quaternion operator-(const Quaternion& inRhs) const noexcept
            {
                return Quaternion(*this) -= inRhs;
            }

            //============================================================================
            // Method Description:
            ///						subtraction assignment operator
            ///
            /// @param
            ///				inRhs
            /// @return
            ///				Quaternion
            ///
            Quaternion& operator-=(const Quaternion& inRhs) noexcept
            {
                data_[0] -= inRhs.data_[0];
                data_[1] -= inRhs.data_[1];
                data_[2] -= inRhs.data_[2];
                data_[3] -= inRhs.data_[3];
                normalize();

                return *this;
            }

            //============================================================================
            // Method Description:
            ///						multiplication operator
            ///
            /// @param
            ///				inRhs
            /// @return
            ///				Quaternion
            ///
            Quaternion operator*(const Quaternion& inRhs) const noexcept
            {
                return Quaternion(*this) *= inRhs;
            }

            //============================================================================
            // Method Description:
            ///						multiplication operator, only useful for multiplying
            ///						by negative 1, all others will be renormalized back out
            ///
            /// @param
            ///				inScalar
            /// @return
            ///				Quaternion
            ///
            Quaternion operator*(double inScalar) const noexcept
            {
                return Quaternion(*this) *= inScalar;
            }

            //============================================================================
            // Method Description:
            ///						multiplication operator
            ///
            /// @param
            ///				inVec
            /// @return
            ///				NdArray<double>
            ///
            template<typename dtype>
            NdArray<double> operator*(const NdArray<dtype>& inVec) const
            {
                if (inVec.size() != 3)
                {
                    std::string errStr = "ERROR: rotations::Quaternion::operator*: input vector must be a cartesion vector of length = 3.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                return toDCM().template dot<double>(inVec.template astype<double>());
            }

            //============================================================================
            // Method Description:
            ///						multiplication assignment operator
            ///
            /// @param
            ///				inRhs
            /// @return
            ///				Quaternion
            ///
            Quaternion& operator*=(const Quaternion& inRhs) noexcept
            {
                const double q0 = inRhs.data_[3] * data_[0] + inRhs.data_[0] * data_[3] - inRhs.data_[1] * data_[2] + inRhs.data_[2] * data_[1];
                const double q1 = inRhs.data_[3] * data_[1] + inRhs.data_[0] * data_[2] + inRhs.data_[1] * data_[3] - inRhs.data_[2] * data_[0];
                const double q2 = inRhs.data_[3] * data_[2] - inRhs.data_[0] * data_[1] + inRhs.data_[1] * data_[0] + inRhs.data_[2] * data_[3];
                const double q3 = inRhs.data_[3] * data_[3] - inRhs.data_[0] * data_[0] - inRhs.data_[1] * data_[1] - inRhs.data_[2] * data_[2];

                data_[0] = q0;
                data_[1] = q1;
                data_[2] = q2;
                data_[3] = q3;
                normalize();

                return *this;
            }

            //============================================================================
            // Method Description:
            ///						multiplication operator, only useful for multiplying
            ///						by negative 1, all others will be renormalized back out
            ///
            /// @param
            ///				inScalar
            /// @return
            ///				Quaternion
            ///
            Quaternion& operator*=(double inScalar) noexcept
            {
                data_[0] *= inScalar;
                data_[1] *= inScalar;
                data_[2] *= inScalar;
                data_[3] *= inScalar;
                normalize();

                return *this;
            }

            //============================================================================
            // Method Description:
            ///						division operator
            ///
            /// @param
            ///				inRhs
            /// @return
            ///				Quaternion
            ///
            Quaternion operator/(const Quaternion& inRhs) const noexcept
            {
                return Quaternion(*this) /= inRhs;
            }

            //============================================================================
            // Method Description:
            ///						division assignment operator
            ///
            /// @param
            ///				inRhs
            /// @return
            ///				Quaternion
            ///
            Quaternion& operator/=(const Quaternion& inRhs) noexcept
            {
                return *this *= inRhs.conjugate();
            }

            //============================================================================
            // Method Description:
            ///						IO operator for the Quaternion class
            ///
            /// @param      inOStream
            /// @param 		inQuat
            /// @return
            ///				std::ostream&
            ///
            friend std::ostream& operator<<(std::ostream& inOStream, const Quaternion& inQuat)
            {
                inOStream << inQuat.str();
                return inOStream;
            }
        };

        //================================================================================
        /// Factory methods for generating direction cosine matrices and vectors
        template<typename dtype>
        class DCM
        {
        public:
            //============================================================================
            // Method Description:
            ///						returns a direction cosine matrix that rotates about
            ///						the input axis by the input angle
            ///
            /// @param				inArray (cartesian vector with x,y,z)
            /// @param				inAngle (in radians)
            /// @return
            ///				NdArray
            ///
            static NdArray<double> angleAxisRotation(const NdArray<dtype>& inArray, double inAngle)
            {
                if (inArray.size() != 3)
                {
                    std::string errStr = "ERROR: rotations::Quaternion::angleAxisRotationDcm: input array must be of size = 3.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                return Quaternion::angleAxisRotation(inArray, inAngle).toDCM();
            }

            //============================================================================
            // Method Description:
            ///						returns whether the input array is a direction cosine
            ///						matrix
            ///
            /// @param
            ///				inArray
            /// @return
            ///				bool
            ///
            static bool isValid(const NdArray<dtype>& inArray)
            {
                const Shape inShape = inArray.shape();
                if (!(inShape.rows == inShape.cols &&
                    round(linalg::det<dtype>(inArray), 2) == 1 &&
                    round(linalg::det<dtype>(inArray.transpose()), 2) == 1))
                {
                    return false;
                }
                return true;
            }

            //============================================================================
            // Method Description:
            ///						returns a direction cosine matrix that rotates about
            ///						the x axis by the input angle
            ///
            /// @param
            ///				inAngle (in radians)
            /// @return
            ///				NdArray<double>
            ///
            static NdArray<double> xRotation(double inAngle)
            {
                return DCM<dtype>::angleAxisRotation({ 1.0, 0.0, 0.0 }, inAngle);
            }

            //============================================================================
            // Method Description:
            ///						returns a direction cosine matrix that rotates about
            ///						the x axis by the input angle
            ///
            /// @param
            ///				inAngle (in radians)
            /// @return
            ///				NdArray<double>
            ///
            static NdArray<double> yRotation(double inAngle)
            {
                return DCM<dtype>::angleAxisRotation({ 0.0, 1.0, 0.0 }, inAngle);
            }

            //============================================================================
            // Method Description:
            ///						returns a direction cosine matrix that rotates about
            ///						the x axis by the input angle
            ///
            /// @param
            ///				inAngle (in radians)
            /// @return
            ///				NdArray<double>
            ///
            static NdArray<double> zRotation(double inAngle)
            {
                return DCM<dtype>::angleAxisRotation({ 0.0, 0.0, 1.0 }, inAngle);
            }
        };
    }
}
