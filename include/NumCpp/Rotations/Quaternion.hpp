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
/// Holds a unit quaternion
///
#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/argmax.hpp"
#include "NumCpp/Functions/clip.hpp"
#include "NumCpp/Functions/dot.hpp"
#include "NumCpp/Functions/norm.hpp"
#include "NumCpp/Functions/square.hpp"
#include "NumCpp/Linalg/hat.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"
#include "NumCpp/Utils/num2str.hpp"
#include "NumCpp/Utils/sqr.hpp"
#include "NumCpp/Vector/Vec3.hpp"

namespace nc::rotations
{
    //================================================================================
    // Class Description:
    /// Holds a unit quaternion
    class Quaternion
    {
    public:
        //============================================================================
        // Method Description:
        /// Default Constructor
        ///
        Quaternion() = default;

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param roll: euler roll angle in radians
        /// @param pitch: euler pitch angle in radians
        /// @param yaw: euler yaw angle in radians
        ///
        Quaternion(double roll, double pitch, double yaw) noexcept
        {
            eulerToQuat(roll, pitch, yaw);
        }

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param inI
        /// @param inJ
        /// @param inK
        /// @param inS
        ///
        Quaternion(double inI, double inJ, double inK, double inS) noexcept :
            components_{ inI, inJ, inK, inS }
        {
            normalize();
        }

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param components
        ///
        Quaternion(const std::array<double, 4>& components) noexcept :
            components_{ components }
        {
            normalize();
        }

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param inArray: if size = 3 the roll, pitch, yaw euler angles
        /// if size = 4 the i, j, k, s components
        /// if shape = [3, 3] then direction cosine matrix
        ///
        Quaternion(const NdArray<double>& inArray) :
            components_{ 0., 0., 0., 0. }
        {
            if (inArray.size() == 3)
            {
                // euler angles
                eulerToQuat(inArray[0], inArray[1], inArray[2]);
            }
            else if (inArray.size() == 4)
            {
                // quaternion i, j, k, s components
                stl_algorithms::copy(inArray.cbegin(), inArray.cend(), components_.begin());
                normalize();
            }
            else if (inArray.size() == 9)
            {
                // direction cosine matrix
                dcmToQuat(inArray);
            }
            else
            {
                THROW_INVALID_ARGUMENT_ERROR("input array is not a valid size.");
            }
        }

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param inAxis: Euler axis
        /// @param inAngle: Euler angle in radians
        ///
        Quaternion(const Vec3& inAxis, double inAngle) noexcept
        {
            // normalize the input vector
            Vec3 normAxis = inAxis.normalize();

            const double halfAngle    = inAngle / 2.;
            const double sinHalfAngle = std::sin(halfAngle);

            components_[0] = normAxis.x * sinHalfAngle;
            components_[1] = normAxis.y * sinHalfAngle;
            components_[2] = normAxis.z * sinHalfAngle;
            components_[3] = std::cos(halfAngle);
        }

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param inAxis: Euler axis x,y,z vector components
        /// @param inAngle: Euler angle in radians
        ///
        Quaternion(const NdArray<double>& inAxis, double inAngle) :
            Quaternion(Vec3(inAxis), inAngle)
        {
        }

        //============================================================================
        // Method Description:
        /// the angle of rotation around the rotation axis that is described by the quaternion
        ///
        /// @return radians
        ///
        [[nodiscard]] double angleOfRotation() const noexcept
        {
            return 2. * std::acos(s());
        }

        //============================================================================
        // Method Description:
        /// angular velocity vector between the two quaternions. The norm
        /// of the array is the magnitude
        ///
        /// @param inQuat1
        /// @param inQuat2
        /// @param inTime (seperation time)
        /// @return NdArray<double>
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

            NdArray<double> omega = q.transpose().dot(qDot.transpose());
            return omega *= 2.;
        }

        //============================================================================
        // Method Description:
        /// angular velocity vector between the two quaternions. The norm
        /// of the array is the magnitude
        ///
        /// @param inQuat2
        /// @param inTime (seperation time)
        /// @return NdArray<double>
        ///
        [[nodiscard]] NdArray<double> angularVelocity(const Quaternion& inQuat2, double inTime) const
        {
            return angularVelocity(*this, inQuat2, inTime);
        }

        //============================================================================
        // Method Description:
        /// the axis of rotation described by the quaternion
        ///
        /// @return Vec3
        ///
        [[nodiscard]] Vec3 axisOfRotation() const noexcept
        {
            const auto halfAngle    = angleOfRotation() / 2.;
            const auto sinHalfAngle = std::sin(halfAngle);
            auto       axis         = Vec3(i() / sinHalfAngle, j() / sinHalfAngle, k() / sinHalfAngle);

            // shouldn't be necessary, but let's be pedantic
            return axis.normalize();
        }

        //============================================================================
        // Method Description:
        /// quaternion conjugate
        ///
        /// @return Quaternion
        ///
        [[nodiscard]] Quaternion conjugate() const noexcept
        {
            return { -i(), -j(), -k(), s() };
        }

        //============================================================================
        // Method Description:
        /// returns the i component
        ///
        /// @return double
        ///
        [[nodiscard]] double i() const noexcept
        {
            return components_[0];
        }

        //============================================================================
        // Method Description:
        /// quaternion identity (0,0,0,1)
        ///
        /// @return Quaternion
        ///
        static Quaternion identity() noexcept
        {
            return {};
        }

        //============================================================================
        // Method Description:
        /// quaternion inverse
        ///
        /// @return Quaterion
        ///
        [[nodiscard]] Quaternion inverse() const noexcept
        {
            /// for unit quaternions the inverse is equal to the conjugate
            return conjugate();
        }

        //============================================================================
        // Method Description:
        /// returns the j component
        ///
        /// @return double
        ///
        [[nodiscard]] double j() const noexcept
        {
            return components_[1];
        }

        //============================================================================
        // Method Description:
        /// returns the k component
        ///
        /// @return double
        ///
        [[nodiscard]] double k() const noexcept
        {
            return components_[2];
        }

        //============================================================================
        // Method Description:
        /// linearly interpolates between the two quaternions
        ///
        /// @param inQuat1
        /// @param inQuat2
        /// @param inPercent [0, 1]
        /// @return Quaternion
        ///
        static Quaternion nlerp(const Quaternion& inQuat1, const Quaternion& inQuat2, double inPercent)
        {
            if (inPercent < 0. || inPercent > 1.)
            {
                THROW_INVALID_ARGUMENT_ERROR("input percent must be of the range [0,1].");
            }

            if (utils::essentiallyEqual(inPercent, 0.))
            {
                return inQuat1;
            }
            if (utils::essentiallyEqual(inPercent, 1.))
            {
                return inQuat2;
            }

            const double          oneMinus = 1. - inPercent;
            std::array<double, 4> newComponents{};

            stl_algorithms::transform(inQuat1.components_.begin(),
                                      inQuat1.components_.end(),
                                      inQuat2.components_.begin(),
                                      newComponents.begin(),
                                      [inPercent, oneMinus](double component1, double component2) -> double
                                      { return oneMinus * component1 + inPercent * component2; });

            return { newComponents };
        }

        //============================================================================
        // Method Description:
        /// linearly interpolates between the two quaternions
        ///
        /// @param inQuat2
        /// @param inPercent (0, 1)
        /// @return Quaternion
        ///
        [[nodiscard]] Quaternion nlerp(const Quaternion& inQuat2, double inPercent) const
        {
            return nlerp(*this, inQuat2, inPercent);
        }

        //============================================================================
        // Method Description:
        /// The euler pitch angle in radians
        ///
        /// @return euler pitch angle in radians
        ///
        [[nodiscard]] double pitch() const noexcept
        {
            return std::asin(2 * (s() * j() - k() * i()));
        }

        //============================================================================
        // Method Description:
        /// returns a quaternion to rotate about the pitch axis
        ///
        /// @param inAngle (radians)
        /// @return Quaternion
        ///
        static Quaternion pitchRotation(double inAngle) noexcept
        {
            return { 0., inAngle, 0. };
        }

        //============================================================================
        // Method Description:
        /// prints the Quaternion to the console
        ///
        void print() const
        {
            std::cout << *this;
        }

        //============================================================================
        // Method Description:
        /// The euler roll angle in radians
        ///
        /// @return euler roll angle in radians
        ///
        [[nodiscard]] double roll() const noexcept
        {
            return std::atan2(2. * (s() * i() + j() * k()), 1. - 2. * (utils::sqr(i()) + utils::sqr(j())));
        }

        //============================================================================
        // Method Description:
        /// returns a quaternion to rotate about the roll axis
        ///
        /// @param inAngle (radians)
        /// @return Quaternion
        ///
        static Quaternion rollRotation(double inAngle) noexcept
        {
            return { inAngle, 0., 0. };
        }

        //============================================================================
        // Method Description:
        /// rotate a vector using the quaternion
        ///
        /// @param inVector (cartesian vector with x,y,z components)
        /// @return NdArray<double> (cartesian vector with x,y,z components)
        ///
        [[nodiscard]] NdArray<double> rotate(const NdArray<double>& inVector) const
        {
            if (inVector.size() != 3)
            {
                THROW_INVALID_ARGUMENT_ERROR("input inVector must be a cartesion vector of length = 3.");
            }

            return *this * inVector;
        }

        //============================================================================
        // Method Description:
        /// rotate a vector using the quaternion
        ///
        /// @param inVec3
        /// @return Vec3
        ///
        [[nodiscard]] Vec3 rotate(const Vec3& inVec3) const
        {
            return *this * inVec3;
        }

        //============================================================================
        // Method Description:
        /// returns the s component
        ///
        /// @return double
        ///
        [[nodiscard]] double s() const noexcept
        {
            return components_[3];
        }

        //============================================================================
        // Method Description:
        /// spherical linear interpolates between the two quaternions
        ///
        /// @param inQuat1
        /// @param inQuat2
        /// @param inPercent (0, 1)
        /// @return Quaternion
        ///
        static Quaternion slerp(const Quaternion& inQuat1, const Quaternion& inQuat2, double inPercent)
        {
            if (inPercent < 0 || inPercent > 1)
            {
                THROW_INVALID_ARGUMENT_ERROR("input percent must be of the range [0, 1]");
            }

            if (utils::essentiallyEqual(inPercent, 0.))
            {
                return inQuat1;
            }
            if (utils::essentiallyEqual(inPercent, 1.))
            {
                return inQuat2;
            }

            double dotProduct = dot<double>(inQuat1.toNdArray(), inQuat2.toNdArray()).item();

            // If the dot product is negative, the quaternions
            // have opposite handed-ness and slerp won't take
            // the shorter path. Fix by reversing one quaternion.
            Quaternion quat1Copy(inQuat1);
            if (dotProduct < 0.)
            {
                quat1Copy *= -1.;
                dotProduct *= -1.;
            }

            constexpr double DOT_THRESHOLD = 0.9995;
            if (dotProduct > DOT_THRESHOLD)
            {
                // If the inputs are too close for comfort, linearly interpolate
                // and normalize the result.
                return nlerp(inQuat1, inQuat2, inPercent);
            }

            dotProduct          = clip(dotProduct, -1., 1.); // Robustness: Stay within domain of acos()
            const double theta0 = std::acos(dotProduct);     // angle between input vectors
            const double theta  = theta0 * inPercent;        // angle between v0 and result

            const double s0 = std::cos(theta) -
                              dotProduct * std::sin(theta) / std::sin(theta0); // == sin(theta_0 - theta) / sin(theta_0)
            const double s1 = std::sin(theta) / std::sin(theta0);

            NdArray<double> interpQuat = (quat1Copy.toNdArray() * s0) + (inQuat2.toNdArray() * s1);
            return Quaternion(interpQuat); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// spherical linear interpolates between the two quaternions
        ///
        /// @param inQuat2
        /// @param inPercent (0, 1)
        /// @return Quaternion
        ///
        [[nodiscard]] Quaternion slerp(const Quaternion& inQuat2, double inPercent) const
        {
            return slerp(*this, inQuat2, inPercent);
        }

        //============================================================================
        // Method Description:
        /// returns the quaternion as a string representation
        ///
        /// @return std::string
        ///
        [[nodiscard]] std::string str() const
        {
            std::string output = "[" + utils::num2str(i()) + ", " + utils::num2str(j()) + ", " + utils::num2str(k()) +
                                 ", " + utils::num2str(s()) + "]\n";

            return output;
        }

        //============================================================================
        // Method Description:
        /// returns the direction cosine matrix
        ///
        /// @return NdArray<double>
        ///
        [[nodiscard]] NdArray<double> toDCM() const
        {
            NdArray<double> dcm(3);

            const double q0 = i();
            const double q1 = j();
            const double q2 = k();
            const double q3 = s();

            const double q0sqr = utils::sqr(q0);
            const double q1sqr = utils::sqr(q1);
            const double q2sqr = utils::sqr(q2);
            const double q3sqr = utils::sqr(q3);

            dcm(0, 0) = q3sqr + q0sqr - q1sqr - q2sqr;
            dcm(0, 1) = 2. * (q0 * q1 - q3 * q2);
            dcm(0, 2) = 2. * (q0 * q2 + q3 * q1);
            dcm(1, 0) = 2. * (q0 * q1 + q3 * q2);
            dcm(1, 1) = q3sqr + q1sqr - q0sqr - q2sqr;
            dcm(1, 2) = 2. * (q1 * q2 - q3 * q0);
            dcm(2, 0) = 2. * (q0 * q2 - q3 * q1);
            dcm(2, 1) = 2. * (q1 * q2 + q3 * q0);
            dcm(2, 2) = q3sqr + q2sqr - q0sqr - q1sqr;

            return dcm;
        }

        //============================================================================
        // Method Description:
        /// returns the quaternion as an NdArray
        ///
        /// @return NdArray<double>
        ///
        [[nodiscard]] NdArray<double> toNdArray() const
        {
            auto componentsCopy = components_;
            return NdArray<double>(componentsCopy); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// returns a quaternion to rotate about the x-axis by the input angle
        ///
        /// @param inAngle (radians)
        /// @return Quaternion
        ///
        static Quaternion xRotation(double inAngle) noexcept
        {
            const Vec3 eulerAxis = { 1., 0., 0. };
            return Quaternion(eulerAxis, inAngle); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// The euler yaw angle in radians
        ///
        /// @return euler yaw angle in radians
        ///
        [[nodiscard]] double yaw() const noexcept
        {
            return std::atan2(2. * (s() * k() + i() * j()), 1. - 2. * (utils::sqr(j()) + utils::sqr(k())));
        }

        //============================================================================
        // Method Description:
        /// returns a quaternion to rotate about the yaw axis
        ///
        /// @param inAngle (radians)
        /// @return Quaternion
        ///
        static Quaternion yawRotation(double inAngle) noexcept
        {
            return { 0., 0., inAngle };
        }

        //============================================================================
        // Method Description:
        /// returns a quaternion to rotate about the y-axis by the input angle
        ///
        /// @param inAngle (radians)
        /// @return Quaternion
        ///
        static Quaternion yRotation(double inAngle) noexcept
        {
            const Vec3 eulerAxis = { 0., 1., 0. };
            return Quaternion(eulerAxis, inAngle); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// returns a quaternion to rotate about the y-axis by the input angle
        ///
        /// @param inAngle (radians)
        /// @return Quaternion
        ///
        static Quaternion zRotation(double inAngle) noexcept
        {
            const Vec3 eulerAxis = { 0., 0., 1. };
            return Quaternion(eulerAxis, inAngle); // NOLINT(modernize-return-braced-init-list)
        }

        //============================================================================
        // Method Description:
        /// equality operator
        ///
        /// @param inRhs
        /// @return bool
        ///
        bool operator==(const Quaternion& inRhs) const noexcept
        {
            const auto comparitor = [](double value1, double value2) noexcept -> bool
            { return utils::essentiallyEqual(value1, value2); };

            return stl_algorithms::equal(components_.begin(), components_.end(), inRhs.components_.begin(), comparitor);
        }

        //============================================================================
        // Method Description:
        /// equality operator
        ///
        /// @param inRhs
        /// @return bool
        ///
        bool operator!=(const Quaternion& inRhs) const noexcept
        {
            return !(*this == inRhs);
        }

        //============================================================================
        // Method Description:
        /// addition assignment operator
        ///
        /// @param inRhs
        /// @return Quaternion
        ///
        Quaternion& operator+=(const Quaternion& inRhs) noexcept
        {
            stl_algorithms::transform(components_.begin(),
                                      components_.end(),
                                      inRhs.components_.begin(),
                                      components_.begin(),
                                      std::plus<double>()); // NOLINT(modernize-use-transparent-functors)

            normalize();

            return *this;
        }

        //============================================================================
        // Method Description:
        /// addition operator
        ///
        /// @param inRhs
        /// @return Quaternion
        ///
        Quaternion operator+(const Quaternion& inRhs) const noexcept
        {
            return Quaternion(*this) += inRhs;
        }

        //============================================================================
        // Method Description:
        /// subtraction assignment operator
        ///
        /// @param inRhs
        /// @return Quaternion
        ///
        Quaternion& operator-=(const Quaternion& inRhs) noexcept
        {
            stl_algorithms::transform(components_.begin(),
                                      components_.end(),
                                      inRhs.components_.begin(),
                                      components_.begin(),
                                      std::minus<double>()); // NOLINT(modernize-use-transparent-functors)

            normalize();

            return *this;
        }

        //============================================================================
        // Method Description:
        /// subtraction operator
        ///
        /// @param inRhs
        /// @return Quaternion
        ///
        Quaternion operator-(const Quaternion& inRhs) const noexcept
        {
            return Quaternion(*this) -= inRhs;
        }

        //============================================================================
        // Method Description:
        /// negative operator
        ///
        /// @return Quaternion
        ///
        Quaternion operator-() const noexcept
        {
            return Quaternion(*this) *= -1.;
        }

        //============================================================================
        // Method Description:
        /// multiplication assignment operator
        ///
        /// @param inRhs
        /// @return Quaternion
        ///
        Quaternion& operator*=(const Quaternion& inRhs) noexcept
        {
            double q0 = inRhs.s() * i();
            q0 += inRhs.i() * s();
            q0 -= inRhs.j() * k();
            q0 += inRhs.k() * j();

            double q1 = inRhs.s() * j();
            q1 += inRhs.i() * k();
            q1 += inRhs.j() * s();
            q1 -= inRhs.k() * i();

            double q2 = inRhs.s() * k();
            q2 -= inRhs.i() * j();
            q2 += inRhs.j() * i();
            q2 += inRhs.k() * s();

            double q3 = inRhs.s() * s();
            q3 -= inRhs.i() * i();
            q3 -= inRhs.j() * j();
            q3 -= inRhs.k() * k();

            components_[0] = q0;
            components_[1] = q1;
            components_[2] = q2;
            components_[3] = q3;

            normalize();

            return *this;
        }

        //============================================================================
        // Method Description:
        /// multiplication operator, only useful for multiplying
        /// by negative 1, all others will be renormalized back out
        ///
        /// @param inScalar
        /// @return Quaternion
        ///
        Quaternion& operator*=(double inScalar) noexcept
        {
            stl_algorithms::for_each(components_.begin(),
                                     components_.end(),
                                     [inScalar](double& component) { component *= inScalar; });

            normalize();

            return *this;
        }

        //============================================================================
        // Method Description:
        /// multiplication operator
        ///
        /// @param inRhs
        /// @return Quaternion
        ///
        Quaternion operator*(const Quaternion& inRhs) const noexcept
        {
            return Quaternion(*this) *= inRhs;
        }

        //============================================================================
        // Method Description:
        /// multiplication operator, only useful for multiplying
        /// by negative 1, all others will be renormalized back out
        ///
        /// @param inScalar
        /// @return Quaternion
        ///
        Quaternion operator*(double inScalar) const noexcept
        {
            return Quaternion(*this) *= inScalar;
        }

        //============================================================================
        // Method Description:
        /// multiplication operator
        ///
        /// @param inVec
        /// @return NdArray<double>
        ///
        NdArray<double> operator*(const NdArray<double>& inVec) const
        {
            if (inVec.size() != 3)
            {
                THROW_INVALID_ARGUMENT_ERROR("input vector must be a cartesion vector of length = 3.");
            }

            const auto vecNorm = norm(inVec).item();
            if (utils::essentiallyEqual(vecNorm, 0.))
            {
                return inVec;
            }

            const auto p      = Quaternion(inVec[0], inVec[1], inVec[2], 0.);
            const auto pPrime = *this * p * this->inverse();

            NdArray<double> rotatedVec = { pPrime.i(), pPrime.j(), pPrime.k() };
            rotatedVec *= vecNorm;
            return rotatedVec;
        }

        //============================================================================
        // Method Description:
        /// multiplication operator
        ///
        /// @param inVec3
        /// @return Vec3
        ///
        Vec3 operator*(const Vec3& inVec3) const
        {
            return *this * inVec3.toNdArray();
        }

        //============================================================================
        // Method Description:
        /// division assignment operator
        ///
        /// @param inRhs
        /// @return Quaternion
        ///
        Quaternion& operator/=(const Quaternion& inRhs) noexcept
        {
            return *this *= inRhs.conjugate();
        }

        //============================================================================
        // Method Description:
        /// division operator
        ///
        /// @param inRhs
        /// @return Quaternion
        ///
        Quaternion operator/(const Quaternion& inRhs) const noexcept
        {
            return Quaternion(*this) /= inRhs;
        }

        //============================================================================
        // Method Description:
        /// IO operator for the Quaternion class
        ///
        /// @param inOStream
        /// @param inQuat
        /// @return std::ostream&
        ///
        friend std::ostream& operator<<(std::ostream& inOStream, const Quaternion& inQuat)
        {
            inOStream << inQuat.str();
            return inOStream;
        }

    private:
        //====================================Attributes==============================
        std::array<double, 4> components_{ { 0., 0., 0., 1. } };

        //============================================================================
        // Method Description:
        /// renormalizes the quaternion
        ///
        void normalize() noexcept
        {
            double sumOfSquares = 0.;
            std::for_each(components_.begin(),
                          components_.end(),
                          [&sumOfSquares](double component) noexcept -> void
                          { sumOfSquares += utils::sqr(component); });

            const double norm = std::sqrt(sumOfSquares);
            stl_algorithms::for_each(components_.begin(),
                                     components_.end(),
                                     [norm](double& component) noexcept -> void { component /= norm; });
        }

        //============================================================================
        // Method Description:
        /// Converts the euler roll, pitch, yaw angles to quaternion components
        ///
        /// @ param roll: the euler roll angle in radians
        /// @ param pitch: the euler pitch angle in radians
        /// @ param yaw: the euler yaw angle in radians
        ///
        void eulerToQuat(double roll, double pitch, double yaw) noexcept
        {
            const auto halfPhi   = roll / 2.;
            const auto halfTheta = pitch / 2.;
            const auto halfPsi   = yaw / 2.;

            const auto sinHalfPhi = std::sin(halfPhi);
            const auto cosHalfPhi = std::cos(halfPhi);

            const auto sinHalfTheta = std::sin(halfTheta);
            const auto cosHalfTheta = std::cos(halfTheta);

            const auto sinHalfPsi = std::sin(halfPsi);
            const auto cosHalfPsi = std::cos(halfPsi);

            components_[0] = sinHalfPhi * cosHalfTheta * cosHalfPsi;
            components_[0] -= cosHalfPhi * sinHalfTheta * sinHalfPsi;

            components_[1] = cosHalfPhi * sinHalfTheta * cosHalfPsi;
            components_[1] += sinHalfPhi * cosHalfTheta * sinHalfPsi;

            components_[2] = cosHalfPhi * cosHalfTheta * sinHalfPsi;
            components_[2] -= sinHalfPhi * sinHalfTheta * cosHalfPsi;

            components_[3] = cosHalfPhi * cosHalfTheta * cosHalfPsi;
            components_[3] += sinHalfPhi * sinHalfTheta * sinHalfPsi;
        }

        //============================================================================
        // Method Description:
        /// Converts the direction cosine matrix to quaternion components
        ///
        /// @ param dcm: the direction cosine matrix
        ///
        void dcmToQuat(const NdArray<double>& dcm)
        {
            const Shape inShape = dcm.shape();
            if (!(inShape.rows == 3 && inShape.cols == 3))
            {
                THROW_INVALID_ARGUMENT_ERROR("input direction cosine matrix must have shape = (3,3).");
            }

            NdArray<double> checks(1, 4);
            checks[0] = 1 + dcm(0, 0) + dcm(1, 1) + dcm(2, 2);
            checks[1] = 1 + dcm(0, 0) - dcm(1, 1) - dcm(2, 2);
            checks[2] = 1 - dcm(0, 0) + dcm(1, 1) - dcm(2, 2);
            checks[3] = 1 - dcm(0, 0) - dcm(1, 1) + dcm(2, 2);

            const uint32 maxIdx = argmax(checks).item();

            switch (maxIdx)
            {
                case 0:
                {
                    components_[3] = 0.5 * std::sqrt(1 + dcm(0, 0) + dcm(1, 1) + dcm(2, 2));
                    components_[0] = (dcm(2, 1) - dcm(1, 2)) / (4 * components_[3]);
                    components_[1] = (dcm(0, 2) - dcm(2, 0)) / (4 * components_[3]);
                    components_[2] = (dcm(1, 0) - dcm(0, 1)) / (4 * components_[3]);

                    break;
                }
                case 1:
                {
                    components_[0] = 0.5 * std::sqrt(1 + dcm(0, 0) - dcm(1, 1) - dcm(2, 2));
                    components_[1] = (dcm(1, 0) + dcm(0, 1)) / (4 * components_[0]);
                    components_[2] = (dcm(2, 0) + dcm(0, 2)) / (4 * components_[0]);
                    components_[3] = (dcm(2, 1) - dcm(1, 2)) / (4 * components_[0]);

                    break;
                }
                case 2:
                {
                    components_[1] = 0.5 * std::sqrt(1 - dcm(0, 0) + dcm(1, 1) - dcm(2, 2));
                    components_[0] = (dcm(1, 0) + dcm(0, 1)) / (4 * components_[1]);
                    components_[2] = (dcm(2, 1) + dcm(1, 2)) / (4 * components_[1]);
                    components_[3] = (dcm(0, 2) - dcm(2, 0)) / (4 * components_[1]);

                    break;
                }
                case 3:
                {
                    components_[2] = 0.5 * std::sqrt(1 - dcm(0, 0) - dcm(1, 1) + dcm(2, 2));
                    components_[0] = (dcm(2, 0) + dcm(0, 2)) / (4 * components_[2]);
                    components_[1] = (dcm(2, 1) + dcm(1, 2)) / (4 * components_[2]);
                    components_[3] = (dcm(1, 0) - dcm(0, 1)) / (4 * components_[2]);

                    break;
                }
            }
        }
    };
} // namespace nc::rotations
