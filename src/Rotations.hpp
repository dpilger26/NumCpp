// Copyright 2018 David Pilger
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files(the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, 
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
// permit persons to whom the Software is furnished to do so, subject to the following 
// conditions :
//
// The above copyright notice and this permission notice shall be included in all copies 
// or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
// DEALINGS IN THE SOFTWARE.

#pragma once

#include"Methods.hpp"
#include"Linalg.hpp"
#include"NdArray.hpp"
#include"Types.hpp"
#include"Utils.hpp"

#include<cmath>
#include<iostream>
#include<stdexcept>

namespace NumC
{
	//================================Rotations Namespace=============================
	namespace Rotations
	{
		//================================================================================
		// Class Description:
		//						holds a unit quaternion
		//
		class Quaternion
		{
		private:
			//====================================Attributes==============================
			double		data_[4];

			//============================================================================
			// Method Description: 
			//						renormalizes the quaternion
			//		
			// Inputs:
			//				None
			// Outputs:
			//				None
			//
			void normalize()
			{
				double norm = std::sqrt(sqr(data_[0]) + sqr(data_[1]) + sqr(data_[2]) + sqr(data_[3]));
				data_[0] /= norm;
				data_[1] /= norm;
				data_[2] /= norm;
				data_[3] /= norm;
			}

		public:
			//============================================================================
			// Method Description: 
			//						Default Constructor, not super usefull on its own
			//		
			// Inputs:
			//				None
			// Outputs:
			//				None
			//
			Quaternion()
			{
				data_[0] = 0.0;
				data_[1] = 0.0;
				data_[2] = 0.0;
				data_[3] = 1.0;
			}

			//============================================================================
			// Method Description: 
			//						Constructor
			//		
			// Inputs:
			//				i
			//				j
			//				k
			//				s
			// Outputs:
			//				None
			//
			Quaternion(double inI, double inJ, double inK, double inS)
			{
				double norm = std::sqrt(sqr(inI) + sqr(inJ) + sqr(inK) + sqr(inS));
				data_[0] = inI / norm;
				data_[1] = inJ / norm;
				data_[2] = inK / norm;
				data_[3] = inS / norm;
			}

			//============================================================================
			// Method Description: 
			//						Constructor
			//		
			// Inputs:
			//				NdArray, size = 4
			// Outputs:
			//				None
			//
			Quaternion(const NdArray<double>& inArray)
			{
				if (inArray.size() != 4)
				{
					throw std::invalid_argument("ERROR: Rotations:::Quaternion::Quaternion(NdArray): input array must be of size = 4;");
				}

				double norm = std::sqrt(sqr(inArray[0]) + sqr(inArray[1]) + sqr(inArray[2]) + sqr(inArray[3]));
				data_[0] = inArray[0] / norm;
				data_[1] = inArray[1] / norm;
				data_[2] = inArray[2] / norm;
				data_[3] = inArray[3] / norm;
			}

			//============================================================================
			// Method Description: 
			//						returns a quaternion to rotate about the input axis by the input angle
			//		
			// Inputs:
			//				NdArray, x,y,z vector components
			//				angle in radians 
			// Outputs:
			//				Quaternion
			//
			template<typename dtype>
			static Quaternion angleAxisRotation(const NdArray<dtype>& inAxis, double inAngle)
			{
				if (inAxis.size() != 3)
				{
					throw std::invalid_argument("ERROR: Rotations::Quaternion::angleAxisRotation: input axis must be a cartesion vector of length = 3.");
				}
				
				// normalize the input vector
				NdArray<double> normAxis = inAxis.astype<double>() / inAxis.norm<double>().item();

				double i = static_cast<double>(normAxis[0]) * std::sin(inAngle / 2.0);
				double j = static_cast<double>(normAxis[1]) * std::sin(inAngle / 2.0);
				double k = static_cast<double>(normAxis[2]) * std::sin(inAngle / 2.0);
				double s = std::cos(inAngle / 2.0);

				return Quaternion(i, j, k, s);
			}

			//============================================================================
			// Method Description: 
			//						angular velocity between the two quaternions
			//		
			// Inputs:
			//				Quaternion 1
			//				Quaternion 2
			//				seperation time
			// Outputs:
			//				Quaternion
			//
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

				NdArray<double> epsilonHat = Linalg::hat(inQuat2.i(), inQuat2.j(), inQuat2.k());
				NdArray<double> q = eyeTimesScalar + epsilonHat;
				q(3, 0) = -inQuat2.i();
				q(3, 1) = -inQuat2.j();
				q(3, 2) = -inQuat2.k();

				NdArray<double> omega = q.transpose().dot<double>(qDot);
				return std::move(omega *= 2);
			}

			//============================================================================
			// Method Description: 
			//						linearly interpolates between the two quaternions
			//		
			// Inputs:
			//				Quaternion 2
			//				seperation time
			// Outputs:
			//				Quaternion
			//
			NdArray<double> angularVelocity(const Quaternion& inQuat2, double inTime) const
			{
				return angularVelocity(*this, inQuat2, inTime);
			}

			//============================================================================
			// Method Description: 
			//						quaternion conjugate
			//		
			// Inputs:
			//				None
			// Outputs:
			//				s
			//
			Quaternion conjugate() const
			{
				return Quaternion(-i(), -j(), -k(), s());
			}

			//============================================================================
			// Method Description: 
			//						returns the i component
			//		
			// Inputs:
			//				None
			// Outputs:
			//				i
			//
			double i() const
			{
				return data_[0];
			}

			//============================================================================
			// Method Description: 
			//						quaternion identity (0,0,0,1)
			//		
			// Inputs:
			//				None
			// Outputs:
			//				s
			//
			static Quaternion identity()
			{
				return Quaternion(0.0, 0.0, 0.0, 1.0);
			}

			//============================================================================
			// Method Description: 
			//						quaternion inverse
			//		
			// Inputs:
			//				None
			// Outputs:
			//				s
			//
			Quaternion inverse() const
			{
				// for unit quaternions the inverse is equal to the conjugate
				return conjugate();
			}

			//============================================================================
			// Method Description: 
			//						returns the j component
			//		
			// Inputs:
			//				None
			// Outputs:
			//				j
			//
			double j() const
			{
				return data_[1];
			}

			//============================================================================
			// Method Description: 
			//						returns the k component
			//		
			// Inputs:
			//				None
			// Outputs:
			//				k
			//
			double k() const
			{
				return data_[2];
			}

			//============================================================================
			// Method Description: 
			//						converts from a direction cosine matrix to a quaternion
			//		
			// Inputs:
			//				NdArray
			// Outputs:
			//				Quaternion
			//
			template<typename dtype>
			static Quaternion fromDcm(const NdArray<dtype>& inDcm)
			{
				Shape inShape = inDcm.shape();
				if (!(inShape.rows == 3 && inShape.cols == 3))
				{
					throw std::invalid_argument("ERROR: Rotations::Quaternion::fromDcm: input direction cosine matrix must have shape = (3,3).");
				}

				NdArray<double> dcm = inDcm.astype<double>();

				NdArray<double> checks(1, 4);
				checks[0] = dcm(0, 0) + dcm(1, 1) + dcm(2, 2);
				checks[1] = dcm(0, 0) - dcm(1, 1) + dcm(2, 2);
				checks[2] = dcm(0, 0) + dcm(1, 1) - dcm(2, 2);
				checks[3] = dcm(0, 0) - dcm(1, 1) - dcm(2, 2);

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
						q0 = (dcm(0, 2) - dcm(2, 0)) / (4 * q2);
						q1 = (dcm(1, 2) - dcm(2, 1)) / (4 * q2);
						q3 = (dcm(0, 1) - dcm(1, 0)) / (4 * q2);

						break;
					}
					case 2:
					{
						q0 = 0.5 * std::sqrt(1 + dcm(0, 0) - dcm(1, 1) - dcm(2, 2));
						q1 = (dcm(0, 1) - dcm(1, 0)) / (4 * q0);
						q2 = (dcm(0, 2) - dcm(2, 0)) / (4 * q0);
						q3 = (dcm(1, 2) - dcm(2, 1)) / (4 * q0);

						break;
					}
					case 3:
					{
						q1 = 0.5 * std::sqrt(1 - dcm(0, 0) + dcm(1, 1) - dcm(2, 2));
						q0 = (dcm(0, 1) - dcm(1, 0)) / (4 * q1);
						q2 = (dcm(1, 2) - dcm(2, 1)) / (4 * q1);
						q3 = (dcm(2, 0) - dcm(0, 2)) / (4 * q1);

						break;
					}
				}

				return Quaternion(q0, q1, q2, q3);
			}

			//============================================================================
			// Method Description: 
			//						linearly interpolates between the two quaternions
			//		
			// Inputs:
			//				Quaternion 1
			//				Quaternion 2
			//				percent [0, 1]
			// Outputs:
			//				Quaternion
			//
			static Quaternion nlerp(const Quaternion& inQuat1, const Quaternion& inQuat2, double inPercent)
			{
				if (inPercent < 0 || inPercent > 1)
				{
					throw std::invalid_argument("ERROR: Rotations::Quaternion::nlerp: input percent must be of the range [0,1].");
				}

				if (inPercent == 0)
				{
					return inQuat1;
				}
				else if (inPercent == 1)
				{
					return inQuat2;
				}

				double oneMinus = 1.0 - inPercent;
				double i = oneMinus * inQuat1.data_[0] + inPercent * inQuat2.data_[0];
				double j = oneMinus * inQuat1.data_[1] + inPercent * inQuat2.data_[1];
				double k = oneMinus * inQuat1.data_[2] + inPercent * inQuat2.data_[2];
				double s = oneMinus * inQuat1.data_[3] + inPercent * inQuat2.data_[3];

				return Quaternion(i, j, k, s);
			}

			//============================================================================
			// Method Description: 
			//						linearly interpolates between the two quaternions
			//		
			// Inputs:
			//				Quaternion 2
			//				percent (0, 1)
			// Outputs:
			//				Quaternion
			//
			Quaternion nlerp(const Quaternion& inQuat2, double inPercent) const
			{
				return nlerp(*this, inQuat2, inPercent);
			}

			//============================================================================
			// Method Description: 
			//						prints the Quaternion to the console
			//		
			// Inputs:
			//				None
			// Outputs:
			//				None
			//
			void print()
			{
				std::cout << *this;
			}

			//============================================================================
			// Method Description: 
			//						rotate a vector using the quaternion
			//		
			// Inputs:
			//				cartesian vector with x,y,z components
			// Outputs:
			//				cartesian vector with x,y,z components
			//
			template<typename dtype>
			NdArray<double> rotate(const NdArray<dtype>& inVector) const
			{
				if (inVector.size() != 3)
				{
					throw std::invalid_argument("ERROR: Rotations::Quaternion::rotate: input inVector must be a cartesion vector of length = 3.");
				}

				return *this * inVector;
			}

			//============================================================================
			// Method Description: 
			//						returns the s component
			//		
			// Inputs:
			//				None
			// Outputs:
			//				s
			//
			double s() const
			{
				return data_[3];
			}

			//============================================================================
			// Method Description: 
			//						spherical linear interpolates between the two quaternions
			//		
			// Inputs:
			//				Quaternion 1
			//				Quaternion 2
			//				percent (0, 1)
			// Outputs:
			//				Quaternion
			//
			static Quaternion slerp(const Quaternion& inQuat1, const Quaternion& inQuat2, double inPercent)
			{
				if (inPercent < 0 || inPercent > 1)
				{
					throw std::invalid_argument("ERROR: Rotations::Quaternion::slerp: input percent must be of the range [0, 1]");
				}

				if (inPercent == 0)
				{
					return inQuat1;
				}
				else if (inPercent == 1)
				{
					return inQuat2;
				}

				double dotProduct = dot<double, double>(inQuat1.toNdArray(), inQuat2.toNdArray()).item();

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
				double theta0 = std::acos(dotProduct);		// angle between input vectors
				double theta = theta0 * inPercent;			// angle between v0 and result

				double s0 = std::cos(theta) - dotProduct * std::sin(theta) / std::sin(theta0);  // == sin(theta_0 - theta) / sin(theta_0)
				double s1 = std::sin(theta) / std::sin(theta0);

				NdArray<double> interpQuat = (inQuat1.toNdArray() * s0) + (inQuat2.toNdArray() * s1);
				return Quaternion(interpQuat);
			}

			//============================================================================
			// Method Description: 
			//						spherical linear interpolates between the two quaternions
			//		
			// Inputs:
			//				Quaternion 2
			//				percent (0, 1)
			// Outputs:
			//				Quaternion
			//
			Quaternion slerp(const Quaternion& inQuat2, double inPercent) const
			{
				return slerp(*this, inQuat2, inPercent);
			}

			//============================================================================
			// Method Description: 
			//						returns the direction cosine matrix
			//		
			// Inputs:
			//				None
			// Outputs:
			//				NdArray
			//
			NdArray<double> toDCM() const
			{
				NdArray<double> dcm(3);

				double q0 = i();
				double q1 = j();
				double q2 = k();
				double q3 = s();

				dcm(0, 0) = sqr(q3) + sqr(q0) - sqr(q1) - sqr(q2);
				dcm(0, 1) = 2 * (q0 * q1 + q3 * q2);
				dcm(0, 2) = 2 * (q0 * q2 - q3 * q1);
				dcm(1, 0) = 2 * (q0 * q1 - q3 * q2);
				dcm(1, 1) = sqr(q3) - sqr(q0) + sqr(q1) - sqr(q2);;
				dcm(1, 2) = 2 * (q1 * q2 + q3 * q0);
				dcm(2, 0) = 2 * (q0 * q2 + q3 * q1);
				dcm(2, 1) = 2 * (q1 * q2 - q3 * q0);
				dcm(2, 2) = sqr(q3) - sqr(q0) - sqr(q1) + sqr(q2);;

				return std::move(dcm);
			}

			//============================================================================
			// Method Description: 
			//						returns the quaternion as an NdArray
			//		
			// Inputs:
			//				None
			// Outputs:
			//				NdArray
			//
			NdArray<double> toNdArray() const
			{
				NdArray<double> returnArray = {data_[0], data_[1], data_[2], data_[3]};
				return std::move(returnArray);
			}

			//============================================================================
			// Method Description: 
			//						returns a quaternion to rotate about the x-axis by the input angle
			//		
			// Inputs:
			//				angle in radians 
			// Outputs:
			//				Quaternion
			//
			static Quaternion xRotation(double inAngle)
			{
				return angleAxisRotation<double>({ 1.0, 0.0, 0.0 }, inAngle);
			}

			//============================================================================
			// Method Description: 
			//						returns a quaternion to rotate about the y-axis by the input angle
			//		
			// Inputs:
			//				angle in radians 
			// Outputs:
			//				Quaternion
			//
			static Quaternion yRotation(double inAngle)
			{
				return angleAxisRotation<double>({ 0.0, 1.0, 0.0 }, inAngle);
			}

			//============================================================================
			// Method Description: 
			//						returns a quaternion to rotate about the y-axis by the input angle
			//		
			// Inputs:
			//				angle in radians 
			// Outputs:
			//				Quaternion
			//
			static Quaternion zRotation(double inAngle)
			{
				return angleAxisRotation<double>({ 0.0, 0.0, 1.0 }, inAngle);
			}

			//============================================================================
			// Method Description: 
			//						equality operator
			//		
			// Inputs:
			//				None
			// Outputs:
			//				None
			//
			bool operator==(const Quaternion& inRhs) const
			{
				return data_[0] == inRhs.data_[0] &&
					data_[1] == inRhs.data_[1] &&
					data_[2] == inRhs.data_[2] &&
					data_[3] == inRhs.data_[3];
			}

			//============================================================================
			// Method Description: 
			//						equality operator
			//		
			// Inputs:
			//				None
			// Outputs:
			//				None
			//
			bool operator!=(const Quaternion& inRhs) const
			{
				return !(*this == inRhs);
			}

			//============================================================================
			// Method Description: 
			//						addition operator
			//		
			// Inputs:
			//				Quaternion
			// Outputs:
			//				Quaternion
			//
			Quaternion operator+(const Quaternion& inRhs) const
			{
				return Quaternion(*this) += inRhs;
			}

			//============================================================================
			// Method Description: 
			//						addition assignment operator
			//		
			// Inputs:
			//				Quaternion
			// Outputs:
			//				Quaternion
			//
			Quaternion& operator+=(const Quaternion& inRhs)
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
			//						subtraction operator
			//		
			// Inputs:
			//				Quaternion
			// Outputs:
			//				Quaternion
			//
			Quaternion operator-(const Quaternion& inRhs) const
			{
				return Quaternion(*this) -= inRhs;
			}

			//============================================================================
			// Method Description: 
			//						subtraction assignment operator
			//		
			// Inputs:
			//				Quaternion
			// Outputs:
			//				Quaternion
			//
			Quaternion& operator-=(const Quaternion& inRhs)
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
			//						multiplication operator
			//		
			// Inputs:
			//				Quaternion
			// Outputs:
			//				Quaternion
			//
			Quaternion operator*(const Quaternion& inRhs) const
			{
				return Quaternion(*this) *= inRhs;
			}

			//============================================================================
			// Method Description: 
			//						multiplication operator, only useful for multiplying
			//						by negative 1, all others will be renormalized back out
			//		
			// Inputs:
			//				scalar value
			// Outputs:
			//				Quaternion
			//
			Quaternion operator*(double inScalar) const
			{
				return Quaternion(*this) *= inScalar;
			}

			//============================================================================
			// Method Description: 
			//						multiplication operator
			//		
			// Inputs:
			//				NdArray
			// Outputs:
			//				NdArray
			//
			template<typename dtype>
			NdArray<double> operator*(const NdArray<dtype>& inVec) const
			{
				if (inVec.size() != 3)
				{
					throw std::invalid_argument("ERROR: Rotations::Quaternion::operator*: input vector must be a cartesion vector of length = 3.");
				}

				return toDCM().dot<double>(inVec.astype<double>());
			}

			//============================================================================
			// Method Description: 
			//						multiplication assignment operator
			//		
			// Inputs:
			//				Quaternion
			// Outputs:
			//				Quaternion
			//
			Quaternion& operator*=(const Quaternion& inRhs)
			{
				data_[0] = inRhs.data_[3] * data_[0] + inRhs.data_[0] * data_[3] + inRhs.data_[1] * data_[2] + inRhs.data_[2] * data_[1];
				data_[1] = inRhs.data_[3] * data_[1] + inRhs.data_[0] * data_[2] + inRhs.data_[1] * data_[3] + inRhs.data_[2] * data_[0];
				data_[2] = inRhs.data_[3] * data_[2] + inRhs.data_[0] * data_[1] + inRhs.data_[1] * data_[0] + inRhs.data_[2] * data_[3];
				data_[3] = inRhs.data_[3] * data_[3] + inRhs.data_[0] * data_[0] + inRhs.data_[1] * data_[1] + inRhs.data_[2] * data_[2];
				normalize();

				return *this;
			}

			//============================================================================
			// Method Description: 
			//						multiplication operator, only useful for multiplying
			//						by negative 1, all others will be renormalized back out
			//		
			// Inputs:
			//				scalar value
			// Outputs:
			//				Quaternion
			//
			Quaternion& operator*=(double inScalar)
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
			//						division operator
			//		
			// Inputs:
			//				Quaternion
			// Outputs:
			//				Quaternion
			//
			Quaternion operator/(const Quaternion& inRhs) const
			{
				return Quaternion(*this) /= inRhs;
			}

			//============================================================================
			// Method Description: 
			//						division assignment operator
			//		
			// Inputs:
			//				Quaternion
			// Outputs:
			//				Quaternion
			//
			Quaternion& operator/=(const Quaternion& inRhs)
			{
				return *this *= inRhs.conjugate();
			}

			//============================================================================
			// Method Description: 
			//						io operator for the Shape class
			//		
			// Inputs:
			//				None
			// Outputs:
			//				None
			//
			friend std::ostream& operator<<(std::ostream& inOStream, const Quaternion& inQuat)
			{
				std::string output = "[" + num2str(inQuat.i()) + ", " + num2str(inQuat.j()) +
					", " + num2str(inQuat.k()) + ", " + num2str(inQuat.s()) + "]";
				inOStream << output << std::endl;
				return inOStream;
			}
		};

		//================================================================================
		// Factory methods for generating direction cosine matrices and vectors
		//

		//============================================================================
		// Method Description: 
		//						returns a direction cosine matrix that rotates about
		//						the input axis by the input angle
		//		
		// Inputs:
		//				NdArray, cartesian vector with x,y,z
		//				rotation angle, in radians
		// Outputs:
		//				NdArray
		//
		template<typename dtype>
		inline NdArray<double> angleAxisRotationDCM(const NdArray<dtype>& inArray, double inAngle)
		{
			if (inArray.size() != 3)
			{
				throw std::invalid_argument("ERROR: Rotations::Quaternion::angleAxisRotationDcm: input array must be of size = 3.");
			}

			return std::move(Quaternion::angleAxisRotation(inArray, inAngle).toDCM());
		}

		//============================================================================
		// Method Description: 
		//						returns whether the input array is a direction cosine
		//						matrix
		//		
		// Inputs:
		//				NdArray
		// Outputs:
		//				bool
		//
		template<typename dtype>
		inline bool isValidDCM(const NdArray<dtype>& inArray)
		{
			if (!array_equal(inArray, inArray.transpose()) || Linalg::det(inArray) != 1)
			{
				return false;
			}
			return true;
		}

		//============================================================================
		// Method Description: 
		//						returns a direction cosine matrix that rotates about
		//						the x axis by the input angle
		//		
		// Inputs:
		//				rotation angle, in radians
		// Outputs:
		//				NdArray
		//
		template<typename dtype>
		inline NdArray<double> xRotationDCM(double inAngle)
		{
			return std::move(angleAxisRotationDCM<double>({ 1.0, 0.0, 0.0 }, inAngle));
		}

		//============================================================================
		// Method Description: 
		//						returns a direction cosine matrix that rotates about
		//						the x axis by the input angle
		//		
		// Inputs:
		//				rotation angle, in radians
		// Outputs:
		//				NdArray
		//
		template<typename dtype>
		inline NdArray<double> yRotationDCM(double inAngle)
		{
			return std::move(angleAxisRotationDCM<double>({ 0.0, 1.0, 0.0 }, inAngle));
		}

		//============================================================================
		// Method Description: 
		//						returns a direction cosine matrix that rotates about
		//						the x axis by the input angle
		//		
		// Inputs:
		//				rotation angle, in radians
		// Outputs:
		//				NdArray
		//
		template<typename dtype>
		inline NdArray<double> zRotationDCM(double inAngle)
		{
			return std::move(angleAxisRotationDCM<double>({ 0.0, 0.0, 1.0 }, inAngle));
		}
	}
}