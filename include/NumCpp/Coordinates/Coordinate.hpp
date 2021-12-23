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
/// Coordinate Object
///
#pragma once

#include "NumCpp/Coordinates/Dec.hpp"
#include "NumCpp/Coordinates/RA.hpp"
#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/deg2rad.hpp"
#include "NumCpp/Functions/dot.hpp"
#include "NumCpp/Functions/rad2deg.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/sqr.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace nc
{
    namespace coordinates
    {
        //================================================================================
        ///						Holds a full coordinate object
        class Coordinate
        {
        public:
            //============================================================================
            ///						Default Constructor
            ///
            Coordinate() = default;

            //============================================================================
            ///						Constructor
            ///
            /// @param      inRaDegrees
            /// @param      inDecDegrees
            ///
            Coordinate(double inRaDegrees, double inDecDegrees) :
                ra_(inRaDegrees),
                dec_(inDecDegrees)
            {
                polarToCartesian();
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param				inRaHours
            /// @param              inRaMinutes
            /// @param              inRaSeconds
            /// @param              inSign
            /// @param              inDecDegreesWhole
            /// @param              inDecMinutes
            /// @param              inDecSeconds
            ///
            Coordinate(uint8 inRaHours, uint8 inRaMinutes, double inRaSeconds, Sign inSign,
                uint8 inDecDegreesWhole, uint8 inDecMinutes, double inDecSeconds)   :
                ra_(inRaHours, inRaMinutes, inRaSeconds),
                dec_(inSign, inDecDegreesWhole, inDecMinutes, inDecSeconds)
            {
                polarToCartesian();
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param				inRA
            /// @param              inDec
            ///
            Coordinate(const RA& inRA, const Dec& inDec) noexcept :
                ra_(inRA),
                dec_(inDec)
            {
                polarToCartesian();
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param				inX
            /// @param              inY
            /// @param              inZ
            ///
            Coordinate(double inX, double inY, double inZ) noexcept :
                x_(inX),
                y_(inY),
                z_(inZ)
            {
                cartesianToPolar();
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param				inCartesianVector
            ///
            Coordinate(const NdArray<double>& inCartesianVector)
            {
                if (inCartesianVector.size() != 3)
                {
                    THROW_INVALID_ARGUMENT_ERROR("NdArray input must be of length 3.");
                }

                x_ = inCartesianVector[0];
                y_ = inCartesianVector[1];
                z_ = inCartesianVector[2];

                cartesianToPolar();
            }

            //============================================================================
            ///						Returns the Dec object
            ///
            /// @return             Dec
            ///
            const Dec& dec() const noexcept 
            {
                return dec_;
            }

            //============================================================================
            ///						Returns the RA object
            ///
            /// @return     RA
            ///
            const RA& ra() const noexcept 
            {
                return ra_;
            }

            //============================================================================
            ///						Returns the cartesian x value
            ///
            /// @return     x
            ///
            double x() const noexcept 
            {
                return x_;
            }

            //============================================================================
            ///						Returns the cartesian y value
            ///
            /// @return     y
            ///
            double y() const noexcept 
            {
                return y_;
            }

            //============================================================================
            ///						Returns the cartesian z value
            ///
            /// @return     z
            ///
            double z() const noexcept 
            {
                return z_;
            }

            //============================================================================
            ///						Returns the cartesian xyz triplet as an NdArray
            ///
            /// @return     NdArray
            ///
            NdArray<double> xyz() const 
            {
                NdArray<double> out = { x_, y_, z_ };
                return out;
            }

            //============================================================================
            ///						Returns the degree seperation between the two Coordinates
            ///
            /// @param      inOtherCoordinate
            ///
            /// @return     degrees
            ///
            double degreeSeperation(const Coordinate& inOtherCoordinate) const 
            {
                return rad2deg(radianSeperation(inOtherCoordinate));
            }

            //============================================================================
            ///						Returns the degree seperation between the Coordinate
            ///                     and the input vector
            ///
            /// @param      inVector
            ///
            /// @return     degrees
            ///
            double degreeSeperation(const NdArray<double>& inVector) const
            {
                return rad2deg(radianSeperation(inVector));
            }

            //============================================================================
            ///						Returns the radian seperation between the two Coordinates
            ///
            /// @param      inOtherCoordinate
            ///
            /// @return     radians
            ///
            double radianSeperation(const Coordinate& inOtherCoordinate) const 
            {
                return std::acos(dot(xyz(), inOtherCoordinate.xyz()).item());
            }

            //============================================================================
            ///						Returns the radian seperation between the Coordinate
            ///                     and the input vector
            ///
            /// @param      inVector
            ///
            /// @return     radians
            ///
            double radianSeperation(const NdArray<double>& inVector) const
            {
                if (inVector.size() != 3)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input vector must be of length 3.");
                }

                return std::acos(dot(xyz(), inVector.flatten()).item());
            }

            //============================================================================
            ///						Returns coordinate as a string representation
            ///
            /// @return     string
            ///
            std::string str() const
            {
                std::string returnStr;
                returnStr = ra_.str();
                returnStr += dec_.str();
                returnStr += "Cartesian = " + xyz().str();
                return returnStr;
            }

            //============================================================================
            ///						Prints the Coordinate object to the console
            ///
            void print() const
            {
                std::cout << *this;
            }

            //============================================================================
            ///						Equality operator
            ///
            /// @param      inRhs
            ///
            /// @return     bool
            ///
            bool operator==(const Coordinate& inRhs) const noexcept
            {
                return ra_ == inRhs.ra_ && dec_ == inRhs.dec_;
            }

            //============================================================================
            ///						Not equality operator
            ///
            /// @param      inRhs
            ///
            /// @return     bool
            ///
            bool operator!=(const Coordinate& inRhs) const noexcept
            {
                return !(*this == inRhs);
            }

            //============================================================================
            ///						Ostream operator
            ///
            /// @param      inStream
            /// @param      inCoord
            ///
            /// @return     std::ostream
            ///
            friend std::ostream& operator<<(std::ostream& inStream, const Coordinate& inCoord)
            {
                inStream << inCoord.str();
                return inStream;
            }

        private:
            //====================================Attributes==============================
            RA      ra_{};
            Dec     dec_{};
            double  x_{ 1.0 };
            double  y_{ 0.0 };
            double  z_{ 0.0 };

            //============================================================================
            ///						Converts polar coordinates to cartesian coordinates
            ///
            void cartesianToPolar() noexcept
            {
                double degreesRa = rad2deg(std::atan2(y_, x_));
                if (degreesRa < 0)
                {
                    degreesRa += 360;
                }
                ra_ = RA(degreesRa);

                const double r = std::sqrt(utils::sqr(x_) + utils::sqr(y_) + utils::sqr(z_));
                const double degreesDec = rad2deg(std::asin(z_ / r));
                dec_ = Dec(degreesDec);
            }

            //============================================================================
            ///						Converts polar coordinates to cartesian coordinates
            ///
            void polarToCartesian() noexcept
            {
                const double raRadians = deg2rad(ra_.degrees());
                const double decRadians = deg2rad(dec_.degrees());

                x_ = std::cos(raRadians) * std::cos(decRadians);
                y_ = std::sin(raRadians) * std::cos(decRadians);
                z_ = std::sin(decRadians);
            }
        };
    } // namespace coordinates
} // namespace nc
