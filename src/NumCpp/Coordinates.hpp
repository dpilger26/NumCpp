/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
/// Copyright 2018 David Pilger
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
/// A module for holding and working with coordinates in either Ra/Dec or cartesian formats
///
#pragma once

#include"NumCpp/DtypeInfo.hpp"
#include"NumCpp/NdArray.hpp"
#include"NumCpp/Methods.hpp"
#include"NumCpp/Types.hpp"
#include"NumCpp/Utils.hpp"

#include<iostream>
#include<stdexcept>
#include<string>
#include<utility>

namespace nc
{
    //================================Coordinates Namespace=============================
    ///A module for holding and working with coordinates in either Ra/Dec or cartesian formats
    namespace coordinates
    {
        //================================================================================
        ///						Holds a right ascension object
        template<typename dtype>
        class RA
        {
        private:
            //====================================Attributes==============================
            uint8   hours_{ 0 };
            uint8   minutes_{ 0 };
            dtype   seconds_{ 0.0 };
            dtype   degrees_{ 0.0 };
            dtype   radians_{ 0.0 };

        public:
            //============================================================================
            ///						Default Constructor, not super usefull on its own
            ///
            RA() noexcept
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::RA: constructor can only be called with floating point types.");
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param      inDegrees
            ///
            RA(dtype inDegrees) :
                degrees_(inDegrees),
                radians_(static_cast<dtype>(deg2rad(inDegrees)))
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::RA: constructor can only be called with floating point types.");

                if (inDegrees < 0 || inDegrees >= 360)
                {
                    std::string errStr = "ERROR: NC::coordinates::RA: input degrees must be of the range [0, 360)";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                hours_ = static_cast<uint8>(std::floor(degrees_ / 15.0));
                const double decMinutes = (degrees_ - static_cast<double>(hours_) * 15.0) * 4.0;
                minutes_ = static_cast<uint8>(std::floor(decMinutes));
                seconds_ = static_cast<dtype>((decMinutes - static_cast<double>(minutes_)) * 60.0);
            }

            //============================================================================
            ///						Constructor
            ///
            ///	@param			inHours
            /// @param          inMinutes
            /// @param          inSeconds
            ///
            RA(uint8 inHours, uint8 inMinutes, dtype inSeconds)  noexcept :
                hours_(inHours),
                minutes_(inMinutes),
                seconds_(inSeconds)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::RA: constructor can only be called with floating point types.");

                degrees_ = static_cast<dtype>(static_cast<double>(hours_) * 15.0 + static_cast<double>(minutes_) / 4.0 + static_cast<double>(seconds_) / 240.0);
                radians_ = static_cast<dtype>(deg2rad(degrees_));
            }

            //============================================================================
            ///						Returns a copy of the RA object as a different type
            ///
            /// @return     RA
            ///
            template<typename dtypeOut>
            RA<dtypeOut> astype() noexcept
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::RA::astype: method can only be called with floating point types.");

                return RA<dtypeOut>(hours_, minutes_, static_cast<dtypeOut>(seconds_));
            }

            //============================================================================
            ///						Get the radians value
            ///
            /// @return     radians
            ///
            dtype radians() const noexcept
            {
                return radians_;
            }

            //============================================================================
            ///						Get the degrees value
            ///
            /// @return     degrees
            ///
            dtype degrees() const noexcept
            {
                return degrees_;
            }

            //============================================================================
            ///						Get the hour value
            ///
            /// @return     hours
            ///
            uint8 hours() const noexcept
            {
                return hours_;
            }

            //============================================================================
            ///						Get the minute value
            ///
            /// @return     minutes
            ///
            uint8 minutes() const noexcept
            {
                return minutes_;
            }

            //============================================================================
            ///						Get the seconds value
            ///
            /// @return     seconds
            ///
            dtype seconds() const noexcept
            {
                return seconds_;
            }

            //============================================================================
            ///						Return the RA object as a string representation
            ///
            /// @return     std::string
            ///
            std::string str() const
            {
                std::string out = "RA hms: " + utils::num2str(hours_) + " hours, " + utils::num2str(minutes_) + " minutes, ";
                out += utils::num2str(seconds_) + " seconds\nRA degrees: " + utils::num2str(degrees_) + "\n";
                return out;
            }

            //============================================================================
            ///						Prints the RA object to the console
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
            bool operator==(const RA<dtype>& inRhs) const noexcept
            {
                return degrees_ == inRhs.degrees_;
            }

            //============================================================================
            ///						Not equality operator
            ///
            /// @param      inRhs
            ///
            /// @return     bool
            ///
            bool operator!=(const RA<dtype>& inRhs) const noexcept
            {
                return !(*this == inRhs);
            }

            //============================================================================
            ///						Ostream operator
            ///
            /// @param      inStream
            /// @param      inRa
            ///
            friend std::ostream& operator<<(std::ostream& inStream, const RA<dtype>& inRa)
            {
                inStream << inRa.str();
                return inStream;
            }
        };

        //================================================================================
        ///						Struct Enum for positive or negative Dec angle
        enum class Sign { NEGATIVE = 0, POSITIVE };

        //================================================================================
        ///						Holds a Declination object
        template<typename dtype>
        class Dec
        {
        private:
            //====================================Attributes==============================
            Sign            sign_{ Sign::POSITIVE };
            uint8           degreesWhole_{ 0 };
            uint8           minutes_{ 0 };
            dtype           seconds_{ 0.0 };
            dtype           degrees_{ 0.0 };
            dtype           radians_{ 0.0 };

        public:
            //============================================================================
            ///						Default Constructor, not super usefull on its own
            ///
            Dec() noexcept
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::Dec: constructor can only be called with floating point types.");
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param      inDegrees
            ///
            Dec(dtype inDegrees) :
                degrees_(inDegrees),
                radians_(static_cast<dtype>(deg2rad(inDegrees)))
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::Dec: constructor can only be called with floating point types.");

                if (inDegrees < -90 || inDegrees > 90)
                {
                    std::string errStr = "ERROR: NC::coordinates::Dec: input degrees must be of the range [-90, 90]";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                sign_ = degrees_ < 0 ? Sign::NEGATIVE : Sign::POSITIVE;
                dtype absDegrees = std::abs(degrees_);
                degreesWhole_ = static_cast<uint8>(std::floor(absDegrees));

                const double decMinutes = (absDegrees - static_cast<double>(degreesWhole_)) * 60.0;
                minutes_ = static_cast<uint8>(std::floor(decMinutes));
                seconds_ = static_cast<dtype>((decMinutes - static_cast<double>(minutes_)) * 60.0);
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param      inSign
            ///	@param      inDegrees
            /// @param      inMinutes
            /// @param      inSeconds
            ///
            Dec(Sign inSign, uint8 inDegrees, uint8 inMinutes, dtype inSeconds)  noexcept :
                sign_(inSign),
                degreesWhole_(inDegrees),
                minutes_(inMinutes),
                seconds_(inSeconds)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::Dec: constructor can only be called with floating point types.");

                degrees_ = static_cast<dtype>(static_cast<double>(degreesWhole_) + static_cast<double>(minutes_) / 60.0 + static_cast<double>(seconds_) / 3600.0);
                degrees_ *= sign_ == Sign::NEGATIVE ? -1 : 1;

                radians_ = static_cast<dtype>(deg2rad(degrees_));
            }

            //============================================================================
            ///						Returns a copy of the Dec object as a different type
            ///
            /// @return     Dec
            ///
            template<typename dtypeOut>
            Dec<dtypeOut> astype() noexcept
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::Dec::astype: method can only be called with floating point types.");

                return Dec<dtypeOut>(sign_, degreesWhole_, minutes_, static_cast<dtypeOut>(seconds_));
            }

            //============================================================================
            ///						Get the sign of the degrees (positive or negative)
            ///
            /// @return     Sign
            ///
            Sign sign() const noexcept
            {
                return sign_;
            }

            //============================================================================
            ///						Get the degrees value
            ///
            /// @return     degrees
            ///
            dtype degrees() const noexcept
            {
                return degrees_;
            }

            //============================================================================
            ///						Get the radians value
            ///
            /// @return     minutes
            ///
            dtype radians() const noexcept
            {
                return radians_;
            }

            //============================================================================
            ///						Get the whole degrees value
            ///
            /// @return     whole degrees
            ///
            uint8 degreesWhole() const noexcept
            {
                return degreesWhole_;
            }

            //============================================================================
            ///						Get the minute value
            ///
            /// @return     minutes
            ///
            uint8 minutes() const noexcept
            {
                return minutes_;
            }

            //============================================================================
            ///						Get the seconds value
            ///
            /// @return     seconds
            ///
            dtype seconds() const noexcept
            {
                return seconds_;
            }

            //============================================================================
            ///						Return the dec object as a string representation
            ///
            /// @return     std::string
            ///
            std::string str() const
            {
                std::string strSign = sign_ == Sign::NEGATIVE ? "-" : "+";
                std::string out = "Dec dms: " + strSign + utils::num2str(degreesWhole_) + " degrees, " + utils::num2str(minutes_) + " minutes, ";
                out += utils::num2str(seconds_) + " seconds\nDec degrees = " + utils::num2str(degrees_) + "\n";
                return out;
            }

            //============================================================================
            ///						Prints the Dec object to the console
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
            bool operator==(const Dec<dtype>& inRhs) const noexcept
            {
                return degrees_ == inRhs.degrees_;
            }

            //============================================================================
            ///						Not equality operator
            ///
            /// @param      inRhs
            ///
            /// @return     bool
            ///
            bool operator!=(const Dec<dtype>& inRhs) const noexcept
            {
                return !(*this == inRhs);
            }

            //============================================================================
            ///						Ostream operator
            ///
            /// @param      inStream
            /// @param      inDec
            ///
            /// @return     std::ostream
            ///
            friend std::ostream& operator<<(std::ostream& inStream, const Dec<dtype>& inDec)
            {
                inStream << inDec.str();
                return inStream;
            }
        };

        //================================================================================
        ///						Holds a full coordinate object
        template<typename dtype>
        class Coordinate
        {
        private:
            //====================================Attributes==============================
            RA<dtype>       ra_;
            Dec<dtype>      dec_;
            dtype           x_{ 1.0 };
            dtype           y_{ 0.0 };
            dtype           z_{ 0.0 };

            //============================================================================
            ///						Converts polar coordinates to cartesian coordinates
            ///
            void cartesianToPolar()
            {
                dtype degreesRa = static_cast<dtype>(rad2deg(std::atan2(y_, x_)));
                if (degreesRa < 0)
                {
                    degreesRa += 360;
                }
                ra_ = RA<dtype>(degreesRa);

                const double r = std::sqrt(static_cast<double>(utils::sqr(x_)) + static_cast<double>(utils::sqr(y_)) + static_cast<double>(utils::sqr(z_)));
                dtype degreesDec = static_cast<dtype>(rad2deg(std::asin(static_cast<double>(z_) / r)));
                dec_ = Dec<dtype>(degreesDec);
            }

            //============================================================================
            ///						Converts polar coordinates to cartesian coordinates
            ///
            void polarToCartesian() noexcept
            {
                const double raRadians = deg2rad(static_cast<double>(ra_.degrees()));
                const double decRadians = deg2rad(static_cast<double>(dec_.degrees()));

                x_ = static_cast<dtype>(std::cos(raRadians) * std::cos(decRadians));
                y_ = static_cast<dtype>(std::sin(raRadians) * std::cos(decRadians));
                z_ = static_cast<dtype>(std::sin(decRadians));
            }

        public:
            //============================================================================
            ///						Default Constructor, not super usefull on its own
            ///
            Coordinate() noexcept
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::Dec: constructor can only be called with floating point types.");
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param      inRaDegrees
            /// @param      inDecDegrees
            ///
            Coordinate(dtype inRaDegrees, dtype inDecDegrees) :
                ra_(inRaDegrees),
                dec_(inDecDegrees)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::Dec: constructor can only be called with floating point types.");
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
            Coordinate(uint8 inRaHours, uint8 inRaMinutes, dtype inRaSeconds, Sign inSign,
                uint8 inDecDegreesWhole, uint8 inDecMinutes, dtype inDecSeconds)  noexcept :
                ra_(inRaHours, inRaMinutes, inRaSeconds),
                dec_(inSign, inDecDegreesWhole, inDecMinutes, inDecSeconds)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::Dec: constructor can only be called with floating point types.");
                polarToCartesian();
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param				inRA
            /// @param              inDec
            ///
            Coordinate(const RA<dtype>& inRA, const Dec<dtype>& inDec) noexcept :
                ra_(inRA),
                dec_(inDec)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::Dec: constructor can only be called with floating point types.");
                polarToCartesian();
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param				inX
            /// @param              inY
            /// @param              inZ
            ///
            Coordinate(dtype inX, dtype inY, dtype inZ) :
                x_(inX),
                y_(inY),
                z_(inZ)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::Dec: constructor can only be called with floating point types.");
                cartesianToPolar();
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param				inCartesianVector
            ///
            Coordinate(const NdArray<dtype> inCartesianVector)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NC::coordinates::Dec: constructor can only be called with floating point types.");

                if (inCartesianVector.size() != 3)
                {
                    std::string errStr = "ERROR: NC::coordinates::Dec: constructor NdArray input must be of length 3.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                x_ = inCartesianVector[0];
                y_ = inCartesianVector[1];
                z_ = inCartesianVector[2];

                cartesianToPolar();
            }

            //============================================================================
            ///						Returns a new Coordinate object with the specified type
            ///
            /// @return     Coordinate
            ///
            template<typename dtypeOut>
            Coordinate<dtypeOut> astype()
            {
                return Coordinate<dtypeOut>(static_cast<dtypeOut>(ra_.degrees()), static_cast<dtypeOut>(dec_.degrees()));
            }

            //============================================================================
            ///						Returns the Dec object
            ///
            /// @return             Dec
            ///
            const Dec<dtype>& dec() const noexcept
            {
                return dec_;
            }

            //============================================================================
            ///						Returns the RA object
            ///
            /// @return     RA
            ///
            const RA<dtype>& ra() const noexcept
            {
                return ra_;
            }

            //============================================================================
            ///						Returns the cartesian x value
            ///
            /// @return     x
            ///
            dtype x() const noexcept
            {
                return x_;
            }

            //============================================================================
            ///						Returns the cartesian y value
            ///
            /// @return     y
            ///
            dtype y() const noexcept
            {
                return y_;
            }

            //============================================================================
            ///						Returns the cartesian z value
            ///
            /// @return     z
            ///
            dtype z() const noexcept
            {
                return z_;
            }

            //============================================================================
            ///						Returns the cartesian xyz triplet as an NdArray
            ///
            /// @return     NdArray
            ///
            NdArray<dtype> xyz() const
            {
                NdArray<dtype> out = { x_, y_, z_ };
                return std::move(out);
            }

            //============================================================================
            ///						Returns the degree seperation between the two Coordinates
            ///
            /// @param      inOtherCoordinate
            ///
            /// @return     degrees
            ///
            dtype degreeSeperation(const Coordinate<dtype>& inOtherCoordinate) const
            {
                return static_cast<dtype>(rad2deg(radianSeperation(inOtherCoordinate)));
            }

            //============================================================================
            ///						Returns the degree seperation between the Coordinate
            ///                     and the input vector
            ///
            /// @param      inVector
            ///
            /// @return     degrees
            ///
            dtype degreeSeperation(const NdArray<dtype>& inVector) const
            {
                return static_cast<dtype>(rad2deg(radianSeperation(inVector)));
            }

            //============================================================================
            ///						Returns the radian seperation between the two Coordinates
            ///
            /// @param      inOtherCoordinate
            ///
            /// @return     radians
            ///
            dtype radianSeperation(const Coordinate<dtype>& inOtherCoordinate) const
            {
                return static_cast<dtype>(std::acos(dot<double>(xyz(), inOtherCoordinate.xyz()).item()));
            }

            //============================================================================
            ///						Returns the radian seperation between the Coordinate
            ///                     and the input vector
            ///
            /// @param      inVector
            ///
            /// @return     radians
            ///
            dtype radianSeperation(const NdArray<dtype>& inVector) const
            {
                if (inVector.size() != 3)
                {
                    std::string errStr = "ERROR: NC::coordinates::Coordinate::radianSeperation: input vector must be of length 3.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                return static_cast<dtype>(std::acos(dot<dtype>(xyz(), inVector.flatten()).item()));
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
            bool operator==(const Coordinate<dtype>& inRhs) const noexcept
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
            bool operator!=(const Coordinate<dtype>& inRhs) const noexcept
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
            friend std::ostream& operator<<(std::ostream& inStream, const Coordinate<dtype>& inCoord)
            {
                inStream << inCoord.str();
                return inStream;
            }
        };

        //============================================================================
        ///						Returns the degree seperation between the two Coordinates
        ///
        /// @param				inCoordinate1
        /// @param              inCoordinate2
        ///
        /// @return             degrees
        ///
        template<typename dtype>
        dtype degreeSeperation(const Coordinate<dtype>& inCoordinate1, const Coordinate<dtype>& inCoordinate2)
        {
            return inCoordinate1.degreeSeperation(inCoordinate2);
        }

        //============================================================================
        ///						Returns the degree seperation between the Coordinate
        ///                     and the input vector
        ///
        /// @param				inVector1
        /// @param              inVector2
        ///
        /// @return             degrees
        ///
        template<typename dtype>
        dtype degreeSeperation(const NdArray<dtype>& inVector1, const NdArray<dtype>& inVector2)
        {
            Coordinate<dtype> inCoord1(inVector1);
            return inCoord1.degreeSeperation(inVector1);
        }

        //============================================================================
        ///						Returns the radian seperation between the two Coordinates
        ///
        /// @param				inCoordinate1
        /// @param              inCoordinate2
        ///
        /// @return             radians
        ///
        template<typename dtype>
        dtype radianSeperation(const Coordinate<dtype>& inCoordinate1, const Coordinate<dtype>& inCoordinate2)
        {
            return inCoordinate1.radianSeperation(inCoordinate2);
        }

        //============================================================================
        ///						Returns the radian seperation between the Coordinate
        ///                     and the input vector
        ///
        /// @param				inVector1
        /// @param              inVector2
        ///
        /// @return             radians
        ///
        template<typename dtype>
        dtype radianSeperation(const NdArray<dtype>& inVector1, const NdArray<dtype>& inVector2)
        {
            Coordinate<dtype> inCoord1(inVector1);
            return inCoord1.radianSeperation(inVector1);
        }
    }
}
