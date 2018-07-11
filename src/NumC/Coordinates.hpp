/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// @version 1.0
///
/// @section LICENSE
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
/// @section DESCRIPTION
/// A module for holding and working with coordinates in either Ra/Dec or cartesian formats
///
#pragma once

#include"NumC/DtypeInfo.hpp"
#include"NumC/NdArray.hpp"
#include"NumC/Methods.hpp"
#include"NumC/Types.hpp"
#include"NumC/Utils.hpp"

#include<iostream>
#include<stdexcept>
#include<string>
#include<utility>

namespace NumC
{
    //================================Coordinates Namespace=============================
    ///A module for holding and working with coordinates in either Ra/Dec or cartesian formats
    namespace Coordinates
    {
        //================================================================================
        ///						Holds a right ascension object
        template<typename dtype>
        class RA
        {
        private:
            //====================================Attributes==============================
            uint8   hours_;
            uint8   minutes_;
            dtype   seconds_;
            dtype   degrees_;
            dtype   radians_;

        public:
            //============================================================================
            ///						Default Constructor, not super usefull on its own
            ///		
            /// @param      None
            ///
            /// @return     None
            ///
            RA() :
                hours_(0),
                minutes_(0),
                seconds_(0.0),
                degrees_(0.0),
                radians_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::RA: constructor can only be called with floating point types.");
            }

            //============================================================================
            ///						Constructor
            ///		
            /// @param      degrees
            ///
            /// @return     None
            ///
            RA(dtype inDegrees) :
                hours_(0),
                minutes_(0),
                seconds_(0.0),
                degrees_(inDegrees),
                radians_(static_cast<dtype>(Methods<dtype>::deg2rad(inDegrees)))
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::RA: constructor can only be called with floating point types.");

                if (inDegrees < 0 || inDegrees >= 360)
                {
                    throw std::invalid_argument("ERROR: NumC::Coordinates::RA: input degrees must be of the range [0, 360)");
                }

                hours_ = static_cast<uint8>(std::floor(degrees_ / 15.0));
                double decMinutes = (degrees_ - static_cast<double>(hours_) * 15.0) * 4.0;
                minutes_ = static_cast<uint8>(std::floor(decMinutes));
                seconds_ = static_cast<dtype>((decMinutes - static_cast<double>(minutes_)) * 60.0);
            }

            //============================================================================
            ///						Constructor
            ///		
            ///	@param			hours
            /// @param          minutes
            /// @param          seconds
            ///
            /// @return         None
            ///
            RA(uint8 inHours, uint8 inMinutes, dtype inSeconds) :
                hours_(inHours),
                minutes_(inMinutes),
                seconds_(inSeconds),
                degrees_(0.0),
                radians_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::RA: constructor can only be called with floating point types.");

                degrees_ = static_cast<dtype>(static_cast<double>(hours_) * 15.0 + static_cast<double>(minutes_) / 4.0 + static_cast<double>(seconds_) / 240.0);
                radians_ = static_cast<dtype>(Methods<dtype>::deg2rad(degrees_));
            }

            //============================================================================
            ///						Returns a copy of the RA object as a different type
            ///		
            /// @param      None
            ///
            /// @return     RA
            ///
            template<typename dtypeOut>
            RA<dtypeOut> astype()
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::RA::astype: method can only be called with floating point types.");

                return RA<dtypeOut>(hours_, minutes_, static_cast<dtypeOut>(seconds_));
            }

            //============================================================================
            ///						Get the radians value
            ///		
            /// @param      None
            ///
            /// @return     radians
            ///
            dtype radians() const
            {
                return radians_;
            }

            //============================================================================
            ///						Get the degrees value
            ///		
            /// @param      None
            ///
            /// @return     degrees
            ///
            dtype degrees() const
            {
                return degrees_;
            }

            //============================================================================
            ///						Get the hour value
            ///		
            /// @param      None
            ///
            /// @return     hours
            ///
            uint8 hours() const
            {
                return hours_;
            }

            //============================================================================
            ///						Get the minute value
            ///		
            /// @param      None   
            ///
            /// @return     minutes
            ///
            uint8 minutes() const
            {
                return minutes_;
            }

            //============================================================================
            ///						Get the seconds value
            ///		
            /// @param      None
            /// 
            /// @return     seconds
            ///
            dtype seconds() const
            {
                return seconds_;
            }

            //============================================================================
            ///						Return the RA object as a string representation
            ///		
            /// @param      None
            ///
            /// @return     string
            ///
            std::string str() const
            {
                std::string out = "RA hms: " + Utils<uint8>::num2str(hours_) + " hours, " + Utils<uint8>::num2str(minutes_) + " minutes, ";
                out += Utils<dtype>::num2str(seconds_) + " seconds\nRA degrees: " + Utils<dtype>::num2str(degrees_) + "\n";
                return out;
            }

            //============================================================================
            ///						Prints the RA object to the console
            ///		
            /// @param      None
            ///
            /// @return     None
            ///
            void print() const
            {
                std::cout << *this;
            }

            //============================================================================
            ///						Equality operator
            ///		
            /// @param      None
            ///
            /// @return     bool
            ///
            bool operator==(const RA<dtype>& inRhs) const
            {
                return degrees_ == inRhs.degrees_;
            }

            //============================================================================
            ///						Not equality operator
            ///		
            /// @param      None
            ///
            /// @return     bool
            ///
            bool operator!=(const RA<dtype>& inRhs) const
            {
                return !(*this == inRhs);
            }

            //============================================================================
            ///						Ostream operator
            ///		
            /// @param      None
            ///
            /// @return     None
            ///
            friend std::ostream& operator<<(std::ostream& inStream, const RA<dtype>& inRa)
            {
                inStream << inRa.str();
                return inStream;
            }
        };

        //================================================================================
        ///						Struct Enum for positive or negative Dec angle
        struct Sign { enum Type { NEGATIVE = 0, POSITIVE }; };

        //================================================================================
        ///						Holds a Declination object
        template<typename dtype>
        class Dec
        {
        private:
            //====================================Attributes==============================
            Sign::Type      sign_;
            uint8           degreesWhole_;
            uint8           minutes_;
            dtype           seconds_;
            dtype           degrees_;
            dtype           radians_;

        public:
            //============================================================================
            ///						Default Constructor, not super usefull on its own
            ///		
            /// @param      None
            ///
            /// @return     None
            ///
            Dec() :
                sign_(Sign::POSITIVE),
                degreesWhole_(0),
                minutes_(0),
                seconds_(0.0),
                degrees_(0.0),
                radians_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
            }

            //============================================================================
            ///						Constructor
            ///		
            /// @param      degrees
            ///
            /// @return     None
            ///
            Dec(dtype inDegrees) :
                degreesWhole_(0),
                minutes_(0),
                seconds_(0.0),
                degrees_(inDegrees),
                radians_(static_cast<dtype>(Methods<dtype>::deg2rad(inDegrees)))
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");

                if (inDegrees < -90 || inDegrees > 90)
                {
                    throw std::invalid_argument("ERROR: NumC::Coordinates::Dec: input degrees must be of the range [-90, 90]");
                }

                sign_ = degrees_ < 0 ? Sign::NEGATIVE : Sign::POSITIVE;
                dtype absDegrees = std::abs(degrees_);
                degreesWhole_ = static_cast<uint8>(std::floor(absDegrees));

                double decMinutes = (absDegrees - static_cast<double>(degreesWhole_)) * 60.0;
                minutes_ = static_cast<uint8>(std::floor(decMinutes));
                seconds_ = static_cast<dtype>((decMinutes - static_cast<double>(minutes_)) * 60.0);
            }

            //============================================================================
            ///						Constructor
            ///		
            /// @param      Sign::Type
            ///	@param      hours
            /// @param      minutes
            /// @param      seconds
            ///
            /// @return     None
            ///
            Dec(Sign::Type inSign, uint8 inDegrees, uint8 inMinutes, dtype inSeconds) :
                sign_(inSign),
                degreesWhole_(inDegrees),
                minutes_(inMinutes),
                seconds_(inSeconds),
                degrees_(0.0),
                radians_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");

                degrees_ = static_cast<dtype>(static_cast<double>(degreesWhole_) + static_cast<double>(minutes_) / 60.0 + static_cast<double>(seconds_) / 3600.0);
                degrees_ *= sign_ == Sign::NEGATIVE ? -1 : 1;
                
                radians_ = static_cast<dtype>(Methods<dtype>::deg2rad(degrees_));
            }

            //============================================================================
            ///						Returns a copy of the Dec object as a different type
            ///		
            /// @param      None
            ///
            /// @return     Dec
            ///
            template<typename dtypeOut>
            Dec<dtypeOut> astype()
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec::astype: method can only be called with floating point types.");

                return Dec<dtypeOut>(sign_, degreesWhole_, minutes_, static_cast<dtypeOut>(seconds_));
            }

            //============================================================================
            ///						Get the sign of the degrees (positive or negative)
            ///		
            /// @param      None
            ///
            /// @return     Sign::Type
            ///
            Sign::Type sign() const
            {
                return sign_;
            }

            //============================================================================
            ///						Get the degrees value
            ///		
            /// @param      None
            ///
            /// @return     degrees
            ///
            dtype degrees() const
            {
                return degrees_;
            }

            //============================================================================
            ///						Get the radians value
            ///		
            /// @param      None
            ///
            /// @return     minutes
            ///
            dtype radians() const
            {
                return radians_;
            }

            //============================================================================
            ///						Get the whole degrees value
            ///		
            /// @param      None
            ///
            /// @return     whole degrees
            ///
            uint8 degreesWhole() const
            {
                return degreesWhole_;
            }

            //============================================================================
            ///						Get the minute value
            ///		
            /// @param      None
            ///
            /// @return     minutes
            ///
            uint8 minutes() const
            {
                return minutes_;
            }

            //============================================================================
            ///						Get the seconds value
            ///		
            /// @param      None
            ///
            /// @return     seconds
            ///
            dtype seconds() const
            {
                return seconds_;
            }

            //============================================================================
            ///						Return the dec object as a string representation
            ///		
            /// @param      None
            ///
            /// @return     string
            ///
            std::string str() const
            {
                std::string strSign = sign_ == Sign::NEGATIVE ? "-" : "+";
                std::string out = "Dec dms: " + strSign + Utils<dtype>::num2str(degreesWhole_) + " degrees, " + Utils<uint8>::num2str(minutes_) + " minutes, ";
                out += Utils<dtype>::num2str(seconds_) + " seconds\nDec degrees = " + Utils<dtype>::num2str(degrees_) + "\n";
                return out;
            }

            //============================================================================
            ///						Prints the Dec object to the console
            ///		
            /// @param      None
            ///
            /// @return     None
            ///
            void print() const
            {
                std::cout << *this;
            }

            //============================================================================
            ///						Equality operator
            ///		
            /// @param      None
            ///
            /// @return     bool
            ///
            bool operator==(const Dec<dtype>& inRhs) const
            {
                return degrees_ == inRhs.degrees_;
            }

            //============================================================================
            ///						Not equality operator
            ///		
            /// @param      None
            ///
            /// @return     bool
            ///
            bool operator!=(const Dec<dtype>& inRhs) const
            {
                return !(*this == inRhs);
            }

            //============================================================================
            ///						Ostream operator
            ///		
            /// @param      None
            ///
            /// @return     None
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
            dtype           x_;
            dtype           y_;
            dtype           z_;

            //============================================================================
            ///						Converts polar coordinates to cartesian coordinates
            ///		
            /// @param      None
            /// @return     None
            ///
            void cartesianToPolar()
            {
                dtype degreesRa = static_cast<dtype>(Methods<dtype>::rad2deg(std::atan2(y_, x_)));
                if (degreesRa < 0)
                {
                    degreesRa += 360;
                }
                ra_ = RA<dtype>(degreesRa);

                double r = std::sqrt(static_cast<double>(Utils<dtype>::sqr(x_)) + static_cast<double>(Utils<dtype>::sqr(y_)) + static_cast<double>(Utils<dtype>::sqr(z_)));
                dtype degreesDec = static_cast<dtype>(Methods<double>::rad2deg(std::asin(static_cast<double>(z_) / r)));
                dec_ = Dec<dtype>(degreesDec);
            }

            //============================================================================ 
            ///						Converts polar coordinates to cartesian coordinates
            ///		
            /// @param      None
            ///
            /// @return     None
            ///
            void polarToCartesian()
            {
                double raRadians = Methods<double>::deg2rad(static_cast<double>(ra_.degrees()));
                double decRadians = Methods<double>::deg2rad(static_cast<double>(dec_.degrees()));

                x_ = static_cast<dtype>(std::cos(raRadians) * std::cos(decRadians));
                y_ = static_cast<dtype>(std::sin(raRadians) * std::cos(decRadians));
                z_ = static_cast<dtype>(std::sin(decRadians));
            }

        public:
            //============================================================================
            ///						Default Constructor, not super usefull on its own
            ///		
            /// @param      None
            ///
            /// @return     None
            ///
            Coordinate() :
                ra_(),
                dec_(),
                x_(1.0),
                y_(0.0),
                z_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
            }

            //============================================================================
            ///						Constructor
            ///		
            /// @param      RA degrees
            /// @param      Dec degrees
            ///
            /// @return     None
            ///
            Coordinate(dtype inRaDegrees, dtype inDecDegrees) :
                ra_(inRaDegrees),
                dec_(inDecDegrees),
                x_(0.0),
                y_(0.0),
                z_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
                polarToCartesian();
            }

            //============================================================================
            ///						Constructor
            ///		
            /// @param				RA hours
            /// @param              RA minutes
            /// @param              RA seconds
            /// @param              Dec degrees whole
            /// @param              Dec minutes
            /// @param              Dec seconds
            ///
            /// @return             None
            ///
            Coordinate(uint8 inRaHours, uint8 inRaMinutes, dtype inRaSeconds, Sign::Type inSign, uint8 inDecDegreesWhole, uint8 inDecMinutes, dtype inDecSeconds) :
                ra_(inRaHours, inRaMinutes, inRaSeconds),
                dec_(inSign, inDecDegreesWhole, inDecMinutes, inDecSeconds),
                x_(0.0),
                y_(0.0),
                z_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
                polarToCartesian();
            }

            //============================================================================
            ///						Constructor
            ///		
            /// @param				RA 
            /// @param              Dec
            ///
            /// @return             None
            ///
            Coordinate(const RA<dtype>& inRA, const Dec<dtype>& inDec) :
                ra_(inRA),
                dec_(inDec),
                x_(0.0),
                y_(0.0),
                z_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
                polarToCartesian();
            }

            //============================================================================
            ///						Constructor
            ///		
            /// @param				x 
            /// @param              y
            /// @param              z
            ///
            /// @return             None
            ///
            Coordinate(dtype inX, dtype inY, dtype inZ) :
                ra_(),
                dec_(),
                x_(inX),
                y_(inY),
                z_(inZ)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
                cartesianToPolar();
            }

            //============================================================================
            ///						Constructor
            ///		
            /// @param				NdArray
            ///
            /// @return             None
            ///
            Coordinate(const NdArray<dtype> inCartesianVector) :
                ra_(),
                dec_(),
                x_(1.0),
                y_(0.0),
                z_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");

                if (inCartesianVector.size() != 3)
                {
                    throw std::invalid_argument("ERROR: NumC::Coordinates::Dec: constructor NdArray input must be of length 3.");
                }

                x_ = inCartesianVector[0];
                y_ = inCartesianVector[1];
                z_ = inCartesianVector[2];

                cartesianToPolar();
            }

            //============================================================================
            ///						Returns a new Coordinate object with the specified type
            ///		
            /// @param      None
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
            /// @param              None
            ///
            /// @return             Dec
            ///
            const Dec<dtype>& dec() const
            {
                return dec_;
            }

            //============================================================================
            ///						Returns the RA object
            ///		
            /// @param      None
            ///
            /// @return     RA
            ///
            const RA<dtype>& ra() const
            {
                return ra_;
            }

            //============================================================================
            ///						Returns the cartesian x value
            ///		
            /// @param      None
            ///
            /// @return     x
            ///
            dtype x() const
            {
                return x_;
            }

            //============================================================================
            ///						Returns the cartesian y value
            ///		
            /// @param      None
            ///
            /// @return     y
            ///
            dtype y() const
            {
                return y_;
            }

            //============================================================================
            ///						Returns the cartesian z value
            ///		
            /// @param      None
            ///
            /// @return     z
            ///
            dtype z() const
            {
                return z_;
            }

            //============================================================================
            ///						Returns the cartesian xyz triplet as an NdArray
            ///		
            /// @param      None
            ///
            /// @return     NdArray
            ///
            NdArray<dtype> xyz() const
            {
                NdArray<dtype> out = {x_, y_, z_};
                return std::move(out);
            }

            //============================================================================  
            ///						Returns the degree seperation between the two Coordinates
            ///		
            /// @param      Coordinate
            ///
            /// @return     degrees
            ///
            dtype degreeSeperation(const Coordinate<dtype>& inOtherCoordinate) const
            {
                return static_cast<dtype>(Methods<dtype>::rad2deg(radianSeperation(inOtherCoordinate)));
            }

            //============================================================================
            ///						Returns the degree seperation between the Coordinate
            ///                     and the input vector
            ///		
            /// @param      NdArray
            ///
            /// @return     degrees
            ///
            dtype degreeSeperation(const NdArray<dtype>& inVector) const
            {
                return static_cast<dtype>(Methods<dtype>::rad2deg(radianSeperation(inVector)));
            }

            //============================================================================
            ///						Returns the radian seperation between the two Coordinates
            ///		
            /// @param      Coordinate
            ///
            /// @return     radians
            ///
            dtype radianSeperation(const Coordinate<dtype>& inOtherCoordinate) const
            {
                return static_cast<dtype>(std::acos(Methods<dtype>::dot(xyz(), inOtherCoordinate.xyz()).item()));
            }

            //============================================================================
            ///						Returns the radian seperation between the Coordinate
            ///                     and the input vector
            ///		
            /// @param      NdArray
            ///
            /// @return     radians
            ///
            dtype radianSeperation(const NdArray<dtype>& inVector) const
            {
                if (inVector.size() != 3)
                {
                    throw std::invalid_argument("ERROR: NumC::Coordinates::Coordinate::radianSeperation: input vector must be of length 3.");
                }

                return static_cast<dtype>(std::acos(Methods<dtype>::dot(xyz(), inVector.flatten()).item()));
            }

            //============================================================================
            ///						Returns coordinate as a string representation
            ///		
            /// @param      None
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
            /// @param      None
            ///
            /// @return     None
            ///
            void print() const
            {
                std::cout << *this;
            }

            //============================================================================
            ///						Equality operator
            ///		
            /// @param      None
            ///
            /// @return     bool
            ///
            bool operator==(const Coordinate<dtype>& inRhs) const
            {
                return ra_ == inRhs.ra_ && dec_ == inRhs.dec_;
            }

            //============================================================================
            ///						Not equality operator
            ///		
            /// @param      None
            ///
            /// @return     bool
            ///
            bool operator!=(const Coordinate<dtype>& inRhs) const
            {
                return !(*this == inRhs);
            }

            //============================================================================
            ///						Ostream operator
            ///		
            /// @param      None
            ///
            /// @return     None
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
        /// @param				Coordinate
        /// @param              Coordinate
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
        /// @param				NdArray
        /// @param              NdArray
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
        /// @param				Coordinate
        /// @param              Coordinate
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
        /// @param				NdArray
        /// @param              NdArray
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
