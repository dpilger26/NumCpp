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
        class RA
        {
        private:
            //====================================Attributes==============================
            uint8   hours_{ 0 };
            uint8   minutes_{ 0 };
            double  seconds_{ 0.0 };
            double  degrees_{ 0.0 };
            double  radians_{ 0.0 };

        public:
            //============================================================================
            ///						Default Constructor, not super usefull on its own
            ///
            RA() noexcept = default;

            //============================================================================
            ///						Constructor
            ///
            /// @param      inDegrees
            ///
            RA(double inDegrees) :
                degrees_(inDegrees),
                radians_(deg2rad(inDegrees))
            {
                if (inDegrees < 0 || inDegrees >= 360)
                {
                    std::string errStr = "ERROR: NC::coordinates::RA: input degrees must be of the range [0, 360)";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                hours_ = static_cast<uint8>(std::floor(degrees_ / 15.0));
                const double decMinutes = (degrees_ - static_cast<double>(hours_) * 15.0) * 4.0;
                minutes_ = static_cast<uint8>(std::floor(decMinutes));
                seconds_ = static_cast<double>((decMinutes - static_cast<double>(minutes_)) * 60.0);
            }

            //============================================================================
            ///						Constructor
            ///
            ///	@param			inHours
            /// @param          inMinutes
            /// @param          inSeconds
            ///
            RA(uint8 inHours, uint8 inMinutes, double inSeconds)  noexcept :
                hours_(inHours),
                minutes_(inMinutes),
                seconds_(inSeconds)
            {
                degrees_ = static_cast<double>(hours_) * 15.0 + static_cast<double>(minutes_) / 4.0 + seconds_ / 240.0;
                radians_ = deg2rad(degrees_);
            }

            //============================================================================
            ///						Get the radians value
            ///
            /// @return     radians
            ///
            double radians() const noexcept
            {
                return radians_;
            }

            //============================================================================
            ///						Get the degrees value
            ///
            /// @return     degrees
            ///
            double degrees() const noexcept
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
            double seconds() const noexcept
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
            bool operator==(const RA& inRhs) const noexcept
            {
                return utils::essentiallyEqual(degrees_, inRhs.degrees_);
            }

            //============================================================================
            ///						Not equality operator
            ///
            /// @param      inRhs
            ///
            /// @return     bool
            ///
            bool operator!=(const RA& inRhs) const noexcept
            {
                return !(*this == inRhs);
            }

            //============================================================================
            ///						Ostream operator
            ///
            /// @param      inStream
            /// @param      inRa
            ///
            friend std::ostream& operator<<(std::ostream& inStream, const RA& inRa)
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
        class Dec
        {
        private:
            //====================================Attributes==============================
            Sign            sign_{ Sign::POSITIVE };
            uint8           degreesWhole_{ 0 };
            uint8           minutes_{ 0 };
            double          seconds_{ 0.0 };
            double          degrees_{ 0.0 };
            double          radians_{ 0.0 };

        public:
            //============================================================================
            ///						Default Constructor, not super usefull on its own
            ///
            Dec() noexcept = default;

            //============================================================================
            ///						Constructor
            ///
            /// @param      inDegrees
            ///
            Dec(double inDegrees) :
                degrees_(inDegrees),
                radians_(deg2rad(inDegrees))
            {
                if (inDegrees < -90 || inDegrees > 90)
                {
                    std::string errStr = "ERROR: NC::coordinates::Dec: input degrees must be of the range [-90, 90]";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                sign_ = degrees_ < 0 ? Sign::NEGATIVE : Sign::POSITIVE;
                double absDegrees = std::abs(degrees_);
                degreesWhole_ = static_cast<uint8>(std::floor(absDegrees));

                const double decMinutes = (absDegrees - static_cast<double>(degreesWhole_)) * 60.0;
                minutes_ = static_cast<uint8>(std::floor(decMinutes));
                seconds_ = (decMinutes - static_cast<double>(minutes_)) * 60.0;
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param      inSign
            ///	@param      inDegrees
            /// @param      inMinutes
            /// @param      inSeconds
            ///
            Dec(Sign inSign, uint8 inDegrees, uint8 inMinutes, double inSeconds)  noexcept :
                sign_(inSign),
                degreesWhole_(inDegrees),
                minutes_(inMinutes),
                seconds_(inSeconds)
            {
                degrees_ = static_cast<double>(degreesWhole_) + static_cast<double>(minutes_) / 60.0 + seconds_ / 3600.0;
                degrees_ *= sign_ == Sign::NEGATIVE ? -1 : 1;

                radians_ = deg2rad(degrees_);
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
            double degrees() const noexcept
            {
                return degrees_;
            }

            //============================================================================
            ///						Get the radians value
            ///
            /// @return     minutes
            ///
            double radians() const noexcept
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
            double seconds() const noexcept
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
            bool operator==(const Dec& inRhs) const noexcept
            {
                return utils::essentiallyEqual(degrees_, inRhs.degrees_);
            }

            //============================================================================
            ///						Not equality operator
            ///
            /// @param      inRhs
            ///
            /// @return     bool
            ///
            bool operator!=(const Dec& inRhs) const noexcept
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
            friend std::ostream& operator<<(std::ostream& inStream, const Dec& inDec)
            {
                inStream << inDec.str();
                return inStream;
            }
        };

        //================================================================================
        ///						Holds a full coordinate object
        class Coordinate
        {
        private:
            //====================================Attributes==============================
            RA      ra_;
            Dec     dec_;
            double  x_{ 1.0 };
            double  y_{ 0.0 };
            double  z_{ 0.0 };

            //============================================================================
            ///						Converts polar coordinates to cartesian coordinates
            ///
            void cartesianToPolar()
            {
                double degreesRa = rad2deg(std::atan2(y_, x_));
                if (degreesRa < 0)
                {
                    degreesRa += 360;
                }
                ra_ = RA(degreesRa);

                const double r = std::sqrt(utils::sqr(x_) + utils::sqr(y_) + utils::sqr(z_));
                double degreesDec = rad2deg(std::asin(z_ / r));
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

        public:
            //============================================================================
            ///						Default Constructor, not super usefull on its own
            ///
            Coordinate() noexcept = default;

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
                uint8 inDecDegreesWhole, uint8 inDecMinutes, double inDecSeconds)  noexcept :
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
            Coordinate(double inX, double inY, double inZ) :
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
            Coordinate(const NdArray<double> inCartesianVector)
            {
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
                    std::string errStr = "ERROR: NC::coordinates::Coordinate::radianSeperation: input vector must be of length 3.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
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
        };

        //============================================================================
        ///						Returns the degree seperation between the two Coordinates
        ///
        /// @param				inCoordinate1
        /// @param              inCoordinate2
        ///
        /// @return             degrees
        ///
        inline double degreeSeperation(const Coordinate& inCoordinate1, const Coordinate& inCoordinate2) noexcept
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
        inline double degreeSeperation(const NdArray<double>& inVector1, const NdArray<double>& inVector2) noexcept
        {
            Coordinate inCoord1(inVector1);
            return inCoord1.degreeSeperation(inVector2);
        }

        //============================================================================
        ///						Returns the radian seperation between the two Coordinates
        ///
        /// @param				inCoordinate1
        /// @param              inCoordinate2
        ///
        /// @return             radians
        ///
        inline double radianSeperation(const Coordinate& inCoordinate1, const Coordinate& inCoordinate2) noexcept
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
        inline double radianSeperation(const NdArray<double>& inVector1, const NdArray<double>& inVector2) noexcept
        {
            Coordinate inCoord1(inVector1);
            return inCoord1.radianSeperation(inVector2);
        }
    }
}
