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
/// Celestial Object
///
#pragma once

#include <cmath>
#include <iostream>
#include <string>

#include "NumCpp/Coordinates/Cartesian.hpp"
#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/deg2rad.hpp"
#include "NumCpp/Functions/dot.hpp"
#include "NumCpp/Functions/rad2deg.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"
#include "NumCpp/Utils/num2str.hpp"
#include "NumCpp/Utils/sqr.hpp"
#include "NumCpp/Vector/Vec3.hpp"

namespace nc::coordinates::reference_frames
{
    //================================================================================
    /// Holds a right ascension object
    class RA
    {
    public:
        //============================================================================
        /// Default Constructor
        ///
        RA() = default;

        //============================================================================
        /// Constructor
        ///
        /// @param inDegrees
        ///
        explicit RA(double inDegrees) :
            degrees_(inDegrees),
            radians_(deg2rad(inDegrees))
        {
            if (inDegrees < 0 || inDegrees >= 360)
            {
                THROW_INVALID_ARGUMENT_ERROR("input degrees must be of the range [0, 360)");
            }

            hours_                  = static_cast<uint8>(std::floor(degrees_ / 15.));
            const double decMinutes = (degrees_ - static_cast<double>(hours_) * 15.) * 4.;
            minutes_                = static_cast<uint8>(std::floor(decMinutes));
            seconds_                = static_cast<double>((decMinutes - static_cast<double>(minutes_)) * 60.);
        }

        //============================================================================
        /// Constructor
        ///
        /// @param inHours
        /// @param inMinutes
        /// @param inSeconds
        ///
        RA(uint8 inHours, uint8 inMinutes, double inSeconds) noexcept :
            hours_(inHours),
            minutes_(inMinutes),
            seconds_(inSeconds)
        {
            degrees_ = static_cast<double>(hours_) * 15. + static_cast<double>(minutes_) / 4. + seconds_ / 240.;
            radians_ = deg2rad(degrees_);
        }

        //============================================================================
        /// Get the radians value
        ///
        /// @return radians
        ///
        [[nodiscard]] double radians() const noexcept
        {
            return radians_;
        }

        //============================================================================
        /// Get the degrees value
        ///
        /// @return degrees
        ///
        [[nodiscard]] double degrees() const noexcept
        {
            return degrees_;
        }

        //============================================================================
        /// Get the hour value
        ///
        /// @return hours
        ///
        [[nodiscard]] uint8 hours() const noexcept
        {
            return hours_;
        }

        //============================================================================
        /// Get the minute value
        ///
        /// @return minutes
        ///
        [[nodiscard]] uint8 minutes() const noexcept
        {
            return minutes_;
        }

        //============================================================================
        /// Get the seconds value
        ///
        /// @return seconds
        ///
        [[nodiscard]] double seconds() const noexcept
        {
            return seconds_;
        }

        //============================================================================
        /// Return the RA object as a string representation
        ///
        /// @return std::string
        ///
        [[nodiscard]] std::string str() const
        {
            std::string out =
                "RA hms: " + utils::num2str(hours_) + " hours, " + utils::num2str(minutes_) + " minutes, ";
            out += utils::num2str(seconds_) + " seconds\nRA degrees: " + utils::num2str(degrees_) + '\n';
            return out;
        }

        //============================================================================
        /// Prints the RA object to the console
        ///
        void print() const
        {
            std::cout << *this;
        }

        //============================================================================
        /// Equality operator
        ///
        /// @param inRhs
        ///
        /// @return bool
        ///
        bool operator==(const RA& inRhs) const noexcept
        {
            return utils::essentiallyEqual(degrees_, inRhs.degrees_);
        }

        //============================================================================
        /// Not equality operator
        ///
        /// @param inRhs
        ///
        /// @return bool
        ///
        bool operator!=(const RA& inRhs) const noexcept
        {
            return !(*this == inRhs);
        }

        //============================================================================
        /// Ostream operator
        ///
        /// @param inStream
        /// @param inRa
        ///
        friend std::ostream& operator<<(std::ostream& inStream, const RA& inRa)
        {
            inStream << inRa.str();
            return inStream;
        }

    private:
        //====================================Attributes==============================
        uint8  hours_{ 0 };
        uint8  minutes_{ 0 };
        double seconds_{ 0. };
        double degrees_{ 0. };
        double radians_{ 0. };
    };

    //================================================================================
    /// Holds a Declination object
    class Dec
    {
    public:
        //================================================================================
        /// Struct Enum for positive or negative Dec angle
        enum class Sign
        {
            NEGATIVE = 0,
            POSITIVE
        };

        //============================================================================
        /// Default Constructor
        ///
        Dec() = default;

        //============================================================================
        /// Constructor
        ///
        /// @param inDegrees
        ///
        explicit Dec(double inDegrees) :
            degrees_(inDegrees),
            radians_(deg2rad(inDegrees))
        {
            if (inDegrees < -90 || inDegrees > 90)
            {
                THROW_INVALID_ARGUMENT_ERROR("input degrees must be of the range [-90, 90]");
            }

            sign_                   = degrees_ < 0 ? Sign::NEGATIVE : Sign::POSITIVE;
            const double absDegrees = std::abs(degrees_);
            degreesWhole_           = static_cast<uint8>(std::floor(absDegrees));

            const double decMinutes = (absDegrees - static_cast<double>(degreesWhole_)) * 60.;
            minutes_                = static_cast<uint8>(std::floor(decMinutes));
            seconds_                = (decMinutes - static_cast<double>(minutes_)) * 60.;
        }

        //============================================================================
        /// Constructor
        ///
        /// @param inSign
        /// @param inDegrees
        /// @param inMinutes
        /// @param inSeconds
        ///
        Dec(Sign inSign, uint8 inDegrees, uint8 inMinutes, double inSeconds) noexcept :
            sign_(inSign),
            degreesWhole_(inDegrees),
            minutes_(inMinutes),
            seconds_(inSeconds)
        {
            degrees_ = static_cast<double>(degreesWhole_) + static_cast<double>(minutes_) / 60. + seconds_ / 3600.;
            degrees_ *= sign_ == Sign::NEGATIVE ? -1 : 1;

            radians_ = deg2rad(degrees_);
        }

        //============================================================================
        /// Get the sign of the degrees (positive or negative)
        ///
        /// @return Sign
        ///
        [[nodiscard]] Sign sign() const noexcept
        {
            return sign_;
        }

        //============================================================================
        /// Get the degrees value
        ///
        /// @return degrees
        ///
        [[nodiscard]] double degrees() const noexcept
        {
            return degrees_;
        }

        //============================================================================
        /// Get the radians value
        ///
        /// @return minutes
        ///
        [[nodiscard]] double radians() const noexcept
        {
            return radians_;
        }

        //============================================================================
        /// Get the whole degrees value
        ///
        /// @return whole degrees
        ///
        [[nodiscard]] uint8 degreesWhole() const noexcept
        {
            return degreesWhole_;
        }

        //============================================================================
        /// Get the minute value
        ///
        /// @return minutes
        ///
        [[nodiscard]] uint8 minutes() const noexcept
        {
            return minutes_;
        }

        //============================================================================
        /// Get the seconds value
        ///
        /// @return seconds
        ///
        [[nodiscard]] double seconds() const noexcept
        {
            return seconds_;
        }

        //============================================================================
        /// Return the dec object as a string representation
        ///
        /// @return std::string
        ///
        [[nodiscard]] std::string str() const
        {
            std::string strSign = sign_ == Sign::NEGATIVE ? "-" : "+";
            std::string out     = "Dec dms: " + strSign + utils::num2str(degreesWhole_) + " degrees, " +
                              utils::num2str(minutes_) + " minutes, ";
            out += utils::num2str(seconds_) + " seconds\nDec degrees = " + utils::num2str(degrees_) + '\n';
            return out;
        }

        //============================================================================
        /// Prints the Dec object to the console
        ///
        void print() const
        {
            std::cout << *this;
        }

        //============================================================================
        /// Equality operator
        ///
        /// @param inRhs
        ///
        /// @return bool
        ///
        bool operator==(const Dec& inRhs) const noexcept
        {
            return utils::essentiallyEqual(degrees_, inRhs.degrees_);
        }

        //============================================================================
        /// Not equality operator
        ///
        /// @param inRhs
        ///
        /// @return bool
        ///
        bool operator!=(const Dec& inRhs) const noexcept
        {
            return !(*this == inRhs);
        }

        //============================================================================
        /// Ostream operator
        ///
        /// @param inStream
        /// @param inDec
        ///
        /// @return std::ostream
        ///
        friend std::ostream& operator<<(std::ostream& inStream, const Dec& inDec)
        {
            inStream << inDec.str();
            return inStream;
        }

    private:
        //====================================Attributes==============================
        Sign   sign_{ Sign::POSITIVE };
        uint8  degreesWhole_{ 0 };
        uint8  minutes_{ 0 };
        double seconds_{ 0. };
        double degrees_{ 0. };
        double radians_{ 0. };
    };

    //================================================================================
    /// Holds a full celestial Celestial object
    class Celestial
    {
    public:
        //============================================================================
        /// Default Constructor
        ///
        Celestial() = default;

        //============================================================================
        /// Constructor
        ///
        /// @param inRaDegrees
        /// @param inDecDegrees
        ///
        Celestial(double inRaDegrees, double inDecDegrees) :
            ra_(inRaDegrees),
            dec_(inDecDegrees)
        {
            polarToCartesian();
        }

        //============================================================================
        /// Constructor
        ///
        /// @param inRaHours
        /// @param inRaMinutes
        /// @param inRaSeconds
        /// @param inSign
        /// @param inDecDegreesWhole
        /// @param inDecMinutes
        /// @param inDecSeconds
        ///
        Celestial(uint8     inRaHours,
                  uint8     inRaMinutes,
                  double    inRaSeconds,
                  Dec::Sign inSign,
                  uint8     inDecDegreesWhole,
                  uint8     inDecMinutes,
                  double    inDecSeconds) :
            ra_(inRaHours, inRaMinutes, inRaSeconds),
            dec_(inSign, inDecDegreesWhole, inDecMinutes, inDecSeconds)
        {
            polarToCartesian();
        }

        //============================================================================
        /// Constructor
        ///
        /// @param inRA
        /// @param inDec
        ///
        Celestial(const RA& inRA, const Dec& inDec) noexcept :
            ra_(inRA),
            dec_(inDec)
        {
            polarToCartesian();
        }

        //============================================================================
        /// Constructor
        ///
        /// @param inX
        /// @param inY
        /// @param inZ
        ///
        Celestial(double inX, double inY, double inZ) :
            x_(inX),
            y_(inY),
            z_(inZ)
        {
            cartesianToPolar();
        }

        //============================================================================
        /// Constructor
        ///
        /// @param inCartesianVector
        ///
        Celestial(const Cartesian& inCartesianVector) :
            x_(inCartesianVector.x),
            y_(inCartesianVector.y),
            z_(inCartesianVector.z)
        {
            cartesianToPolar();
        }

        //============================================================================
        /// Constructor
        ///
        /// @param inCartesianVector
        ///
        Celestial(const Vec3& inCartesianVector) :
            x_(inCartesianVector.x),
            y_(inCartesianVector.y),
            z_(inCartesianVector.z)
        {
            cartesianToPolar();
        }

        //============================================================================
        /// Constructor
        ///
        /// @param inCartesianVector
        ///
        Celestial(const NdArray<double>& inCartesianVector)
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
        /// Returns the Dec object
        ///
        /// @return Dec
        ///
        [[nodiscard]] const Dec& dec() const noexcept
        {
            return dec_;
        }

        //============================================================================
        /// Returns the RA object
        ///
        /// @return RA
        ///
        [[nodiscard]] const RA& ra() const noexcept
        {
            return ra_;
        }

        //============================================================================
        /// Returns the cartesian x value
        ///
        /// @return x
        ///
        [[nodiscard]] double x() const noexcept
        {
            return x_;
        }

        //============================================================================
        /// Returns the cartesian y value
        ///
        /// @return y
        ///
        [[nodiscard]] double y() const noexcept
        {
            return y_;
        }

        //============================================================================
        /// Returns the cartesian z value
        ///
        /// @return z
        ///
        [[nodiscard]] double z() const noexcept
        {
            return z_;
        }

        //============================================================================
        /// Returns the cartesian xyz triplet as an NdArray
        ///
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<double> xyz() const
        {
            NdArray<double> out = { x_, y_, z_ };
            return out;
        }

        //============================================================================
        /// Returns the degree seperation between the two Celestials
        ///
        /// @param inOtherCelestial
        ///
        /// @return degrees
        ///
        [[nodiscard]] double degreeSeperation(const Celestial& inOtherCelestial) const
        {
            return rad2deg(radianSeperation(inOtherCelestial));
        }

        //============================================================================
        /// Returns the degree seperation between the Celestial
        /// and the input vector
        ///
        /// @param inVector
        ///
        /// @return degrees
        ///
        [[nodiscard]] double degreeSeperation(const NdArray<double>& inVector) const
        {
            return rad2deg(radianSeperation(inVector));
        }

        //============================================================================
        /// Returns the radian seperation between the two Celestials
        ///
        /// @param inOtherCelestial
        ///
        /// @return radians
        ///
        [[nodiscard]] double radianSeperation(const Celestial& inOtherCelestial) const
        {
            return std::acos(dot(xyz(), inOtherCelestial.xyz()).item());
        }

        //============================================================================
        /// Returns the radian seperation between the Celestial
        /// and the input vector
        ///
        /// @param inVector
        ///
        /// @return radians
        ///
        [[nodiscard]] double radianSeperation(const NdArray<double>& inVector) const
        {
            if (inVector.size() != 3)
            {
                THROW_INVALID_ARGUMENT_ERROR("input vector must be of length 3.");
            }

            return std::acos(dot(xyz(), inVector.flatten()).item());
        }

        //============================================================================
        /// Returns Celestial as a string representation
        ///
        /// @return string
        ///
        [[nodiscard]] std::string str() const
        {
            std::string returnStr;
            returnStr = ra_.str();
            returnStr += dec_.str();
            returnStr += "Cartesian = " + xyz().str();
            return returnStr;
        }

        //============================================================================
        /// Prints the Celestial object to the console
        ///
        void print() const
        {
            std::cout << *this;
        }

        //============================================================================
        /// Equality operator
        ///
        /// @param inRhs
        ///
        /// @return bool
        ///
        bool operator==(const Celestial& inRhs) const noexcept
        {
            return ra_ == inRhs.ra_ && dec_ == inRhs.dec_;
        }

        //============================================================================
        /// Not equality operator
        ///
        /// @param inRhs
        ///
        /// @return bool
        ///
        bool operator!=(const Celestial& inRhs) const noexcept
        {
            return !(*this == inRhs);
        }

        //============================================================================
        /// Ostream operator
        ///
        /// @param inStream
        /// @param inCoord
        ///
        /// @return std::ostream
        ///
        friend std::ostream& operator<<(std::ostream& inStream, const Celestial& inCoord)
        {
            inStream << inCoord.str();
            return inStream;
        }

    private:
        //====================================Attributes==============================
        RA     ra_{};
        Dec    dec_{};
        double x_{ 1. };
        double y_{ 0. };
        double z_{ 0. };

        //============================================================================
        /// Converts polar Celestials to cartesian Celestials
        ///
        void cartesianToPolar()
        {
            double degreesRa = rad2deg(std::atan2(y_, x_));
            if (degreesRa < 0)
            {
                degreesRa += 360;
            }
            ra_ = RA(degreesRa);

            const double r          = std::sqrt(utils::sqr(x_) + utils::sqr(y_) + utils::sqr(z_));
            const double degreesDec = rad2deg(std::asin(z_ / r));
            dec_                    = Dec(degreesDec);
        }

        //============================================================================
        /// Converts polar Celestials to cartesian Celestials
        ///
        void polarToCartesian() noexcept
        {
            const double raRadians  = deg2rad(ra_.degrees());
            const double decRadians = deg2rad(dec_.degrees());

            x_ = std::cos(raRadians) * std::cos(decRadians);
            y_ = std::sin(raRadians) * std::cos(decRadians);
            z_ = std::sin(decRadians);
        }
    };
} // namespace nc::coordinates::reference_frames
