/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
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
/// Declination object
///
#pragma once

#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Core/Error.hpp"
#include "NumCpp/Methods/deg2rad.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"
#include "NumCpp/Utils/num2str.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace nc
{
    namespace coordinates
    {
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
                    error::throwInvalidArgument(errStr);
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
    }
}
