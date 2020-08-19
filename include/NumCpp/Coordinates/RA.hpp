/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.2.0
///
/// @section License
/// Copyright 2020 David Pilger
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
/// Right Ascension object
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/deg2rad.hpp"
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
        ///						Holds a right ascension object
        class RA
        {
        public:
            //============================================================================
            ///						Default Constructor
            ///
            RA() = default;

            //============================================================================
            ///						Constructor
            ///
            /// @param      inDegrees
            ///
            explicit RA(double inDegrees) :
                degrees_(inDegrees),
                radians_(deg2rad(inDegrees))
            {
                if (inDegrees < 0 || inDegrees >= 360)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input degrees must be of the range [0, 360)");
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
            RA(uint8 inHours, uint8 inMinutes, double inSeconds) noexcept :
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

        private:
            //====================================Attributes==============================
            uint8   hours_{ 0 };
            uint8   minutes_{ 0 };
            double  seconds_{ 0.0 };
            double  degrees_{ 0.0 };
            double  radians_{ 0.0 };
        };
    }  // namespace coordinates
}  // namespace nc
