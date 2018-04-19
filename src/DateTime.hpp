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
#define _CRT_SECURE_NO_WARNINGS

#include"Constants.hpp"
#include"Types.hpp"
#include"Utils.hpp"

#include<time.h>
#include<chrono>
#include<iostream>
#include<string>
#include<stdexcept>

namespace NumC
{
    //================================================================================
    // Class Description:
    //						class for working with dates and times
    //
    class DateTime
    {
    private:
        //==================================Attributes================================//
        std::tm     datetime_;

        // ============================================================================= 
        // Description:
        //              Default Constructor, almost completely useless
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              None
        //
        DateTime(const std::time_t inTime)
        {
            localtime_s(&datetime_, &inTime);
        }

        // ============================================================================= 
        // Description:
        //              zero pads the input string
        // 
        // Parameter(s): 
        //              string
        //              width
        // 
        // Return: 
        //              None
        //
        void zeroPad(std::string& inString, uint8 inWidth) const
        {
            if (inString.size() < inWidth)
            {
                std::string pad(inWidth - inString.size(), '0');
                inString = pad + inString;
            }
        }

    public:
        // ============================================================================= 
        // Description:
        //              Default Constructor, almost completely useless
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              None
        //
        DateTime()
        {};

        // ============================================================================= 
        // Description:
        //              Constructor
        // 
        // Parameter(s): 
        //              year
        //              month
        //              day
        //              hour
        //              minute
        //              second
        //              millisecond
        // 
        // Return: 
        //              None
        //
        DateTime(uint32 inYear, uint32 inMonth, uint32 inDay, uint32 inHour, uint32 inMinute, uint32 inSecond)
        {
            datetime_.tm_year = inYear - 1900;
            datetime_.tm_mon = inMonth - 1;
            datetime_.tm_mday = inDay;
            datetime_.tm_hour = inHour;
            datetime_.tm_min = inMinute;
            datetime_.tm_sec = inSecond;
        }

        // ============================================================================= 
        // Description:
        //              returns number of seconds since Jan 1, 1970 @ midnight
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              double
        //
        uint32 datetime() const
        {
            std::tm datetimeCopy = datetime_;
            return static_cast<uint32>(std::mktime(&datetimeCopy));
        }

        // ============================================================================= 
        // Description:
        //              returns the year
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 year
        //
        uint32 year() const
        {
            return datetime_.tm_year + 1900;
        }

        // ============================================================================= 
        // Description:
        //              returns the month
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 month
        //
        uint32 month() const
        {
            return datetime_.tm_mon + 1;
        }

        // ============================================================================= 
        // Description:
        //              returns the day of the year [0, 366]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 day of year
        //
        uint32 day() const
        {
            return datetime_.tm_mday;
        }

        // ============================================================================= 
        // Description:
        //              returns the hour of the day [0, 23]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 hour
        //
        uint32 hour() const
        {
            return datetime_.tm_hour;
        }

        // ============================================================================= 
        // Description:
        //              returns the minute of the hour [0, 59]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 minute
        //
        uint32 minute() const
        {
            return datetime_.tm_min;
        }

        // ============================================================================= 
        // Description:
        //              returns the second of the minute [0, 59]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 second
        //
        uint32 second() const
        {
            datetime_.tm_sec;
        }

        // ============================================================================= 
        // Description:
        //              returns the seconds past midnight
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 seconds
        //
        uint32 secondsPastMidnight() const
        {
            return static_cast<uint32>(hour() * Constants::SECONDS_PER_HOUR + minute() * Constants::SECONDS_PER_MINUTE + second());
        }

        // ============================================================================= 
        // Description:
        //              returns a new DateTime object with GMT time
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 seconds
        //
        DateTime gmtTime() const
        {
            std::tm newDatetime;
            std::tm datetimeCopy = datetime_;
            std::time_t newTimeT = std::mktime(&datetimeCopy);
            gmtime_s(&newDatetime, &newTimeT);

            return DateTime(newDatetime);
        }

        // ============================================================================= 
        // Description:
        //              returns the difference of the two DateTime objects in seconds. The 
        //              socond input DateTime must be larger than the first
        // 
        // Parameter(s): 
        //              DateTime earlier
        //              DateTime later
        // 
        // Return: 
        //              uint32 seconds
        //
        static uint32 diffSeconds(const DateTime& inDateTime1, const DateTime& inDateTime2)
        {
            std::tm* dt1Ptr = const_cast<std::tm*>(&inDateTime1.datetime_);
            std::tm* dt2Ptr = const_cast<std::tm*>(&inDateTime2.datetime_);
            return std::difftime(std::mktime(dt1Ptr), std::mktime(dt2Ptr));
        }

        // ============================================================================= 
        // Description:
        //              returns the difference of the two DateTime objects in seconds.  The input
        //              DateTime must be larger (more recent) than the object DateTime
        // 
        // Parameter(s): 
        //              DateTime
        // 
        // Return: 
        //              uint32 seconds
        //
        uint32 diffSeconds(const DateTime& inOtherDateTime) const
        {
            return diffSeconds(*this, inOtherDateTime);
        }

        // ============================================================================= 
        // Description:
        //              interpolates between two DateTime objects
        // 
        // Parameter(s): 
        //              DateTime
        // 
        // Return: 
        //              DateTime
        //
        static DateTime interpolate(const DateTime& inDateTime1, const DateTime& inDateTime2, double inPercent)
        {
            if (inPercent < 0 || inPercent > 1)
            {
                throw std::invalid_argument("ERROR: NumC::DateTime::interpolate: input percent value must be of the range [0, 1].");
            }

            std::tm* dt1Ptr = const_cast<std::tm*>(&inDateTime1.datetime_);
            std::tm* dt2Ptr = const_cast<std::tm*>(&inDateTime2.datetime_);

            std::time_t time1 = std::mktime(dt1Ptr);
            std::time_t time2 = std::mktime(dt2Ptr);


            double dateTime = 0;
            if (inDateTime1 < inDateTime2)
            {
                dateTime = inDateTime1.dateTime_ * (1.0 - inPercent) + inDateTime2.dateTime_ * inPercent;
            }
            else
            {
                dateTime = inDateTime2.dateTime_ * (1.0 - inPercent) + inDateTime1.dateTime_ * inPercent;
            }

            return DateTime(dateTime);
        }

        // ============================================================================= 
        // Description:
        //              interpolates between two DateTime objects
        // 
        // Parameter(s): 
        //              DateTime
        // 
        // Return: 
        //              DateTime
        //
        DateTime interpolate(const DateTime& inOtherDateTime, double inPercent) const
        {
            return interpolate(*this, inOtherDateTime, inPercent);
        }

        // ============================================================================= 
        // Description:
        //              returns a DateTime object for the system now
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              DateTime
        //
        static DateTime now()
        {
            std::chrono::system_clock::time_point today = std::chrono::system_clock::now();
            std::time_t tt = std::chrono::system_clock::to_time_t(today); // seconds since 1/1/1970 @ midnight
            std::tm ltm;
            localtime_s(&ltm, &tt);

            uint16 year = static_cast<uint16>(1900 + ltm.tm_year);
            int8 month = static_cast<uint8>(ltm.tm_mon + 1);
            int8 day = static_cast<uint8>(ltm.tm_mday);
            int8 hour = static_cast<uint8>(ltm.tm_hour);
            int8 minute = static_cast<uint8>(ltm.tm_min);
            int8 second = static_cast<uint8>(ltm.tm_sec);

            return DateTime(year, month, day, hour, minute, second);
        }

        // ============================================================================= 
        // Description:
        //              prints the DateTime to the console
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              None
        //
        void print() const
        {
            std::cout << *this;
        }

        // ============================================================================= 
        // Description:
        //              returns a string time representation. The format is:
        //              yyyy_mm_dd_HH_MM_SS_fff
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              string
        //
        std::string str() const
        {
            std::string monthStr = Utils::num2str(static_cast<uint16>(month()));
            zeroPad(monthStr, 2);
            std::string dayStr = Utils::num2str(static_cast<uint16>(day()));
            zeroPad(dayStr, 2);
            std::string hourStr = Utils::num2str(static_cast<uint16>(hour()));
            zeroPad(hourStr, 2);
            std::string minuteStr = Utils::num2str(static_cast<uint16>(minute()));
            zeroPad(minuteStr, 2);
            std::string secondStr = Utils::num2str(static_cast<uint16>(second()));
            zeroPad(secondStr, 2);

            std::string str = Utils::num2str(year());
            str += "_" + monthStr;
            str += "_" + dayStr;
            str += "_" + hourStr;
            str += "_" + minuteStr;
            str += "_" + secondStr;
            str += '\n';

            return str;
        }

        // ============================================================================= 
        // Description:
        //              addition operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              DateTime
        //
        DateTime operator+(const DateTime& inOtherDateTime) const 
        {
            return DateTime(*this) += inOtherDateTime;
        }

        // ============================================================================= 
        // Description:
        //              addition assignment operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              DateTime
        //
        DateTime& operator+=(const DateTime& inOtherDateTime)
        {
            dateTime_ += inOtherDateTime.dateTime_;
            setDate();
            return *this;
        }

        // ============================================================================= 
        // Description:
        //              subtraction operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              DateTime
        //
        DateTime operator-(const DateTime& inOtherDateTime) const
        {
            return DateTime(*this) -= inOtherDateTime;
        }

        // ============================================================================= 
        // Description:
        //              subtraction assignment operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              DateTime
        //
        DateTime& operator-=(const DateTime& inOtherDateTime)
        {
            dateTime_ -= inOtherDateTime.dateTime_;
            if (dateTime_ < 0)
            {
                throw std::runtime_error("ERROR: DateTime subtraction results in a negative date!");
            }
            setDate();
            return *this;
        }

        // ============================================================================= 
        // Description:
        //              less than operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              bool
        //
        bool operator<(const DateTime& inOtherDateTime) const
        {
            return dateTime_ < inOtherDateTime.dateTime_;
        }

        // ============================================================================= 
        // Description:
        //              less or equal than operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              bool
        //
        bool operator<=(const DateTime& inOtherDateTime) const
        {
            return dateTime_ < inOtherDateTime.dateTime_;
        }

        // ============================================================================= 
        // Description:
        //              greater than operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              bool
        //
        bool operator>(const DateTime& inOtherDateTime) const 
        {
            return dateTime_ > inOtherDateTime.dateTime_;
        }

        // ============================================================================= 
        // Description:
        //              greater or equal than operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              bool
        //
        bool operator>=(const DateTime& inOtherDateTime) const
        {
            return dateTime_ >= inOtherDateTime.dateTime_;
        }

        // ============================================================================= 
        // Description:
        //              equality operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              bool
        //
        bool operator==(const DateTime& inOtherDateTime) const
        {
            return dateTime_ == inOtherDateTime.dateTime_;
        }

        // ============================================================================= 
        // Description:
        //              not equality operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              bool
        //
        bool operator!=(const DateTime& inOtherDateTime) const
        {
            return !(*this == inOtherDateTime);
        }

        // ============================================================================= 
        // Description:
        //              io operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              bool
        //
        friend std::ostream& operator<<(std::ostream& inOstream, const DateTime& inDateTime)
        {
            inOstream << inDateTime.str();
            return inOstream;
        }
    };
}
