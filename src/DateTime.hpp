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

#include"Constants.hpp"
#include"Types.hpp"
#include"Utils.hpp"

#include<ctime>
#include<iostream>
#include<string>

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
        uint16      year_;
        uint8       month_;
        uint8       day_;
        uint8       hour_;
        uint8       minute_;
        uint8       second_;
        uint16      millisecond_;

        double      dateTime_;

        // ============================================================================= 
        // Description:
        //              Constructor
        // 
        // Parameter(s): 
        //              double
        // 
        // Return: 
        //              None
        //
        DateTime(double inDateTime) :
            dateTime_(inDateTime)
        {
            setDate();
        }

        // ============================================================================= 
        // Description:
        //              Converts a date and time to modified julian date.  Valid for 
        //              Gregorian dates from 17-Nov-1858. Adapted from sci.astro FAQ.
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
        //              double
        //
        double getModifiedJulianDay(uint16 inYear, uint8 inMonth, uint8 inDay, uint8 inHour, uint8 inMinute, uint8 inSecond, uint16 inMillisecond = 0) 
        {
            return
                367 * inYear
                - 7 * (inYear + (inMonth + 9) / 12) / 4
                - 3 * ((inYear + (inMonth - 9) / 7) / 100 + 1) / 4
                + 275 * inMonth / 9
                + inDay
                + 1721028
                - 2400000
                + static_cast<double>(inHour) / Constants::HOURS_PER_DAY
                + static_cast<double>(inMinute) / Constants::MINUTES_PER_DAY
                + static_cast<double>(inSecond) / Constants::SECONDS_PER_DAY
                + static_cast<double>(inMillisecond) / Constants::MILLISECONDS_PER_DAY;
        }

        // ============================================================================= 
        // Description:
        //              Converts from a modified julian date to a date.Assumes Gregorian calendar.  
        //              Adapted from Fliegel / van Flandern ACM 11 / #10 p 657 Oct 1968.
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              None
        //
        void setDate()
        {
            uint32 J, C, Y, M;

            J = static_cast<uint32>(std::floor(dateTime_)) + 2400001 + 68569;
            C = 4 * J / 146097;
            J = J - (146097 * C + 3) / 4;
            Y = 4000 * (J + 1) / 1461001;
            J = J - 1461 * Y / 4 + 31;
            M = 80 * J / 2447;
            day_ = static_cast<uint8>(J - 2447 * M / 80);
            J = M / 11;
            month_ = static_cast<uint8>(M + 2 - (12 * J));
            year_ = static_cast<uint16>(100 * (C - 49) + Y + J);

            double fractionalDay = dateTime_ - std::floor(dateTime_);
            hour_ = static_cast<uint8>(std::floor(fractionalDay * Constants::HOURS_PER_DAY));

            double fractinalHour = fractionalDay - static_cast<double>(hour_) / Constants::HOURS_PER_DAY;
            minute_ = static_cast<uint8>(std::floor(fractinalHour * Constants::MINUTES_PER_DAY));

            double fractionalMinute = fractinalHour - static_cast<double>(minute_) / Constants::MINUTES_PER_DAY;
            second_ = static_cast<uint8>(std::floor(fractionalMinute * Constants::SECONDS_PER_DAY));

            double fractionalSecond = fractionalMinute - static_cast<double>(second_) / Constants::SECONDS_PER_DAY;
            millisecond_ = static_cast<uint16>(std::floor(fractionalSecond * Constants::MILLISECONDS_PER_DAY));
        }

        // ============================================================================= 
        // Description:
        //              Pads the input string with zeros
        // 
        // Parameter(s): 
        //              None
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
        DateTime() :
            year_(0),
            month_(0),
            day_(0),
            hour_(0),
            minute_(0),
            second_(0),
            millisecond_(0),
            dateTime_(0)
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
        DateTime(uint16 inYear, uint8 inMonth, uint8 inDay, uint8 inHour, uint8 inMinute, uint8 inSecond, uint16 inMillisecond = 0) 
        {
            dateTime_ = getModifiedJulianDay(inYear, inMonth, inDay, inHour, inMinute, inSecond, inMillisecond);
            setDate();
        }

        // ============================================================================= 
        // Description:
        //              returns the year
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint16 year
        //
        uint16 year() const
        {
            return year_;
        }

        // ============================================================================= 
        // Description:
        //              returns the month
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint8 month
        //
        uint8 month() const
        {
            return month_;
        }

        // ============================================================================= 
        // Description:
        //              returns the day of the year [0, 366]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint16 day of year
        //
        uint8 day() const
        {
            return day_;
        }

        // ============================================================================= 
        // Description:
        //              returns the hour of the day [0, 23]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint16 hour
        //
        uint8 hour() const
        {
            return hour_;
        }

        // ============================================================================= 
        // Description:
        //              returns the minute of the hour [0, 59]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint16 minute
        //
        uint8 minute() const
        {
            return minute_;
        }

        // ============================================================================= 
        // Description:
        //              returns the second of the minute [0, 59]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint16 second
        //
        uint8 second() const 
        {
            return second_;
        }

        // ============================================================================= 
        // Description:
        //              returns the millisecond of the second [0, 999]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint16 millisecond
        //
        uint16 millisecond() const
        {
            return millisecond_;
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
            return static_cast<uint32>(hour_ * Constants::SECONDS_PER_HOUR + minute_ * Constants::SECONDS_PER_MINUTE + second_);;
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
            double dateDiff = std::abs(inDateTime2.dateTime_ - inDateTime1.dateTime_);
            double fractionalDay = dateDiff - std::floor(dateDiff);
            return static_cast<uint32>(fractionalDay * Constants::SECONDS_PER_DAY);
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
            std::time_t now = std::time(0);
            std::tm *ltm = std::localtime(&now);

            uint16 year = static_cast<uint16>(1970 + ltm->tm_year);
            uint8 month = static_cast<uint8>(1 + ltm->tm_mon);
            uint8 day = static_cast<uint8>(ltm->tm_mday);
            uint8 hour = static_cast<uint8>(1 + ltm->tm_hour);
            uint8 minute = static_cast<uint8>(1 + ltm->tm_min);
            uint8 second = static_cast<uint8>(1 + ltm->tm_sec);

            return DateTime(year, month, day, hour, minute, second);
            
            //return DateTime(static_cast<double>(std::time(0)));
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
            std::string monthStr = Utils::num2str(static_cast<uint16>(month_));
            zeroPad(monthStr, 2);
            std::string dayStr = Utils::num2str(static_cast<uint16>(day_));
            zeroPad(dayStr, 2);
            std::string hourStr = Utils::num2str(static_cast<uint16>(hour_));
            zeroPad(hourStr, 2);
            std::string minuteStr = Utils::num2str(static_cast<uint16>(minute_));
            zeroPad(minuteStr, 2);
            std::string secondStr = Utils::num2str(static_cast<uint16>(second_));
            zeroPad(secondStr, 2);
            std::string millisecondStr = Utils::num2str(static_cast<uint16>(millisecond_));
            zeroPad(millisecondStr, 3);

            std::string str = Utils::num2str(year_);
            str += "_" + monthStr;
            str += "_" + dayStr;
            str += "_" + hourStr;
            str += "_" + minuteStr;
            str += "_" + secondStr;
            str += "_" + millisecondStr;
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
