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
    public:
        struct TimeZone { enum Zone { GMT = 0, LOCAL }; };

    private:
        //==================================Attributes================================//
        std::time_t     datenum_;
        std::tm         datetime_;
        TimeZone::Zone  timeZone_;

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
        DateTime(const std::time_t inTime, TimeZone::Zone inTimeZone):
            datenum_(inTime),
            timeZone_(inTimeZone)
        {
            if (inTimeZone == TimeZone::LOCAL)
            {
                localtime_s(&datetime_, &datenum_);
            }
            else
            {
                gmtime_s(&datetime_, &datenum_);
                if (!datetime_.tm_isdst)
                {
                    datenum_ += 3600;
                    gmtime_s(&datetime_, &datenum_);
                }
            }
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
        DateTime():
            datenum_(0),
            timeZone_(TimeZone::LOCAL)
        {
            localtime_s(&datetime_, &datenum_);
        }

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
        //              (Optional) TimeZone::Zone, default GMT
        // 
        // Return: 
        //              None
        //
        DateTime(uint32 inYear, uint32 inMonth, uint32 inDay, uint32 inHour, uint32 inMinute, uint32 inSecond, TimeZone::Zone inTimeZone = TimeZone::LOCAL):
            timeZone_(inTimeZone)
        {
            datetime_.tm_year = inYear - 1900;
            datetime_.tm_mon = inMonth - 1;
            datetime_.tm_mday = inDay;
            datetime_.tm_hour = inHour + 1;
            datetime_.tm_min = inMinute;
            datetime_.tm_sec = inSecond;

            datenum_ = std::mktime(&datetime_);
            if (datenum_ == -1)
            {
                throw std::invalid_argument("ERROR: NumC::DateTime(): invalid date.");
            }
            
            localtime_s(&datetime_, &datenum_);
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
        std::time_t datetime() const
        {
            return datenum_;
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
        //              returns the day of the month [1, 31]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 day of month
        //
        uint32 day() const
        {
            return datetime_.tm_mday;
        }

        // ============================================================================= 
        // Description:
        //              returns the days since Sunday [0, 6]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 day of week
        //
        uint32 dayOfWeek() const
        {
            return datetime_.tm_wday;
        }

        // ============================================================================= 
        // Description:
        //              returns the day of the year [0, 365]
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 day of year
        //
        uint32 dayOfYear() const
        {
            return datetime_.tm_yday;
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
            return datetime_.tm_sec;
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
        //              returns the timezone
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              TimeZone::Zone
        //
        TimeZone::Zone timeZone() const
        {
            return timeZone_;
        }

        // ============================================================================= 
        // Description:
        //              returns a new DateTime object as the input timezone
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              uint32 seconds
        //
        DateTime toTimeZone(TimeZone::Zone inTimeZone) const
        {
            DateTime newDateTime(datenum_, inTimeZone);
            if (inTimeZone == TimeZone::LOCAL && newDateTime.datetime_.tm_isdst)
            {
                return DateTime(newDateTime.datenum_ - 3600, inTimeZone);
            }

            return newDateTime;
        }

        // ============================================================================= 
        // Description:
        //              returns the difference of the two DateTime objects in seconds.
        // 
        // Parameter(s): 
        //              int64
        // 
        // Return: 
        //              uint32 seconds
        //
        static int64 diffSeconds(const DateTime& inDateTime1, const DateTime& inDateTime2)
        {
            if (inDateTime1.timeZone_ != inDateTime2.timeZone_)
            {
                throw std::invalid_argument("ERROR: NumC::diffSeconds::interpolate: input DateTime objects need to be the same timezone.");
            }

            std::tm* dt1Ptr = const_cast<std::tm*>(&inDateTime1.datetime_);
            std::tm* dt2Ptr = const_cast<std::tm*>(&inDateTime2.datetime_);

            if (inDateTime1.datenum_ < inDateTime2.datenum_)
            {
                return static_cast<int64>(std::difftime(std::mktime(dt1Ptr), std::mktime(dt2Ptr)));
            }
            else
            {
                return -static_cast<int64>(std::difftime(std::mktime(dt2Ptr), std::mktime(dt1Ptr)));
            }
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
        int64 diffSeconds(const DateTime& inOtherDateTime) const
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

            if (inDateTime1.timeZone_ != inDateTime2.timeZone_)
            {
                throw std::invalid_argument("ERROR: NumC::DateTime::interpolate: input DateTime objects need to be the same timezone.");
            }

            std::tm* dt1Ptr = const_cast<std::tm*>(&inDateTime1.datetime_);
            std::tm* dt2Ptr = const_cast<std::tm*>(&inDateTime2.datetime_);

            std::time_t time1 = std::mktime(dt1Ptr);
            std::time_t time2 = std::mktime(dt2Ptr);

            std::time_t dateTime;
            if (time1 < time2)
            {
                dateTime = static_cast<std::time_t>(time1 * (1.0 - inPercent) + time2 * inPercent);
            }
            else
            {
                dateTime = static_cast<std::time_t>(time2 * (1.0 - inPercent) + time1 * inPercent);
            }

            return DateTime(dateTime, inDateTime1.timeZone_);
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
        static DateTime now(TimeZone::Zone inTimeZone)
        {
            std::chrono::system_clock::time_point today = std::chrono::system_clock::now();
            std::time_t tt = std::chrono::system_clock::to_time_t(today); // seconds since 1/1/1970 @ midnight
            return DateTime(tt, inTimeZone);
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
            std::string monthStr = Utils::num2str(month());
            zeroPad(monthStr, 2);
            std::string dayStr = Utils::num2str(day());
            zeroPad(dayStr, 2);
            std::string hourStr = Utils::num2str(hour());
            zeroPad(hourStr, 2);
            std::string minuteStr = Utils::num2str(minute());
            zeroPad(minuteStr, 2);
            std::string secondStr = Utils::num2str(second());
            zeroPad(secondStr, 2);

            std::string str = Utils::num2str(year());
            str += "_" + monthStr;
            str += "_" + dayStr;
            str += "_" + hourStr;
            str += "_" + minuteStr;
            str += "_" + secondStr;
            
            if (timeZone_ == TimeZone::GMT)
            {
                str += "_GMT";
            }
            else
            {
                str += "_Local";
            }
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
            if (timeZone_ != inOtherDateTime.timeZone_)
            {
                throw std::invalid_argument("ERROR: NumC::DateTime::addition: input DateTime objects need to be the same timezone.");
            }

            datenum_ += inOtherDateTime.datenum_;

            if (timeZone_ == TimeZone::LOCAL)
            {
                localtime_s(&datetime_, &datenum_);
            }
            else
            {
                gmtime_s(&datetime_, &datenum_);
            }

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
            if (timeZone_ != inOtherDateTime.timeZone_)
            {
                throw std::invalid_argument("ERROR: NumC::DateTime::addition: input DateTime objects need to be the same timezone.");
            }

            datenum_ -= inOtherDateTime.datenum_;
            if (datenum_ < 0)
            {
                throw std::runtime_error("ERROR: NumC::DateTime subtraction results in a negative date!");
            }

            if (timeZone_ == TimeZone::LOCAL)
            {
                localtime_s(&datetime_, &datenum_);
            }
            else
            {
                gmtime_s(&datetime_, &datenum_);
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
            if (timeZone_ != inOtherDateTime.timeZone_)
            {
                throw std::invalid_argument("ERROR: NumC::DateTime::addition: input DateTime objects need to be the same timezone.");
            }

            return datenum_ < inOtherDateTime.datenum_;
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
            if (timeZone_ != inOtherDateTime.timeZone_)
            {
                throw std::invalid_argument("ERROR: NumC::DateTime::addition: input DateTime objects need to be the same timezone.");
            }

            return datenum_ < inOtherDateTime.datenum_;
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
            if (timeZone_ != inOtherDateTime.timeZone_)
            {
                throw std::invalid_argument("ERROR: NumC::DateTime::addition: input DateTime objects need to be the same timezone.");
            }

            return datenum_ > inOtherDateTime.datenum_;
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
            if (timeZone_ != inOtherDateTime.timeZone_)
            {
                throw std::invalid_argument("ERROR: NumC::DateTime::addition: input DateTime objects need to be the same timezone.");
            }

            return datenum_ >= inOtherDateTime.datenum_;
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
            if (timeZone_ != inOtherDateTime.timeZone_)
            {
                throw std::invalid_argument("ERROR: NumC::DateTime::addition: input DateTime objects need to be the same timezone.");
            }

            return datenum_ == inOtherDateTime.datenum_;
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
            if (timeZone_ != inOtherDateTime.timeZone_)
            {
                throw std::invalid_argument("ERROR: NumC::DateTime::addition: input DateTime objects need to be the same timezone.");
            }

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
