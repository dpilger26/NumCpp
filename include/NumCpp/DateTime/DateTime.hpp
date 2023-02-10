/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
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
/// DateTime module
///
#pragma once

#ifndef NUMCPP_NO_USE_BOOST

#include <chrono>
#include <ctime>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "boost/date_time/posix_time/posix_time.hpp"

#include "NumCpp/DateTime/Clock.hpp"

namespace nc
{
    //================================================================================
    // Class Description:
    /// Date Time class for working with iso formatted date times
    class DateTime
    {
    public:
        static constexpr int MAX_MONTH  = 12;
        static constexpr int MAX_DAY    = 31;
        static constexpr int MAX_HOUR   = 23;
        static constexpr int MAX_MINUTE = 59;
        static constexpr int MAX_SECOND = 59;

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        DateTime() = default;

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param tp: a timepoint object
        ///
        explicit DateTime(const TimePoint& tp)
        {
            auto tpSubSeconds     = std::chrono::duration_cast<Duration>(tp.time_since_epoch());
            auto fractionalSecond = static_cast<double>(tpSubSeconds.count() % Duration::period::den) /
                                    static_cast<double>(Duration::period::den);
            auto    time = Clock::to_time_t(std::chrono::time_point_cast<Clock::duration>(tp));
            std::tm tm;
#ifdef _MSC_VER
            gmtime_s(&tm, &time);
#else
            gmtime_r(&time, &tm);
#endif

            setYear(tm.tm_year + TM_EPOCH_YEAR);
            setMonth(tm.tm_mon + 1);
            setDay(tm.tm_mday);
            setHour(tm.tm_hour);
            setMinute(tm.tm_min);
            setSecond(tm.tm_sec);
            setFractionalSecond(fractionalSecond);
        }

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        /// @param timestamp: an iso formatted datetime string (0001-01-01T00:00:00.00000Z)
        ///
        explicit DateTime(const std::string& timestamp) :
            DateTime(strToTimepoint(timestamp))
        {
        }

        //============================================================================
        // Method Description:
        /// Constructor
        ///
        ///@param year: year value
        ///@param month: month value
        ///@param day: day value
        ///@param hour: hour value
        ///@param minute: minute value
        ///@param second: second value
        ///@param fractionalSecond: fractionalSecond value
        ///
        DateTime(int year, int month, int day, int hour, int minute, int second, double fractionalSecond = 0.0) noexcept
            :
            year_(year),
            month_(month),
            day_(day),
            hour_(hour),
            minute_(minute),
            second_(second),
            fractionalSecond_(fractionalSecond)
        {
        }

        //============================================================================
        // Method Description:
        ///@brief year getter
        ///
        ///@return int
        ///
        [[nodiscard]] int year() const noexcept
        {
            return year_;
        }

        //============================================================================
        // Method Description:
        ///@brief year setter
        ///
        ///@param year: year value
        ///
        void setYear(int year)
        {
            if (year < 0)
            {
                throw std::invalid_argument("input year must be greater than zero");
            }
            year_ = year;
        }

        //============================================================================
        // Method Description:
        ///@brief month getter
        ///
        ///@return int
        ///
        [[nodiscard]] int month() const noexcept
        {
            return month_;
        }

        //============================================================================
        // Method Description:
        ///@brief month setter
        ///
        ///@param month: month value
        ///
        void setMonth(int month)
        {
            if (month < 1)
            {
                throw std::invalid_argument("input month must be greater than one");
            }
            if (month > MAX_MONTH)
            {
                throw std::invalid_argument("input month must be less than DateTime::MAX_MONTH");
            }
            month_ = month;
        }

        //============================================================================
        // Method Description:
        ///@brief day getter
        ///
        ///@return int
        ///
        [[nodiscard]] int day() const noexcept
        {
            return day_;
        }

        //============================================================================
        // Method Description:
        ///@brief day setter
        ///
        ///@param day: day value
        ///
        void setDay(int day)
        {
            if (day < 1)
            {
                throw std::invalid_argument("input day must be greater than one");
            }
            if (day > MAX_DAY)
            {
                throw std::invalid_argument("input day must be less than DateTime::MAX_DAY");
            }
            day_ = day;
        }

        //============================================================================
        // Method Description:
        ///@brief hour getter
        ///
        ///@return int
        ///
        [[nodiscard]] int hour() const noexcept
        {
            return hour_;
        }

        //============================================================================
        // Method Description:
        ///@brief hour setter
        ///
        ///@param hour: hour value
        ///
        void setHour(int hour)
        {
            if (hour < 0)
            {
                throw std::invalid_argument("input hour must be greater than zero");
            }
            if (hour > MAX_HOUR)
            {
                throw std::invalid_argument("input hour must be less than DateTime::MAX_HOUR");
            }
            hour_ = hour;
        }

        //============================================================================
        // Method Description:
        ///@brief minute getter
        ///
        ///@return int
        ///
        [[nodiscard]] int minute() const noexcept
        {
            return minute_;
        }

        //============================================================================
        // Method Description:
        ///@brief minute setter
        ///
        ///@param minute: minute value
        ///
        void setMinute(int minute)
        {
            if (minute < 0)
            {
                throw std::invalid_argument("input minute must be greater than zero");
            }
            if (minute > MAX_MINUTE)
            {
                throw std::invalid_argument("input minute must be less than DateTime::MAX_MINUTE");
            }
            minute_ = minute;
        }

        //============================================================================
        // Method Description:
        ///@brief second getter
        ///
        ///@return int
        ///
        [[nodiscard]] int second() const noexcept
        {
            return second_;
        }

        //============================================================================
        // Method Description:
        ///@brief second setter
        ///
        ///@param second: second value
        ///
        void setSecond(int second)
        {
            if (second < 0)
            {
                throw std::invalid_argument("input second must be greater than zero");
            }
            if (second > MAX_SECOND)
            {
                throw std::invalid_argument("input second must be less than DateTime::MAX_SECOND");
            }
            second_ = second;
        }

        //============================================================================
        // Method Description:
        ///@brief fractionalSecond getter
        ///
        ///@return double
        ///
        [[nodiscard]] double fractionalSecond() const noexcept
        {
            return fractionalSecond_;
        }

        //============================================================================
        // Method Description:
        ///@brief fractionalSecond setter
        ///
        ///@param fractionalSecond: fractionalSecond value
        ///
        void setFractionalSecond(double fractionalSecond)
        {
            if (fractionalSecond < 0. || fractionalSecond >= 1.)
            {
                throw std::invalid_argument("input fractionalSecond must be in the range [0, 1)");
            }
            fractionalSecond_ = fractionalSecond;
        }

        //============================================================================
        // Method Description:
        ///@brief Converts the struct to a TimePoint
        ///
        ///@returns TimePoint
        ///
        [[nodiscard]] TimePoint toTimePoint() const
        {
            std::tm t{};
            t.tm_year      = year_ - TM_EPOCH_YEAR;
            t.tm_mon       = month_ - 1; // tm is 0 based months
            t.tm_mday      = day_;
            t.tm_hour      = hour_;
            t.tm_min       = minute_;
            t.tm_sec       = second_;
            auto timePoint = Clock::from_time_t(
#ifdef _MSC_VER
                _mkgmtime
#else
                timegm
#endif
                (&t));
            return std::chrono::time_point_cast<TimePoint::duration>(timePoint) +
                   std::chrono::nanoseconds(static_cast<int64_t>(fractionalSecond_ * SECONDS_TO_NANOSECONDS));
        }

        //============================================================================
        // Method Description:
        ///@brief Converts the struct to an iso string
        ///
        ///@returns std::string
        ///
        [[nodiscard]] std::string toStr() const
        {
            const auto timePoint         = toTimePoint();
            const auto timeSinceEpoch    = timePoint.time_since_epoch().count();
            time_t     secondsFromEpoch  = timeSinceEpoch / Duration::period::den;
            const auto fractionalSeconds = static_cast<double>(timeSinceEpoch % Duration::period::den) /
                                           static_cast<double>(Duration::period::den);

            std::tm tm;
#ifdef _MSC_VER
            gmtime_s(&tm, &secondsFromEpoch);
#else
            gmtime_r(&secondsFromEpoch, &tm);
#endif

            std::stringstream ss;
            if (fractionalSeconds > 0)
            {
                const auto        format = "%Y-%m-%dT%H:%M:%S.%msZ";
                std::stringstream ssFractionalSecond;
                ssFractionalSecond.precision(NANO_SECOND_PRECESION);
                ssFractionalSecond << std::fixed << fractionalSeconds;
                auto fractionalSecondStr = ssFractionalSecond.str();
                // strip of the preceding "0." and any trailing zeros
                fractionalSecondStr = fractionalSecondStr.substr(2, fractionalSecondStr.size());
                fractionalSecondStr = fractionalSecondStr.substr(0, fractionalSecondStr.find_last_not_of('0') + 1);
                const auto fractionalSecondsFormat = std::regex_replace(format, std::regex("%ms"), fractionalSecondStr);
                ss << std::put_time(&tm, fractionalSecondsFormat.c_str());
            }
            else
            {
                const auto format = "%Y-%m-%dT%H:%M:%SZ";
                ss << std::put_time(&tm, format);
            }

            return ss.str();
        }

        //============================================================================
        // Method Description:
        ///@brief Factory static method for returning a DateTime object
        ///       cooresponding to the system clock now.
        ///
        ///@returns DateTime
        ///
        [[nodiscard]] static DateTime now() noexcept
        {
            return DateTime(Clock::now());
        }

        //============================================================================
        // Method Description:
        ///@brief Converts the struct to an iso string
        ///@param timestamp: an iso formatted datetime string (0001-01-01T00:00:00.00000Z)
        ///@returns Timepoint
        ///
        static TimePoint strToTimepoint(const std::string& timestamp)
        {
            const std::regex regexIsoTime{ "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(.\\d+)?Z" };
            if (!std::regex_match(timestamp, regexIsoTime))
            {
                throw std::invalid_argument("Invalid iso timestamp format");
            }

            auto convertedTime = boost::posix_time::ptime{};
            try
            {
                convertedTime = boost::posix_time::from_iso_extended_string(timestamp.substr(0, timestamp.size() - 1));
            }
            catch (...)
            {
                throw std::invalid_argument("Invalid iso timestamp format");
            }

            const auto fromEpoch = convertedTime - POSIX_EPOCH;
            return TimePoint{ Duration{ fromEpoch.total_nanoseconds() } };
        }

    private:
        static constexpr int                         TM_EPOCH_YEAR    = 1900;
        static constexpr int                         POSIX_EPOCH_YEAR = 1970;
        static inline const std::string              POSIX_EPOCH_STR{ "1970-01-01T00:00:00" };
        static inline const boost::posix_time::ptime POSIX_EPOCH{ boost::posix_time::from_iso_extended_string(
            POSIX_EPOCH_STR) };
        static constexpr double                      SECONDS_TO_NANOSECONDS = 1e9;
        static constexpr int                         NANO_SECOND_PRECESION  = 9;

        /// years since 1
        int year_{ POSIX_EPOCH_YEAR };
        /// [1, 12]
        int month_{ 1 };
        /// [1, 31]
        int day_{ 1 };
        /// [0, 23]
        int hour_{ 0 };
        /// [0, 59]
        int minute_{ 0 };
        /// [0, 59]
        int second_{ 0 };
        /// [0, 1)
        double fractionalSecond_{ 0.0 };
    };

    //============================================================================
    // Method Description:
    ///@brief Equality operator for DateTime
    ///
    ///@param lhs: the left hand side value
    ///@param rhs: the right hand side value
    ///@returns bool
    ///
    [[nodiscard]] inline bool operator==(const DateTime& lhs, const DateTime& rhs) noexcept
    {
        return lhs.toTimePoint() == rhs.toTimePoint();
    }

    //============================================================================
    // Method Description:
    ///@brief Non Equality operator for DateTime
    ///
    ///@param lhs: the left hand side value
    ///@param rhs: the right hand side value
    ///@returns bool
    ///
    [[nodiscard]] inline bool operator!=(const DateTime& lhs, const DateTime& rhs) noexcept
    {
        return !(lhs == rhs);
    }

    //============================================================================
    // Method Description:
    ///@brief Less than operator
    ///
    ///@param lhs: the left hand side value
    ///@param rhs: the right hand side value
    ///@returns bool
    ///
    [[nodiscard]] inline bool operator<(const DateTime& lhs, const DateTime& rhs) noexcept
    {
        return lhs.toTimePoint() < rhs.toTimePoint();
    }

    //============================================================================
    // Method Description:
    ///@brief Less than or equal operator
    ///
    ///@param lhs: the left hand side value
    ///@param rhs: the right hand side value
    ///@returns bool
    ///
    [[nodiscard]] inline bool operator<=(const DateTime& lhs, const DateTime& rhs) noexcept
    {
        return lhs.toTimePoint() <= rhs.toTimePoint();
    }

    //============================================================================
    // Method Description:
    ///@brief Greater than operator
    ///
    ///@param lhs: the left hand side value
    ///@param rhs: the right hand side value
    ///@returns bool
    ///
    [[nodiscard]] inline bool operator>(const DateTime& lhs, const DateTime& rhs) noexcept
    {
        return lhs.toTimePoint() > rhs.toTimePoint();
    }

    //============================================================================
    // Method Description:
    ///@brief Greater than or equal operator
    ///
    ///@param lhs: the left hand side value
    ///@param rhs: the right hand side value
    ///@returns bool
    ///
    [[nodiscard]] inline bool operator>=(const DateTime& lhs, const DateTime& rhs) noexcept
    {
        return lhs.toTimePoint() >= rhs.toTimePoint();
    }

    //============================================================================
    // Method Description:
    ///@brief Subtraction operator
    ///
    ///@param lhs: the left hand side value
    ///@param rhs: the right hand side value
    ///@returns bool
    ///
    [[nodiscard]] inline Duration operator-(const DateTime& lhs, const DateTime& rhs) noexcept
    {
        return lhs.toTimePoint() - rhs.toTimePoint();
    }

    //============================================================================
    // Method Description:
    /// @brief Stream operator
    ///
    /// @param os: the output stream
    /// @param datetime: the datetime object
    /// @returns ostream
    ///
    inline std::ostream& operator<<(std::ostream& os, const DateTime& datetime) noexcept
    {
        os << "DateTime:\n";
        os << "\tyear: " << datetime.year() << '\n';
        os << "\tmonth: " << datetime.month() << '\n';
        os << "\tday: " << datetime.day() << '\n';
        os << "\thour: " << datetime.hour() << '\n';
        os << "\tminute: " << datetime.minute() << '\n';
        os << "\tsecond: " << datetime.second() << '\n';
        os << "\tfractionalSecond: " << datetime.fractionalSecond() << '\n';
        return os;
    }
} // namespace nc

#endif
