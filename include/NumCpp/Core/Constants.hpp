/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2025 David Pilger
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
/// Holds usefull constants
///
#pragma once

#include <cmath>
#include <complex>
#include <limits>

namespace nc::constants
{
    constexpr double c     = 3.e8;                                    ///< speed of light
    constexpr double e     = 2.718281828459045;                       ///< eulers number
    constexpr double inf   = std::numeric_limits<double>::infinity(); ///< infinity
    constexpr double pi    = 3.141592653589793238462643383279502884;  ///< Pi
    constexpr double twoPi = 2. * pi;                                 ///< 2Pi
    const double     nan   = std::nan("1");                           ///< NaN
    constexpr auto   j     = std::complex<double>(0, 1);              // sqrt(-1) unit imaginary number

    constexpr double DAYS_PER_WEEK           = 7;    ///< Number of days in a week
    constexpr double MINUTES_PER_HOUR        = 60;   ///< Number of minutes in an hour
    constexpr double SECONDS_PER_MINUTE      = 60;   ///< Number of seconds in a minute
    constexpr double MILLISECONDS_PER_SECOND = 1000; ///< Number of milliseconds in a second
    constexpr double SECONDS_PER_HOUR        = MINUTES_PER_HOUR * SECONDS_PER_MINUTE; ///< Number of seconds in an hour
    constexpr double HOURS_PER_DAY           = 24;                                    ///< Number of hours in a day
    constexpr double MINUTES_PER_DAY         = HOURS_PER_DAY * MINUTES_PER_HOUR;      ///< Number of minutes in a day
    constexpr double SECONDS_PER_DAY         = MINUTES_PER_DAY * SECONDS_PER_MINUTE;  ///< Number of seconds in a day
    constexpr double MILLISECONDS_PER_DAY =
        SECONDS_PER_DAY * MILLISECONDS_PER_SECOND;                       ///< Number of milliseconds in a day
    constexpr double SECONDS_PER_WEEK = SECONDS_PER_DAY * DAYS_PER_WEEK; ///< Number of seconds in a week
} // namespace nc::constants
