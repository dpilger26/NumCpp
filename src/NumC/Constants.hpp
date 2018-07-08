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
/// Holds usefull constants
///
#pragma once

#include"NumC/Types.hpp"

#include<cmath>
#include<string>

namespace NumC
{
    //================================Constants====================================
    /// Holds usefull constants
    namespace Constants
    {
        const double        c = 3.0e8; ///< speed of light 
        const double        e = 2.718281828459045;  ///< eulers number 
        const double        pi = 3.14159265358979323846; ///< Pi 
        const double        nan = std::nan("1"); ///< NaN 

        const double        DAYS_PER_WEEK = 7; ///< Number of days in a week
        const double        MINUTES_PER_HOUR = 60; ///< Number of minutes in an hour
        const double        SECONDS_PER_MINUTE = 60; ///< Number of seconds in a minute
        const double        MILLISECONDS_PER_SECOND = 1000; ///< Number of milliseconds in a second
        const double        SECONDS_PER_HOUR = MINUTES_PER_HOUR * SECONDS_PER_MINUTE; ///< Number of seconds in an hour
        const double        HOURS_PER_DAY = 24; ///< Number of hours in a day
        const double        MINUTES_PER_DAY = HOURS_PER_DAY * MINUTES_PER_HOUR; ///< Number of minutes in a day
        const double        SECONDS_PER_DAY = MINUTES_PER_DAY * SECONDS_PER_MINUTE; ///< Number of seconds in a day
        const double        MILLISECONDS_PER_DAY = SECONDS_PER_DAY * MILLISECONDS_PER_SECOND; ///< Number of milliseconds in a day
        const double        SECONDS_PER_WEEK = SECONDS_PER_DAY * DAYS_PER_WEEK; ///< Number of seconds in a week

        const std::string   VERSION = "1.0"; ///< Current NumC version number
    }
}
