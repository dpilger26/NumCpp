/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2019 Benjamin Mahr
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
/// Numerical Integration
///
/// Code modified under MIT license from https://github.com/Ben1980/numericalIntegration
/// as posted in https://thoughts-on-coding.com/2019/04/17/numerical-methods-in-c-part-1-newton-cotes-integration/
///
#pragma once

#include <functional>

#include "NumCpp/Core/Types.hpp"

namespace nc
{
    namespace integrate
    {
        //============================================================================
        // Method Description:
        /// Performs Newton-Cotes trapazoidal integration of the input function
        ///
        /// @param low: the lower bound of the integration
        /// @param high: the upper bound of the integration
        /// @param n: the number of subdivisions
        /// @param f: the function to integrate over
        ///
        /// @return double
        ///
        inline double trapazoidal(const double                         low,
                                  const double                         high,
                                  const uint32                         n,
                                  const std::function<double(double)>& f) noexcept
        {
            const double width = (high - low) / static_cast<double>(n);

            double trapezoidal_integral = 0.;
            for (uint32 step = 0; step < n; ++step)
            {
                const double x1 = low + static_cast<double>(step) * width;
                const double x2 = low + static_cast<double>(step + 1) * width;

                trapezoidal_integral += 0.5 * (x2 - x1) * (f(x1) + f(x2));
            }

            return trapezoidal_integral;
        }
    } // namespace integrate
} // namespace nc
