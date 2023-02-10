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
#include <vector>

#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Integrate/trapazoidal.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/power.hpp"

namespace nc
{
    namespace integrate
    {
        //============================================================================
        // Method Description:
        /// Performs Newton-Cotes Romberg integration of the input function
        ///
        /// @param low: the lower bound of the integration
        /// @param high: the upper bound of the integration
        /// @param n: the number of iterations
        /// @param f: the function to integrate over
        ///
        /// @return double
        ///
        inline double
            romberg(const double low, const double high, const uint8 n, const std::function<double(double)>& f)
        {
            NdArray<double> rombergIntegral(n);

            // R(0,0) Start with trapezoidal integration with N = 1
            rombergIntegral(0, 0) = trapazoidal(low, high, 1, f);

            double h = high - low;
            for (uint8 step = 1; step < n; step++)
            {
                h *= 0.5;

                // R(step, 0) Improve trapezoidal integration with decreasing h
                double       trapezoidal_integration = 0.;
                const uint32 stepEnd                 = utils::power(2, step - 1);
                for (uint32 tzStep = 1; tzStep <= stepEnd; ++tzStep)
                {
                    const double deltaX = (2. * static_cast<double>(tzStep - 1)) * h;
                    trapezoidal_integration += f(low + deltaX);
                }

                rombergIntegral(step, 0) = 0.5 * rombergIntegral(step - 1, 0);
                rombergIntegral(step, 0) += trapezoidal_integration * h;

                // R(m,n) Romberg integration with R(m,1) -> Simpson rule, R(m,2) -> Boole's rule
                for (uint8 rbStep = 1; rbStep <= step; ++rbStep)
                {
                    const double k                = utils::power(4, rbStep);
                    rombergIntegral(step, rbStep) = k * rombergIntegral(step, rbStep - 1);
                    rombergIntegral(step, rbStep) -= rombergIntegral(step - 1, rbStep - 1);
                    rombergIntegral(step, rbStep) /= (k - 1.);
                }
            }

            return rombergIntegral.back();
        }
    } // namespace integrate
} // namespace nc
