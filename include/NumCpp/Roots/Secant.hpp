/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2019 Benjamin Mahr
/// Copyright 2018-2022 David Pilger
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
/// Finds the roots of the polynomial
///
/// Code modified under MIT license from https://github.com/Ben1980/rootApproximation
/// as posted in
/// https://thoughts-on-coding.com/2019/06/06/numerical-methods-with-cpp-part-3-root-approximation-algorithms/
///
#pragma once

#include <cmath>
#include <functional>
#include <utility>

#include "NumCpp/Core/DtypeInfo.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Roots/Iteration.hpp"

namespace nc
{
    namespace roots
    {
        //================================================================================
        // Class Description:
        /// Secant root finding method
        ///
        class Secant : public Iteration
        {
        public:
            //============================================================================
            // Method Description:
            /// Constructor
            ///
            /// @param epsilon: the epsilon value
            /// @param f: the function
            ///
            Secant(const double epsilon, std::function<double(double)> f) noexcept :
                Iteration(epsilon),
                f_(std::move(f))
            {
            }

            //============================================================================
            // Method Description:
            /// Constructor
            ///
            /// @param epsilon: the epsilon value
            /// @param maxNumIterations: the maximum number of iterations to perform
            /// @param f: the function
            ///
            Secant(const double epsilon, const uint32 maxNumIterations, std::function<double(double)> f) noexcept :
                Iteration(epsilon, maxNumIterations),
                f_(std::move(f))
            {
            }

            //============================================================================
            // Method Description:
            /// Destructor
            ///
            ~Secant() override = default;

            //============================================================================
            // Method Description:
            /// Solves for the root in the range [a, b]
            ///
            /// @param a: the lower bound
            /// @param b: the upper bound
            /// @return root between the bound
            ///
            double solve(double a, double b)
            {
                resetNumberOfIterations();

                if (f_(a) > f_(b))
                {
                    std::swap(a, b);
                }

                double x      = b;
                double lastX  = a;
                double fx     = f_(b);
                double lastFx = f_(a);

                while (std::fabs(fx) >= epsilon_)
                {
                    const double x_tmp = calculateX(x, lastX, fx, lastFx);

                    lastFx = fx;
                    lastX  = x;
                    x      = x_tmp;

                    fx = f_(x);

                    incrementNumberOfIterations();
                }

                return x;
            }

        private:
            //============================================================================
            const std::function<double(double)> f_;

            //============================================================================
            // Method Description:
            /// Calculates x
            ///
            /// @param x: the current x value
            /// @param lastX: the previous x value
            /// @param fx: the function evaluated at the current x value
            /// @param lastFx: the function evaluated at the previous x value
            /// @return x
            ///
            static double calculateX(double x, double lastX, double fx, double lastFx) noexcept
            {
                const double functionDifference = fx - lastFx;
                return x - fx * (x - lastX) / functionDifference;
            }
        };
    } // namespace roots
} // namespace nc
