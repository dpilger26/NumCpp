/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.2.0
///
/// @section License
/// Copyright 2019 Benjamin Mahr
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
/// Finds the roots of the polynomial
///
/// Code modified under MIT license from https://github.com/Ben1980/rootApproximation
/// as posted in https://thoughts-on-coding.com/2019/06/06/numerical-methods-with-cpp-part-3-root-approximation-algorithms/
///
#pragma once

#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Roots/Iteration.hpp"

#include <cmath>
#include <functional>
#include <utility>

namespace nc
{
    namespace roots
    {
        //================================================================================
        // Class Description:
        ///	Bisection root finding method
        ///
        class Bisection : public Iteration
        {
        public:
            //============================================================================
            // Method Description:
            ///	Constructor
            ///
            /// @param epsilon: the epsilon value
            /// @param f: the function 
            ///
            Bisection(const double epsilon, 
                std::function<double(double)>  f) noexcept :
                Iteration(epsilon),
                f_(std::move(f))
            {}

            //============================================================================
            // Method Description:
            ///	Constructor
            ///
            /// @param epsilon: the epsilon value
            /// @param maxNumIterations: the maximum number of iterations to perform
            /// @param f: the function 
            ///
            Bisection(const double epsilon, 
                const uint32 maxNumIterations,
                std::function<double(double)>  f) noexcept :
                Iteration(epsilon, maxNumIterations),
                f_(std::move(f))
            {}

            //============================================================================
            // Method Description:
            ///	Destructor
            ///
            ~Bisection() override = default;

            //============================================================================
            // Method Description:
            ///	Solves for the root in the range [a, b]
            ///
            /// @param a: the lower bound
            /// @param b: the upper bound
            /// @return root between the bound
            ///
            double solve(double a, double b)
            {
                resetNumberOfIterations();
                checkAndFixAlgorithmCriteria(a, b);

                double x = 0.5 * (a + b);
                double fx = f_(x);

                while (std::fabs(fx) >= epsilon_)
                {
                    x = calculateX(x, a, b, fx);
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
            ///	Checks the bounds criteria
            ///
            /// @param a: the lower bound
            /// @param b: the upper bound
            ///
            void checkAndFixAlgorithmCriteria(double &a, double &b) const noexcept
            {
                //Algorithm works in range [a,b] if criteria f(a)*f(b) < 0 and f(a) > f(b) is fulfilled
                if (f_(a) < f_(b))
                {
                    std::swap(a, b);
                }
            }

            //============================================================================
            // Method Description:
            ///	Calculates the bisection point
            ///
            /// @param x: the evaluation point
            /// @param a: the lower bound
            /// @param b: the upper bound
            /// @param fx: the function evaluated at x
            /// @return x
            ///
            static double calculateX(double x, double &a, double &b, double fx) noexcept 
            {
                if (fx < 0)
                {
                    b = x;
                }
                else
                {
                    a = x;
                }

                return 0.5 * (a + b);
            }
        };
    }  // namespace roots
}  // namespace nc
