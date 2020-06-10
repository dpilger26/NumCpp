/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.0.0
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

namespace nc
{
    namespace roots
    {
        //================================================================================
        // Class Description:
        ///	Newton root finding method
        ///
        class Newton : public Iteration
        {
        public:
            //============================================================================
            // Method Description:
            ///	Constructor
            ///
            /// @param epsilon: the epsilon value
            /// @param f: the function 
            /// @param fPrime: the derivative of the function 
            ///
            Newton(const double epsilon,
                const std::function<double(double)>& f,
                const std::function<double(double)>& fPrime)  :
                Iteration(epsilon),
                f_(f),
                fPrime_(fPrime)
            {}

            //============================================================================
            // Method Description:
            ///	Constructor
            ///
            /// @param epsilon: the epsilon value
            /// @param maxNumIterations: the maximum number of iterations to perform
            /// @param f: the function 
            /// @param fPrime: the derivative of the function 
            ///
            Newton(const double epsilon,
                const uint32 maxNumIterations,
                const std::function<double(double)>& f,
                const std::function<double(double)>& fPrime)  :
                Iteration(epsilon, maxNumIterations),
                f_(f),
                fPrime_(fPrime)
            {}

            //============================================================================
            // Method Description:
            ///	Destructor
            ///
            ~Newton()  = default;

            //============================================================================
            // Method Description:
            ///	Solves for the root in the range [a, b]
            ///
            /// @param x: the starting point
            /// @return root nearest the starting point
            ///
            double solve(double x)
            {
                resetNumberOfIterations();

                double fx = f_(x);
                double fxPrime = fPrime_(x);

                while (std::fabs(fx) >= epsilon_)
                {
                    x = calculateX(x, fx, fxPrime);

                    fx = f_(x);
                    fxPrime = fPrime_(x);

                    incrementNumberOfIterations();
                }

                return x;
            }

        private:
            //============================================================================
            const std::function<double(double)> f_;
            const std::function<double(double)> fPrime_;

            //============================================================================
            // Method Description:
            ///	Calculates x
            ///
            /// @param x: the current x value
            /// @param fx: the function evaluated at the current x value
            /// @param fxPrime: the derivate of the function evaluated at the current x value
            /// @return x
            ///
            double calculateX(double x, double fx, double fxPrime) noexcept 
            {
                return x - fx / fxPrime;
            }
        };
    }
}
