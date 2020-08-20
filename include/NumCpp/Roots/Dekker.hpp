/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
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
/// Description
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
        ///	Dekker root finding method
        ///
        class Dekker : public Iteration
        {
        public:
            //============================================================================
            // Method Description:
            ///	Constructor
            ///
            /// @param epsilon: the epsilon value
            /// @param f: the function 
            ///
            Dekker(const double epsilon,
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
            Dekker(const double epsilon, 
                const uint32 maxNumIterations, 
                std::function<double(double)>  f) noexcept :
                Iteration(epsilon, maxNumIterations),
                f_(std::move(f))
            {}

            //============================================================================
            // Method Description:
            ///	Destructor
            ///
            ~Dekker() override = default;

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

                double fa = f_(a);
                double fb = f_(b);

                checkAndFixAlgorithmCriteria(a, b, fa, fb);

                double lastB = a;
                double lastFb = fa;

                while (std::fabs(fb) > epsilon_ && std::fabs(b - a) > epsilon_) 
                {
                    const double s = calculateSecant(b, fb, lastB, lastFb);
                    const double m = calculateBisection(a, b);

                    lastB = b;

                    b = useSecantMethod(b, s, m) ? s : m;

                    lastFb = fb;
                    fb = f_(b);

                    if (fa * fb > 0 && fb * lastFb < 0) 
                    {
                        a = lastB;
                    }

                    fa = f_(a);
                    checkAndFixAlgorithmCriteria(a, b, fa, fb);

                    incrementNumberOfIterations();
                }

                return b;
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
            /// @param fa: the function evalulated at the lower bound
            /// @param fb: the function evalulated at the upper bound
            ///
            static void checkAndFixAlgorithmCriteria(double &a, double &b, double &fa, double &fb) noexcept 
            {
                //Algorithm works in range [a,b] if criteria f(a)*f(b) < 0 and f(a) > f(b) is fulfilled
                if (std::fabs(fa) < std::fabs(fb)) 
                {
                    std::swap(a, b);
                    std::swap(fa, fb);
                }
            }

            //============================================================================
            // Method Description:
            ///	Calculates secant
            ///
            /// @param b: the upper bound
            /// @param fb: the function evalulated at the upper bound
            /// @param lastB: the last upper bound
            /// @param lastFb: the function evalulated at the last upper bound
            /// @ return secant value
            ///
            static double calculateSecant(double b, double fb, double lastB, double lastFb) noexcept 
            {
                //No need to check division by 0, in this case the method returns NAN which is taken care by useSecantMethod method
                return b - fb * (b - lastB) / (fb - lastFb);
            }

            //============================================================================
            // Method Description:
            ///	Calculate the bisection point
            ///
            /// @param a: the lower bound
            /// @param b: the upper bound
            /// @return bisection point
            ///
            static double calculateBisection(double a, double b) noexcept 
            {
                return 0.5 * (a + b);
            }

            //============================================================================
            // Method Description:
            ///	Whether or not to use the secant method
            ///
            /// @param b: the upper bound
            /// @param s:
            /// @param m:
            /// @ return bool
            ///
            static bool useSecantMethod(double b, double s, double m) noexcept 
            {
                //Value s calculated by secant method has to be between m and b
                return (b > m && s > m && s < b) ||
                    (b < m && s > b && s < m);
            }
        };
    }  // namespace roots
}  // namespace nc
