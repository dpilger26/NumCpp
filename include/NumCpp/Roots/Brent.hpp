/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.2
///
/// @section License
/// Copyright 2019 David Pilger
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
#pragma once

#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Roots/Iteration.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <utility>

namespace nc
{
    namespace roots
    {
        //================================================================================
        // Class Description:
        ///	Bisection root finding method
        ///
        class Brent : public Iteration
        {
        public:
            //============================================================================
            // Method Description:
            ///	Constructor
            ///
            /// @param epsilon: the epsilon value
            /// @param f: the function 
            ///
            Brent(const double epsilon, const std::function<double(double)>& f) noexcept :
                Iteration(epsilon),
                f_(f)
            {}

            //============================================================================
            // Method Description:
            ///	Constructor
            ///
            /// @param epsilon: the epsilon value
            /// @param f: the function 
            ///
            Brent(const double epsilon, const uint32 maxNumIterations, const std::function<double(double)>& f) noexcept :
                Iteration(epsilon, maxNumIterations),
                f_(f)
            {}

            //============================================================================
            // Method Description:
            ///	Destructor
            ///
            /// @param epsilon: the epsilon value
            /// @param f: the function 
            ///
            ~Brent() noexcept override = default;

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

                double lastB = a; // b_{k-1}
                double lastFb = fa;
                double s = std::numeric_limits<double>::max();
                double fs = std::numeric_limits<double>::max();
                double penultimateB = a; // b_{k-2}

                bool bisection = true;
                while(std::fabs(fb) > epsilon_ && std::fabs(fs) > epsilon_ && std::fabs(b-a) > epsilon_)
                {
                    if(useInverseQuadraticInterpolation(fa, fb, lastFb))
                    {
                        s = calculateInverseQuadraticInterpolation(a, b, lastB, fa, fb, lastFb);
                    }
                    else 
                    {
                        s = calculateSecant(a, b, fa, fb);
                    }

                    if(useBisection(bisection, b, lastB, penultimateB, s))
                    {
                        s = calculateBisection(a, b);
                        bisection = true;
                    }
                    else
                    {
                        bisection = false;
                    }

                    fs = f_(s);
                    penultimateB = lastB;
                    lastB = b;

                    if(fa*fs < 0)
                    {
                        b = s;
                    }
                    else {
                        a = s;
                    }

                    fa = f_(a);
                    lastFb = fb;
                    fb = f_(b);
                    checkAndFixAlgorithmCriteria(a, b, fa, fb);
                }

                return fb < fs ? b : s;
            }

        private:
            //============================================================================
            const std::function<double(double)> f_;

            //============================================================================
            // Method Description:
            ///	Calculates the bisection point
            ///
            /// @param a: the lower bound
            /// @param b: the upper bound
            /// @return x
            ///
            double calculateBisection(const double a, const double b) const noexcept
            {
                return 0.5 * (a + b);
            }

            //============================================================================
            // Method Description:
            ///	Calculates the secant point
            ///
            /// @param a: the lower bound
            /// @param b: the upper bound
            /// @param fa: the function evaluated at a
            /// @param fb: the function evaluated at b
            /// @return the secant point
            ///
            double calculateSecant(const double a, const double b, const double fa, const double fb) const noexcept
            {
                //No need to check division by 0, in this case the method returns NAN which is taken care by useSecantMethod method
                return b - fb * (b - a) / (fb - fa);
            }

            //============================================================================
            // Method Description:
            ///	Calculates the inverse quadratic interpolation
            ///
            /// @param a: the lower bound
            /// @param b: the upper bound
            /// @param lastB: the previous upper bound
            /// @param fa: the function evaluated at a
            /// @param fb: the function evaluated at b
            /// @param lastB: the previous function evaluated at the upper bound
            /// @return the inverse quadratic interpolation
            ///
            double calculateInverseQuadraticInterpolation(const double a, const double b, const double lastB, 
                const double fa, const double fb, const double lastFb) const noexcept
            {
                return a * fb * lastFb / ((fa - fb) * (fa - lastFb)) +
                    b * fa * lastFb / ((fb - fa) * (fb - lastFb)) +
                    lastB * fa * fb / ((lastFb - fa) * (lastFb - fb));
            }

            //============================================================================
            // Method Description:
            ///	Uses the inverse quadratic interpolation
            ///
            /// @param fa: the function evaluated at a
            /// @param fb: the function evaluated at b
            /// @param lastB: the previous function evaluated at the upper bound
            /// @return bool
            ///
            bool useInverseQuadraticInterpolation(const double fa, const double fb, const double lastFb) const noexcept
            {
                return fa != lastFb && fb != lastFb;
            }

            //============================================================================
            // Method Description:
            ///	Checks the algorithm criteria
            ///
            /// @param a: the lower bound
            /// @param b: the upper bound
            /// @param fa: the function evaluated at a
            /// @param fb: the function evaluated at b
            ///
            void checkAndFixAlgorithmCriteria(double &a, double &b, double &fa, double &fb) noexcept
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
            ///	Uses the bisection
            ///
            /// @param bisection: the bisection point
            /// @param b: the upper bound
            /// @param lastB: the previous upper bound
            /// @param penultimateB:
            /// @param s: 
            /// @return the inverse quadratic interpolation
            ///
            bool useBisection(const bool bisection, const double b, const double lastB,
                const double penultimateB, const double s) const
            {
                const double DELTA = epsilon_ + std::numeric_limits<double>::min();

                return (bisection && std::fabs(s - b) >= 0.5 * std::fabs(b - lastB)) || //Bisection was used in last step but |s-b|>=|b-lastB|/2 <- Interpolation step would be to rough, so still use bisection
                    (!bisection && std::fabs(s - b) >= 0.5 * std::fabs(lastB - penultimateB)) || //Interpolation was used in last step but |s-b|>=|lastB-penultimateB|/2 <- Interpolation step would be to small
                    (bisection  && std::fabs(b - lastB) < DELTA) || //If last iteration was using bisection and difference between b and lastB is < delta use bisection for next iteration
                    (!bisection && std::fabs(lastB - penultimateB) < DELTA); //If last iteration was using interpolation but difference between lastB ond penultimateB is < delta use biscetion for next iteration
            }
        };
    }
}
