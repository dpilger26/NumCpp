/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.1.0
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

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Types.hpp"

#include <functional>

namespace nc
{
    namespace roots
    {
        //================================================================================
        // Class Description:
        ///	ABC for iteration classes to derive from
        class Iteration 
        {
        public:
            //============================================================================
            // Method Description:
            ///	Constructor
            ///
            /// @param epsilon: the epsilon value
            ///
            explicit Iteration(double epsilon) noexcept : 
                epsilon_(epsilon)
            {}

            //============================================================================
            // Method Description:
            ///	Constructor
            ///
            /// @param epsilon: the epsilon value
            /// @param maxNumIterations: the maximum number of iterations to perform
            ///
            Iteration(double epsilon, uint32 maxNumIterations) noexcept : 
                epsilon_(epsilon),
                maxNumIterations_(maxNumIterations)
            {}

            //============================================================================
            // Method Description:
            ///	Destructor
            ///
            virtual ~Iteration() noexcept = default;

            //============================================================================
            // Method Description:
            ///	Returns the number of iterations
            ///
            /// @return: number of iterations
            ///
            uint32 numIterations() const noexcept 
            { 
                return numIterations_;
            }

        protected:
            //============================================================================
            // Method Description:
            ///	Resets the number of iterations
            ///
            void resetNumberOfIterations() noexcept 
            { 
                numIterations_ = 0;
            }

            //============================================================================
            // Method Description:
            ///	Incraments the number of iterations
            ///
            /// @return the number of iterations prior to incramenting
            ///
            void incrementNumberOfIterations()
            { 
                ++numIterations_;
                if (numIterations_ > maxNumIterations_)
                {
                    THROW_RUNTIME_ERROR("Maximum number of iterations has been reached; no root has been found within epsilon.");
                }
            }

            //====================================Attributes==============================
            const double    epsilon_;
            uint32          maxNumIterations_{1000};
            uint32          numIterations_{0};
        };
    }  // namespace roots
} // namespace nc
