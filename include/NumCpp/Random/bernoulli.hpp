/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
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
/// "bernoulli" distribution.
///
#pragma once

#include <algorithm>
#include <random>
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/generator.hpp"

namespace nc
{
    namespace random
    {
        namespace detail
        {
            //============================================================================
            // Method Description:
            /// Single random value sampled from the "bernoulli" distribution.
            ///
            /// @param generator: instance of a random number generator
            /// @param inP (probability of success [0, 1]). Default 0.5
            /// @return NdArray
            ///
            template<typename GeneratorType = std::mt19937>
            bool bernoulli(GeneratorType& generator, double inP = 0.5)
            {
                if (inP < 0 || inP > 1)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input probability of success must be of the range [0, 1].");
                }

                std::bernoulli_distribution dist(inP);
                return dist(generator);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from the "bernoulli" distribution.
            ///
            /// @param generator: instance of a random number generator
            /// @param inShape
            /// @param inP (probability of success [0, 1]). Default 0.5
            /// @return NdArray
            ///
            template<typename GeneratorType = std::mt19937>
            NdArray<bool> bernoulli(GeneratorType& generator, const Shape& inShape, double inP = 0.5)
            {
                if (inP < 0 || inP > 1)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input probability of success must be of the range [0, 1].");
                }

                NdArray<bool> returnArray(inShape);

                std::bernoulli_distribution dist(inP);

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&generator, &dist](bool& value) -> void { value = dist(generator); });

                return returnArray;
            }
        } // namespace detail

        //============================================================================
        // Method Description:
        /// Single random value sampled from the "bernoulli" distribution.
        ///
        /// @param inP (probability of success [0, 1]). Default 0.5
        /// @return NdArray
        ///
        inline bool bernoulli(double inP = 0.5)
        {
            return detail::bernoulli(generator_, inP);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from the "bernoulli" distribution.
        ///
        /// @param inShape
        /// @param inP (probability of success [0, 1]). Default 0.5
        /// @return NdArray
        ///
        inline NdArray<bool> bernoulli(const Shape& inShape, double inP = 0.5)
        {
            return detail::bernoulli(generator_, inShape, inP);
        }
    } // namespace random
} // namespace nc
