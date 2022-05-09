/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
///
/// License
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
/// Random Number Generater Class with non-global state
///
#pragma once

#include <random>

#include "NumCpp/Random/randInt.hpp"

namespace nc
{
    namespace random
    {
        //============================================================================
        // Class Description:
        /// Random Number Generater Class with non-global state
        ///
        template<typename GeneratorType = std::mt19937_64>
        class RNG
        {
        public:
            //============================================================================
            // Method Description:
            /// Defualt Constructor
            ///
            RNG() = default;

            //============================================================================
            // Method Description:
            /// Seed Constructor
            ///
            /// @param seed: the seed value
            ///
            RNG(int seed) :
                generator_(seed){};

            //============================================================================
            // Method Description:
            /// Seed Constructor
            ///
            /// @param seed: the seed value
            ///
            void seed(int value) noexcept
            {
                generator_.seed(value);
            }

            //============================================================================
            // Method Description:
            /// Return random integer from low (inclusive) to high (exclusive),
            /// with the given shape. If no high value is input then the range will
            /// go from [0, low).
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html#numpy.random.randint
            ///
            /// @param inLow
            /// @param inHigh default 0.
            /// @return NdArray
            ///
            template<typename dtype>
            dtype randInt(dtype inLow, dtype inHigh = 0)
            {
                return detail::randInt(generator_, inLow, inHigh);
            }

            //============================================================================
            // Method Description:
            /// Return random integers from low (inclusive) to high (exclusive),
            /// with the given shape. If no high value is input then the range will
            /// go from [0, low).
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html#numpy.random.randint
            ///
            /// @param inShape
            /// @param inLow
            /// @param inHigh default 0.
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> randInt(const Shape& inShape, dtype inLow, dtype inHigh = 0)
            {
                return detail::randInt(generator_, inShape, inLow, inHigh);
            }

        private:
            GeneratorType generator_{};
        };
    } // namespace random
} // namespace nc
