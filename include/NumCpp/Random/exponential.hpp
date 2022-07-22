/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
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
/// "exponential" distrubution
///
#pragma once

#include <algorithm>
#include <random>

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
            /// Single random value sampled from the "exponential" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential
            ///
            /// @param generator: instance of a random number generator
            /// @param inScaleValue (default 1)
            /// @return NdArray
            ///
            template<typename dtype, typename GeneratorType = std::mt19937>
            dtype exponential(GeneratorType& generator, dtype inScaleValue = 1)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                std::exponential_distribution<dtype> dist(inScaleValue);
                return dist(generator);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "exponential" distrubution.
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential
            ///
            /// @param generator: instance of a random number generator
            /// @param inShape
            /// @param inScaleValue (default 1)
            /// @return NdArray
            ///
            template<typename dtype, typename GeneratorType = std::mt19937>
            NdArray<dtype> exponential(GeneratorType& generator, const Shape& inShape, dtype inScaleValue = 1)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                NdArray<dtype> returnArray(inShape);

                std::exponential_distribution<dtype> dist(inScaleValue);

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&generator, &dist](dtype& value) -> void { value = dist(generator); });

                return returnArray;
            }
        } // namespace detail

        //============================================================================
        // Method Description:
        /// Single random value sampled from the "exponential" distrubution.
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential
        ///
        /// @param inScaleValue (default 1)
        /// @return NdArray
        ///
        template<typename dtype>
        dtype exponential(dtype inScaleValue = 1)
        {
            return detail::exponential(generator_, inScaleValue);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from a "exponential" distrubution.
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential
        ///
        /// @param inShape
        /// @param inScaleValue (default 1)
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<dtype> exponential(const Shape& inShape, dtype inScaleValue = 1)
        {
            return detail::exponential(generator_, inShape, inScaleValue);
        }
    } // namespace random
} // namespace nc
