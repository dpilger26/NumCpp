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
/// Return random integers from low (inclusive) to high (exclusive),
/// with the given shape
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
            /// Return random integer from low (inclusive) to high (exclusive),
            /// with the given shape. If no high value is input then the range will
            /// go from [0, low).
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html#numpy.random.randint
            ///
            /// @param generator: instance of a random number generator
            /// @param inLow
            /// @param inHigh default 0.
            /// @return NdArray
            ///
            template<typename dtype, typename GeneratorType = std::mt19937>
            dtype randInt(GeneratorType generator, dtype inLow, dtype inHigh = 0)
            {
                STATIC_ASSERT_INTEGER(dtype);

                if (inLow == inHigh)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input low value must be less than the input high value.");
                }
                else if (inLow > inHigh - 1)
                {
                    std::swap(inLow, inHigh);
                }

                std::uniform_int_distribution<dtype> dist(inLow, inHigh - 1);
                return dist(generator);
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
            template<typename dtype, typename GeneratorType = std::mt19937>
            NdArray<dtype> randInt(GeneratorType generator, const Shape& inShape, dtype inLow, dtype inHigh = 0)
            {
                STATIC_ASSERT_INTEGER(dtype);

                if (inLow == inHigh)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input low value must be less than the input high value.");
                }
                else if (inLow > inHigh - 1)
                {
                    std::swap(inLow, inHigh);
                }

                NdArray<dtype> returnArray(inShape);

                std::uniform_int_distribution<dtype> dist(inLow, inHigh - 1);

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&dist, &generator](dtype& value) -> void { value = dist(generator); });

                return returnArray;
            }
        } // namespace detail

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
    } // namespace random
} // namespace nc
