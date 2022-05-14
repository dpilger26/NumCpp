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
/// "F" distrubution.
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
            /// Single random value sampled from the "F" distrubution.
            ///
            /// NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.f.html#numpy.random.f
            ///
            /// @param generator: instance of a random number generator
            /// @param inDofN: Degrees of freedom in numerator. Should be greater than zero.
            /// @param inDofD: Degrees of freedom in denominator. Should be greater than zero.
            /// @return NdArray
            ///
            template<typename dtype, typename GeneratorType = std::mt19937>
            dtype f(GeneratorType& generator, dtype inDofN, dtype inDofD)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inDofN <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("numerator degrees of freedom should be greater than zero.");
                }

                if (inDofD <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("denominator degrees of freedom should be greater than zero.");
                }

                std::fisher_f_distribution<dtype> dist(inDofN, inDofD);
                return dist(generator);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "F" distrubution.
            ///
            /// NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.f.html#numpy.random.f
            ///
            /// @param generator: instance of a random number generator
            /// @param inShape
            /// @param inDofN: Degrees of freedom in numerator. Should be greater than zero.
            /// @param inDofD: Degrees of freedom in denominator. Should be greater than zero.
            /// @return NdArray
            ///
            template<typename dtype, typename GeneratorType = std::mt19937>
            NdArray<dtype> f(GeneratorType& generator, const Shape& inShape, dtype inDofN, dtype inDofD)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inDofN <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("numerator degrees of freedom should be greater than zero.");
                }

                if (inDofD <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("denominator degrees of freedom should be greater than zero.");
                }

                NdArray<dtype> returnArray(inShape);

                std::fisher_f_distribution<dtype> dist(inDofN, inDofD);

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&generator, &dist](dtype& value) -> void { value = dist(generator); });

                return returnArray;
            }
        } // namespace detail

        //============================================================================
        // Method Description:
        /// Single random value sampled from the "F" distrubution.
        ///
        /// NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.f.html#numpy.random.f
        ///
        /// @param inDofN: Degrees of freedom in numerator. Should be greater than zero.
        /// @param inDofD: Degrees of freedom in denominator. Should be greater than zero.
        /// @return NdArray
        ///
        template<typename dtype>
        dtype f(dtype inDofN, dtype inDofD)
        {
            return detail::f(generator_, inDofN, inDofD);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from a "F" distrubution.
        ///
        /// NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.f.html#numpy.random.f
        ///
        /// @param inShape
        /// @param inDofN: Degrees of freedom in numerator. Should be greater than zero.
        /// @param inDofD: Degrees of freedom in denominator. Should be greater than zero.
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<dtype> f(const Shape& inShape, dtype inDofN, dtype inDofD)
        {
            return detail::f(generator_, inShape, inDofN, inDofD);
        }
    } // namespace random
} // namespace nc
