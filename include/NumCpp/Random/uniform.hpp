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
/// Draw samples from a uniform distribution.
///
#pragma once

#include "NumCpp/Core/DtypeInfo.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Random/randFloat.hpp"

namespace nc
{
    namespace random
    {
        namespace detail
        {
            //============================================================================
            // Method Description:
            /// Draw sample from a uniform distribution.
            ///
            /// Samples are uniformly distributed over the half -
            /// open interval[low, high) (includes low, but excludes high)
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html#numpy.random.uniform
            ///
            /// @param generator: instance of a random number generator
            /// @param inLow
            /// @param inHigh
            /// @return NdArray
            ///
            template<typename dtype, typename GeneratorType = std::mt19937>
            dtype uniform(GeneratorType& generator, dtype inLow, dtype inHigh)
            {
                STATIC_ASSERT_FLOAT(dtype);

                return detail::randFloat(generator, inLow, inHigh);
            }

            //============================================================================
            // Method Description:
            /// Draw samples from a uniform distribution.
            ///
            /// Samples are uniformly distributed over the half -
            /// open interval[low, high) (includes low, but excludes high)
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html#numpy.random.uniform
            ///
            /// @param generator: instance of a random number generator
            /// @param inShape
            /// @param inLow
            /// @param inHigh
            /// @return NdArray
            ///
            template<typename dtype, typename GeneratorType = std::mt19937>
            NdArray<dtype> uniform(GeneratorType& generator, const Shape& inShape, dtype inLow, dtype inHigh)
            {
                STATIC_ASSERT_FLOAT(dtype);

                return detail::randFloat(generator, inShape, inLow, inHigh);
            }
        } // namespace detail

        //============================================================================
        // Method Description:
        /// Draw sample from a uniform distribution.
        ///
        /// Samples are uniformly distributed over the half -
        /// open interval[low, high) (includes low, but excludes high)
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html#numpy.random.uniform
        ///
        /// @param inLow
        /// @param inHigh
        /// @return NdArray
        ///
        template<typename dtype>
        dtype uniform(dtype inLow, dtype inHigh)
        {
            return detail::uniform(generator_, inLow, inHigh);
        }

        //============================================================================
        // Method Description:
        /// Draw samples from a uniform distribution.
        ///
        /// Samples are uniformly distributed over the half -
        /// open interval[low, high) (includes low, but excludes high)
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html#numpy.random.uniform
        ///
        /// @param inShape
        /// @param inLow
        /// @param inHigh
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<dtype> uniform(const Shape& inShape, dtype inLow, dtype inHigh)
        {
            return detail::uniform(generator_, inShape, inLow, inHigh);
        }
    } // namespace random
} // namespace nc
