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
/// "standard normal" distrubution
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/normal.hpp"

namespace nc
{
    namespace random
    {
        namespace detail
        {
            //============================================================================
            // Method Description:
            /// Single random value sampled from the "standard normal" distrubution with
            /// mean = 0 and std = 1
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_normal.html#numpy.random.standard_normal
            ///
            /// @param generator: instance of a random number generator
            /// @return NdArray
            ///
            template<typename dtype, typename GeneratorType = std::mt19937>
            dtype standardNormal(GeneratorType& generator)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                return detail::normal<dtype>(generator, 0, 1);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a "standard normal" distrubution with
            /// mean = 0 and std = 1
            ///
            /// NumPy Reference:
            /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_normal.html#numpy.random.standard_normal
            ///
            /// @param generator: instance of a random number generator
            /// @param inShape
            /// @return NdArray
            ///
            template<typename dtype, typename GeneratorType = std::mt19937>
            NdArray<dtype> standardNormal(GeneratorType& generator, const Shape& inShape)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                return detail::normal<dtype>(generator, inShape, 0, 1);
            }
        } // namespace detail

        //============================================================================
        // Method Description:
        /// Single random value sampled from the "standard normal" distrubution with
        /// mean = 0 and std = 1
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_normal.html#numpy.random.standard_normal
        ///
        /// @return NdArray
        ///
        template<typename dtype>
        dtype standardNormal()
        {
            return detail::standardNormal<dtype>(generator_);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from a "standard normal" distrubution with
        /// mean = 0 and std = 1
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_normal.html#numpy.random.standard_normal
        ///
        /// @param inShape
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<dtype> standardNormal(const Shape& inShape)
        {
            return detail::standardNormal<dtype>(generator_, inShape);
        }
    } // namespace random
} // namespace nc
