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
/// "standard normal" distribution.
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/generator.hpp"

#include <algorithm>
#include <random>

namespace nc
{
    namespace random
    {
        //============================================================================
        // Method Description:
        /// Returns a single random value sampled from the "standard normal" distribution.
        ///
        /// NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html#numpy.random.randn
        ///
        /// @return dtype
        ///
        template<typename dtype>
        dtype randN() 
        {
            STATIC_ASSERT_FLOAT(dtype);

            std::normal_distribution<dtype> dist;
            return dist(generator_);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from the "standard normal" distribution.
        ///
        /// NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html#numpy.random.randn
        ///
        /// @param inShape
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<dtype> randN(const Shape& inShape) 
        {
            STATIC_ASSERT_FLOAT(dtype);

            NdArray<dtype> returnArray(inShape);

            std::normal_distribution<dtype> dist;

            std::for_each(returnArray.begin(), returnArray.end(),
                [&dist](dtype& value) -> void
                { 
                    value = dist(generator_);
                });

            return returnArray;
        }
    } // namespace random
} // namespace nc
