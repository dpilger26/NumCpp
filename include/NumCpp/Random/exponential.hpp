/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.0.0
///
/// @section License
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
/// "exponential" distrubution
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Random/generator.hpp"

#include "boost/random/exponential_distribution.hpp"

#include <memory>

namespace nc
{
    namespace random
    {
        //============================================================================
        // Method Description:
        ///						Single random value sampled from the "exponential" distrubution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential
        ///
        /// @param				inScaleValue (default 1)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        dtype exponential(dtype inScaleValue = 1) noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            const boost::random::exponential_distribution<dtype> dist(inScaleValue);
            return dist(generator_); 
        }

        //============================================================================
        // Method Description:
        ///						Create an array of the given shape and populate it with
        ///						random samples from a "exponential" distrubution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential
        ///
        /// @param				inShape
        /// @param				inScaleValue (default 1)
        /// @return
        ///				NdArray
        ///
        template<typename dtype, class Alloc = std::allocator<dtype>>
        NdArray<dtype, Alloc> exponential(const Shape& inShape, dtype inScaleValue = 1) noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            NdArray<dtype, Alloc> returnArray(inShape);

            const boost::random::exponential_distribution<dtype> dist(inScaleValue);

            stl_algorithms::for_each(returnArray.begin(), returnArray.end(),
                [&dist](dtype& value) noexcept -> void
                {
                    value = dist(generator_); 
                });

            return returnArray;
        }
    }
}
