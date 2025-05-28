/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2025 David Pilger
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
/// "extreme value" distrubution.
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

namespace nc::random
{
    namespace detail
    {
        //============================================================================
        // Method Description:
        /// Single random value sampled from the "extreme value" distrubution.
        ///
        /// @param generator: instance of a random number generator
        /// @param inA (default 1)
        /// @param inB (default 1)
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        dtype extremeValue(GeneratorType& generator, dtype inA = 1, dtype inB = 1)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (inA <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input a must be greater than zero.");
            }

            if (inB <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input b must be greater than zero.");
            }

            std::extreme_value_distribution<dtype> dist(inA, inB);
            return dist(generator);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from a "extreme value" distrubution.
        ///
        /// @param generator: instance of a random number generator
        /// @param inShape
        /// @param inA (default 1)
        /// @param inB (default 1)
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        NdArray<dtype> extremeValue(GeneratorType& generator, const Shape& inShape, dtype inA = 1, dtype inB = 1)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (inA <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input a must be greater than zero.");
            }

            if (inB <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input b must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            std::extreme_value_distribution<dtype> dist(inA, inB);

            std::for_each(returnArray.begin(),
                          returnArray.end(),
                          [&generator, &dist](dtype& value) -> void { value = dist(generator); });

            return returnArray;
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Single random value sampled from the "extreme value" distrubution.
    ///
    /// @param inA (default 1)
    /// @param inB (default 1)
    /// @return NdArray
    ///
    template<typename dtype>
    dtype extremeValue(dtype inA = 1, dtype inB = 1)
    {
        return detail::extremeValue<dtype>(generator_, inA, inB);
    }

    //============================================================================
    // Method Description:
    /// Create an array of the given shape and populate it with
    /// random samples from a "extreme value" distrubution.
    ///
    /// @param inShape
    /// @param inA (default 1)
    /// @param inB (default 1)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> extremeValue(const Shape& inShape, dtype inA = 1, dtype inB = 1)
    {
        return detail::extremeValue<dtype>(generator_, inShape, inA, inB);
    }
} // namespace nc::random
