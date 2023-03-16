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
/// "beta" distribution.
///
#pragma once

#ifndef NUMCPP_NO_USE_BOOST

#include <algorithm>
#include <string>

#include "boost/random/beta_distribution.hpp"

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
        /// Single random value sampled from the from the "beta" distribution.
        /// NOTE: Use of this function requires using the Boost includes.
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.beta.html#numpy.random.beta
        ///
        /// @param generator: instance of a random number generator
        /// @param inAlpha
        /// @param inBeta
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        dtype beta(GeneratorType& generator, dtype inAlpha, dtype inBeta)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (inAlpha < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input alpha must be greater than zero.");
            }

            if (inBeta < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input beta must be greater than zero.");
            }

            boost::random::beta_distribution<dtype> dist(inAlpha, inBeta);
            return dist(generator);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from the "beta" distribution.
        /// NOTE: Use of this function requires using the Boost includes.
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.beta.html#numpy.random.beta
        ///
        /// @param generator: instance of a random number generator
        /// @param inShape
        /// @param inAlpha
        /// @param inBeta
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        NdArray<dtype> beta(GeneratorType& generator, const Shape& inShape, dtype inAlpha, dtype inBeta)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (inAlpha < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input alpha must be greater than zero.");
            }

            if (inBeta < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input beta must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::beta_distribution<dtype> dist(inAlpha, inBeta);

            std::for_each(returnArray.begin(),
                          returnArray.end(),
                          [&generator, &dist](dtype& value) -> void { value = dist(generator); });

            return returnArray;
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Single random value sampled from the from the "beta" distribution.
    /// NOTE: Use of this function requires using the Boost includes.
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.beta.html#numpy.random.beta
    ///
    /// @param inAlpha
    /// @param inBeta
    /// @return NdArray
    ///
    template<typename dtype>
    dtype beta(dtype inAlpha, dtype inBeta)
    {
        return detail::beta(generator_, inAlpha, inBeta);
    }

    //============================================================================
    // Method Description:
    /// Create an array of the given shape and populate it with
    /// random samples from the "beta" distribution.
    /// NOTE: Use of this function requires using the Boost includes.
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.beta.html#numpy.random.beta
    ///
    /// @param inShape
    /// @param inAlpha
    /// @param inBeta
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> beta(const Shape& inShape, dtype inAlpha, dtype inBeta)
    {
        return detail::beta(generator_, inShape, inAlpha, inBeta);
    }
} // namespace nc::random

#endif // #ifndef NUMCPP_NO_USE_BOOST
