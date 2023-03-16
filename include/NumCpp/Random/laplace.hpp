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
/// "laplace" distrubution.
///
#pragma once

#ifndef NUMCPP_NO_USE_BOOST

#include <algorithm>

#include "boost/random/laplace_distribution.hpp"

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
        /// Single random value sampled from the "laplace" distrubution.
        /// NOTE: Use of this function requires using the Boost includes.
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html#numpy.random.laplace
        ///
        /// @param generator: instance of a random number generator
        /// @param inLoc: (The position, mu, of the distribution peak. Default is 0)
        /// @param inScale: (float optional the exponential decay. Default is 1)
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        dtype laplace(GeneratorType& generator, dtype inLoc = 0, dtype inScale = 1)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            boost::random::laplace_distribution<dtype> dist(inLoc, inScale);
            return dist(generator);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from a "laplace" distrubution.
        /// NOTE: Use of this function requires using the Boost includes.
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html#numpy.random.laplace
        ///
        /// @param generator: instance of a random number generator
        /// @param inShape
        /// @param inLoc: (The position, mu, of the distribution peak. Default is 0)
        /// @param inScale: (float optional the exponential decay. Default is 1)
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        NdArray<dtype> laplace(GeneratorType& generator, const Shape& inShape, dtype inLoc = 0, dtype inScale = 1)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            NdArray<dtype> returnArray(inShape);

            boost::random::laplace_distribution<dtype> dist(inLoc, inScale);

            std::for_each(returnArray.begin(),
                          returnArray.end(),
                          [&generator, &dist](dtype& value) -> void { value = dist(generator); });

            return returnArray;
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Single random value sampled from the "laplace" distrubution.
    /// NOTE: Use of this function requires using the Boost includes.
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html#numpy.random.laplace
    ///
    /// @param inLoc: (The position, mu, of the distribution peak. Default is 0)
    /// @param inScale: (float optional the exponential decay. Default is 1)
    /// @return NdArray
    ///
    template<typename dtype>
    dtype laplace(dtype inLoc = 0, dtype inScale = 1)
    {
        return detail::laplace(generator_, inLoc, inScale);
    }

    //============================================================================
    // Method Description:
    /// Create an array of the given shape and populate it with
    /// random samples from a "laplace" distrubution.
    /// NOTE: Use of this function requires using the Boost includes.
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html#numpy.random.laplace
    ///
    /// @param inShape
    /// @param inLoc: (The position, mu, of the distribution peak. Default is 0)
    /// @param inScale: (float optional the exponential decay. Default is 1)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> laplace(const Shape& inShape, dtype inLoc = 0, dtype inScale = 1)
    {
        return detail::laplace(generator_, inShape, inLoc, inScale);
    }
} // namespace nc::random

#endif // #ifndef NUMCPP_NO_USE_BOOST
