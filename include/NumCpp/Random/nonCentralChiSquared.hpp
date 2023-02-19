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
/// "non central chi squared" distrubution.
///
#pragma once

#ifndef NUMCPP_NO_USE_BOOST

#include <algorithm>
#include <string>

#include "boost/random/non_central_chi_squared_distribution.hpp"

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
        /// Single random value sampled from the "non central chi squared" distrubution.
        /// NOTE: Use of this function requires using the Boost includes.
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.noncentral_chisquare.html#numpy.random.noncentral_chisquare
        ///
        /// @param generator: instance of a random number generator
        /// @param inK (default 1)
        /// @param inLambda (default 1)
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        dtype nonCentralChiSquared(GeneratorType& generator, dtype inK = 1, dtype inLambda = 1)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (inK <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input k must be greater than zero.");
            }

            if (inLambda <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input lambda must be greater than zero.");
            }

            boost::random::non_central_chi_squared_distribution<dtype> dist(inK, inLambda);
            return dist(generator);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from a "non central chi squared" distrubution.
        /// NOTE: Use of this function requires using the Boost includes.
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.noncentral_chisquare.html#numpy.random.noncentral_chisquare
        ///
        /// @param generator: instance of a random number generator
        /// @param inShape
        /// @param inK (default 1)
        /// @param inLambda (default 1)
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        NdArray<dtype>
            nonCentralChiSquared(GeneratorType& generator, const Shape& inShape, dtype inK = 1, dtype inLambda = 1)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (inK <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input k must be greater than zero.");
            }

            if (inLambda <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input lambda must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::non_central_chi_squared_distribution<dtype> dist(inK, inLambda);

            std::for_each(returnArray.begin(),
                          returnArray.end(),
                          [&generator, &dist](dtype& value) -> void { value = dist(generator); });

            return returnArray;
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Single random value sampled from the "non central chi squared" distrubution.
    /// NOTE: Use of this function requires using the Boost includes.
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.noncentral_chisquare.html#numpy.random.noncentral_chisquare
    ///
    /// @param inK (default 1)
    /// @param inLambda (default 1)
    /// @return NdArray
    ///
    template<typename dtype>
    dtype nonCentralChiSquared(dtype inK = 1, dtype inLambda = 1)
    {
        return detail::nonCentralChiSquared(generator_, inK, inLambda);
    }

    //============================================================================
    // Method Description:
    /// Create an array of the given shape and populate it with
    /// random samples from a "non central chi squared" distrubution.
    /// NOTE: Use of this function requires using the Boost includes.
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.noncentral_chisquare.html#numpy.random.noncentral_chisquare
    ///
    /// @param inShape
    /// @param inK (default 1)
    /// @param inLambda (default 1)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> nonCentralChiSquared(const Shape& inShape, dtype inK = 1, dtype inLambda = 1)
    {
        return detail::nonCentralChiSquared(generator_, inShape, inK, inLambda);
    }
} // namespace nc::random

#endif // #ifndef NUMCPP_NO_USE_BOOST
