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
/// "discrete" distrubution.
///
#pragma once

#include <algorithm>
#include <random>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/generator.hpp"
#include "NumCpp/Core/Internal/Concepts.hpp"

namespace nc::random
{
    namespace detail
    {
        //============================================================================
        // Method Description:
        /// Single random value sampled from the from the
        /// "discrete" distrubution.  It produces integers in the
        /// range [0, n) with the probability of producing each value
        /// is specified by the parameters of the distribution.
        ///
        /// @param generator: instance of a random number generator
        /// @param inWeights
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        dtype discrete(GeneratorType& generator, const NdArray<double>& inWeights)
        {
            STATIC_ASSERT_INTEGER(dtype);

            std::discrete_distribution<dtype> dist(inWeights.cbegin(), inWeights.cend());
            return dist(generator);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from a "discrete" distrubution.  It produces
        /// integers in the range [0, n) with the probability of
        /// producing each value is specified by the parameters
        /// of the distribution.
        ///
        /// @param generator: instance of a random number generator
        /// @param inShape
        /// @param inWeights
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        NdArray<dtype> discrete(GeneratorType& generator, const Shape& inShape, const NdArray<double>& inWeights)
        {
            STATIC_ASSERT_INTEGER(dtype);

            NdArray<dtype> returnArray(inShape);

            std::discrete_distribution<dtype> dist(inWeights.cbegin(), inWeights.cend());

            std::for_each(returnArray.begin(),
                          returnArray.end(),
                          [&generator, &dist](dtype& value) -> void { value = dist(generator); });

            return returnArray;
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Single random value sampled from the from the
    /// "discrete" distrubution.  It produces integers in the
    /// range [0, n) with the probability of producing each value
    /// is specified by the parameters of the distribution.
    ///
    /// @param inWeights
    /// @return NdArray
    ///
    template<nc::Integer dtype>
    dtype discrete(const NdArray<double>& inWeights)
    {
        return detail::discrete<dtype>(generator_, inWeights);
    }

    //============================================================================
    // Method Description:
    /// Create an array of the given shape and populate it with
    /// random samples from a "discrete" distrubution.  It produces
    /// integers in the range [0, n) with the probability of
    /// producing each value is specified by the parameters
    /// of the distribution.
    ///
    /// @param inShape
    /// @param inWeights
    /// @return NdArray
    ///
    template<nc::Integer dtype>
    NdArray<dtype> discrete(const Shape& inShape, const NdArray<double>& inWeights)
    {
        return detail::discrete<dtype>(generator_, inShape, inWeights);
    }
} // namespace nc::random
