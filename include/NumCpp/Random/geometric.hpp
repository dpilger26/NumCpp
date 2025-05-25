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
/// "geometric" distrubution.
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
#include "NumCpp/Core/Internal/Concepts.hpp"

namespace nc::random
{
    namespace detail
    {
        //============================================================================
        // Method Description:
        /// Single random value sampled from the "geometric" distrubution.
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.geometric.html#numpy.random.geometric
        ///
        /// @param generator: instance of a random number generator
        /// @param inP (probablity of success [0, 1])
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        dtype geometric(GeneratorType& generator, double inP = 0.5)
        {
            if (inP < 0 || inP > 1)
            {
                THROW_INVALID_ARGUMENT_ERROR("input probability of sucess must be of the range [0, 1].");
            }

            std::geometric_distribution<dtype> dist(inP);
            return dist(generator);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from a "geometric" distrubution.
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.geometric.html#numpy.random.geometric
        ///
        /// @param generator: instance of a random number generator
        /// @param inShape
        /// @param inP (probablity of success [0, 1])
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        NdArray<dtype> geometric(GeneratorType& generator, const Shape& inShape, double inP = 0.5)
        {
            if (inP < 0 || inP > 1)
            {
                THROW_INVALID_ARGUMENT_ERROR("input probability of sucess must be of the range [0, 1].");
            }

            NdArray<dtype> returnArray(inShape);

            std::geometric_distribution<dtype> dist(inP);

            std::for_each(returnArray.begin(),
                          returnArray.end(),
                          [&generator, &dist](dtype& value) -> void { value = dist(generator); });

            return returnArray;
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Single random value sampled from the "geometric" distrubution.
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.geometric.html#numpy.random.geometric
    ///
    /// @param inP (probablity of success [0, 1])
    /// @return NdArray
    ///
    template<nc::Integer dtype>
    dtype geometric(double inP = 0.5)
    {
        return detail::geometric<dtype>(generator_, inP);
    }

    //============================================================================
    // Method Description:
    /// Create an array of the given shape and populate it with
    /// random samples from a "geometric" distrubution.
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.geometric.html#numpy.random.geometric
    ///
    /// @param inShape
    /// @param inP (probablity of success [0, 1])
    /// @return NdArray
    ///
    template<nc::Integer dtype>
    NdArray<dtype> geometric(const Shape& inShape, double inP = 0.5)
    {
        return detail::geometric<dtype>(generator_, inShape, inP);
    }
} // namespace nc::random
