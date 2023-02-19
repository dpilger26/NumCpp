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
/// Create an array of the given shape and populate it with
/// random samples from a uniform distribution over [0, 1).
///
#pragma once

#include <algorithm>
#include <random>

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
        /// Single random value sampled from the uniform distribution over [0, 1).
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html#numpy.random.rand
        ///
        /// @param generator: instance of a random number generator
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        dtype rand(GeneratorType& generator)
        {
            STATIC_ASSERT_FLOAT(dtype);

            std::uniform_real_distribution<dtype> dist(static_cast<dtype>(0.),
                                                       static_cast<dtype>(1.) - DtypeInfo<dtype>::epsilon());
            return dist(generator);
        }

        //============================================================================
        // Method Description:
        /// Create an array of the given shape and populate it with
        /// random samples from a uniform distribution over [0, 1).
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html#numpy.random.rand
        ///
        /// @param generator: instance of a random number generator
        /// @param inShape
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        NdArray<dtype> rand(GeneratorType& generator, const Shape& inShape)
        {
            STATIC_ASSERT_FLOAT(dtype);

            NdArray<dtype> returnArray(inShape);

            std::uniform_real_distribution<dtype> dist(static_cast<dtype>(0.),
                                                       static_cast<dtype>(1.) - DtypeInfo<dtype>::epsilon());

            std::for_each(returnArray.begin(),
                          returnArray.end(),
                          [&generator, &dist](dtype& value) -> void { value = dist(generator); });

            return returnArray;
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Single random value sampled from the uniform distribution over [0, 1).
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html#numpy.random.rand
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    dtype rand()
    {
        return detail::rand<dtype>(generator_);
    }

    //============================================================================
    // Method Description:
    /// Create an array of the given shape and populate it with
    /// random samples from a uniform distribution over [0, 1).
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html#numpy.random.rand
    ///
    /// @param inShape
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> rand(const Shape& inShape)
    {
        return detail::rand<dtype>(generator_, inShape);
    }
} // namespace nc::random
