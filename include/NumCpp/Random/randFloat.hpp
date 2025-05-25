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
/// Return random floats from low (inclusive) to high (exclusive),
/// with the given shape
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
#include "NumCpp/Utils/essentiallyEqual.hpp"
#include "NumCpp/Core/Internal/Concepts.hpp"

namespace nc::random
{
    namespace detail
    {
        //============================================================================
        // Method Description:
        /// Return a single random float from low (inclusive) to high (exclusive),
        /// with the given shape. If no high value is input then the range will
        /// go from [0, low).
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.ranf.html#numpy.random.ranf
        ///
        /// @param generator: instance of a random number generator
        /// @param inLow
        /// @param inHigh default 0.
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        dtype randFloat(GeneratorType& generator, dtype inLow, dtype inHigh = 0.)
        {
            if (utils::essentiallyEqual(inLow, inHigh))
            {
                THROW_INVALID_ARGUMENT_ERROR("input low value must be less than the input high value.");
            }
            else if (inLow > inHigh)
            {
                std::swap(inLow, inHigh);
            }

            std::uniform_real_distribution<dtype> dist(inLow, inHigh - DtypeInfo<dtype>::epsilon());
            return dist(generator);
        }

        //============================================================================
        // Method Description:
        /// Return random floats from low (inclusive) to high (exclusive),
        /// with the given shape. If no high value is input then the range will
        /// go from [0, low).
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.ranf.html#numpy.random.ranf
        ///
        /// @param generator: instance of a random number generator
        /// @param inShape
        /// @param inLow
        /// @param inHigh default 0.
        /// @return NdArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        NdArray<dtype> randFloat(GeneratorType& generator, const Shape& inShape, dtype inLow, dtype inHigh = 0.)
        {
            if (utils::essentiallyEqual(inLow, inHigh))
            {
                THROW_INVALID_ARGUMENT_ERROR("input low value must be less than the input high value.");
            }
            else if (inLow > inHigh)
            {
                std::swap(inLow, inHigh);
            }

            NdArray<dtype> returnArray(inShape);

            std::uniform_real_distribution<dtype> dist(inLow, inHigh - DtypeInfo<dtype>::epsilon());

            std::for_each(returnArray.begin(),
                          returnArray.end(),
                          [&generator, &dist](dtype& value) -> void { value = dist(generator); });

            return returnArray;
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Return a single random float from low (inclusive) to high (exclusive),
    /// with the given shape. If no high value is input then the range will
    /// go from [0, low).
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.ranf.html#numpy.random.ranf
    ///
    /// @param inLow
    /// @param inHigh default 0.
    /// @return NdArray
    ///
    template<nc::Float dtype>
    dtype randFloat(dtype inLow, dtype inHigh = 0.)
    {
        return detail::randFloat(generator_, inLow, inHigh);
    }

    //============================================================================
    // Method Description:
    /// Return random floats from low (inclusive) to high (exclusive),
    /// with the given shape. If no high value is input then the range will
    /// go from [0, low).
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.ranf.html#numpy.random.ranf
    ///
    /// @param inShape
    /// @param inLow
    /// @param inHigh default 0.
    /// @return NdArray
    ///
    template<nc::Float dtype>
    NdArray<dtype> randFloat(const Shape& inShape, dtype inLow, dtype inHigh = 0.)
    {
        return detail::randFloat(generator_, inShape, inLow, inHigh);
    }
} // namespace nc::random
