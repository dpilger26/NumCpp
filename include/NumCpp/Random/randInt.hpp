/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
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
/// Return random integers from low (inclusive) to high (exclusive),
///	with the given shape
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Random/generator.hpp"

#include "boost/random/uniform_int_distribution.hpp"

#include <string>

namespace nc
{
    namespace random
    {
        //============================================================================
        // Method Description:
        ///						Return random integer from low (inclusive) to high (exclusive),
        ///						with the given shape. If no high value is input then the range will 
        ///                     go from [0, low).
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html#numpy.random.randint
        ///
        /// @param				inLow
        /// @param				inHigh default 0.
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        dtype randInt(dtype inLow, dtype inHigh = 0)
        {
            STATIC_ASSERT_INTEGER(dtype);

            if (inLow == inHigh)
            {
                THROW_INVALID_ARGUMENT_ERROR("input low value must be less than the input high value.");
            }
            else if (inLow > inHigh - 1)
            {
                std::swap(inLow, inHigh);
            }

            const boost::random::uniform_int_distribution<dtype> dist(inLow, inHigh - 1);
            return dist(generator_);
        }

        //============================================================================
        // Method Description:
        ///						Return random integers from low (inclusive) to high (exclusive),
        ///						with the given shape. If no high value is input then the range will 
        ///                     go from [0, low).
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html#numpy.random.randint
        ///
        /// @param				inShape
        /// @param				inLow
        /// @param				inHigh default 0.
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> randInt(const Shape& inShape, dtype inLow, dtype inHigh = 0)
        {
            STATIC_ASSERT_INTEGER(dtype);

            if (inLow == inHigh)
            {
                THROW_INVALID_ARGUMENT_ERROR("input low value must be less than the input high value.");
            }
            else if (inLow > inHigh - 1)
            {
                std::swap(inLow, inHigh);
            }

            NdArray<dtype> returnArray(inShape);

            const boost::random::uniform_int_distribution<dtype> dist(inLow, inHigh - 1);

            stl_algorithms::for_each(returnArray.begin(), returnArray.end(),
                [&dist](dtype& value) noexcept -> void
                { 
                    value = dist(generator_); 
                });

            return returnArray;
        }
    }
}
