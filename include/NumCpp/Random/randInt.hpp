/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
/// Copyright 2019 David Pilger
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
/// A module for generating random numbers
///
#pragma once

#include "NumCpp/Core/DtypeInfo.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/generator.hpp"

#include "boost/random/uniform_int_distribution.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace nc
{
    namespace random
    {
        //============================================================================
        // Method Description:
        ///						Return random integers from low (inclusive) to high (exclusive),
        ///						with the given shape
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html#numpy.random.randint
        ///
        /// @param				inShape
        /// @param				inLow
        /// @param				inHigh
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> randInt(const Shape& inShape, dtype inLow, dtype inHigh)
        {
            // only works with integer input types
            static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: randInt: can only use with integer types.");

            if (inLow == inHigh)
            {
                std::string errStr = "Error: randint: input low value must be less than the input high value.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }
            else if (inLow > inHigh - 1)
            {
                std::swap(inLow, inHigh);
            }

            NdArray<dtype> returnArray(inShape);

            const boost::random::uniform_int_distribution<dtype> dist(inLow, inHigh - 1);

            std::for_each(returnArray.begin(), returnArray.end(), 
                [&dist](dtype& value) noexcept -> void
                { value = dist(generator_); });

            return returnArray;
        }
    }
}
