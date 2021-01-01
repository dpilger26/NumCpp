/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
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
/// Description
/// "chi square" distribution.
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/generator.hpp"

#include <random>
#include <string>

namespace nc
{
    namespace random
    {
        //============================================================================
        // Method Description:
        ///						Single random value sampled from the from the "chi square" distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.chisquare.html#numpy.random.chisquare
        ///
        /// @param				inDof (independent random variables)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        dtype chiSquare(dtype inDof)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (inDof <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("numerator degrees of freedom must be greater than zero.");
            }

            std::chi_squared_distribution<dtype> dist(inDof);
            return dist(generator_);
        }

        //============================================================================
        // Method Description:
        ///						Create an array of the given shape and populate it with
        ///						random samples from the "chi square" distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.chisquare.html#numpy.random.chisquare
        ///
        /// @param				inShape
        /// @param				inDof (independent random variables)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> chiSquare(const Shape& inShape, dtype inDof)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (inDof <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("numerator degrees of freedom must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            std::chi_squared_distribution<dtype> dist(inDof);

            stl_algorithms::for_each(returnArray.begin(), returnArray.end(),
                [&dist](dtype& value)  -> void
                { 
                    value = dist(generator_);
                });

            return returnArray;
        }
    }  // namespace random
} // namespace nc
