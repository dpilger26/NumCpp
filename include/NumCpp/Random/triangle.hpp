/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.4
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
/// Create an array of the given shape and populate it with
///	random samples from the "triangle" distribution.
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/generator.hpp"

#include "boost/random/triangle_distribution.hpp"

#include <string>

namespace nc
{
    namespace random
    {
        //============================================================================
        // Method Description:
        ///						Single random value sampled from the "triangle" distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.triangular.html#numpy.random.triangular
        ///
        /// @param				inA
        /// @param				inB
        /// @param				inC
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        dtype triangle(dtype inA = 0, dtype inB = 0.5, dtype inC = 1)
        {
            if (inA < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input A must be greater than or equal to zero.");
            }

            if (inB < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input B must be greater than or equal to zero.");
            }

            if (inC < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input C must be greater than or equal to zero.");
            }

            const bool aLessB = inA <= inB;
            const bool bLessC = inB <= inC;
            if (!(aLessB && bLessC))
            {
                THROW_INVALID_ARGUMENT_ERROR("inputs must be a <= b <= c.");
            }

            boost::random::triangle_distribution<dtype> dist(inA, inB, inC);
            return dist(generator_);
        }

        //============================================================================
        // Method Description:
        ///						Create an array of the given shape and populate it with
        ///						random samples from the "triangle" distribution.
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.triangular.html#numpy.random.triangular
        ///
        /// @param				inShape
        /// @param				inA
        /// @param				inB
        /// @param				inC
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> triangle(const Shape& inShape, dtype inA = 0, dtype inB = 0.5, dtype inC = 1)
        {
            if (inA < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input A must be greater than or equal to zero.");
            }

            if (inB < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input B must be greater than or equal to zero.");
            }

            if (inC < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input C must be greater than or equal to zero.");
            }

            const bool aLessB = inA <= inB;
            const bool bLessC = inB <= inC;
            if (!(aLessB && bLessC))
            {
                THROW_INVALID_ARGUMENT_ERROR("inputs must be a <= b <= c.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::triangle_distribution<dtype> dist(inA, inB, inC);

            stl_algorithms::for_each(returnArray.begin(), returnArray.end(),
                [&dist](dtype& value) noexcept -> void
                {
                    value = dist(generator_);
                });

            return returnArray;
        }
    }
}
