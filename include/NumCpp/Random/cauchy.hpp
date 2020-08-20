/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
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
/// "cauchy" distrubution.
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/generator.hpp"

#include "boost/random/cauchy_distribution.hpp"

#include <string>

namespace nc
{
    namespace random
    {
        //============================================================================
        // Method Description:
        ///						Single random value sampled from the from the "cauchy" distrubution.
        ///
        /// @param				inMean: Mean value of the underlying normal distribution. Default is 0.
        /// @param				inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero. Default is 1.
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        dtype cauchy(dtype inMean = 0, dtype inSigma = 1)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (inSigma <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input sigma must be greater than zero.");
            }

            boost::random::cauchy_distribution<dtype> dist(inMean, inSigma);
            return dist(generator_);
        }

        //============================================================================
        // Method Description:
        ///						Create an array of the given shape and populate it with
        ///						random samples from a "cauchy" distrubution.
        ///
        /// @param              inShape
        /// @param				inMean: Mean value of the underlying normal distribution. Default is 0.
        /// @param				inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero. Default is 1.
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> cauchy(const Shape& inShape, dtype inMean = 0, dtype inSigma = 1)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (inSigma <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input sigma must be greater than zero.");
            }

            NdArray<dtype> returnArray(inShape);

            boost::random::cauchy_distribution<dtype> dist(inMean, inSigma);

            stl_algorithms::for_each(returnArray.begin(), returnArray.end(),
                [&dist](dtype& value)  -> void
                {
                    value = dist(generator_); 
                });

            return returnArray;
        }
    }  // namespace random
} // namespace nc
