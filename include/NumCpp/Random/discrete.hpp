/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.0.0
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
/// "discrete" distrubution.
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Random/generator.hpp"

#include "boost/random/discrete_distribution.hpp"


namespace nc
{
    namespace random
    {
        //============================================================================
        // Method Description:
        ///						Single random value sampled from the from the 
        ///                     "discrete" distrubution.  It produces integers in the 
        ///                     range [0, n) with the probability of producing each value
        ///                     is specified by the parameters of the distribution.
        ///
        ///	@param		inWeights
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        dtype discrete(const NdArray<double>& inWeights)
        {
            STATIC_ASSERT_INTEGER(dtype);

            boost::random::discrete_distribution<dtype> dist(inWeights.cbegin(), inWeights.cend());
            return dist(generator_);
        }

        //============================================================================
        // Method Description:
        ///						Create an array of the given shape and populate it with
        ///						random samples from a "discrete" distrubution.  It produces
        ///						integers in the range [0, n) with the probability of
        ///						producing each value is specified by the parameters
        ///						of the distribution.
        ///
        /// @param      inShape
        ///	@param		inWeights
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> discrete(const Shape& inShape, const NdArray<double>& inWeights)
        {
            STATIC_ASSERT_INTEGER(dtype);

            NdArray<dtype> returnArray(inShape);

            boost::random::discrete_distribution<dtype> dist(inWeights.cbegin(), inWeights.cend());

            stl_algorithms::for_each(returnArray.begin(), returnArray.end(),
                [&dist](dtype& value)  -> void
                { 
                    value = dist(generator_);
                });

            return returnArray;
        }
    }
}
