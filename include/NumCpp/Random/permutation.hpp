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
/// Randomly permute a sequence, or return a permuted range
///
#pragma once

#include <algorithm>

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Functions/arange.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/generator.hpp"

namespace nc
{
    namespace random
    {
        namespace detail
        {
            //============================================================================
            // Method Description:
            /// Randomly permute a sequence, or return a permuted range.
            /// If x is an integer, randomly permute np.arange(x).
            /// If x is an array, make a copy and shuffle the elements randomly.
            ///
            /// @param generator: instance of a random number generator
            /// @param inValue
            /// @return NdArray
            ///
            template<typename dtype, typename GeneratorType = std::mt19937>
            NdArray<dtype> permutation(GeneratorType& generator, dtype inValue)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                NdArray<dtype> returnArray = arange(inValue);
                std::shuffle(returnArray.begin(), returnArray.end(), generator);
                return returnArray;
            }

            //============================================================================
            // Method Description:
            /// Randomly permute a sequence, or return a permuted range.
            /// If x is an integer, randomly permute np.arange(x).
            /// If x is an array, make a copy and shuffle the elements randomly.
            ///
            /// @param generator: instance of a random number generator
            /// @param inArray
            /// @return NdArray
            ///
            template<typename dtype, typename GeneratorType = std::mt19937>
            NdArray<dtype> permutation(GeneratorType& generator, const NdArray<dtype>& inArray)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                NdArray<dtype> returnArray(inArray);
                std::shuffle(returnArray.begin(), returnArray.end(), generator);
                return returnArray;
            }
        } // namespace detail

        //============================================================================
        // Method Description:
        /// Randomly permute a sequence, or return a permuted range.
        /// If x is an integer, randomly permute np.arange(x).
        /// If x is an array, make a copy and shuffle the elements randomly.
        ///
        /// @param inValue
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<dtype> permutation(dtype inValue)
        {
            return detail::permutation(generator_, inValue);
        }

        //============================================================================
        // Method Description:
        /// Randomly permute a sequence, or return a permuted range.
        /// If x is an integer, randomly permute np.arange(x).
        /// If x is an array, make a copy and shuffle the elements randomly.
        ///
        /// @param inArray
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<dtype> permutation(const NdArray<dtype>& inArray)
        {
            return detail::permutation(generator_, inArray);
        }
    } // namespace random
} // namespace nc
