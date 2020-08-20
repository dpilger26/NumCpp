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
/// Randomly permute a sequence, or return a permuted range
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/generator.hpp"

#include <algorithm>

namespace nc
{
    namespace random
    {
        //============================================================================
        // Method Description:
        ///						Randomly permute a sequence, or return a permuted range.
        ///						If x is an integer, randomly permute np.arange(x).
        ///						If x is an array, make a copy and shuffle the elements randomly.
        ///
        /// @param
        ///				inValue
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> permutation(dtype inValue) 
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            NdArray<dtype> returnArray = arange(inValue);
            std::shuffle(returnArray.begin(), returnArray.end(), generator_);
            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						Randomly permute a sequence, or return a permuted range.
        ///						If x is an integer, randomly permute np.arange(x).
        ///						If x is an array, make a copy and shuffle the elements randomly.
        ///
        /// @param
        ///				inArray
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> permutation(const NdArray<dtype>& inArray) 
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            NdArray<dtype> returnArray(inArray);
            std::shuffle(returnArray.begin(), returnArray.end(), generator_);
            return returnArray;
        }
    }  // namespace random
} // namespace nc
