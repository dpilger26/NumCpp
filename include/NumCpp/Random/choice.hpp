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
/// Chooses a random sample from an input array.
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/permutation.hpp"
#include "NumCpp/Random/randInt.hpp"

#include <algorithm>

namespace nc
{
    namespace random
    {
        //============================================================================
        // Method Description:
        ///						Chooses a random sample from an input array.
        ///
        /// @param      inArray
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        dtype choice(const NdArray<dtype>& inArray)
        {
            uint32 randIdx = random::randInt<uint32>(0, inArray.size());
            return inArray[randIdx];
        }

        //============================================================================
        // Method Description:
        ///						Chooses inNum random samples from an input array.
        ///
        /// @param      inArray
        /// @param      inNum
        /// @param      replace: Whether the sample is with or without replacement
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> choice(const NdArray<dtype>& inArray, uint32 inNum, bool replace = true)
        {
            if (!replace && inNum > inArray.size())
            {
                THROW_INVALID_ARGUMENT_ERROR("when 'replace' == false 'inNum' must be <= inArray.size()");
            }

            if (replace)
            {
                NdArray<dtype> outArray(1, inNum);
                stl_algorithms::for_each(outArray.begin(), outArray.end(),
                    [&inArray](dtype& value) -> void
                    { 
                        value = choice(inArray); 
                    });

                return outArray;
            }

            return permutation(inArray)[Slice(inNum)];
        }
    }  // namespace random
}  // namespace nc
