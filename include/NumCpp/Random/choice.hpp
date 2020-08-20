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

#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/randInt.hpp"

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
            uint32 randIdx = random::randInt<uint32>(Shape(1), 0, inArray.size()).item();
            return inArray[randIdx];
        }

        //============================================================================
        // Method Description:
        ///						Chooses inNum random samples from an input array. Samples
        ///                     are in no way guarunteed to be unique.
        ///
        /// @param      inArray
        /// @param      inNum
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> choice(const NdArray<dtype>& inArray, uint32 inNum)
        {
            NdArray<dtype> outArray(1, inNum);
            for (uint32 i = 0; i < inNum; ++i)
            {
                uint32 randIdx = random::randInt<uint32>(Shape(1), 0, inArray.size()).item();
                outArray[i] = inArray[randIdx];
            }

            return outArray;
        }
    }  // namespace random
}  // namespace nc
