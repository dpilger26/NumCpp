/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
///
/// License
/// Copyright 2018-2025 David Pilger
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
/// Modify a sequence in-place by shuffling its contents.
///
#pragma once

#include <algorithm>

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/generator.hpp"

namespace nc::random
{
    namespace detail
    {
        //============================================================================
        // Method Description:
        /// Modify a sequence in-place by shuffling its contents.
        ///
        /// @param generator: instance of a random number generator
        /// @param inArray
        ///
        template<typename dtype, typename GeneratorType = std::mt19937>
        void shuffle(GeneratorType& generator, NdArray<dtype>& inArray)
        {
            std::shuffle(inArray.begin(), inArray.end(), generator);
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Modify a sequence in-place by shuffling its contents.
    ///
    /// @param inArray
    ///
    template<typename dtype>
    void shuffle(NdArray<dtype>& inArray)
    {
        return detail::shuffle(generator_, inArray);
    }
} // namespace nc::random
