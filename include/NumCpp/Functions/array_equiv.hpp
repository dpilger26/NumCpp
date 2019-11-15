/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.2
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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/StlAlgorithms.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Returns True if input arrays are shape consistent and all elements equal.
    ///
    ///						Shape consistent means they are either the same shape, or one input array
    ///						can be broadcasted to create the same shape as the other one.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.array_equiv.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				bool
    ///
    template<typename dtype>
    bool array_equiv(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2) noexcept
    {
        if (inArray1.size() != inArray2.size())
        {
            return false;
        }

        return stl_algorithms::equal(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin());
    }
}
