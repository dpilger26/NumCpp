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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Rearranges the elements in the array in such a way that
    ///						value of the element in kth position is in the position it
    ///						would be in a sorted array. All elements smaller than the kth
    ///						element are moved before this element and all equal or greater
    ///						are moved behind it. The ordering of the elements in the two
    ///						partitions is undefined.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.partition.html
    ///
    /// @param              inArray
    /// @param				inKth: kth element
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype, Alloc> partition(const NdArray<dtype, Alloc>& inArray, uint32 inKth, Axis inAxis = Axis::NONE)
    {
        NdArray<dtype, Alloc> returnArray(inArray);
        returnArray.partition(inKth, inAxis);
        return returnArray;
    }
}
