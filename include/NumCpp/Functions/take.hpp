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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Evenly round to the given number of decimals.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.take.html
    ///
    /// @param inArray
    /// @param inIndices
    /// @param inAxis
    /// @return NdArray
    ///
    template<
        typename dtype,
        typename Indices,
        std::enable_if_t<std::is_same_v<Indices, NdArray<int32>> || std::is_same_v<Indices, NdArray<uint32>>, int> = 0>
    NdArray<dtype> take(const NdArray<dtype>& inArray, const Indices& inIndices, Axis inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                return inArray[inIndices];
            }
            case Axis::ROW:
            {
                return inArray(inIndices, inArray.cSlice());
            }
            case Axis::COL:
            {
                return inArray(inArray.rSlice(), inIndices);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }

} // namespace nc
