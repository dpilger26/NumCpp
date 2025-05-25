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

#include <vector>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Functions/hsplit.hpp"
#include "NumCpp/Functions/vsplit.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Split an array into multiple sub-arrays as views into array.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.split.html
    ///
    /// @param inArray
    /// @param indices: the indices to split
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return NdArray
    ///
    template<typename dtype, NdArrayInt Indices>
    std::vector<NdArray<dtype>> split(const NdArray<dtype>& inArray, const Indices& indices, Axis inAxis = Axis::ROW)
    {
        switch (inAxis)
        {
            case Axis::ROW:
            {
                return vsplit(inArray, indices);
            }
            case Axis::COL:
            {
                return hsplit(inArray, indices);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("input inAxis must be either Axis::ROW or Axis::COL");
            }
        }

        return {}; // get rid of compiler warning
    }
} // namespace nc
