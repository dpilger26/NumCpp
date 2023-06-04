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

#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Append values to the end of an array.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.append.html
    ///
    /// @param inArray
    /// @param inAppendValues
    /// @param inAxis (Optional, default NONE): The axis along which values are appended.
    /// If axis is not given, both inArray and inAppendValues
    /// are flattened before use.
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> append(const NdArray<dtype>& inArray, const NdArray<dtype>& inAppendValues, Axis inAxis = Axis::NONE)
    {
        if (inArray.shape().isnull())
        {
            return inAppendValues;
        }
        else if (inAppendValues.shape().isnull())
        {
            return inArray;
        }

        switch (inAxis)
        {
            case Axis::NONE:
            {
                NdArray<dtype> returnArray(1, inArray.size() + inAppendValues.size());
                stl_algorithms::copy(inArray.cbegin(), inArray.cend(), returnArray.begin());
                stl_algorithms::copy(inAppendValues.cbegin(),
                                     inAppendValues.cend(),
                                     returnArray.begin() + inArray.size());

                return returnArray;
            }
            case Axis::ROW:
            {
                const Shape inShape     = inArray.shape();
                const Shape appendShape = inAppendValues.shape();
                if (inShape.cols != appendShape.cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "all the input array dimensions except for the concatenation axis must match exactly");
                }

                NdArray<dtype> returnArray(inShape.rows + appendShape.rows, inShape.cols);
                stl_algorithms::copy(inArray.cbegin(), inArray.cend(), returnArray.begin());
                stl_algorithms::copy(inAppendValues.cbegin(),
                                     inAppendValues.cend(),
                                     returnArray.begin() + inArray.size());

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape inShape     = inArray.shape();
                const Shape appendShape = inAppendValues.shape();
                if (inShape.rows != appendShape.rows)
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "all the input array dimensions except for the concatenation axis must match exactly");
                }

                NdArray<dtype> returnArray(inShape.rows, inShape.cols + appendShape.cols);
                for (uint32 row = 0; row < returnArray.shape().rows; ++row)
                {
                    stl_algorithms::copy(inArray.cbegin(row), inArray.cend(row), returnArray.begin(row));
                    stl_algorithms::copy(inAppendValues.cbegin(row),
                                         inAppendValues.cend(row),
                                         returnArray.begin(row) + inShape.cols);
                }

                return returnArray;
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
} // namespace nc
