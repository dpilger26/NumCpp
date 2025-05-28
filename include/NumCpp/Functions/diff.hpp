/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
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
/// Functions for working with NdArrays
///
#pragma once

#include <string>

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Calculate the n-th discrete difference along given axis.
    /// Unsigned dtypes will give you weird results...obviously.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.diff.html
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> diff(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const Shape inShape = inArray.shape();

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inArray.size() < 2)
                {
                    return NdArray<dtype>(0);
                }

                NdArray<dtype> returnArray(1, inArray.size() - 1);
                stl_algorithms::transform(inArray.cbegin(),
                                          inArray.cend() - 1,
                                          inArray.cbegin() + 1,
                                          returnArray.begin(),
                                          [](dtype inValue1, dtype inValue2) noexcept -> dtype
                                          { return inValue2 - inValue1; });

                return returnArray;
            }
            case Axis::COL:
            {
                if (inShape.cols < 2)
                {
                    return NdArray<dtype>(0);
                }

                NdArray<dtype> returnArray(inShape.rows, inShape.cols - 1);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    stl_algorithms::transform(inArray.cbegin(row),
                                              inArray.cend(row) - 1,
                                              inArray.cbegin(row) + 1,
                                              returnArray.begin(row),
                                              [](dtype inValue1, dtype inValue2) noexcept -> dtype
                                              { return inValue2 - inValue1; });
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return diff(inArray.transpose(), Axis::COL).transpose();
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
} // namespace nc
