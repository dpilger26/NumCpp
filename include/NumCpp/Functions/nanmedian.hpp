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

#include <cmath>
#include <vector>

#include "NumCpp/Core/DtypeInfo.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/max.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Compute the median along the specified axis ignoring NaNs.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanmedian.html
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> nanmedian(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_FLOAT(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::vector<dtype> values;
                for (auto value : inArray)
                {
                    if (!std::isnan(value))
                    {
                        values.push_back(value);
                    }
                }

                const uint32 middle = static_cast<uint32>(values.size()) / 2;
                stl_algorithms::nth_element(values.begin(), values.begin() + middle, values.end());
                NdArray<dtype> returnArray = { values[middle] };

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape    inShape = inArray.shape();
                NdArray<dtype> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    std::vector<dtype> values;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (!std::isnan(inArray(row, col)))
                        {
                            values.push_back(inArray(row, col));
                        }
                    }

                    const uint32 middle = static_cast<uint32>(values.size()) / 2;
                    stl_algorithms::nth_element(values.begin(), values.begin() + middle, values.end());
                    returnArray(0, row) = values[middle];
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return nanmedian(inArray.transpose(), Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
} // namespace nc
