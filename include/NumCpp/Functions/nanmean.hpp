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

#include <algorithm>
#include <cmath>

#include "NumCpp/Core/DtypeInfo.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/max.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Compute the mean along the specified axis ignoring NaNs.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanmean.html
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> nanmean(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_FLOAT(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                auto sum = static_cast<double>(
                    std::accumulate(inArray.cbegin(),
                                    inArray.cend(),
                                    0.,
                                    [](dtype inValue1, dtype inValue2) -> dtype
                                    { return std::isnan(inValue2) ? inValue1 : inValue1 + inValue2; }));

                const auto numberNonNan =
                    static_cast<double>(std::accumulate(inArray.cbegin(),
                                                        inArray.cend(),
                                                        0.,
                                                        [](dtype inValue1, dtype inValue2) -> dtype
                                                        { return std::isnan(inValue2) ? inValue1 : inValue1 + 1; }));

                NdArray<double> returnArray = { sum /= numberNonNan };

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape     inShape = inArray.shape();
                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    auto sum = static_cast<double>(
                        std::accumulate(inArray.cbegin(row),
                                        inArray.cend(row),
                                        0.,
                                        [](dtype inValue1, dtype inValue2) -> dtype
                                        { return std::isnan(inValue2) ? inValue1 : inValue1 + inValue2; }));

                    auto numberNonNan = static_cast<double>(
                        std::accumulate(inArray.cbegin(row),
                                        inArray.cend(row),
                                        0.,
                                        [](dtype inValue1, dtype inValue2) -> dtype
                                        { return std::isnan(inValue2) ? inValue1 : inValue1 + 1; }));

                    returnArray(0, row) = sum / numberNonNan;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return nanmean(inArray.transpose(), Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
} // namespace nc
