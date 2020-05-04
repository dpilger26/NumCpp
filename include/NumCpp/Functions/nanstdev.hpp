/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.4
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

#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/nanmean.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/sqr.hpp"

#include <algorithm>
#include <cmath>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Compute the standard deviation along the specified axis, while ignoring NaNs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanstd.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> nanstdev(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE) noexcept
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                double meanValue = nanmean(inArray, inAxis).item();
                double sum = 0;
                double counter = 0;
                for (auto value : inArray)
                {
                    if (std::isnan(value))
                    {
                        continue;
                    }

                    sum += utils::sqr(static_cast<double>(value) - meanValue);
                    ++counter;
                }
                NdArray<double> returnArray = { std::sqrt(sum / counter) };
                return returnArray;
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();
                NdArray<double> meanValue = nanmean(inArray, inAxis);
                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    double sum = 0;
                    double counter = 0;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (std::isnan(inArray(row, col)))
                        {
                            continue;
                        }

                        sum += utils::sqr(static_cast<double>(inArray(row, col)) - meanValue[row]);
                        ++counter;
                    }
                    returnArray(0, row) = std::sqrt(sum / counter);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<double> meanValue = nanmean(inArray, inAxis);
                NdArray<dtype> transposedArray = inArray.transpose();
                const Shape inShape = transposedArray.shape();
                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    double sum = 0;
                    double counter = 0;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (std::isnan(transposedArray(row, col)))
                        {
                            continue;
                        }

                        sum += utils::sqr(static_cast<double>(transposedArray(row, col)) - meanValue[row]);
                        ++counter;
                    }
                    returnArray(0, row) = std::sqrt(sum / counter);
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<double>(0);
            }
        }
    }

}
