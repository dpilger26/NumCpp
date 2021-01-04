/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2021 David Pilger
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
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Integrate along the given axis using the composite trapezoidal rule.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.trapz.html
    ///
    /// @param				inArray
    /// @param              dx: (Optional defaults to 1.0)
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> trapz(const NdArray<dtype>& inArray, double dx = 1.0, Axis inAxis = Axis::NONE) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const Shape inShape = inArray.shape();
        switch (inAxis)
        {
            case Axis::COL:
            {
                NdArray<double> returnArray(inShape.rows, 1);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    double sum = 0;
                    for (uint32 col = 0; col < inShape.cols - 1; ++col)
                    {
                        sum += static_cast<double>(inArray(row, col + 1) - inArray(row, col)) / 2.0 +
                            static_cast<double>(inArray(row, col));
                    }

                    returnArray[row] = sum * dx;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> arrayTranspose = inArray.transpose();
                const Shape transShape = arrayTranspose.shape();
                NdArray<double> returnArray(transShape.rows, 1);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    double sum = 0;
                    for (uint32 col = 0; col < transShape.cols - 1; ++col)
                    {
                        sum += static_cast<double>(arrayTranspose(row, col + 1) - arrayTranspose(row, col)) / 2.0 +
                            static_cast<double>(arrayTranspose(row, col));
                    }

                    returnArray[row] = sum * dx;
                }

                return returnArray;
            }
            case Axis::NONE:
            {
                double sum = 0.0;
                for (uint32 i = 0; i < inArray.size() - 1; ++i)
                {
                    sum += static_cast<double>(inArray[i + 1] - inArray[i]) / 2.0 + static_cast<double>(inArray[i]);
                }

                NdArray<double> returnArray = { sum * dx };
                return returnArray;
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Integrate along the given axis using the composite trapezoidal rule.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.trapz.html
    ///
    /// @param				inArrayY
    /// @param				inArrayX
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> trapz(const NdArray<dtype>& inArrayY, const NdArray<dtype>& inArrayX, Axis inAxis = Axis::NONE)
    {
        const Shape inShapeY = inArrayY.shape();
        const Shape inShapeX = inArrayX.shape();

        if (inShapeY != inShapeX)
        {
            THROW_INVALID_ARGUMENT_ERROR("input x and y arrays should be the same shape.");
        }

        switch (inAxis)
        {
            case Axis::COL:
            {
                NdArray<double> returnArray(inShapeY.rows, 1);
                for (uint32 row = 0; row < inShapeY.rows; ++row)
                {
                    double sum = 0;
                    for (uint32 col = 0; col < inShapeY.cols - 1; ++col)
                    {
                        const auto dx = static_cast<double>(inArrayX(row, col + 1) - inArrayX(row, col));
                        sum += dx * (static_cast<double>(inArrayY(row, col + 1) - inArrayY(row, col)) / 2.0 +
                            static_cast<double>(inArrayY(row, col)));
                    }

                    returnArray[row] = sum;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> arrayYTranspose = inArrayY.transpose();
                NdArray<dtype> arrayXTranspose = inArrayX.transpose();
                const Shape transShape = arrayYTranspose.shape();
                NdArray<double> returnArray(transShape.rows, 1);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    double sum = 0;
                    for (uint32 col = 0; col < transShape.cols - 1; ++col)
                    {
                        const auto dx = static_cast<double>(arrayXTranspose(row, col + 1) - arrayXTranspose(row, col));
                        sum += dx * (static_cast<double>(arrayYTranspose(row, col + 1) - arrayYTranspose(row, col)) / 2.0 +
                            static_cast<double>(arrayYTranspose(row, col)));
                    }

                    returnArray[row] = sum;
                }

                return returnArray;
            }
            case Axis::NONE:
            {
                double sum = 0.0;
                for (uint32 i = 0; i < inArrayY.size() - 1; ++i)
                {
                    const auto dx = static_cast<double>(inArrayX[i + 1] - inArrayX[i]);
                    sum += dx * (static_cast<double>(inArrayY[i + 1] - inArrayY[i]) / 2.0 + static_cast<double>(inArrayY[i]));
                }

                NdArray<double> returnArray = { sum };
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
