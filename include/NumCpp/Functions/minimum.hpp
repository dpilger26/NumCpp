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

#include <cmath>
#include <complex>
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Element-wise minimum of array elements.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.minimum.html
    ///
    ///
    /// @param inArray1
    /// @param inArray2
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> minimum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto comparitor = [](const dtype& lhs, const dtype& rhs) noexcept -> bool { return lhs < rhs; };

        if (inArray1.shape() == inArray2.shape())
        {
            return [&inArray1, &inArray2, &comparitor]
            {
                NdArray<dtype> returnArray(inArray1.shape());
                stl_algorithms::transform(inArray1.cbegin(),
                                          inArray1.cend(),
                                          inArray2.cbegin(),
                                          returnArray.begin(),
                                          [comparitor](const dtype& inValue1, const dtype& inValue2) -> dtype
                                          { return std::min(inValue1, inValue2, comparitor); });

                return returnArray;
            }();
        }
        else if (inArray1.isscalar())
        {
            return minimum(inArray2, inArray1);
        }
        else if (inArray2.isscalar())
        {
            const auto value = inArray2.item();
            return [&inArray1, &value, &comparitor]
            {
                NdArray<dtype> returnArray(inArray1.shape());
                stl_algorithms::transform(inArray1.cbegin(),
                                          inArray1.cend(),
                                          returnArray.begin(),
                                          [&value, comparitor](const dtype& inValue) -> dtype
                                          { return std::min(inValue, value, comparitor); });
                return returnArray;
            }();
        }
        else if (inArray1.isflat() && inArray2.isflat())
        {
            return [&inArray1, &inArray2, &comparitor]
            {
                const auto     numRows = std::max(inArray1.numRows(), inArray2.numRows(), comparitor);
                const auto     numCols = std::max(inArray1.numCols(), inArray2.numCols(), comparitor);
                NdArray<dtype> returnArray(numRows, numCols);
                if (inArray1.numRows() > 1)
                {
                    for (uint32 row = 0; row < inArray1.numRows(); ++row)
                    {
                        for (uint32 col = 0; col < inArray2.numCols(); ++col)
                        {
                            returnArray(row, col) = std::min(inArray1[row], inArray2[col], comparitor);
                        }
                    }
                }
                else
                {
                    for (uint32 row = 0; row < inArray2.numRows(); ++row)
                    {
                        for (uint32 col = 0; col < inArray1.numCols(); ++col)
                        {
                            returnArray(row, col) = std::min(inArray1[col], inArray2[row], comparitor);
                        }
                    }
                }
                return returnArray;
            }();
        }
        else if (inArray1.isflat())
        {
            return minimum(inArray2, inArray1);
        }
        else if (inArray2.isflat())
        {
            if (inArray2.numRows() > 1 && inArray2.numRows() == inArray1.numRows())
            {
                return [&inArray1, &inArray2, &comparitor]
                {
                    NdArray<dtype> returnArray(inArray1.shape());
                    for (uint32 row = 0; row < inArray1.numRows(); ++row)
                    {
                        const auto value = inArray2[row];
                        stl_algorithms::transform(inArray1.cbegin(row),
                                                  inArray1.cend(row),
                                                  returnArray.begin(row),
                                                  [&value, comparitor](const dtype& inValue) -> dtype
                                                  { return std::min(inValue, value, comparitor); });
                    }
                    return returnArray;
                }();
            }
            else if (inArray2.numCols() > 1 && inArray2.numCols() == inArray1.numCols())
            {
                return minimum(inArray1.transpose(), inArray2.transpose()).transpose();
            }
            else
            {
                THROW_INVALID_ARGUMENT_ERROR("operands could not be broadcast together");
            }
        }
        else
        {
            THROW_INVALID_ARGUMENT_ERROR("operands could not be broadcast together");
        }

        return {}; // get rid of compiler warning
    }

    //============================================================================
    // Method Description:
    /// Element-wise minimum of array elements.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.minimum.html
    ///
    ///
    /// @param inArray
    /// @param inScalar
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> minimum(const NdArray<dtype>& inArray, const dtype& inScalar)
    {
        const NdArray<dtype> inArray2 = { inScalar };
        return minimum(inArray, inArray2);
    }

    //============================================================================
    // Method Description:
    /// Element-wise minimum of array elements.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.minimum.html
    ///
    ///
    /// @param inScalar
    /// @param inArray
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> minimum(const dtype& inScalar, const NdArray<dtype>& inArray)
    {
        return minimum(inArray, inScalar);
    }
} // namespace nc
