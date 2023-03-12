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
/// Broadcasting for NdArray functions
///
#pragma once

#include <cmath>
#include <utility>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray/NdArrayCore.hpp"

namespace nc::broadcast
{
    //============================================================================
    // Method Description:
    /// Broadcasting template function
    ///
    /// @param function
    /// @param inArray1
    /// @param inArray2
    ///
    /// @return NdArray
    ///
    template<typename dtypeOut,
             typename dtypeIn1,
             typename dtypeIn2,
             typename Function,
             typename... AdditionalFunctionArgs>
    NdArray<dtypeOut> broadcaster(const NdArray<dtypeIn1>& inArray1,
                                  const NdArray<dtypeIn2>& inArray2,
                                  const Function&          function,
                                  const AdditionalFunctionArgs&&... additionalFunctionArgs)
    {
        if (inArray1.shape() == inArray2.shape())
        {
            return [&inArray1, &inArray2, &function, &additionalFunctionArgs...]
            {
                NdArray<dtypeOut> returnArray(inArray1.shape());
                stl_algorithms::transform(
                    inArray1.cbegin(),
                    inArray1.cend(),
                    inArray2.cbegin(),
                    returnArray.begin(),
                    [&function, &additionalFunctionArgs...](const dtypeIn1& inValue1,
                                                            const dtypeIn2& inValue2) -> dtypeOut {
                        return function(inValue1,
                                        inValue2,
                                        std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                    });

                return returnArray;
            }();
        }
        else if (inArray1.isscalar())
        {
            return broadcaster<dtypeOut>(inArray2,
                                         inArray1,
                                         function,
                                         std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
        }
        else if (inArray2.isscalar())
        {
            const auto value = inArray2.item();
            return [&inArray1, &value, &function, &additionalFunctionArgs...]
            {
                NdArray<dtypeOut> returnArray(inArray1.shape());
                stl_algorithms::transform(
                    inArray1.cbegin(),
                    inArray1.cend(),
                    returnArray.begin(),
                    [&value, &function, &additionalFunctionArgs...](const dtypeIn1& inValue) -> dtypeOut {
                        return function(inValue,
                                        value,
                                        std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                    });
                return returnArray;
            }();
        }
        else if (inArray1.isflat() && inArray2.isflat())
        {
            return [&inArray1, &inArray2, &function, &additionalFunctionArgs...]
            {
                const auto        numRows = std::max(inArray1.numRows(), inArray2.numRows());
                const auto        numCols = std::max(inArray1.numCols(), inArray2.numCols());
                NdArray<dtypeOut> returnArray(numRows, numCols);
                if (inArray1.numRows() > 1)
                {
                    for (uint32 row = 0; row < inArray1.numRows(); ++row)
                    {
                        for (uint32 col = 0; col < inArray2.numCols(); ++col)
                        {
                            returnArray(row, col) =
                                function(inArray1[row],
                                         inArray2[col],
                                         std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                        }
                    }
                }
                else
                {
                    for (uint32 row = 0; row < inArray2.numRows(); ++row)
                    {
                        for (uint32 col = 0; col < inArray1.numCols(); ++col)
                        {
                            returnArray(row, col) =
                                function(inArray1[col],
                                         inArray2[row],
                                         std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                        }
                    }
                }
                return returnArray;
            }();
        }
        else if (inArray1.isflat())
        {
            return broadcaster<dtypeOut>(inArray2,
                                         inArray1,
                                         function,
                                         std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
        }
        else if (inArray2.isflat())
        {
            if (inArray2.numRows() > 1 && inArray2.numRows() == inArray1.numRows())
            {
                return [&inArray1, &inArray2, &function, &additionalFunctionArgs...]
                {
                    NdArray<dtypeOut> returnArray(inArray1.shape());
                    for (uint32 row = 0; row < inArray1.numRows(); ++row)
                    {
                        const auto value = inArray2[row];
                        stl_algorithms::transform(
                            inArray1.cbegin(row),
                            inArray1.cend(row),
                            returnArray.begin(row),
                            [&value, &function, &additionalFunctionArgs...](const dtypeIn1& inValue) -> dtypeOut {
                                return function(inValue,
                                                value,
                                                std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                            });
                    }
                    return returnArray;
                }();
            }
            else if (inArray2.numCols() > 1 && inArray2.numCols() == inArray1.numCols())
            {
                return broadcaster<dtypeOut>(inArray1.transpose(),
                                             inArray2.transpose(),
                                             function,
                                             std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...)
                    .transpose();
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
} // namespace nc::broadcast
