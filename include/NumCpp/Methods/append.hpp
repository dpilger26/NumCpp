/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
/// Copyright 2019 David Pilger
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
/// Methods for working with NdArrays
///
#pragma once

#include"NumCpp/Core/Types.hpp"
#include"NumCpp/NdArray/NdArray.hpp"

#include<algorithm>
#include<iostream>
#include<string>
#include<stdexcept>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Append values to the end of an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.append.html
    ///
    /// @param				inArray
    /// @param				inAppendValues
    /// @param				inAxis (Optional, default NONE): The axis along which values are appended.
    ///									If axis is not given, both inArray and inAppendValues
    ///									are flattened before use.
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> append(const NdArray<dtype>& inArray, const NdArray<dtype>& inAppendValues, Axis inAxis)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                NdArray<dtype> returnArray(1, inArray.size() + inAppendValues.size());
                std::copy(inArray.cbegin(), inArray.cend(), returnArray.begin());
                std::copy(inAppendValues.cbegin(), inAppendValues.cend(), returnArray.begin() + inArray.size());

                return returnArray;
            }
            case Axis::ROW:
            {
                const Shape inShape = inArray.shape();
                const Shape appendShape = inAppendValues.shape();
                if (inShape.cols != appendShape.cols)
                {
                    std::string errStr = "ERROR: append: all the input array dimensions except for the concatenation axis must match exactly";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                NdArray<dtype> returnArray(inShape.rows + appendShape.rows, inShape.cols);
                std::copy(inArray.cbegin(), inArray.cend(), returnArray.begin());
                std::copy(inAppendValues.cbegin(), inAppendValues.cend(), returnArray.begin() + inArray.size());

                return returnArray;
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();
                const Shape appendShape = inAppendValues.shape();
                if (inShape.rows != appendShape.rows)
                {
                    std::string errStr = "ERROR: append: all the input array dimensions except for the concatenation axis must match exactly";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                NdArray<dtype> returnArray(inShape.rows, inShape.cols + appendShape.cols);
                for (uint32 row = 0; row < returnArray.shape().rows; ++row)
                {
                    std::copy(inArray.cbegin(row), inArray.cend(row), returnArray.begin(row));
                    std::copy(inAppendValues.cbegin(row), inAppendValues.cend(row), returnArray.begin(row) + inShape.cols);
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtype>(0);
            }
        }
    }
}
