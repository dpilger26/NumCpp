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

#include <vector>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Return the elements of an array that satisfy some condition.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.extract.html
    ///
    /// @param condition: An array whose nonzero or True entries indicate the elements of arr to extract.
    /// @param arr: Input array of the same size as condition
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> extract(const NdArray<bool>& condition, const NdArray<dtype>& arr)
    {
        if (condition.size() != arr.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("Input arguments 'condition' and 'arr' must have the same size.");
        }

        std::vector<dtype> values;
        for (decltype(arr.size()) i = 0; i < arr.size(); ++i)
        {
            if (condition[i])
            {
                values.push_back(arr[i]);
            }
        }

        return NdArray<dtype>(values.begin(), values.end());
    }
} // namespace nc
