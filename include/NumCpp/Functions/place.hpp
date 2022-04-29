/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2022 David Pilger
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
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Change elements of an array based on conditional and input values.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.place.html
    ///
    /// @param arr: Array to put data into.
    /// @param mask: Boolean mask array. Must have the same size as arr
    /// @param vals: Values to put into a. Only the first N elements are used, where N is the
    /// number of True values in mask. If vals is smaller than N, it will be repeated.
    /// @return NdArray
    ///
    template<typename dtype>
    void place(NdArray<dtype>& arr, const NdArray<bool>& mask, const NdArray<dtype>& vals)
    {
        if (mask.size() != arr.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("Input arguments 'arr' and 'mask' must have the same size.");
        }

        if (vals.isempty())
        {
            return;
        }

        auto valIdx = 0;
        for (decltype(arr.size()) i = 0; i < arr.size(); ++i)
        {
            if (mask[i])
            {
                arr[i] = vals[valIdx++ % vals.size()];
            }
        }
    }
} // namespace nc
