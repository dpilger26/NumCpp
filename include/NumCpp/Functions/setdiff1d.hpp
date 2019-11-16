/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.2
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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/NdArray.hpp"

#include <set>
#include <vector>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Find the set difference of two arrays.
    ///
    ///						Return the sorted, unique values in ar1 that are not in ar2.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.setdiff1d.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> setdiff1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        std::vector<dtype> res(inArray1.size() + inArray2.size());
        const std::set<dtype> in1(inArray1.cbegin(), inArray1.cend());
        const std::set<dtype> in2(inArray2.cbegin(), inArray2.cend());

        const auto iter = std::set_difference(in1.begin(), in1.end(),
            in2.begin(), in2.end(), res.begin());
        res.resize(iter - res.begin());
        return NdArray<dtype>(res);
    }
}
