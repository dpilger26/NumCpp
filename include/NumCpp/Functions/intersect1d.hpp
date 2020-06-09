/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.0.0
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

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"

#include <functional>
#include <set>
#include <vector>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Find the intersection of two arrays.
    ///
    ///						Return the sorted, unique values that are in both of the input arrays.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.intersect1d.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype, class Alloc>
    NdArray<dtype, Alloc> intersect1d(const NdArray<dtype, Alloc>& inArray1, const NdArray<dtype, Alloc>& inArray2)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::vector<dtype, Alloc> res(inArray1.size() + inArray2.size());
        const std::set<dtype, std::less<dtype>, Alloc> in1(inArray1.cbegin(), inArray1.cend());
        const std::set<dtype, std::less<dtype>, Alloc> in2(inArray2.cbegin(), inArray2.cend());

        const auto iter = stl_algorithms::set_intersection(in1.begin(), in1.end(),
            in2.begin(), in2.end(), res.begin());
        res.resize(iter - res.begin());
        return NdArray<dtype, Alloc>(res);
    }
}
