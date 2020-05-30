/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.4.0
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
#include "NumCpp/Functions/sort.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"

#include <complex>
#include <vector>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Find the unique elements of an array.
    ///
    ///						Returns the sorted unique elements of an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.unique.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> unique(const NdArray<dtype>& inArray) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto comp = [](const dtype lhs, const dtype rhs) noexcept -> bool
        {
            return utils::essentiallyEqual(lhs, rhs);
        };

        const auto sorted = sort(inArray);

        std::vector<dtype> res(sorted.size());
        const auto last = stl_algorithms::unique_copy(sorted.begin(), sorted.end(), res.begin(), comp);

        return NdArray<dtype>(res.begin(), last);
    }
}
