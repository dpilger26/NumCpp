/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2026 David Pilger
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
/// matrix svd
///
#pragma once

#include <utility>

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Linalg/svd.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::linalg
{
    //============================================================================
    // Method Description:
    /// Singular Value Decomposition.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    ///
    /// @param inArray: NdArray to be SVDed
    ///
    /// @returns array of singular values
    ///
    template<typename dtype>
    NdArray<double> svdvals(const NdArray<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        auto u  = NdArray<double>{};
        auto s  = NdArray<double>{};
        auto vT = NdArray<double>{};

        svd(inArray, u, s, vT);

        return s;
    }
} // namespace nc::linalg
