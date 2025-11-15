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

#include <complex>
#include <unordered_set>

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"
#include "NumCpp/Functions/sort.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Find duplication in a array
    ///
    /// Return an array of duplicated elements
    ///
    /// NumPy Reference: https://numpy.org/doc/stable//reference/generated/numpy.rec.find_duplicate.html
    ///
    /// @param inArray
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> find_duplicates(const NdArray<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        auto repeats = std::unordered_set<dtype>{};
        auto count   = std::unordered_set<dtype>{};

        for (const auto& value : inArray)
        {
            if (count.count(value) > 0)
            {
                repeats.insert(value);
            }
            else
            {
                count.insert(value);
            }
        }

        return sort(NdArray<dtype>{ repeats.begin(), repeats.end() });
    }

    //============================================================================
    // Method Description:
    /// Find duplication in a array
    ///
    /// Return an array of duplicated elements
    ///
    /// NumPy Reference: https://numpy.org/doc/stable//reference/generated/numpy.rec.find_duplicate.html
    ///
    /// @param inArray
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> find_duplicates(const NdArray<std::complex<dtype>>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        auto repeats = std::unordered_set<std::complex<dtype>, ComplexHash<dtype>>{};
        auto count   = std::unordered_set<std::complex<dtype>, ComplexHash<dtype>>{};

        for (const auto& value : inArray)
        {
            if (count.count(value) > 0)
            {
                repeats.insert(value);
            }
            else
            {
                count.insert(value);
            }
        }

        return sort(NdArray<std::complex<dtype>>{ repeats.begin(), repeats.end() });
    }
} // namespace nc
