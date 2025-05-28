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

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "NumCpp/Functions/unique.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Return the indices of the bins to which each value in input array belongs.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
    ///
    /// @param x: Input array to be binned.
    /// @param bins: Array of bins.
    ///
    /// @return NdArray
    ///
    template<typename dtype1, typename dtype2>
    NdArray<uint32> digitize(const NdArray<dtype1>& x, const NdArray<dtype2>& bins)
    {
        const auto uniqueBins = unique(bins);
        auto       result     = NdArray<uint32>(1, x.size());
        result.fill(0);

        typename decltype(result)::size_type idx{ 0 };
        std::for_each(x.begin(),
                      x.end(),
                      [&uniqueBins, &result, &idx](const auto value)
                      {
                          const auto upperBin = std::upper_bound(uniqueBins.begin(), uniqueBins.end(), value);
                          result[idx++]       = static_cast<uint32>(std::distance(uniqueBins.begin(), upperBin));
                      });

        return result;
    }
} // namespace nc
