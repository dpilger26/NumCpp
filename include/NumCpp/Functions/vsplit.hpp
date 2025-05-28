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

#include "NumCpp/Functions/split.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Split an array into multiple sub-arrays vertically (row-wise).
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html
    ///
    /// @param inArray
    /// @param indices: the indices to split
    ///
    /// @return NdArray
    ///
    template<typename dtype, typename Indices, type_traits::ndarray_int_concept<Indices> = 0>
    std::vector<NdArray<dtype>> vsplit(const NdArray<dtype>& inArray, const Indices& indices)
    {
        const auto     numRows = static_cast<int32>(inArray.numRows());
        NdArray<int32> uniqueIndices(1, indices.size());
        stl_algorithms::transform(indices.begin(),
                                  indices.end(),
                                  uniqueIndices.begin(),
                                  [numRows](auto index) noexcept -> int32
                                  {
                                      if constexpr (type_traits::is_ndarray_signed_int_v<Indices>)
                                      {
                                          if (index < 0)
                                          {
                                              index = std::max(index + numRows, int32{ 0 });
                                          }
                                      }
                                      if (index > numRows - 1)
                                      {
                                          index = numRows - 1;
                                      }

                                      return static_cast<int32>(index);
                                  });
        uniqueIndices = unique(uniqueIndices);

        std::vector<NdArray<dtype>> splits{};
        splits.reserve(uniqueIndices.size() + 1);

        const auto cSlice   = inArray.cSlice();
        int32      lowerIdx = 0;
        for (const auto index : uniqueIndices)
        {
            if (index == 0)
            {
                splits.push_back(NdArray<dtype>(Shape(0, inArray.numCols())));
                continue;
            }
            else
            {
                splits.push_back(inArray(Slice(lowerIdx, index), cSlice));
            }

            lowerIdx = index;
        }

        if (lowerIdx < numRows - 1)
        {
            splits.push_back(inArray(Slice(lowerIdx, numRows), cSlice));
        }
        else
        {
            splits.push_back(inArray(-1, cSlice));
        }

        return splits;
    }
} // namespace nc
