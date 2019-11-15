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

#include "NumCpp/Functions/arange.hpp"
#include "NumCpp/NdArray.hpp"

#include <utility>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Return coordinate matrices from coordinate vectors.
    ///                     Make 2D coordinate arrays for vectorized evaluations of 2D scalar
    ///                     vector fields over 2D grids, given one - dimensional coordinate arrays x1, x2, ..., xn.
    ///                     If input arrays are not one dimensional they will be flattened.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.meshgrid.html
    ///
    /// @param				inICoords
    /// @param  			inJCoords
    ///
    /// @return
    ///				std::pair<NdArray<dtype>, NdArray<dtype> >, i and j matrices
    ///
    template<typename dtype>
    std::pair<NdArray<dtype>, NdArray<dtype> > meshgrid(const NdArray<dtype>& inICoords, const NdArray<dtype>& inJCoords) noexcept
    {
        const uint32 numRows = inJCoords.size();
        const uint32 numCols = inICoords.size();
        auto returnArrayI = NdArray<dtype>(numRows, numCols);
        auto returnArrayJ = NdArray<dtype>(numRows, numCols);

        // first the I array
        for (uint32 row = 0; row < numRows; ++row)
        {
            for (uint32 col = 0; col < numCols; ++col)
            {
                returnArrayI(row, col) = inICoords[col];
            }
        }

        // then the I array
        for (uint32 col = 0; col < numCols; ++col)
        {
            for (uint32 row = 0; row < numRows; ++row)
            {
                returnArrayJ(row, col) = inJCoords[row];
            }
        }

        return std::make_pair(returnArrayI, returnArrayJ);
    }

    //============================================================================
// Method Description:
///						Return coordinate matrices from coordinate vectors.
///                     Make 2D coordinate arrays for vectorized evaluations of 2D scalar
///                     vector fields over 2D grids, given one - dimensional coordinate arrays x1, x2, ..., xn.
///
///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.meshgrid.html
///
/// @param				inSlice1
/// @param  			inSlice2
///
/// @return
///				std::pair<NdArray<dtype>, NdArray<dtype> >, i and j matrices
///
    template<typename dtype>
    std::pair<NdArray<dtype>, NdArray<dtype> > meshgrid(const Slice& inSlice1, const Slice& inSlice2) noexcept
    {
        return meshgrid(arange<dtype>(inSlice1), arange<dtype>(inSlice2));
    }

}
