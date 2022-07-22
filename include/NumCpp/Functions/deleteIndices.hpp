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

#include <string>
#include <vector>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Functions/unique.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Return a new array with sub-arrays along an axis deleted.
    ///
    /// @param inArray
    /// @param inArrayIdxs
    /// @param inAxis (Optional, default NONE) if none the indices will be applied to the flattened array
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype>
        deleteIndices(const NdArray<dtype>& inArray, const NdArray<uint32>& inArrayIdxs, Axis inAxis = Axis::NONE)
    {
        // make sure that the indices are unique first
        NdArray<uint32> indices = unique(inArrayIdxs);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::vector<dtype> values;
                for (uint32 i = 0; i < inArray.size(); ++i)
                {
                    if (indices.contains(i).item())
                    {
                        continue;
                    }

                    values.push_back(inArray[i]);
                }

                return NdArray<dtype>(values);
            }
            case Axis::ROW:
            {
                const Shape inShape = inArray.shape();
                if (indices.max().item() >= inShape.rows)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input index value is greater than the number of rows in the array.");
                }

                const uint32   numNewRows = inShape.rows - indices.size();
                NdArray<dtype> returnArray(numNewRows, inShape.cols);

                uint32 rowCounter = 0;
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    if (indices.contains(row).item())
                    {
                        continue;
                    }

                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        returnArray(rowCounter, col) = inArray(row, col);
                    }
                    ++rowCounter;
                }

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();
                if (indices.max().item() >= inShape.cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input index value is greater than the number of cols in the array.");
                }

                const uint32   numNewCols = inShape.cols - indices.size();
                NdArray<dtype> returnArray(inShape.rows, numNewCols);

                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    uint32 colCounter = 0;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (indices.contains(col).item())
                        {
                            continue;
                        }

                        returnArray(row, colCounter++) = inArray(row, col);
                    }
                }

                return returnArray;
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }

    //============================================================================
    // Method Description:
    /// Return a new array with sub-arrays along an axis deleted.
    ///
    /// @param inArray
    /// @param inIndicesSlice
    /// @param inAxis (Optional, default NONE) if none the indices will be applied to the flattened array
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, const Slice& inIndicesSlice, Axis inAxis = Axis::NONE)
    {
        Slice sliceCopy(inIndicesSlice);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                sliceCopy.makePositiveAndValidate(inArray.size());
                break;
            }
            case Axis::ROW:
            {
                sliceCopy.makePositiveAndValidate(inArray.shape().cols);
                break;
            }
            case Axis::COL:
            {
                sliceCopy.makePositiveAndValidate(inArray.shape().rows);
                break;
            }
        }

        std::vector<uint32> indices;
        for (auto i = static_cast<uint32>(sliceCopy.start); i < static_cast<uint32>(sliceCopy.stop);
             i += sliceCopy.step)
        {
            indices.push_back(i);
        }

        return deleteIndices(inArray, NdArray<uint32>(indices), inAxis);
    }

    //============================================================================
    // Method Description:
    /// Return a new array with sub-arrays along an axis deleted.
    ///
    /// @param inArray
    /// @param inIndex
    /// @param inAxis (Optional, default NONE) if none the indices will be applied to the flattened array
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, uint32 inIndex, Axis inAxis = Axis::NONE)
    {
        NdArray<uint32> inIndices = { inIndex };
        return deleteIndices(inArray, inIndices, inAxis);
    }
} // namespace nc
