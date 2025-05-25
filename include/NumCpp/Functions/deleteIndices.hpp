/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
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
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Functions/unique.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    namespace detail
    {
        //============================================================================
        // Method Description:
        /// Return a new array with sub-arrays deleted.
        ///
        /// @param inArray
        /// @param inIndices
        /// @return NdArray
        ///
        template<typename dtype, NdArrayInt Indices>
        NdArray<dtype> deleteFlatIndices(const NdArray<dtype>& inArray, Indices inIndices)
        {
            if constexpr (NdArraySignedInt<Indices>)
            {
                const auto arraySize = inArray.size();
                stl_algorithms::for_each(inIndices.begin(),
                                         inIndices.end(),
                                         [arraySize](auto& value)
                                         {
                                             if (value < 0)
                                             {
                                                 value += arraySize;
                                             }
                                         });
            }

            auto indices = unique(inIndices);

            std::vector<dtype> values;
            values.reserve(indices.size());
            for (int32 i = 0; i < static_cast<int32>(inArray.size()); ++i)
            {
                if (std::binary_search(indices.begin(), indices.end(), i))
                {
                    continue;
                }

                values.push_back(inArray[i]);
            }

            return NdArray<dtype>(values);
        }

        //============================================================================
        // Method Description:
        /// Return a new array with sub-arrays along the row axis deleted.
        ///
        /// @param inArray
        /// @param inIndices
        /// @return NdArray
        ///
        template<typename dtype, NdArrayInt Indices>
        NdArray<dtype> deleteRowIndices(const NdArray<dtype>& inArray, Indices inIndices)
        {
            const auto arrayRows = static_cast<int32>(inArray.numRows());
            if constexpr (NdArraySignedInt<Indices>)
            {
                stl_algorithms::for_each(inIndices.begin(),
                                         inIndices.end(),
                                         [arrayRows](auto& value)
                                         {
                                             if (value < 0)
                                             {
                                                 value += arrayRows;
                                             }
                                         });
            }

            auto indices = unique(inIndices);

            uint32 indicesSize = 0;
            std::for_each(indices.begin(),
                          indices.end(),
                          [arrayRows, &indicesSize](const auto& value)
                          {
                              if constexpr (std::is_signed_v<decltype(value)>)
                              {
                                  if (value >= 0 && value < arrayRows)
                                  {
                                      ++indicesSize;
                                  }
                              }
                              else
                              {
                                  if (value < arrayRows)
                                  {
                                      ++indicesSize;
                                  }
                              }
                          });

            const auto     arrayCols = static_cast<int32>(inArray.numCols());
            NdArray<dtype> returnArray(arrayRows - indicesSize, arrayCols);

            uint32 rowCounter = 0;
            for (int32 row = 0; row < arrayRows; ++row)
            {
                if (std::binary_search(indices.begin(), indices.end(), row))
                {
                    continue;
                }

                for (int32 col = 0; col < arrayCols; ++col)
                {
                    returnArray(rowCounter, col) = inArray(row, col);
                }

                ++rowCounter;
            }

            return returnArray;
        }

        //============================================================================
        // Method Description:
        /// Return a new array with sub-arrays along the col axis deleted.
        ///
        /// @param inArray
        /// @param inIndices
        /// @return NdArray
        ///
        template<typename dtype, NdArrayInt Indices>
        NdArray<dtype> deleteColumnIndices(const NdArray<dtype>& inArray, Indices inIndices)
        {
            const auto arrayCols = static_cast<int32>(inArray.numCols());
            if constexpr (NdArraySignedInt<Indices>)
            {
                stl_algorithms::for_each(inIndices.begin(),
                                         inIndices.end(),
                                         [arrayCols](auto& value)
                                         {
                                             if (value < 0)
                                             {
                                                 value += arrayCols;
                                             }
                                         });
            }

            auto indices = unique(inIndices);

            uint32 indicesSize = 0;
            std::for_each(indices.begin(),
                          indices.end(),
                          [arrayCols, &indicesSize](const auto& value)
                          {
                              if constexpr (std::is_signed_v<decltype(value)>)
                              {
                                  if (value >= 0 && value < arrayCols)
                                  {
                                      ++indicesSize;
                                  }
                              }
                              else
                              {
                                  if (value < arrayCols)
                                  {
                                      ++indicesSize;
                                  }
                              }
                          });

            const auto     arrayRows = static_cast<int32>(inArray.numRows());
            NdArray<dtype> returnArray(arrayRows, arrayCols - indicesSize);

            uint32 colCounter = 0;
            for (int32 col = 0; col < arrayCols; ++col)
            {
                if (std::binary_search(indices.begin(), indices.end(), col))
                {
                    continue;
                }

                for (int32 row = 0; row < arrayRows; ++row)
                {
                    returnArray(row, colCounter) = inArray(row, col);
                }

                ++colCounter;
            }

            return returnArray;
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Return a new array with sub-arrays along an axis deleted.
    ///
    /// @param inArray
    /// @param inIndices
    /// @param inAxis (Optional, default NONE) if NONE the indices will be applied to the flattened array
    /// @return NdArray
    ///
    template<typename dtype, NdArrayInt Indices>
    NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, const Indices& inIndices, Axis inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                return detail::deleteFlatIndices(inArray, inIndices);
            }
            case Axis::ROW:
            {
                return detail::deleteRowIndices(inArray, inIndices);
            }
            case Axis::COL:
            {
                return detail::deleteColumnIndices(inArray, inIndices);
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
    NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, Slice inIndicesSlice, Axis inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                inIndicesSlice.makePositiveAndValidate(inArray.size());
                break;
            }
            case Axis::ROW:
            {
                inIndicesSlice.makePositiveAndValidate(inArray.numRows());
                break;
            }
            case Axis::COL:
            {
                inIndicesSlice.makePositiveAndValidate(inArray.numCols());
                break;
            }
        }

        std::vector<int32> indices;
        for (auto i = inIndicesSlice.start; i < inIndicesSlice.stop; i += inIndicesSlice.step)
        {
            indices.push_back(i);
        }

        return deleteIndices(inArray, NdArray<int32>(indices.data(), indices.size(), PointerPolicy::SHELL), inAxis);
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
    NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, int32 inIndex, Axis inAxis = Axis::NONE)
    {
        NdArray<int32> inIndices = { inIndex };
        return deleteIndices(inArray, inIndices, inAxis);
    }
} // namespace nc
