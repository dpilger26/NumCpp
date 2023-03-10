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

#include <cmath>
#include <utility>
#include <vector>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/ones_like.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Insert values before the given indices.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    ///
    /// @param arr: input array.
    /// @param index: index to insert the value before in the flattened
    /// @param value: value to insert
    /// @return index: index before which values are inserted.
    ///
    template<typename dtype>
    NdArray<dtype> insert(const NdArray<dtype>& arr, int32 index, const dtype& value)
    {
        const NdArray<dtype> values = { value };
        return insert(arr, index, values);
    }

    //============================================================================
    // Method Description:
    /// Insert values before the given indices.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    ///
    /// @param arr: input array.
    /// @param index: index to insert the values before in the flattened
    /// @param values: value to insert
    /// @return index: index before which values are inserted.
    ///
    template<typename dtype>
    NdArray<dtype> insert(const NdArray<dtype>& arr, int32 index, const NdArray<dtype>& values)
    {
        if (index < 0)
        {
            index += arr.size();
            // still
            if (index < 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("index out of range");
            }
        }
        else if (index > static_cast<int32>(arr.size()))
        {
            THROW_INVALID_ARGUMENT_ERROR("index out of range");
        }

        const auto valuesSlice = Slice(index, index + values.size());
        auto       result      = NdArray<dtype>(1, arr.size() + values.size());
        result.put(valuesSlice, values);

        NdArray<bool> mask(result.shape());
        mask.fill(true);
        mask.put(valuesSlice, false);
        result.putMask(mask, arr.flatten());

        return result;
    }

    //============================================================================
    // Method Description:
    /// Insert values along the given axis before the given indices.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    ///
    /// @param arr: input array.
    /// @param index: index to insert the values before
    /// @param value: value to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    ///
    template<typename dtype>
    NdArray<dtype> insert(const NdArray<dtype>& arr, int32 index, const dtype& value, Axis axis)
    {
        const NdArray<dtype> values = { value };
        return insert(arr, index, values, axis);
    }

    //============================================================================
    // Method Description:
    /// Insert values along the given axis before the given indices.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    ///
    /// @param arr: input array.
    /// @param index: index to insert the values before
    /// @param values: values to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    ///
    template<typename dtype>
    NdArray<dtype> insert(const NdArray<dtype>& arr, int32 index, const NdArray<dtype>& values, Axis axis)
    {
        switch (axis)
        {
            case Axis::NONE:
            {
                return insert(arr, index, values);
            }
            case Axis::ROW:
            {
                if (!(values.size() == arr.numCols() || values.isscalar() || values.numCols() == arr.numCols()))
                {
                    THROW_INVALID_ARGUMENT_ERROR("input values shape cannot be broadcast to input array dimensions");
                }

                if (index < 0)
                {
                    index += arr.numRows();
                    // still
                    if (index < 0)
                    {
                        THROW_INVALID_ARGUMENT_ERROR("index out of range");
                    }
                }
                else if (index > static_cast<int32>(arr.numRows()))
                {
                    THROW_INVALID_ARGUMENT_ERROR("index out of range");
                }

                auto  result = NdArray<dtype>();
                int32 valuesSize{};
                if (values.size() == arr.numCols() || values.isscalar())
                {
                    result.resizeFast(arr.numRows() + 1, arr.numCols());
                    valuesSize = 1;
                }
                else if (values.numCols() == arr.numCols())
                {
                    result.resizeFast(arr.numRows() + values.numRows(), arr.numCols());
                    valuesSize = values.numRows();
                }

                auto mask = ones_like<bool>(result);
                mask.put(Slice(index, index + valuesSize), mask.cSlice(), false);

                result.putMask(mask, arr);
                result.putMask(!mask, values);

                return result;
            }
            case Axis::COL:
            {
                if (!(values.size() == arr.numRows() || values.isscalar() || values.numRows() == arr.numRows()))
                {
                    THROW_INVALID_ARGUMENT_ERROR("input values shape cannot be broadcast to input array dimensions");
                }

                if (index < 0)
                {
                    index += arr.numCols();
                    // still
                    if (index < 0)
                    {
                        THROW_INVALID_ARGUMENT_ERROR("index out of range");
                    }
                }
                else if (index > static_cast<int32>(arr.numCols()))
                {
                    THROW_INVALID_ARGUMENT_ERROR("index out of range");
                }

                auto  result = NdArray<dtype>();
                int32 valuesSize{};
                if (values.size() == arr.numRows() || values.isscalar())
                {
                    result.resizeFast(arr.numRows(), arr.numCols() + 1);
                    valuesSize = 1;
                }
                else if (values.numRows() == arr.numRows())
                {
                    result.resizeFast(arr.numRows(), arr.numCols() + values.numCols());
                    valuesSize = values.numCols();
                }

                auto mask = ones_like<bool>(result);
                mask.put(mask.rSlice(), Slice(index, index + valuesSize), false);

                result.putMask(mask, arr);
                result.putMask(!mask, values);

                return result;
            }
            default:
            {
                // get rid of compiler warning
                return {};
            }
        }
    }

    //============================================================================
    // Method Description:
    /// Insert values along the given axis before the given indices.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    ///
    /// @param arr: input array.
    /// @param indices: indices to insert the values before
    /// @param value: value to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    ///
    template<typename dtype, typename Indices, type_traits::ndarray_int_concept<Indices> = 0>
    NdArray<dtype> insert(const NdArray<dtype>& arr, const Indices& indices, const dtype& value, Axis axis = Axis::NONE)
    {
        const NdArray<dtype> values = { value };
        return insert(arr, indices, values, axis);
    }

    //============================================================================
    // Method Description:
    /// Insert values along the given axis before the given indices.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    ///
    /// @param arr: input array.
    /// @param slice: slice to insert the values before
    /// @param value: values to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    ///
    template<typename dtype>
    NdArray<dtype> insert(const NdArray<dtype>& arr, Slice slice, const dtype& value, Axis axis = Axis::NONE)
    {
        auto sliceIndices = slice.toIndices(arr.dimSize(axis));
        return insert(arr, NdArray<uint32>(sliceIndices.data(), sliceIndices.size(), false), value, axis);
    }

    //============================================================================
    // Method Description:
    /// Insert values along the given axis before the given indices.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    ///
    /// @param arr: input array.
    /// @param indices: indices to insert the values before
    /// @param values: values to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    ///
    template<typename dtype, typename Indices, type_traits::ndarray_int_concept<Indices> = 0>
    NdArray<dtype>
        insert(const NdArray<dtype>& arr, const Indices& indices, const NdArray<dtype>& values, Axis axis = Axis::NONE)
    {
        const auto isScalarValue = values.isscalar();

        switch (axis)
        {
            case Axis::NONE:
            {
                if (!isScalarValue && indices.size() != values.size())
                {
                    THROW_INVALID_ARGUMENT_ERROR("could not broadcast values into indices");
                }

                const auto arrSize = static_cast<int32>(arr.size());

                std::vector<std::pair<int32, dtype>> indexValues(indices.size());
                if (isScalarValue)
                {
                    const auto value = values.front();
                    stl_algorithms::transform(indices.begin(),
                                              indices.end(),
                                              indexValues.begin(),
                                              [arrSize, value](auto index) -> std::pair<int32, dtype>
                                              {
                                                  if constexpr (type_traits::is_ndarray_signed_int_v<Indices>)
                                                  {
                                                      if (index < 0)
                                                      {
                                                          index += arrSize;
                                                      }
                                                      // still
                                                      if (index < 0)
                                                      {
                                                          THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                                      }
                                                  }
                                                  if (static_cast<int32>(index) > arrSize)
                                                  {
                                                      THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                                  }

                                                  return std::make_pair(static_cast<int32>(index), value);
                                              });
                }
                else
                {
                    stl_algorithms::transform(indices.begin(),
                                              indices.end(),
                                              values.begin(),
                                              indexValues.begin(),
                                              [arrSize](auto index, const auto& value) -> std::pair<int32, dtype>
                                              {
                                                  if constexpr (type_traits::is_ndarray_signed_int_v<Indices>)
                                                  {
                                                      if (index < 0)
                                                      {
                                                          index += arrSize;
                                                      }
                                                      // still
                                                      if (index < 0)
                                                      {
                                                          THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                                      }
                                                  }
                                                  if (static_cast<int32>(index) > arrSize)
                                                  {
                                                      THROW_INVALID_ARGUMENT_ERROR("index out of range");
                                                  }

                                                  return std::make_pair(static_cast<int32>(index), value);
                                              });
                }

                stl_algorithms::sort(indexValues.begin(),
                                     indexValues.end(),
                                     [](const auto& indexValue1, const auto& indexValue2) noexcept -> bool
                                     { return indexValue1.first < indexValue2.first; });
                auto indexValuesUnique = std::vector<typename decltype(indexValues)::value_type>{};
                std::unique_copy(indexValues.begin(),
                                 indexValues.end(),
                                 std::back_inserter(indexValuesUnique),
                                 [](const auto& indexValue1, const auto& indexValue2) noexcept -> bool
                                 { return indexValue1.first == indexValue2.first; });

                auto result = NdArray<dtype>(1, arr.size() + indexValuesUnique.size());

                auto  mask    = ones_like<bool>(result);
                int32 counter = 0;
                std::for_each(indexValuesUnique.begin(),
                              indexValuesUnique.end(),
                              [&counter, &mask](auto& indexValue) noexcept -> void
                              { mask[indexValue.first + counter++] = false; });

                result.putMask(mask, arr.flatten());

                auto valuesSorted = [&indexValuesUnique]
                {
                    std::vector<dtype> values_;
                    values_.reserve(indexValuesUnique.size());
                    std::transform(indexValuesUnique.begin(),
                                   indexValuesUnique.end(),
                                   std::back_inserter(values_),
                                   [](const auto& indexValue) { return indexValue.second; });
                    return values_;
                }();

                result.putMask(!mask, NdArray<dtype>(valuesSorted.data(), valuesSorted.size(), false));

                return result;
            }
            case Axis::ROW:
            {
                if (!(values.size() == arr.numCols() || values.isscalar() ||
                      (values.numCols() == arr.numCols() && values.numRows() == indices.size())))
                {
                    THROW_INVALID_ARGUMENT_ERROR("input values shape cannot be broadcast to input array dimensions");
                }

                // const auto arrNumRows = static_cast<int32>(arr.numRows());

                return {};
            }
            case Axis::COL:
            {
                if (!(values.size() == arr.numRows() || values.isscalar() || values.numRows() == arr.numRows() ||
                      (values.numRows() == arr.numRows() && values.numCols() == indices.size())))
                {
                    THROW_INVALID_ARGUMENT_ERROR("input values shape cannot be broadcast to input array dimensions");
                }

                // const auto arrNumCols = static_cast<int32>(arr.numCols());

                return {};
            }
            default:
            {
                // get rid of compiler warning
                return {};
            }
        }
    }

    //============================================================================
    // Method Description:
    /// Insert values along the given axis before the given indices.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    ///
    /// @param arr: input array.
    /// @param slice: slice to insert the values before
    /// @param values: values to insert
    /// @param axis: axis along which to insert values
    /// @return index: index before which values are inserted.
    ///
    template<typename dtype>
    NdArray<dtype> insert(const NdArray<dtype>& arr, Slice slice, const NdArray<dtype>& values, Axis axis = Axis::NONE)
    {
        auto sliceIndices = slice.toIndices(arr.dimSize(axis));
        return insert(arr, NdArray<uint32>(sliceIndices.data(), sliceIndices.size(), false), values, axis);
    }
} // namespace nc
