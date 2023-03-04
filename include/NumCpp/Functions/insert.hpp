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

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
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
            if (index < 0)
            {
                index = 0;
            }
        }
        else if (index > static_cast<int32>(arr.size()))
        {
            index = arr.size();
        }

        auto result = NdArray<dtype>(1, arr.size() + values.size());

        if (index > 0)
        {
            const auto sliceFront = Slice(index);
            result.put(sliceFront, arr[sliceFront]);
        }

        result.put(Slice(index, index + values.size()), values.flatten());

        if (index < static_cast<int32>(arr.size()))
        {
            result.put(result.cSlice(index + values.size()), arr[Slice(index, arr.size())]);
        }

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
                if (!(values.size() == arr.numCols() || values.size() == 1 || values.numCols() == arr.numCols()))
                {
                    THROW_INVALID_ARGUMENT_ERROR("input values shape cannot be broadcast to input array dimensions");
                }

                if (index < 0)
                {
                    index += arr.numRows();
                    if (index < 0)
                    {
                        index = 0;
                    }
                }
                else if (index > static_cast<int32>(arr.numRows()))
                {
                    index = arr.numRows();
                }

                auto  result = NdArray<dtype>();
                int32 valuesSize{};
                if (values.size() == arr.numCols() || values.size() == 1)
                {
                    result.resizeFast(arr.numRows() + 1, arr.numCols());
                    valuesSize = 1;
                }
                else if (values.numCols() == arr.numCols())
                {
                    result.resizeFast(arr.numRows() + values.numRows(), arr.numCols());
                    valuesSize = values.numRows();
                }

                if (index > 0)
                {
                    const auto sliceFront = Slice(index);
                    result.put(sliceFront, result.cSlice(), arr(sliceFront, arr.cSlice()));
                }

                result.put(Slice(index, index + valuesSize), result.cSlice(), values);

                if (index < static_cast<int32>(arr.numRows()))
                {
                    result.put(result.rSlice(index + valuesSize),
                               result.cSlice(),
                               arr(arr.rSlice(index), arr.cSlice()));
                }

                return result;
            }
            case Axis::COL:
            {
                if (!(values.size() == arr.numRows() || values.size() == 1 || values.numRows() == arr.numRows()))
                {
                    THROW_INVALID_ARGUMENT_ERROR("input values shape cannot be broadcast to input array dimensions");
                }

                if (index < 0)
                {
                    index += arr.numCols();
                    if (index < 0)
                    {
                        index = 0;
                    }
                }
                else if (index > static_cast<int32>(arr.numCols()))
                {
                    index = arr.numCols();
                }

                auto  result = NdArray<dtype>();
                int32 valuesSize{};
                if (values.size() == arr.numRows() || values.size() == 1)
                {
                    result.resizeFast(arr.numRows(), arr.numCols() + 1);
                    valuesSize = 1;
                }
                else if (values.numRows() == arr.numRows())
                {
                    result.resizeFast(arr.numRows(), arr.numCols() + values.numCols());
                    valuesSize = values.numCols();
                }

                if (index > 0)
                {
                    const auto sliceFront = Slice(index);
                    result.put(result.rSlice(), sliceFront, arr(arr.rSlice(), sliceFront));
                }

                result.put(result.rSlice(), Slice(index, index + valuesSize), values);

                if (index < static_cast<int32>(arr.numCols()))
                {
                    result.put(result.rSlice(),
                               result.cSlice(index + valuesSize),
                               arr(arr.rSlice(), arr.cSlice(index)));
                }

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
    NdArray<dtype> insert(const NdArray<dtype>& /*arr*/,
                          const Indices& /*indices*/,
                          const NdArray<dtype>& /*values*/,
                          Axis /*axis = Axis::NONE*/)
    {
        return {};
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
