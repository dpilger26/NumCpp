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

#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// set the flat index element to the value
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inIndex
    /// @param inValue
    ///
    template<typename dtype>
    NdArray<dtype>& put(NdArray<dtype>& inArray, int32 inIndex, const dtype& inValue)
    {
        inArray.put(inIndex, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// set the 2D row/col index element to the value
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRow
    /// @param inCol
    /// @param inValue
    ///
    template<typename dtype>
    NdArray<dtype>& put(NdArray<dtype>& inArray, int32 inRow, int32 inCol, const dtype& inValue)
    {
        inArray.put(inRow, inCol, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set a.flat[n] = values for all n in indices.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inIndices
    /// @param inValue
    /// @return reference to self
    ///
    template<typename dtype, typename Indices, type_traits::ndarray_int_concept<Indices> = 0>
    NdArray<dtype>& put(NdArray<dtype>& inArray, const Indices& inIndices, const dtype& inValue)
    {
        inArray.put(inIndices, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set a.flat[n] = values[n] for all n in indices.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inIndices
    /// @param inValues
    /// @return reference to self
    ///
    template<typename dtype, typename Indices, type_traits::ndarray_int_concept<Indices> = 0>
    NdArray<dtype>& put(NdArray<dtype>& inArray, const Indices& inIndices, const NdArray<dtype>& inValues)
    {
        inArray.put(inIndices, inValues);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input value.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inSlice
    /// @param inValue
    /// @return reference to self
    ///
    template<typename dtype>
    NdArray<dtype>& put(NdArray<dtype>& inArray, const Slice& inSlice, const dtype& inValue)
    {
        inArray.put(inSlice, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input values.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inSlice
    /// @param inValues
    /// @return reference to self
    ///
    template<typename dtype>
    NdArray<dtype>& put(NdArray<dtype>& inArray, const Slice& inSlice, const NdArray<dtype>& inValues)
    {
        inArray.put(inSlice, inValues);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input value.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param rowIndices
    /// @param colIndices
    /// @param inValue
    /// @return reference to self
    ///
    template<typename dtype,
             typename RowIndices,
             typename ColIndices,
             type_traits::ndarray_int_concept<RowIndices> = 0,
             type_traits::ndarray_int_concept<ColIndices> = 0>
    NdArray<dtype>& put(NdArray<dtype>&   inArray,
                        const RowIndices& inRowIndices,
                        const ColIndices& inColIndices,
                        const dtype&      inValue)
    {
        inArray.put(inRowIndices, inColIndices, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input value.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param rowIndices
    /// @param inColSlice
    /// @param inValue
    /// @return reference to self
    ///
    template<typename dtype, typename RowIndices, type_traits::ndarray_int_concept<RowIndices> = 0>
    NdArray<dtype>&
        put(NdArray<dtype>& inArray, const RowIndices& inRowIndices, const Slice& inColSlice, const dtype& inValue)
    {
        inArray.put(inRowIndices, inColSlice, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input value.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowSlice
    /// @param colIndices
    /// @param inValue
    /// @return reference to self
    ///
    template<typename dtype, typename ColIndices, type_traits::ndarray_int_concept<ColIndices> = 0>
    NdArray<dtype>&
        put(NdArray<dtype>& inArray, const Slice& inRowSlice, const ColIndices& inColIndices, const dtype& inValue)
    {
        inArray.put(inRowSlice, inColIndices, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input value.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowSlice
    /// @param inColSlice
    /// @param inValue
    /// @return reference to self
    ///
    template<typename dtype>
    NdArray<dtype>& put(NdArray<dtype>& inArray, const Slice& inRowSlice, const Slice& inColSlice, const dtype& inValue)
    {
        inArray.put(inRowSlice, inColSlice, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input value.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowIndices
    /// @param inColIndex
    /// @param inValue
    /// @return reference to self
    ///
    template<typename dtype, typename Indices, type_traits::ndarray_int_concept<Indices> = 0>
    NdArray<dtype>& put(NdArray<dtype>& inArray, const Indices& inRowIndices, int32 inColIndex, const dtype& inValue)
    {
        inArray.put(inRowIndices, inColIndex, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input value.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowSlice
    /// @param inColIndex
    /// @param inValue
    /// @return reference to self
    ///
    template<typename dtype>
    NdArray<dtype>& put(NdArray<dtype>& inArray, const Slice& inRowSlice, int32 inColIndex, const dtype& inValue)
    {
        inArray.put(inRowSlice, inColIndex, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input value.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowIndex
    /// @param inColIndices
    /// @param inValue
    /// @return reference to self
    ///
    template<typename dtype, typename Indices, type_traits::ndarray_int_concept<Indices> = 0>
    NdArray<dtype>& put(NdArray<dtype>& inArray, int32 inRowIndex, const Indices& inColIndices, const dtype& inValue)
    {
        inArray.put(inRowIndex, inColIndices, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input value.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowIndex
    /// @param inColSlice
    /// @param inValue
    /// @return reference to self
    ///
    template<typename dtype>
    NdArray<dtype>& put(NdArray<dtype>& inArray, int32 inRowIndex, const Slice& inColSlice, const dtype& inValue)
    {
        inArray.put(inRowIndex, inColSlice, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input values.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowIndices
    /// @param inColIndices
    /// @param inValues
    /// @return reference to self
    ///
    template<typename dtype,
             typename RowIndices,
             typename ColIndices,
             type_traits::ndarray_int_concept<RowIndices> = 0,
             type_traits::ndarray_int_concept<ColIndices> = 0>
    NdArray<dtype>& put(NdArray<dtype>&       inArray,
                        const RowIndices&     inRowIndices,
                        const ColIndices&     inColIndices,
                        const NdArray<dtype>& inValues)
    {
        inArray.put(inRowIndices, inColIndices, inValues);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input values.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowIndices
    /// @param inColSlice
    /// @param inValues
    /// @return reference to self
    ///
    template<typename dtype, typename RowIndices, type_traits::ndarray_int_concept<RowIndices> = 0>
    NdArray<dtype>& put(NdArray<dtype>&       inArray,
                        const RowIndices&     inRowIndices,
                        const Slice&          inColSlice,
                        const NdArray<dtype>& inValues)
    {
        inArray.put(inRowIndices, inColSlice, inValues);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input values.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowSlice
    /// @param inColIndices
    /// @param inValues
    /// @return reference to self
    ///
    template<typename dtype, typename ColIndices, type_traits::ndarray_int_concept<ColIndices> = 0>
    NdArray<dtype>& put(NdArray<dtype>&       inArray,
                        const Slice&          inRowSlice,
                        const ColIndices&     inColIndices,
                        const NdArray<dtype>& inValues)
    {
        inArray.put(inRowSlice, inColIndices, inValues);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input values.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowSlice
    /// @param inColSlice
    /// @param inValues
    /// @return reference to self
    ///
    template<typename dtype>
    NdArray<dtype>&
        put(NdArray<dtype>& inArray, const Slice& inRowSlice, const Slice& inColSlice, const NdArray<dtype>& inValues)
    {
        inArray.put(inRowSlice, inColSlice, inValues);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input values.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowIndices
    /// @param inColIndex
    /// @param inValues
    /// @return reference to self
    ///
    template<typename dtype, typename Indices, type_traits::ndarray_int_concept<Indices> = 0>
    NdArray<dtype>&
        put(NdArray<dtype>& inArray, const Indices& inRowIndices, int32 inColIndex, const NdArray<dtype>& inValues)
    {
        inArray.put(inRowIndices, inColIndex, inValues);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input values.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowSlice
    /// @param inColIndex
    /// @param inValues
    /// @return reference to self
    ///
    template<typename dtype>
    NdArray<dtype>&
        put(NdArray<dtype>& inArray, const Slice& inRowSlice, int32 inColIndex, const NdArray<dtype>& inValues)
    {
        inArray.put(inRowSlice, inColIndex, inValues);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input values.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inRowIndex
    /// @param inColSlice
    /// @param inValues
    /// @return reference to self
    ///
    /// @param inArray
    template<typename dtype, typename Indices, type_traits::ndarray_int_concept<Indices> = 0>
    NdArray<dtype>&
        put(NdArray<dtype>& inArray, int32 inRowIndex, const Indices& inColIndices, const NdArray<dtype>& inValues)
    {
        inArray.put(inRowIndex, inColIndices, inValues);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Set the slice indices to the input values.
    ///
    /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.put.html
    ///
    /// @param inArray
    /// @param inRowIndex
    /// @param inColSlice
    /// @param inValues
    /// @return reference to self
    ///
    template<typename dtype>
    NdArray<dtype>&
        put(NdArray<dtype>& inArray, int32 inRowIndex, const Slice& inColSlice, const NdArray<dtype>& inValues)
    {
        inArray.put(inRowIndex, inColSlice, inValues);
        return inArray;
    }
} // namespace nc
