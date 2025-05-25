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

#include <array>
#include <deque>
#include <forward_list>
#include <initializer_list>
#include <iterator>
#include <list>
#include <set>
#include <type_traits>
#include <vector>

#include "NumCpp/Core/Enums.hpp"
#include "NumCpp/Core/Internal/Concepts.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Convert the list initializer to an array.
    /// eg: NdArray<int> myArray = NC::asarray<int>({1,2,3});
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param inList
    /// @return NdArray
    ///
    template<ValidDtype dtype>
    NdArray<dtype> asarray(std::initializer_list<dtype> inList)
    {
        return NdArray<dtype>(inList);
    }

    //============================================================================
    // Method Description:
    /// Convert the list initializer to an array.
    /// eg: NdArray<int> myArray = NC::asarray<int>({{1,2,3}, {4, 5, 6}});
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param inList
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(std::initializer_list<std::initializer_list<dtype>> inList)
    {
        return NdArray<dtype>(inList);
    }

    //============================================================================
    // Method Description:
    /// Convert the std::array to an array.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param inArray
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return NdArray
    ///
    template<ValidDtype dtype, size_t ArraySize>
    NdArray<dtype> asarray(std::array<dtype, ArraySize>& inArray, PointerPolicy pointerPolicy = PointerPolicy::COPY)
    {
        return NdArray<dtype>(inArray, pointerPolicy);
    }

    //============================================================================
    // Method Description:
    /// Convert the std::array to an array.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param inArray
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return NdArray
    ///
    template<typename dtype, size_t Dim0Size, size_t Dim1Size>
    NdArray<dtype> asarray(std::array<std::array<dtype, Dim1Size>, Dim0Size>& inArray,
                           PointerPolicy                                      pointerPolicy = PointerPolicy::COPY)
    {
        return NdArray<dtype>(inArray, pointerPolicy);
    }

    //============================================================================
    // Method Description:
    /// Convert the vector to an array.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param inVector
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return NdArray
    ///
    template<ValidDtype dtype>
    NdArray<dtype> asarray(std::vector<dtype>& inVector, PointerPolicy pointerPolicy = PointerPolicy::COPY)
    {
        return NdArray<dtype>(inVector, pointerPolicy);
    }

    //============================================================================
    // Method Description:
    /// Convert the vector to an array. Makes a copy of the data.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param inVector
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(const std::vector<std::vector<dtype>>& inVector)
    {
        return NdArray<dtype>(inVector);
    }

    //============================================================================
    // Method Description:
    /// Convert the vector to an array.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param inVector
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return NdArray
    ///
    template<typename dtype, size_t Dim1Size>
    NdArray<dtype> asarray(std::vector<std::array<dtype, Dim1Size>>& inVector,
                           PointerPolicy                             pointerPolicy = PointerPolicy::COPY)
    {
        return NdArray<dtype>(inVector, pointerPolicy);
    }

    //============================================================================
    // Method Description:
    /// Convert the vector to an array.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param inDeque
    /// @return NdArray
    ///
    template<ValidDtype dtype>
    NdArray<dtype> asarray(const std::deque<dtype>& inDeque)
    {
        return NdArray<dtype>(inDeque);
    }

    //============================================================================
    // Method Description:
    /// Convert the vector to an array. Makes a copy of the data.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param inDeque
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(const std::deque<std::deque<dtype>>& inDeque)
    {
        return NdArray<dtype>(inDeque);
    }

    //============================================================================
    // Method Description:
    /// Convert the set to an array. Makes a copy of the data.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param inSet
    /// @return NdArray
    ///
    template<typename dtype, typename dtypeComp>
    NdArray<dtype> asarray(const std::set<dtype, dtypeComp>& inSet)
    {
        return NdArray<dtype>(inSet);
    }

    //============================================================================
    // Method Description:
    /// Convert the list to an array. Makes a copy of the data.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param inList
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(const std::list<dtype>& inList)
    {
        return NdArray<dtype>(inList);
    }

    //============================================================================
    // Method Description:
    /// Convert the forward_list to an array. Makes a copy of the data.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param iterBegin
    /// @param iterEnd
    /// @return NdArray
    ///
    template<typename Iterator>
    auto asarray(Iterator iterBegin, Iterator iterEnd)
    {
        return NdArray<typename std::iterator_traits<Iterator>::value_type>(iterBegin, iterEnd);
    }

    //============================================================================
    // Method Description:
    /// Convert the forward_list to an array. Makes a copy of the data.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param iterBegin
    /// @param iterEnd
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(const dtype* iterBegin, const dtype* iterEnd)
    {
        return NdArray<dtype>(iterBegin, iterEnd);
    }

    //============================================================================
    // Method Description:
    /// Convert the c-style array to an array. Makes a copy of the data.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param ptr to array
    /// @param size: the number of elements in the array
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(const dtype* ptr, uint32 size)
    {
        return NdArray<dtype>(ptr, size);
    }

    //============================================================================
    // Method Description:
    /// Convert the c-style array to an array. Makes a copy of the data.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param ptr to array
    /// @param numRows: number of rows of the buffer
    /// @param numCols: number of cols of the buffer
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(const dtype* ptr, uint32 numRows, uint32 numCols)
    {
        return NdArray<dtype>(ptr, numRows, numCols);
    }

    //============================================================================
    // Method Description:
    /// Convert the c-style array to an array. Makes a copy of the data.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param ptr to array
    /// @param size: the number of elements in the array
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return NdArray
    ///
    template<typename dtype, std::integral UIntType>
    requires (!std::is_same_v<UIntType, bool>)
    NdArray<dtype> asarray(dtype* ptr, UIntType size, PointerPolicy pointerPolicy = PointerPolicy::COPY) noexcept
    {
        return NdArray<dtype>(ptr, size, pointerPolicy);
    }

    //============================================================================
    // Method Description:
    /// Convert the c-style array to an array. Makes a copy of the data.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param ptr to array
    /// @param numRows: number of rows of the buffer
    /// @param numCols: number of cols of the buffer
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return NdArray
    ///
    template<typename dtype, std::integral UIntType1, std::integral UIntType2>
    requires (!std::is_same_v<UIntType1, bool> && !std::is_same_v<UIntType2, bool>)
    NdArray<dtype> asarray(dtype*        ptr,
                           UIntType1     numRows,
                           UIntType2     numCols,
                           PointerPolicy pointerPolicy = PointerPolicy::COPY) noexcept
    {
        return NdArray<dtype>(ptr, numRows, numCols, pointerPolicy);
    }
} // namespace nc
