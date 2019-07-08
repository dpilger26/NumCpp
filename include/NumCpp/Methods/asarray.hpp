/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
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
/// Methods for working with NdArrays
///
#pragma once

#include "NumCpp/NdArray.hpp"

#include <initializer_list>
#include <deque>
#include <set>
#include <vector>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Convert the vector to an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param
    ///				inVector
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(const std::vector<dtype>& inVector) noexcept
    {
        return NdArray<dtype>(inVector);
    }

    //============================================================================
    // Method Description:
    ///						Convert the deque to an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param
    ///				inDeque
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(const std::deque<dtype>& inDeque) noexcept
    {
        return NdArray<dtype>(inDeque);
    }

    //============================================================================
    // Method Description:
    ///						Convert the set to an array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param
    ///				inSet
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(const std::deque<dtype>& inSet) noexcept
    {
        return NdArray<dtype>(inSet);
    }

    //============================================================================
    // Method Description:
    ///						Convert the list initializer to an array.
    ///						eg: NdArray<int> myArray = NC::asarray<int>({1,2,3});
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param
    ///				inList
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(std::initializer_list<dtype>& inList) noexcept
    {
        return NdArray<dtype>(inList);
    }

        //============================================================================
    // Method Description:
    ///						Convert the list initializer to an array.
    ///						eg: NdArray<int> myArray = NC::asarray<int>({{1,2,3}, {4, 5, 6}});
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html
    ///
    /// @param
    ///				inList
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> asarray(std::initializer_list<std::initializer_list<dtype> >& inList) noexcept
    {
        return NdArray<dtype>(inList);
    }
}
