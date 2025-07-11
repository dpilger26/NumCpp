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
/// Compute the dot product of two or more arrays in a single function call.
///
#pragma once

#include <initializer_list>
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Functions/dot.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::linalg
{
    //============================================================================
    // Method Description:
    /// Compute the dot product of two or more arrays in a single
    /// function call.
    ///
    /// NumPy Reference:
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.multi_dot.html#numpy.linalg.multi_dot
    ///
    /// @param inList: list of arrays
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> multi_dot(const std::initializer_list<NdArray<dtype>>& inList)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        typename std::initializer_list<NdArray<dtype>>::iterator iter = inList.begin();

        if (inList.size() == 0)
        {
            THROW_INVALID_ARGUMENT_ERROR("input empty list of arrays.");
        }
        else if (inList.size() == 1)
        {
            return iter->copy();
        }

        NdArray<dtype> returnArray = dot<dtype>(*iter, *(iter + 1));
        iter += 2;
        for (; iter < inList.end(); ++iter)
        {
            returnArray = dot(returnArray, *iter);
        }

        return returnArray;
    }
} // namespace nc::linalg
