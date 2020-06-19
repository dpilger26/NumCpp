/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.0.0
///
/// @section License
/// Copyright 2020 David Pilger
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

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/column_stack.hpp"
#include "NumCpp/Functions/row_stack.hpp"
#include "NumCpp/NdArray.hpp"

#include <initializer_list>
#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Compute the variance along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.stack.html
    ///
    /// @param      inArrayList: {list} of arrays to stack
    /// @param      inAxis: axis to stack the input NdArrays
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> stack(const std::initializer_list<NdArray<dtype> >& inArrayList, Axis inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::ROW:
            {
                return row_stack(inArrayList);
            }
            case Axis::COL:
            {
                return column_stack(inArrayList);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("inAxis must be either ROW or COL.");
            }
        }

        return NdArray<dtype>();  // getting rid of compiler warning
    }
}
