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

#include <initializer_list>
#include <vector>

#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/column_stack.hpp"
#include "NumCpp/Functions/row_stack.hpp"

namespace nc
{
    namespace detail
    {
        //============================================================================
        // Method Description:
        /// Join a sequence of arrays along an existing axis.
        ///
        /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.concatenate.html
        ///
        /// @param begin: the begin iterator
        /// @param end: the end iterator
        /// @param inAxis (Optional, default NONE)
        /// @return NdArray
        ///
        template<typename dtype, typename Iterator>
        NdArray<dtype> concatenate(Iterator begin, Iterator end, Axis inAxis = Axis::NONE)
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    uint32 finalSize = 0;
                    auto   iter      = begin;
                    while (iter != end)
                    {
                        const auto& ndarray = *iter++;
                        finalSize += ndarray.size();
                    }

                    NdArray<dtype> returnArray(1, finalSize);
                    uint32         offset = 0;
                    iter                  = begin;
                    while (iter != end)
                    {
                        const auto& ndarray = *iter++;
                        stl_algorithms::copy(ndarray.cbegin(), ndarray.cend(), returnArray.begin() + offset);
                        offset += ndarray.size();
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    return row_stack<dtype>(begin, end);
                }
                case Axis::COL:
                {
                    return column_stack<dtype>(begin, end);
                }
                default:
                {
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
                }
            }
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Join a sequence of arrays along an existing axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.concatenate.html
    ///
    /// @param inArrayList
    /// @param inAxis (Optional, default NONE)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> concatenate(const std::initializer_list<NdArray<dtype>>& inArrayList, Axis inAxis = Axis::NONE)
    {
        return detail::concatenate<dtype>(inArrayList.begin(), inArrayList.end(), inAxis);
    }

    //============================================================================
    // Method Description:
    /// Join a sequence of arrays along an existing axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.concatenate.html
    ///
    /// @param inArrayList
    /// @param inAxis (Optional, default NONE)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> concatenate(const std::vector<NdArray<dtype>>& inArrayList, Axis inAxis = Axis::NONE)
    {
        return detail::concatenate<dtype>(inArrayList.begin(), inArrayList.end(), inAxis);
    }
} // namespace nc
