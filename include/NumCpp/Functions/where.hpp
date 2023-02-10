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

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Return elements, either from x or y, depending on the input mask.
    /// The output array contains elements of x where mask is True, and
    /// elements from y elsewhere.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.where.html
    ///
    /// @param inMask
    /// @param inA
    /// @param inB
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> where(const NdArray<bool>& inMask, const NdArray<dtype>& inA, const NdArray<dtype>& inB)
    {
        const auto shapeMask = inMask.shape();
        const auto shapeA    = inA.shape();
        if (shapeA != inB.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("input inA and inB must be the same shapes.");
        }

        if (shapeMask != shapeA)
        {
            THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the input arrays.");
        }

        auto outArray = NdArray<dtype>(shapeMask);

        uint32 idx = 0;
        for (auto maskValue : inMask)
        {
            if (maskValue)
            {
                outArray[idx] = inA[idx];
            }
            else
            {
                outArray[idx] = inB[idx];
            }
            ++idx;
        }

        return outArray;
    }

    //============================================================================
    // Method Description:
    /// Return elements, either from x or y, depending on the input mask.
    /// The output array contains elements of x where mask is True, and
    /// elements from y elsewhere.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.where.html
    ///
    /// @param inMask
    /// @param inA
    /// @param inB
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> where(const NdArray<bool>& inMask, const NdArray<dtype>& inA, dtype inB)
    {
        const auto shapeMask = inMask.shape();
        const auto shapeA    = inA.shape();
        if (shapeMask != shapeA)
        {
            THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the input arrays.");
        }

        auto outArray = NdArray<dtype>(shapeMask);

        uint32 idx = 0;
        for (auto maskValue : inMask)
        {
            if (maskValue)
            {
                outArray[idx] = inA[idx];
            }
            else
            {
                outArray[idx] = inB;
            }
            ++idx;
        }

        return outArray;
    }

    //============================================================================
    // Method Description:
    /// Return elements, either from x or y, depending on the input mask.
    /// The output array contains elements of x where mask is True, and
    /// elements from y elsewhere.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.where.html
    ///
    /// @param inMask
    /// @param inA
    /// @param inB
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> where(const NdArray<bool>& inMask, dtype inA, const NdArray<dtype>& inB)
    {
        const auto shapeMask = inMask.shape();
        const auto shapeB    = inB.shape();
        if (shapeMask != shapeB)
        {
            THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the input arrays.");
        }

        auto outArray = NdArray<dtype>(shapeMask);

        uint32 idx = 0;
        for (auto maskValue : inMask)
        {
            if (maskValue)
            {
                outArray[idx] = inA;
            }
            else
            {
                outArray[idx] = inB[idx];
            }
            ++idx;
        }

        return outArray;
    }

    //============================================================================
    // Method Description:
    /// Return elements, either from x or y, depending on the input mask.
    /// The output array contains elements of x where mask is True, and
    /// elements from y elsewhere.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.where.html
    ///
    /// @param inMask
    /// @param inA
    /// @param inB
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> where(const NdArray<bool>& inMask, dtype inA, dtype inB)
    {
        auto outArray = NdArray<dtype>(inMask.shape());

        uint32 idx = 0;
        for (auto maskValue : inMask)
        {
            if (maskValue)
            {
                outArray[idx] = inA;
            }
            else
            {
                outArray[idx] = inB;
            }
            ++idx;
        }

        return outArray;
    }
} // namespace nc
