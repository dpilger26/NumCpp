/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2026 David Pilger
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

#include <type_traits>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Unpacks elements of a uint8 array into a binary-valued output array.
    ///
    /// Each element of a represents a bit - field that should be unpacked into a binary -
    /// valued output array.The shape of the output array is either 1 - D(if axis is None) or
    /// the same shape as the input array with unpacking done along the axis specified.
    ///
    /// Numpy Reference: https://numpy.org/doc/stable/reference/generated/numpy.unpackbits.html
    ///
    /// @param a: An array of uint8 whose elements should be unpacked to bits.
    /// @param axis: The dimension over which bit-unpacking is done. None implies unpacking the flattened array.
    /// @return NdArray<uint8>
    ///
    inline NdArray<uint8> unpackbitsLittleEndian(const NdArray<uint8>& a, Axis axis = Axis::NONE)
    {
        switch (axis)
        {
            case Axis::NONE:
            {
                NdArray<uint8> result(1, a.size() * 8);

                for (NdArray<uint8>::size_type byte = 0; byte < a.size(); ++byte)
                {
                    const auto startIdx  = byte * 8;
                    const auto byteValue = a[byte];

                    for (uint8 bit = 0; bit < 8; ++bit)
                    {
                        result[startIdx + bit] = static_cast<uint8>((byteValue & (uint8{ 1 } << bit)) >> bit);
                    }
                }

                return result;
            }
            case Axis::COL:
            {
                const auto     aShape = a.shape();
                NdArray<uint8> result(aShape.rows, aShape.cols * 8);
                const auto     resultCSlice = result.cSlice();
                const auto     aCSlice      = a.cSlice();

                for (NdArray<uint8>::size_type row = 0; row < aShape.rows; ++row)
                {
                    result.put(row, resultCSlice, unpackbitsLittleEndian(a(row, aCSlice)));
                }

                return result;
            }
            case Axis::ROW:
            {
                return unpackbitsLittleEndian(a.transpose(), Axis::COL).transpose();
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
    /// Unpacks elements of a uint8 array into a binary-valued output array.
    ///
    /// Each element of a represents a bit - field that should be unpacked into a binary -
    /// valued output array.The shape of the output array is either 1 - D(if axis is None) or
    /// the same shape as the input array with unpacking done along the axis specified.
    ///
    /// Numpy Reference: https://numpy.org/doc/stable/reference/generated/numpy.unpackbits.html
    ///
    /// @param a: An array of uint8 whose elements should be unpacked to bits.
    /// @param axis: The dimension over which bit-unpacking is done. None implies unpacking the flattened array.
    /// @return NdArray<uint8>
    ///
    inline NdArray<uint8> unpackbitsBigEndian(const NdArray<uint8>& a, Axis axis = Axis::NONE)
    {
        switch (axis)
        {
            case Axis::NONE:
            {
                NdArray<uint8> result(1, a.size() * 8);

                for (NdArray<uint8>::size_type byte = 0; byte < a.size(); ++byte)
                {
                    const auto startIdx  = byte * 8;
                    const auto byteValue = a[byte];

                    for (uint8 bit = 0; bit < 8; ++bit)
                    {
                        const auto bitToMask = static_cast<uint8>(7 - bit);
                        result[startIdx + bit] =
                            static_cast<uint8>((byteValue & (uint8{ 1 } << bitToMask)) >> bitToMask);
                    }
                }

                return result;
            }
            case Axis::COL:
            {
                const auto     aShape = a.shape();
                NdArray<uint8> result(aShape.rows, aShape.cols * 8);
                const auto     resultCSlice = result.cSlice();
                const auto     aCSlice      = a.cSlice();

                for (NdArray<uint8>::size_type row = 0; row < aShape.rows; ++row)
                {
                    result.put(row, resultCSlice, unpackbitsBigEndian(a(row, aCSlice)));
                }

                return result;
            }
            case Axis::ROW:
            {
                return unpackbitsBigEndian(a.transpose(), Axis::COL).transpose();
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }

} // namespace nc
