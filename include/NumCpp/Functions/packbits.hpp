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
    /// Packs the elements of a binary-valued array into bits in a uint8 array.
    ///
    /// Numpy Reference: https://numpy.org/doc/stable/reference/generated/numpy.packbits.html
    ///
    /// @param a: An array of integers or booleans whose elements should be packed to bits.
    /// @param axis: The dimension over which bit-packing is done. None implies packing the flattened array.
    /// @return NdArray<uint8>
    ///
    template<typename dtype, std::enable_if_t<std::is_integral_v<dtype> || std::is_same_v<dtype, bool>, int> = 0>
    NdArray<uint8> packbitsLittleEndian(const NdArray<dtype>& a, Axis axis = Axis::NONE)
    {
        switch (axis)
        {
            case Axis::NONE:
            {
                const auto numFullValues = a.size() / 8;
                const auto leftOvers     = a.size() % 8;
                const auto resultSize    = leftOvers == 0 ? numFullValues : numFullValues + 1;

                NdArray<uint8> result(1, resultSize);
                result.fill(0);

                for (typename NdArray<dtype>::size_type i = 0; i < numFullValues; ++i)
                {
                    const auto startIdx = i * 8;
                    for (auto bit = 0; bit < 8; ++bit)
                    {
                        auto value = static_cast<uint8>(a[startIdx + bit]);
                        value      = value == 0 ? 0 : 1;
                        result[i] |= (value << bit);
                    }
                }

                if (leftOvers != 0)
                {
                    const auto startIdx = numFullValues * 8;
                    for (std::remove_const_t<decltype(leftOvers)> bit = 0; bit < leftOvers; ++bit)
                    {
                        auto value = static_cast<uint8>(a[startIdx + bit]);
                        value      = value == 0 ? 0 : 1;
                        result.back() |= (value << bit);
                    }
                }

                return result;
            }
            case Axis::COL:
            {
                const auto aShape        = a.shape();
                const auto numFullValues = aShape.cols / 8;
                const auto leftOvers     = aShape.cols % 8;
                const auto resultSize    = leftOvers == 0 ? numFullValues : numFullValues + 1;

                NdArray<uint8> result(aShape.rows, resultSize);
                const auto     resultCSlice = result.cSlice();
                const auto     aCSlice      = a.cSlice();

                for (typename NdArray<dtype>::size_type row = 0; row < aShape.rows; ++row)
                {
                    result.put(row, resultCSlice, packbitsLittleEndian(a(row, aCSlice)));
                }

                return result;
            }
            case Axis::ROW:
            {
                return packbitsLittleEndian(a.transpose(), Axis::COL).transpose();
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
    /// Packs the elements of a binary-valued array into bits in a uint8 array.
    ///
    /// Numpy Reference: https://numpy.org/doc/stable/reference/generated/numpy.packbits.html
    ///
    /// @param a: An array of integers or booleans whose elements should be packed to bits.
    /// @param axis: The dimension over which bit-packing is done. None implies packing the flattened array.
    /// @return NdArray<uint8>
    ///
    template<typename dtype, std::enable_if_t<std::is_integral_v<dtype> || std::is_same_v<dtype, bool>, int> = 0>
    NdArray<uint8> packbitsBigEndian(const NdArray<dtype>& a, Axis axis = Axis::NONE)
    {
        switch (axis)
        {
            case Axis::NONE:
            {
                const auto numFullValues = a.size() / 8;
                const auto leftOvers     = a.size() % 8;
                const auto resultSize    = leftOvers == 0 ? numFullValues : numFullValues + 1;

                NdArray<uint8> result(1, resultSize);
                result.fill(0);

                for (typename NdArray<dtype>::size_type i = 0; i < numFullValues; ++i)
                {
                    const auto startIdx = i * 8;
                    for (auto bit = 0; bit < 8; ++bit)
                    {
                        auto value = static_cast<uint8>(a[startIdx + bit]);
                        value      = value == 0 ? 0 : 1;
                        result[i] |= (value << (7 - bit));
                    }
                }

                if (leftOvers != 0)
                {
                    const auto startIdx = numFullValues * 8;
                    for (std::remove_const_t<decltype(leftOvers)> bit = 0; bit < leftOvers; ++bit)
                    {
                        auto value = static_cast<uint8>(a[startIdx + bit]);
                        value      = value == 0 ? 0 : 1;
                        result.back() |= (value << (7 - bit));
                    }
                }

                return result;
            }
            case Axis::COL:
            {
                const auto aShape        = a.shape();
                const auto numFullValues = aShape.cols / 8;
                const auto leftOvers     = aShape.cols % 8;
                const auto resultSize    = leftOvers == 0 ? numFullValues : numFullValues + 1;

                NdArray<uint8> result(aShape.rows, resultSize);
                const auto     resultCSlice = result.cSlice();
                const auto     aCSlice      = a.cSlice();

                for (typename NdArray<dtype>::size_type row = 0; row < aShape.rows; ++row)
                {
                    result.put(row, resultCSlice, packbitsBigEndian(a(row, aCSlice)));
                }

                return result;
            }
            case Axis::ROW:
            {
                return packbitsBigEndian(a.transpose(), Axis::COL).transpose();
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }

} // namespace nc
