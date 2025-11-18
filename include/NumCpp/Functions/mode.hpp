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

#include <algorithm>
#include <complex>
#include <unordered_map>
#include <utility>

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //===========================================================================
    // Method Description:
    /// Compute the mode along the specified axis.
    ///
    /// NumPy Reference: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html>
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return NdArray
    ///
    template<typename dtype, typename HashFunction = std::hash<dtype>>
    NdArray<dtype> mode(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        const auto modeFunction =
            [](typename NdArray<dtype>::const_iterator iterBegin, typename NdArray<dtype>::const_iterator iterEnd)
        {
            std::unordered_map<dtype, int, HashFunction> counts{};
            auto                                         greatestCount = int{ 0 };
            dtype                                        mode{};
            for (auto iter = iterBegin; iter != iterEnd; ++iter)
            {
                const auto& value = *iter;

                if (counts.count(value) > 0)
                {
                    auto& count = counts[value];
                    ++count;
                    if (count > greatestCount)
                    {
                        greatestCount = count;
                        mode          = value;
                    }
                }
                else
                {
                    counts.insert({ value, 1 });
                }
            }

            return mode;
        };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                NdArray<dtype> returnArray = { modeFunction(inArray.cbegin(), inArray.cend()) };
                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<dtype> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    returnArray(0, row) = modeFunction(inArray.cbegin(row), inArray.cend(row));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return mode(inArray.transpose(), Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {};
            }
        }
    }

    //============================================================================
    // Method Description:
    /// Compute the mode along the specified axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.mode.html
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<dtype>> mode(const NdArray<std::complex<dtype>>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return mode<std::complex<dtype>, ComplexHash<dtype>>(inArray, inAxis);
    }
} // namespace nc
