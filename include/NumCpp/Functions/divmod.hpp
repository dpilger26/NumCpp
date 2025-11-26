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

#include <cmath>
#include <type_traits>

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //===========================================================================
    // Method Description:
    /// Return element-wise quotient and remainder simultaneously along the specified axis.
    ///
    /// NumPy Reference: https://numpy.org/doc/2.3/reference/generated/numpy.divmod.html#numpy-divmod
    ///
    /// @param inArray1
    /// @param inArray2
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    std::pair<NdArray<dtype>, NdArray<dtype>> divmod(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (inArray1.size() != inArray2.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("Arrays must have the same size.");
        }

        auto div = NdArray<dtype>(inArray1.shape());
        auto mod = NdArray<dtype>(inArray1.shape());

        for (auto i = 0u; i < inArray1.size(); ++i)
        {
            if constexpr (std::is_floating_point_v<dtype>)
            {
                div[i] = std::floor(inArray1[i] / inArray2[i]);
                mod[i] = std::fmod(inArray1[i], inArray2[i]);
            }
            else
            {
                div[i] = inArray1[i] / inArray2[i];
                mod[i] = inArray1[i] % inArray2[i];
            }
        }

        return std::make_pair(div, mod);
    }
} // namespace nc
