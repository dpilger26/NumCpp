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

#include <sstream>
#include <string_view>

#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Construct an array from data in a string
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fromstring.html
    ///
    /// @param inStr
    /// @param inSep: Delimiter separator between values in the string
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fromstring(const std::string& inStr, const char inSep = ' ')
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::istringstream inputStream(inStr);
        auto               values = std::vector<dtype>{};
        dtype              value{};
        for (std::string segment; std::getline(inputStream, segment, inSep);)
        {
            if (!inputStream.fail())
            {
                std::istringstream segmentStream(segment);
                while (segmentStream >> value)
                {
                    if (!inputStream.fail())
                    {
                        values.push_back(value);
                    }
                }
            }
        }

        return NdArray<dtype>(values);
    }
} // namespace nc
