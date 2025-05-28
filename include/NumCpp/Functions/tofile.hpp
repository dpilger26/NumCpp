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

#include <string>

#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Write array to a file as binary.
    /// The data produced by this method can be recovered
    /// using the function fromfile().
    ///
    /// @param inArray
    /// @param inFilename
    ///
    template<typename dtype>
    void tofile(const NdArray<dtype>& inArray, const std::string& inFilename)
    {
        return inArray.tofile(inFilename);
    }

    //============================================================================
    // Method Description:
    /// Write array to a file as text.
    /// The data produced by this method can be recovered
    /// using the function fromfile().
    ///
    /// @param inArray
    /// @param inFilename
    /// @param inSep: Separator between array items for text output.
    ///
    template<typename dtype>
    void tofile(const NdArray<dtype>& inArray, const std::string& inFilename, const char inSep)
    {
        return inArray.tofile(inFilename, inSep);
    }
} // namespace nc
