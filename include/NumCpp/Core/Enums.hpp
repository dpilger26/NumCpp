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
/// Enumerations
///
#pragma once

namespace nc
{
    //================================================================================
    // Class Description:
    /// Enum To describe an axis
    enum class Axis
    {
        NONE = 0,
        ROW,
        COL
    };

    //================================================================================
    // Class Description:
    /// Enum for endianess
    enum class Endian
    {
        NATIVE = 0,
        BIG,
        LITTLE
    };

    //================================================================================
    // Class Description:
    /// Policy for NdArray constructor that takes in a pointer to data
    enum class PointerPolicy
    {
        COPY,
        SHELL
    };

    //================================================================================
    // Class Description:
    /// Bias boolean
    enum class Bias : bool
    {
        YES = true,
        NO  = false
    };

    //================================================================================
    // Class Description:
    /// End Point boolean
    enum class EndPoint : bool
    {
        YES = true,
        NO  = false
    };

    //================================================================================
    // Class Description:
    /// Increasing boolean
    enum class Increasing : bool
    {
        YES = true,
        NO  = false
    };

    //================================================================================
    // Class Description:
    /// Is Roots boolean
    enum class IsRoots : bool
    {
        YES = true,
        NO  = false
    };

    //================================================================================
    // Class Description:
    /// Replace boolean
    enum class Replace : bool
    {
        YES = true,
        NO  = false
    };

    //================================================================================
    // Class Description:
    /// Print Elapsed Time boolean
    enum class PrintElapsedTime : bool
    {
        YES = true,
        NO  = false
    };

    //================================================================================
    // Class Description:
    /// Print Results boolean
    enum class PrintResults : bool
    {
        YES = true,
        NO  = false
    };

    //============================================================================
    // Class Description:
    /// Right or Left side
    ///
    enum class Side
    {
        LEFT,
        RIGHT
    };

    //============================================================================
    // Class Description:
    /// Interpolation method
    ///
    enum class InterpolationMethod
    {
        LINEAR,
        LOWER,
        HIGHER,
        NEAREST,
        MIDPOINT
    };
} // namespace nc