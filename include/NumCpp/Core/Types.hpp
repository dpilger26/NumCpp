/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
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
/// Usefull types
///
#pragma once

#include <cstdint>

namespace nc
{
    //====================================Typedefs====================================
    using int64 = std::int64_t;
    using int32 = std::int32_t;
    using int16 = std::int16_t;
    using int8 = std::int8_t;
    using uint64 = std::uint64_t;
    using uint32 = std::uint32_t;
    using uint16 = std::uint16_t;
    using uint8 = std::uint8_t;

    //================================================================================
    ///						Enum To describe an axis
    enum class Axis { NONE = 0, ROW, COL };

    //================================================================================
    ///						Enum for endianess
    enum class Endian { NATIVE = 0, BIG, LITTLE };
} // namespace nc
