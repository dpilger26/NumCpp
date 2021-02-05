/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2021 David Pilger
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
/// Functions for determining and swaping endianess
///
#pragma once

#include "NumCpp/Core/Types.hpp"

#include <climits>

namespace nc
{
    namespace endian
    {
        //============================================================================
        // Function Description:
        ///	Determines the endianess of the system
        ///
        /// @return bool true if the system is little endian
        ///
        inline bool isLittleEndian() noexcept
        {
            union
            {
                uint32  i;
                char    c[4];
            } fourBytes = { 0x01020304 };

            return fourBytes.c[0] == 4;
        }

        //============================================================================
        // Function Description:
        ///	Swaps the bytes of the input value
        ///
        /// @param	value
        /// @return byte swapped value
        ///
        template <typename dtype>
        dtype byteSwap(dtype value) noexcept
        {
            STATIC_ASSERT_INTEGER(dtype);
            static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");

            union
            {
                dtype   value;
                uint8   value8[sizeof(dtype)];
            } source, dest;

            source.value = value;

            for (size_t k = 0; k < sizeof(dtype); ++k)
            {
                dest.value8[k] = source.value8[sizeof(dtype) - k - 1];
            }

            return dest.value;
        }
    }  // namespace endian
}  // namespace nc
