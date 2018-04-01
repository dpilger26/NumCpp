// Copyright 2018 David Pilger
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files(the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, 
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
// permit persons to whom the Software is furnished to do so, subject to the following 
// conditions :
//
// The above copyright notice and this permission notice shall be included in all copies 
// or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
// DEALINGS IN THE SOFTWARE.

#pragma once

#include<cstdint>

namespace NumC
{
    //====================================Typedefs====================================
    typedef int64_t		int64;
    typedef int32_t		int32;
    typedef int16_t		int16;
    typedef int8_t		int8;
    typedef uint64_t	uint64;
    typedef uint32_t	uint32;
    typedef uint16_t	uint16;
    typedef uint8_t		uint8;

    //================================================================================
    // Class Description:
    //						Enum To describe an axis
    //
    struct Axis { enum Type { NONE = 0, ROW, COL }; };

    //================================================================================
    // Class Description:
    //						Enum for endianess
    //
    struct Endian { enum Type { NATIVE = 0, BIG, LITTLE }; };
}
