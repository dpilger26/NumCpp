/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
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
/// Some helper routines for checking types
///
#pragma once

#include <type_traits>

#include "NumCpp/Core/Internal/TypeTraits.hpp"

#define STATIC_ASSERT_VALID_DTYPE(dtype) \
    static_assert(nc::is_valid_dtype_v<dtype>, "Template type is not a valid dtype for NdArray")

#define STATIC_ASSERT_ARITHMETIC(dtype) \
    static_assert(nc::is_arithmetic_v<dtype>, "Can only be used with arithmetic types")

#define STATIC_ASSERT_INTEGER(dtype) static_assert(nc::is_integral_v<dtype>, "Can only be used with integer types")

#define STATIC_ASSERT_UNSIGNED_INTEGER(dtype) \
    static_assert(nc::is_integral_v<dtype> && nc::is_unsigned_v<dtype>, "Can only be used with integer types")

#define STATIC_ASSERT_FLOAT(dtype) static_assert(nc::is_floating_point_v<dtype>, "Can only be used with float types")

#define STATIC_ASSERT_COMPLEX(dtype) static_assert(nc::is_complex_v<dtype>, "Can only be used with std::complex types")

#define STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype)                       \
    static_assert(nc::is_arithmetic_v<dtype> || nc::is_complex_v<dtype>, \
                  "Can only be used with arithmetic types or std::complex types")
