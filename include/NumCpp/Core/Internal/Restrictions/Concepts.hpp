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
/// Some helper routines for checking types
///
#pragma once

#if __cplusplus >= 202002L

#include <complex>
#include <concepts>
#include <type_traits>

namespace nc
{
    // Forward declare
    template<typename dtype, class Allocator>
    class NdArray;

    //============================================================================
    // Concept Description:
    /// Checks that all template parameter types are arithmetic
    ///
    template<typename... Args>
    concept all_arithmetic = (std::is_arithmetic_v<Args> && ...);

    //============================================================================
    // Concept Description:
    /// Checks that all template parameter types are the same
    ///
    template<typename T, typename... Args>
    concept all_same = (std::same_as<T, Args> && ...);

    //============================================================================
    // Concept Description:
    /// Checks that dtype is a valid dtype for NdArray
    ///
    template<typename dtype>
    concept is_valid_dtype =
        (std::is_default_constructible<dtype>::value && std::is_nothrow_copy_constructible<dtype>::value &&
         std::is_nothrow_move_constructible<dtype>::value && std::is_nothrow_copy_assignable<dtype>::value &&
         std::is_nothrow_move_assignable<dtype>::value && std::is_nothrow_destructible<dtype>::value &&
         !std::is_void<dtype>::value && !std::is_pointer<dtype>::value && !std::is_array<dtype>::value &&
         !std::is_union<dtype>::value && !std::is_function<dtype>::value && !std::is_abstract<dtype>::value);

    //============================================================================
    // Concept Description:
    /// Concept for determining if dtype is a valid index type for NdArray
    ///
    template<typename NdArrayType>
    concept is_ndarray_int = requires(NdArrayType array) {
        typename NdArrayType::value_type;
        std::integral<typename NdArrayType::value_type>;
    };

    //============================================================================
    // Concept Description:
    /// Concept for determining if dtype is a valid index type for NdArray
    ///
    template<typename NdArrayType>
    concept is_ndarray_signed_int = requires(NdArrayType array) {
        typename NdArrayType::value_type;
        std::is_signed_v<typename NdArrayType::value_type>;
    };

    //============================================================================
    // Class Description:
    /// Concept for determining if type is std::complex<>
    ///
    template<typename T>
    concept is_complex = requires(T) { typename std::complex<std::remove_cvref_t<T>>::value_type; };

    //============================================================================
    // Class Description:
    /// type trait to test if one value is larger than another at compile time
    ///
    template<std::size_t Value1, std::size_t Value2>
    concept greater_than = (Value1 > Value2);
} // namespace nc

#endif
