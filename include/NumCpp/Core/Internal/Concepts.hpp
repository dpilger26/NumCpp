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
/// Modern C++ concepts for type checking and constraints
///
#pragma once

#include <concepts>
#include <complex>
#include <type_traits>

namespace nc
{
    //============================================================================
    // Concept Description:
    /// Concept for checking if all types are arithmetic
    ///
    template<typename... Ts>
    concept AllArithmetic = (std::is_arithmetic_v<Ts> && ...);

    //============================================================================
    // Concept Description:
    /// Concept for checking if all types are the same as the first type
    ///
    template<typename T1, typename... Ts>
    concept AllSame = (std::is_same_v<T1, Ts> && ...);

    //============================================================================
    // Concept Description:
    /// Concept for checking if a type is a valid dtype for NdArray
    ///
    template<typename dtype>
    concept ValidDtype = requires
    {
        requires std::is_default_constructible_v<dtype>;
        requires std::is_nothrow_copy_constructible_v<dtype>;
        requires std::is_nothrow_move_constructible_v<dtype>;
        requires std::is_nothrow_copy_assignable_v<dtype>;
        requires std::is_nothrow_move_assignable_v<dtype>;
        requires std::is_nothrow_destructible_v<dtype>;
        requires !std::is_void_v<dtype>;
        requires !std::is_pointer_v<dtype>;
        requires !std::is_array_v<dtype>;
        requires !std::is_union_v<dtype>;
        requires !std::is_function_v<dtype>;
        requires !std::is_abstract_v<dtype>;
    };

    // Forward declare
    template<typename dtype, class Allocator>
    class NdArray;

    //============================================================================
    // Concept Description:
    /// Concept for checking if a type is an NdArray of integral type (for index arrays)
    ///
    template<typename T>
    concept NdArrayInt = requires {
        typename T::value_type;
        typename T::allocator_type;
        requires std::is_integral_v<typename T::value_type>;
        requires std::is_same_v<T, NdArray<typename T::value_type, typename T::allocator_type>>;
    };

    //============================================================================
    // Concept Description:
    /// Concept for checking if a type is an NdArray of signed integral type (for signed index arrays)
    ///
    template<typename T>
    concept NdArraySignedInt = requires {
        typename T::value_type;
        typename T::allocator_type;
        requires std::is_signed_v<typename T::value_type>;
        requires std::is_integral_v<typename T::value_type>;
        requires std::is_same_v<T, NdArray<typename T::value_type, typename T::allocator_type>>;
    };

    //============================================================================
    // Concept Description:
    /// Concept for checking if a type is std::complex<>
    ///
    template<typename T>
    concept IsComplex = requires
    {
        requires std::is_same_v<T, std::complex<typename T::value_type>>;
    };

    //============================================================================
    // Concept Description:
    /// Concept for checking if one value is greater than another at compile time
    ///
    template<std::size_t Value1, std::size_t Value2>
    concept GreaterThan = (Value1 > Value2);

} // namespace nc 