/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.3
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
/// Some helper routines for checking types
///
#pragma once

#include <type_traits>

namespace nc
{
    //============================================================================
    // Class Description:
    ///	std::enable_if helper, for c++11 compatibility
    ///
    template<bool B, class T = void>
    using enable_if_t = typename std::enable_if<B, T>::type;

    //============================================================================
    // Class Description:
    ///	Template class for determining if all of the types are arithmetic
    ///
    template <typename... Ts>
    struct all_arithmetic;

    //============================================================================
    // Class Description:
    ///	Template class specialization for determining if all of the types are arithmetic
    ///
    template <typename Head, typename... Tail>
    struct all_arithmetic<Head, Tail...>
    {
        static constexpr bool value = std::is_arithmetic<Head>::value && all_arithmetic<Tail...>::value;
    };

    //============================================================================
    // Class Description:
    ///	Template class specialization for determining if all of the types are arithmetic
    ///
    template <typename T>
    struct all_arithmetic<T>
    {
        static constexpr bool value = std::is_arithmetic<T>::value;
    };

    //============================================================================
    // Class Description:
    ///	Template class for determining if all of the types are the same as another type
    ///
    template <typename T1, typename... Ts>
    struct all_same;

    //============================================================================
    // Class Description:
    ///	Template class specialization for determining if all of the types are the same as another type
    ///
    template <typename T1, typename Head, typename... Tail>
    struct all_same<T1, Head, Tail...>
    {
        static constexpr bool value = std::is_same<T1, Head>::value && all_same<T1, Tail...>::value;
    };

    //============================================================================
    // Class Description:
    ///	Template class specialization for determining if all of the types are the same as another type
    ///
    template <typename T1, typename T2>
    struct all_same<T1, T2>
    {
        static constexpr bool value = std::is_same<T1, T2>::value;
    };
}
