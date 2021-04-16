#pragma once

#include "NumCpp/Core/Internal/TypeTraits.hpp"

#include <initializer_list>
#include <type_traits>

namespace nc_develop
{
    //============================================================================
    // Class Description:
    ///	Template class for determining if type is std::initializer_list<>
    ///
    template<class T>
    struct is_initializer_list : public std::false_type {};

    //============================================================================
    // Class Description:
    ///	Template class specialization for determining if type is std::initializer_list<>
    ///
    template<class T>
    struct is_initializer_list<std::initializer_list<T>> : public std::true_type {};

    //============================================================================
    // Class Description:
    ///	std::is_initializer_list helper
    ///
    template<class T>
    constexpr bool is_initializer_list_v = is_initializer_list<T>::value;

    //============================================================================
    // Class Description:
    ///	Template class for determining if dtype is a valid dtype for NdArray
    ///
    template<typename dtype>
    struct is_valid_dtype 
    {
        static constexpr bool value = std::is_default_constructible<dtype>::value &&
            std::is_nothrow_copy_constructible<dtype>::value &&
            std::is_nothrow_move_constructible<dtype>::value &&
            std::is_nothrow_copy_assignable<dtype>::value &&
            std::is_nothrow_move_assignable<dtype>::value &&
            std::is_nothrow_destructible<dtype>::value &&
            !std::is_void<dtype>::value &&
            !std::is_pointer<dtype>::value &&
            !is_initializer_list_v<dtype> &&
            !std::is_array<dtype>::value &&
            !std::is_union<dtype>::value &&
            !std::is_function<dtype>::value &&
            !std::is_abstract<dtype>::value;
    };

    //============================================================================
    // Class Description:
    ///	is_valid_dtype helper
    ///
    template<class dtype>
    constexpr bool is_valid_dtype_v = is_valid_dtype<dtype>::value;

    //============================================================================
    // Class Description:
    ///	Template class for determining if all of the types are convertable to a type
    ///
    template <typename ToType, typename... Ts>
    struct all_convertable;

    //============================================================================
    // Class Description:
    ///	Template class specialization for determining if all of the types are convertable to std::size_t
    ///
    template <typename ToType, typename Head, typename... Tail>
    struct all_convertable<ToType, Head, Tail...>
    {
        static constexpr bool value = std::is_convertible_v<Head, ToType> && all_convertable<ToType, Tail...>::value;
    };

    //============================================================================
    // Class Description:
    ///	Template class specialization for determining if all of the types are convertable to std::size_t
    ///
    template <typename ToType, typename T>
    struct all_convertable<ToType, T>
    {
        static constexpr bool value = std::is_convertible_v<T, ToType>;
    };

    //============================================================================
    // Class Description:
    ///	all_convertable_size_t helper
    ///
    template<typename ToType, typename... Ts>
    constexpr bool all_convertable_v = all_convertable<ToType, Ts...>::value;

    //============================================================================
    // Class Description:
    ///	Checks if value is greater than zero
    ///
    template<std::size_t value>
    inline constexpr bool greater_than_zero_v = value > 0;

    //============================================================================
    // Class Description:
    ///	Checks if container is an STL conforming container
    ///
    template <typename ContainerType>
    class is_conforming_container
    {
        struct no {};
        struct yes {};

        template <typename T,
            std::enable_if_t<std::is_convertible_v<typename T::value_type, std::size_t>, int> = 0,
            std::enable_if_t<std::is_same_v<typename T::const_iterator, decltype(std::declval<const T>().begin())>, int> = 0,
            std::enable_if_t<std::is_same_v<typename T::const_iterator, decltype(std::declval<const T>().end())>, int> = 0>
            static yes test(int) { return yes{}; }

        template <typename T>
        static no test(...) { return no{}; }

    public:
        enum { value = std::is_same<yes, decltype(test<ContainerType>(0))>::value };
    };

    template<typename ContainerType>
    inline constexpr bool is_conforming_container_v = is_conforming_container<ContainerType>::value;
}
