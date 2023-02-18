# TODO

* issue #177: slice with integer index types
* issue #170: insert()
* issue #74, #147, #174: broadcasting

* setup job for clang-tidy
* setup job for cpp-check

```cpp
template<typename RowIndices,
            typename ColIndices,
            ndarray_int_concept<RowIndices> = 0,
            ndarray_int_concept<ColIndices> = 0>
                 
//============================================================================
// Class Description:
/// Template class for determining if dtype is a valid index type for NdArray
///
template<typename>
struct is_ndarray_int : std::false_type
{
};

//============================================================================
// Class Description:
/// Template class for determining if dtype is a valid index typefor NdArray
///

template<typename dtype, typename Allocator>
struct is_ndarray_int<NdArray<dtype, Allocator>>
{
    static constexpr bool value = std::is_integral_v<dtype>;
};

//============================================================================
// Class Description:
/// is_ndarray_int helper
///
template<typename T>
constexpr bool is_ndarray_int_v = is_ndarray_int<T>::value;

//============================================================================
// Class Description:
/// is_ndarray_int
///
template<typename T>
using ndarray_int_concept = std::enable_if_t<is_ndarray_int_v<T>, int>;
```
