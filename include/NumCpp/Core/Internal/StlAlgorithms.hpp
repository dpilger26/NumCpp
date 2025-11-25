/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2026 David Pilger
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
/// Macro to define whether or not c++17 parallel algorithm policies are supported
///
#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>

#if defined(__cpp_lib_parallel_algorithm) && defined(NUMCPP_USE_MULTITHREAD)
#define PARALLEL_ALGORITHMS_SUPPORTED
#define CONDITIONAL_NO_EXCEPT
#include <execution>
#else
#define CONDITIONAL_NO_EXCEPT noexcept
#endif

namespace nc::stl_algorithms
{
    //============================================================================
    // Method Description:
    /// Tests if all of the elements of a range satisy a predicate
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param p: unary predicate function
    /// @return bool
    ///
    template<class InputIt, class UnaryPredicate>
    bool all_of(InputIt first, InputIt last, UnaryPredicate p) CONDITIONAL_NO_EXCEPT
    {
        return std::all_of(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            p);
    }

    //============================================================================
    // Method Description:
    /// Tests if any of the elements of a range satisy a predicate
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param p: unary predicate function
    /// @return bool
    ///
    template<class InputIt, class UnaryPredicate>
    bool any_of(InputIt first, InputIt last, UnaryPredicate p) CONDITIONAL_NO_EXCEPT
    {
        return std::any_of(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            p);
    }

    //============================================================================
    // Method Description:
    /// Copies from one container to another
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param destination: the first iterator of the destination
    /// @return OutputIt
    ///
    template<class InputIt, class OutputIt>
    OutputIt copy(InputIt first, InputIt last, OutputIt destination) CONDITIONAL_NO_EXCEPT
    {
        return std::copy(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            destination);
    }

    //============================================================================
    // Method Description:
    /// Counts the values in the range
    ///
    /// @param first: the first iterator of container
    /// @param last: the last iterator of container
    /// @param value: the initial value
    /// @return count
    ///
    template<class InputIt, class T>
    typename std::iterator_traits<InputIt>::difference_type
        count(InputIt first, InputIt last, const T& value) CONDITIONAL_NO_EXCEPT
    {
        return std::count(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            value);
    }

    //============================================================================
    // Method Description:
    /// Test if two ranges are equal
    ///
    /// @param first1: the first iterator of first container
    /// @param last1: the last iterator of first container
    /// @param first2: the first iterator of second container
    /// @return bool
    ///
    template<class InputIt1, class InputIt2>
    bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2) CONDITIONAL_NO_EXCEPT
    {
        return std::equal(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first1,
            last1,
            first2);
    }

    //============================================================================
    // Method Description:
    /// Test if two ranges are equal
    ///
    /// @param first1: the first iterator of first container
    /// @param last1: the last iterator of first container
    /// @param first2: the first iterator of second container
    /// @param p: binary predicate to compare the elements
    /// @return bool
    ///
    template<class InputIt1, class InputIt2, class BinaryPredicate>
    bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, BinaryPredicate p) CONDITIONAL_NO_EXCEPT
    {
        return std::equal(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first1,
            last1,
            first2,
            p);
    }

    //============================================================================
    // Method Description:
    /// Fills the range with the value
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param value: the function to apply to the input iterators
    ///
    template<class ForwardIt, class T>
    void fill(ForwardIt first, ForwardIt last, const T& value) CONDITIONAL_NO_EXCEPT
    {
        return std::fill(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            value);
    }

    //============================================================================
    // Method Description:
    /// Returns the first element in the range [first, last)
    /// that satisfies specific criteria:
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param value: the value to find
    /// @return InputIt
    ///
    template<class InputIt, class T>
    InputIt find(InputIt first, InputIt last, const T& value) CONDITIONAL_NO_EXCEPT
    {
        return std::find(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            value);
    }

    //============================================================================
    // Method Description:
    /// Runs the function on each element of the range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param f: the function to apply to the input iterators
    ///
    template<class InputIt, class UnaryFunction>
    void for_each(InputIt first, InputIt last, UnaryFunction f)
    {
        std::for_each(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            f);
    }

    //============================================================================
    // Method Description:
    /// Returns true if the array is sorted
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @return bool true if sorted
    ///
    template<class ForwardIt>
    bool is_sorted(ForwardIt first, ForwardIt last) CONDITIONAL_NO_EXCEPT
    {
        return std::is_sorted(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last);
    }

    //============================================================================
    // Method Description:
    /// Returns true if the array is sorted
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param comp: comparitor function
    /// @return bool true if sorted
    ///
    template<class ForwardIt, class Compare>
    bool is_sorted(ForwardIt first, ForwardIt last, Compare comp) CONDITIONAL_NO_EXCEPT
    {
        return std::is_sorted(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            comp);
    }

    //============================================================================
    // Method Description:
    /// Returns the maximum element of the range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @return ForwordIt
    ///
    template<class ForwardIt>
    ForwardIt max_element(ForwardIt first, ForwardIt last) CONDITIONAL_NO_EXCEPT
    {
        return std::max_element(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last);
    }

    //============================================================================
    // Method Description:
    /// Returns the maximum element of the range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param comp: the comparitor function
    /// @return ForwordIt
    ///
    template<class ForwardIt, class Compare>
    ForwardIt max_element(ForwardIt first, ForwardIt last, Compare comp) CONDITIONAL_NO_EXCEPT
    {
        return std::max_element(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            comp);
    }

    //============================================================================
    // Method Description:
    /// Returns the minimum element of the range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @return ForwardIt
    template<class ForwardIt>
    ForwardIt min_element(ForwardIt first, ForwardIt last) CONDITIONAL_NO_EXCEPT
    {
        return std::min_element(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last);
    }

    //============================================================================
    // Method Description:
    /// Returns the minimum element of the range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param comp: the comparitor function
    /// @return ForwordIt
    ///
    template<class ForwardIt, class Compare>
    ForwardIt min_element(ForwardIt first, ForwardIt last, Compare comp) CONDITIONAL_NO_EXCEPT
    {
        return std::min_element(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            comp);
    }

    //============================================================================
    // Method Description:
    /// Runs the minimum and maximum elements of the range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @return std::pair
    ///
    template<class ForwardIt>
    std::pair<ForwardIt, ForwardIt> minmax_element(ForwardIt first, ForwardIt last) CONDITIONAL_NO_EXCEPT
    {
        return std::minmax_element(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last);
    }

    //============================================================================
    // Method Description:
    /// Runs the minimum and maximum elements of the range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param comp: the comparitor function
    /// @return std::pair
    ///
    template<class ForwardIt, class Compare>
    std::pair<ForwardIt, ForwardIt> minmax_element(ForwardIt first, ForwardIt last, Compare comp) CONDITIONAL_NO_EXCEPT
    {
        return std::minmax_element(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            comp);
    }

    //============================================================================
    // Method Description:
    /// Tests if none of the elements of a range satisy a predicate
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param p: unary predicate function
    /// @return bool
    ///
    template<class InputIt, class UnaryPredicate>
    bool none_of(InputIt first, InputIt last, UnaryPredicate p) CONDITIONAL_NO_EXCEPT
    {
        return std::none_of(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            p);
    }

    //============================================================================
    // Method Description:
    /// Sorts up to the nth element
    ///
    /// @param first: the first iterator of the range
    /// @param nth: the element that should be sorted
    /// @param last: the last iterator of the range
    ///
    template<class RandomIt>
    void nth_element(RandomIt first, RandomIt nth, RandomIt last) CONDITIONAL_NO_EXCEPT
    {
        std::nth_element(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            nth,
            last);
    }

    //============================================================================
    // Method Description:
    /// Sorts up to the nth element
    ///
    /// @param first: the first iterator of the range
    /// @param nth: the element that should be sorted
    /// @param last: the last iterator of the range
    /// @param comp: the comparitor function
    ///
    template<class RandomIt, class Compare>
    void nth_element(RandomIt first, RandomIt nth, RandomIt last, Compare comp) CONDITIONAL_NO_EXCEPT
    {
        std::nth_element(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            nth,
            last,
            comp);
    }

    //============================================================================
    // Method Description:
    /// replaces a value in the range with another value
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param oldValue: the value to replace
    /// @param newValue: the replacement value
    ///
    template<class ForwardIt, class T>
    void replace(ForwardIt first, ForwardIt last, const T& oldValue, const T& newValue) CONDITIONAL_NO_EXCEPT
    {
        std::replace(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            oldValue,
            newValue);
    }

    //============================================================================
    // Method Description:
    /// reverses the range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    ///
    template<class BidirIt>
    void reverse(BidirIt first, BidirIt last) CONDITIONAL_NO_EXCEPT
    {
        std::reverse(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last);
    }

    //============================================================================
    // Method Description:
    /// Rotates the elements of a range
    ///
    /// @param first: the first iterator of the range
    /// @param firstN: the element that should appear at the beginning of the rotated range
    /// @param last: the last iterator of the range
    ///
    template<class ForwardIt>
    void rotate(ForwardIt first, ForwardIt firstN, ForwardIt last) CONDITIONAL_NO_EXCEPT
    {
        std::rotate(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            firstN,
            last);
    }

    //============================================================================
    // Method Description:
    /// finds the difference of two ranges
    ///
    /// @param first1: the first iterator of the source
    /// @param last1: the last iterator of the source
    /// @param first2: the first iterator of the second source
    /// @param last2: the first iterator of the destination
    /// @param destination: the function to apply to the input iterators
    /// @return OutputIt
    ///
    template<class InputIt1, class InputIt2, class OutputIt>
    OutputIt set_difference(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt destination)
    {
        return std::set_difference(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first1,
            last1,
            first2,
            last2,
            destination);
    }

    //============================================================================
    // Method Description:
    /// finds the difference of two ranges
    ///
    /// @param first1: the first iterator of the source
    /// @param last1: the last iterator of the source
    /// @param first2: the first iterator of the second source
    /// @param last2: the first iterator of the destination
    /// @param destination: the function to apply to the input iterators
    /// @param comp: comparitor function
    /// @return OutputIt
    ///
    template<class InputIt1, class InputIt2, class OutputIt, class Compare>
    OutputIt set_difference(InputIt1 first1,
                            InputIt1 last1,
                            InputIt2 first2,
                            InputIt2 last2,
                            OutputIt destination,
                            Compare  comp) CONDITIONAL_NO_EXCEPT
    {
        return std::set_difference(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first1,
            last1,
            first2,
            last2,
            destination,
            comp);
    }

    //============================================================================
    // Method Description:
    /// finds the intersection of two ranges
    ///
    /// @param first1: the first iterator of the source
    /// @param last1: the last iterator of the source
    /// @param first2: the first iterator of the second source
    /// @param last2: the first iterator of the destination
    /// @param destination: the function to apply to the input iterators
    /// @return OutputIt
    ///
    template<class InputIt1, class InputIt2, class OutputIt>
    OutputIt set_intersection(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt destination)
        CONDITIONAL_NO_EXCEPT
    {
        return std::set_intersection(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first1,
            last1,
            first2,
            last2,
            destination);
    }

    //============================================================================
    // Method Description:
    /// finds the intersection of two ranges
    ///
    /// @param first1: the first iterator of the source
    /// @param last1: the last iterator of the source
    /// @param first2: the first iterator of the second source
    /// @param last2: the first iterator of the destination
    /// @param destination: the function to apply to the input iterators
    /// @param comp: comparitor function
    /// @return OutputIt
    ///
    template<class InputIt1, class InputIt2, class OutputIt, class Compare>
    OutputIt set_intersection(InputIt1 first1,
                              InputIt1 last1,
                              InputIt2 first2,
                              InputIt2 last2,
                              OutputIt destination,
                              Compare  comp) CONDITIONAL_NO_EXCEPT
    {
        return std::set_intersection(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first1,
            last1,
            first2,
            last2,
            destination,
            comp);
    }

    //============================================================================
    // Method Description:
    /// finds the union of two ranges
    ///
    /// @param first1: the first iterator of the source
    /// @param last1: the last iterator of the source
    /// @param first2: the first iterator of the second source
    /// @param last2: the first iterator of the destination
    /// @param destination: the function to apply to the input iterators
    /// @return OutputIt
    ///
    template<class InputIt1, class InputIt2, class OutputIt>
    OutputIt set_union(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt destination)
        CONDITIONAL_NO_EXCEPT
    {
        return std::set_union(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first1,
            last1,
            first2,
            last2,
            destination);
    }

    //============================================================================
    // Method Description:
    /// finds the union of two ranges
    ///
    /// @param first1: the first iterator of the source
    /// @param last1: the last iterator of the source
    /// @param first2: the first iterator of the second source
    /// @param last2: the first iterator of the destination
    /// @param destination: the function to apply to the input iterators
    /// @param comp: comparitor function
    /// @return OutputIt
    ///
    template<class InputIt1, class InputIt2, class OutputIt, class Compare>
    OutputIt
        set_union(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt destination, Compare comp)
            CONDITIONAL_NO_EXCEPT
    {
        return std::set_union(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first1,
            last1,
            first2,
            last2,
            destination,
            comp);
    }

    //============================================================================
    // Method Description:
    /// Sorts the range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    ///
    template<class RandomIt>
    void sort(RandomIt first, RandomIt last) CONDITIONAL_NO_EXCEPT
    {
        return std::sort(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last);
    }

    //============================================================================
    // Method Description:
    /// Sorts the range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param comp: the comparitor function
    ///
    template<class RandomIt, class Compare>
    void sort(RandomIt first, RandomIt last, Compare comp) CONDITIONAL_NO_EXCEPT
    {
        return std::sort(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            comp);
    }

    //============================================================================
    // Method Description:
    /// Sorts the range preserving order
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    ///
    template<class RandomIt>
    void stable_sort(RandomIt first, RandomIt last) CONDITIONAL_NO_EXCEPT
    {
        std::stable_sort(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last);
    }

    //============================================================================
    // Method Description:
    /// Sorts the range preserving order
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param comp: the comparitor function
    ///
    template<class RandomIt, class Compare>
    void stable_sort(RandomIt first, RandomIt last, Compare comp) CONDITIONAL_NO_EXCEPT
    {
        std::stable_sort(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            comp);
    }

    //============================================================================
    // Method Description:
    /// Transforms the elements of the range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param destination: the first iterator of the destination
    /// @param unaryFunction: the function to apply to the input iterators
    /// @return OutputIt
    ///
    template<class InputIt, class OutputIt, class UnaryOperation>
    OutputIt transform(InputIt first, InputIt last, OutputIt destination, UnaryOperation unaryFunction)
    {
        return std::transform(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            destination,
            unaryFunction);
    }

    //============================================================================
    // Method Description:
    /// Transforms the elements of the range
    ///
    /// @param first1: the first iterator of the source
    /// @param last1: the last iterator of the source
    /// @param first2: the first iterator of the second source
    /// @param destination: the first iterator of the destination
    /// @param unaryFunction: the function to apply to the input iterators
    /// @return OutputIt
    ///
    template<class InputIt1, class InputIt2, class OutputIt, class BinaryOperation>
    OutputIt
        transform(InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt destination, BinaryOperation unaryFunction)
    {
        return std::transform(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first1,
            last1,
            first2,
            destination,
            unaryFunction);
    }

    //============================================================================
    // Method Description:
    /// Copies the unique elements of a range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param destination: the first iterator of the destination
    /// @return OutputIt
    ///
    template<class InputIt, class OutputIt>
    constexpr OutputIt unique_copy(InputIt first, InputIt last, OutputIt destination) CONDITIONAL_NO_EXCEPT
    {
        return std::unique_copy(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            destination);
    }

    //============================================================================
    // Method Description:
    /// Copies the unique elements of a range
    ///
    /// @param first: the first iterator of the source
    /// @param last: the last iterator of the source
    /// @param destination: the first iterator of the destination
    /// @param binaryFunction: the function to apply to the input iterators
    /// @return OutputIt
    ///
    template<class InputIt, class OutputIt, class BinaryPredicate>
    constexpr OutputIt unique_copy(InputIt first, InputIt last, OutputIt destination, BinaryPredicate binaryFunction)
        CONDITIONAL_NO_EXCEPT
    {
        return std::unique_copy(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::execution::par_unseq,
#endif
            first,
            last,
            destination,
            binaryFunction);
    }
} // namespace nc::stl_algorithms
