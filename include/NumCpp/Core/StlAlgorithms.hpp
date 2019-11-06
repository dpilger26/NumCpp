/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
///
/// @section License
/// Copyright 2019 David Pilger
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
/// Macro to define whether or not c++17 parallel algorithm policies are supported
///
#pragma once

#include <algorithm>

#if defined(__cpp_lib_execution) && defined(__cpp_lib_parallel_algorithm)
#define PARALLEL_ALGORITHMS_SUPPORTED
#include <execution>
#endif

namespace nc
{
    namespace stl_algorithms
    {
        //============================================================================
        // Method Description:
        ///						Copies from one container to another
        ///
        /// @param first: the first iterator of the source
        /// @param last: the last iterator of the source
        /// @param destination: the first iterator of the destination
        ///
        template<class InputIt, class OutputIt>
        void copy(InputIt first, InputIt last, OutputIt destination)
        {
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::copy(std::execution::par_unseq, first, last, destination);
#else
            std::copy(first, last, destination);
#endif
        }

        //============================================================================
        // Method Description:
        ///						Copies from one container to another
        ///
        /// @param first: the first iterator of the source
        /// @param last: the last iterator of the source
        /// @param destination: the first iterator of the destination
        /// @param unaryFunction: the function to apply to the input iterators
        ///
        template< class InputIt, class OutputIt, class UnaryOperation >
        void transform(InputIt first, InputIt last, OutputIt destination,
            UnaryOperation unaryFunction)
        {
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::transform(std::execution::par_unseq, first, last, destination, unaryFunction);
#else
            std::transform(first, last, destination, unaryFunction);
#endif
        }

        //============================================================================
        // Method Description:
        ///						Copies from one container to another
        ///
        /// @param first1: the first iterator of the source
        /// @param last1: the last iterator of the source
        /// @param first2: the first iterator of the second source
        /// @param destination: the first iterator of the destination
        /// @param unaryFunction: the function to apply to the input iterators
        template< class InputIt1, class InputIt2, class OutputIt, class BinaryOperation >
        void transform( InputIt1 first1, InputIt1 last1, InputIt2 first2,
            OutputIt destination, BinaryOperation unaryFunction )
        {
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
            std::transform(std::execution::par_unseq, 
                first1, last1, first2, destination, unaryFunction);
#else
            std::transform(first1, last1, first2, destination, unaryFunction);
#endif
        }
    }
}
