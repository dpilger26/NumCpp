/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
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
/// Description
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/NdArray.hpp"

#ifndef NO_USE_BOOST
#include "boost/integer/common_factor_rt.hpp"
#endif

#ifdef __cpp_lib_gcd_lcm
#include <numeric>
#endif

namespace nc
{
#if defined(__cpp_lib_gcd_lcm) || !defined(NO_USE_BOOST)
    //============================================================================
    // Method Description:
    ///						Returns the greatest common divisor of |x1| and |x2|.
    ///                     NOTE: Use of this function requires either using the Boost
    ///                     includes or a C++17 complient compiler.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.gcd.html
    ///
    /// @param      inValue1
    /// @param      inValue2
    /// @return
    ///				dtype
    ///
    template<typename dtype>
    dtype gcd(dtype inValue1, dtype inValue2) noexcept 
    {
        STATIC_ASSERT_INTEGER(dtype);

#ifdef __cpp_lib_gcd_lcm
        return std::gcd(inValue1, inValue2);
#else
        return boost::integer::gcd(inValue1, inValue2);
#endif
    }
#endif

#ifndef NO_USE_BOOST
    //============================================================================
    // Method Description:
    ///						Returns the greatest common divisor of the values in the
    ///                     input array.
    ///                     NOTE: Use of this function requires using the Boost includes.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.gcd.html
    ///
    /// @param      inArray
    /// @return
    ///				NdArray<double>
    ///
    template<typename dtype>
    dtype gcd(const NdArray<dtype>& inArray)
    {
        STATIC_ASSERT_INTEGER(dtype);
        return boost::integer::gcd_range(inArray.cbegin(), inArray.cend()).first;
    }
#endif
} // namespace nc
