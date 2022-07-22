/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2022 David Pilger
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
/// Special Functions
///
#pragma once

#ifndef NUMCPP_NO_USE_BOOST

#include <complex>
#include <type_traits>

#include "boost/math/special_functions/hankel.hpp"

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        /// Hankel funcion of the first kind.
        /// NOTE: Use of this function requires using the Boost includes.
        ///
        /// @param inV: the order of the bessel function
        /// @param inX: the input value
        /// @return std::complex<double>
        ///
        template<typename dtype1, typename dtype2>
        auto cyclic_hankel_1(dtype1 inV, dtype2 inX)
        {
            STATIC_ASSERT_ARITHMETIC(dtype1);
            STATIC_ASSERT_ARITHMETIC(dtype2);

            return boost::math::cyl_hankel_1(inV, inX);
        }

        //============================================================================
        // Method Description:
        /// Hankel funcion of the first kind.
        /// NOTE: Use of this function requires using the Boost includes.
        ///
        /// @param inV: the order of the bessel function
        /// @param inX: the input array
        /// @return NdArray<std::complex>
        ///
        template<typename dtype1, typename dtype2>
        auto cyclic_hankel_1(dtype1 inV, const NdArray<dtype2>& inX)
        {
            NdArray<decltype(cyclic_hankel_1(dtype1{ 0 }, dtype2{ 0 }))> returnArray(inX.shape());

            stl_algorithms::transform(
                inX.cbegin(),
                inX.cend(),
                returnArray.begin(),
                [inV](dtype2 x) -> auto{ return cyclic_hankel_1(inV, x); });

            return returnArray;
        }
    } // namespace special
} // namespace nc

#endif // #ifndef NUMCPP_NO_USE_BOOST
