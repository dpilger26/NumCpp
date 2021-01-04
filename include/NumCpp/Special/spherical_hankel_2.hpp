/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2021 David Pilger
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

#ifndef NO_USE_BOOST

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/NdArray.hpp"

#include "boost/math/special_functions/hankel.hpp"

#include <complex>
#include <type_traits>

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        ///	Spherical Hankel funcion of the second kind.
        /// NOTE: Use of this function requires using the Boost includes.
        ///
        /// @param      inV: the order of the bessel function
        /// @param      inX: the input value
        /// @return
        ///				double
        ///
        template<typename dtype1, typename dtype2>
        std::complex<double> spherical_hankel_2(dtype1 inV, dtype2 inX)
        {
            STATIC_ASSERT_ARITHMETIC(dtype1);
            STATIC_ASSERT_ARITHMETIC(dtype2);

            return boost::math::sph_hankel_2(inV, inX);
        }

        //============================================================================
        // Method Description:
        /// Spherical Hankel funcion of the second kind.
        /// NOTE: Use of this function requires using the Boost includes.
        ///
        /// @param      inV: the order of the bessel function
        /// @param      inArray: the input value
        /// @return
        ///				NdArray
        ///
        template<typename dtype1, typename dtype2>
        auto spherical_hankel_2(dtype1 inV, const NdArray<dtype2>& inArray) 
        {
            NdArray<decltype(spherical_hankel_2(dtype1{0}, dtype2{0}))> returnArray(inArray.shape());

            stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
                [inV](dtype2 inValue) -> auto
                { 
                    return spherical_hankel_2(inV, inValue); 
                });

            return returnArray;
        }
    } // namespace special
} // namespace nc

#endif
