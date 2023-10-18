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
/// Special Functions
///
#pragma once

#ifndef NUMCPP_NO_USE_BOOST

#include "boost/math/special_functions/erf.hpp"

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::special
{
    //============================================================================
    // Method Description:
    /// Returns the inverse complentary error function of z, that is a value x such that:
    /// z = erfc(x).
    /// NOTE: Use of this function requires using the Boost includes.
    ///
    /// @param inValue
    /// @return calculated-result-type
    ///
    template<typename dtype>
    auto erfc_inv(dtype inValue)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return boost::math::erfc_inv(inValue);
    }

    //============================================================================
    // Method Description:
    /// Returns the inverse complementary error function of z, that is a value x such that:
    /// z = erfc(x).
    /// NOTE: Use of this function requires using the Boost includes.
    ///
    /// @param inArray
    /// @return NdArray
    ///
    template<typename dtype>
    auto erfc_inv(const NdArray<dtype>& inArray)
    {
        NdArray<decltype(erfc_inv(dtype{ 0 }))> returnArray(inArray.shape());

        stl_algorithms::transform(inArray.cbegin(),
                                  inArray.cend(),
                                  returnArray.begin(),
                                  [](dtype inValue) -> auto { return erfc_inv(inValue); });

        return returnArray;
    }
} // namespace nc::special

#endif // #ifndef NUMCPP_NO_USE_BOOST
