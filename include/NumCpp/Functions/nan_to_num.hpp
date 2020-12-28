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

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/DtypeInfo.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Functions/isinf.hpp"
#include "NumCpp/Functions/isnan.hpp"

#include <utility>

namespace nc
{
    //============================================================================
    // Method Description:
    ///	Replace NaN with zero and infinity with large finite numbers (default behaviour)
    /// or with the numbers defined by the user using the nan, posinf and/or neginf keywords.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
    ///
    /// @param			inArray
    /// @param			copy: whether to create a copy of inArray (true) or replace values 
    ///                       in-place (false)
    ///                 nan: value to be used to fill NaN values, default 0
    ///                 posInf: value to be used to fill positive infinity values, default a very large number
    ///                 negInf: value to be used to fill negative infinity values, default a very large negative number
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> nan_to_num(NdArray<dtype> inArray, 
        dtype nan = static_cast<dtype>(0.0),
        dtype posInf = DtypeInfo<dtype>::max(), 
        dtype negInf = DtypeInfo<dtype>::min()) 
    {
        STATIC_ASSERT_FLOAT(dtype);

        stl_algorithms::for_each(inArray.begin(), inArray.end(),
            [nan, posInf, negInf](dtype& value)
            {
                if (isnan(value))
                {
                    value = nan;
                }
                else if (isinf(value))
                {
                    if (value > static_cast<dtype>(0.0))
                    {
                        value = posInf;
                    }
                    else
                    {
                        value = negInf;
                    }
                }
            }
        );

        return inArray;
    }
}  // namespace nc
