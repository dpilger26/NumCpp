/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2025 David Pilger
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

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Functions/cov.hpp"
#include "NumCpp/Functions/empty_like.hpp"
#include "NumCpp/Functions/sqrt.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Return Pearson product-moment correlation coefficients.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
    ///
    /// @param x: A 1-D or 2-D array containing multiple variables and observations.
    /// Each row of x represents a variable, and each column a single observation
    /// of all those variables.
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> corrcoef(const NdArray<dtype>& x)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto covariance           = cov(x);
        auto       normalizedCovariance = empty_like(covariance);
        for (decltype(covariance.numRows()) i = 0; i < covariance.numRows(); ++i)
        {
            for (decltype(covariance.numCols()) j = 0; j < covariance.numCols(); ++j)
            {
                normalizedCovariance(i, j) = covariance(i, j) / sqrt(covariance(i, i) * covariance(j, j));
            }
        }

        return normalizedCovariance;
    }
} // namespace nc
