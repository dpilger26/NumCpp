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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Functions/mean.hpp"

#include <type_traits>

namespace nc
{
    //============================================================================
    // Method Description:
    ///	Estimate a covariance matrix, given data and weights.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.cov.html
    ///
    /// @param	x: A 1-D or 2-D array containing multiple variables and observations.
    ///            Each row of x represents a variable, and each column a single observation 
    ///            of all those variables.
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> cov(const NdArray<dtype>& x)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto varMeans = mean(x, Axis::COL);
        const auto numVars = x.numRows();
        const auto numObs = x.numCols();
        using IndexType = std::remove_const<decltype(numVars)>::type;

        // upper triangle
        auto covariance = NdArray<double>(numVars);
        for (IndexType i = 0; i < numVars; ++i)
        {
            const auto var1Mean = varMeans[i];

            for (IndexType j = i; j < numVars; ++j)
            {
                const auto var2Mean = varMeans[j];

                double sum = 0.0;
                for (IndexType iObs = 0; iObs < numObs; ++iObs)
                {
                    sum += (x(i, iObs) - var1Mean) * (x(j, iObs) - var2Mean);
                }

                covariance(i, j) = sum / static_cast<double>(numObs - 1);
            }
        }

        // fill in the rest of the symmetric matrix
        for (IndexType j = 0; j < numVars; ++j)
        {
            for (IndexType i = j + 1; i < numVars; ++i)
            {
                covariance(i, j) = covariance(j, i);
            }
        }

        return covariance;
    }
}  // namespace nc
