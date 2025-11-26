/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2019 Benjamin Mahr
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
/// matrix pivot LU decomposition
///
/// Code modified under MIT license from https://github.com/Ben1980/linAlg
/// as posted in
/// https://thoughts-on-coding.com/2019/06/12/numerical-methods-with-c-part-4-introduction-into-decomposition-methods-of-linear-equation-systems/
///
#pragma once

#include <cmath>
#include <tuple>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/eye.hpp"
#include "NumCpp/Functions/zeros_like.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"

namespace nc::linalg
{
    //============================================================================
    // Method Description:
    /// matrix pivot LU decomposition PA = LU
    ///
    /// @param inMatrix: NdArray to be decomposed
    ///
    /// @return std::tuple<NdArray, NdArray, NdArray> of the decomposed L, U, and P matrices
    ///
    template<typename dtype>
    std::tuple<NdArray<double>, NdArray<double>, NdArray<double>> pivotLU_decomposition(const NdArray<dtype>& inMatrix)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto shape = inMatrix.shape();

        if (!shape.issquare())
        {
            THROW_RUNTIME_ERROR("Input matrix should be square.");
        }

        NdArray<double> lMatrix = zeros_like<double>(inMatrix);
        NdArray<double> uMatrix = inMatrix.template astype<double>();
        NdArray<double> pMatrix = eye<double>(shape.rows);

        for (uint32 k = 0; k < shape.rows; ++k)
        {
            double max = 0.;
            uint32 pk  = 0;
            for (uint32 i = k; i < shape.rows; ++i)
            {
                double s = 0.;
                for (uint32 j = k; j < shape.cols; ++j)
                {
                    s += std::fabs(uMatrix(i, j));
                }

                const double q = std::fabs(uMatrix(i, k)) / s;
                if (q > max)
                {
                    max = q;
                    pk  = i;
                }
            }

            if (utils::essentiallyEqual(max, double{ 0. }))
            {
                THROW_RUNTIME_ERROR("Division by 0.");
            }

            if (pk != k)
            {
                for (uint32 j = 0; j < shape.cols; ++j)
                {
                    std::swap(pMatrix(k, j), pMatrix(pk, j));
                    std::swap(lMatrix(k, j), lMatrix(pk, j));
                    std::swap(uMatrix(k, j), uMatrix(pk, j));
                }
            }

            for (uint32 i = k + 1; i < shape.rows; ++i)
            {
                lMatrix(i, k) = uMatrix(i, k) / uMatrix(k, k);

                for (uint32 j = k; j < shape.cols; ++j)
                {
                    uMatrix(i, j) = uMatrix(i, j) - lMatrix(i, k) * uMatrix(k, j);
                }
            }
        }

        for (uint32 k = 0; k < shape.rows; ++k)
        {
            lMatrix(k, k) = 1.;
        }

        return std::make_tuple(lMatrix, uMatrix, pMatrix);
    }
} // namespace nc::linalg
