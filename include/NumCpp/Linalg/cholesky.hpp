/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2019 Benjamin Mahr
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
/// matrix cholesky decomposition
///
/// Code modified under MIT license from https://github.com/Ben1980/linAlg
/// as posted in
/// https://thoughts-on-coding.com/2019/06/12/numerical-methods-with-c-part-4-introduction-into-decomposition-methods-of-linear-equation-systems/
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    namespace linalg
    {
        //============================================================================
        // Method Description:
        /// matrix cholesky decomposition A = L * L.transpose()
        ///
        /// NumPy Reference:
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.cholesky.html#numpy.linalg.cholesky
        ///
        /// @param inMatrix: NdArray to be decomposed
        ///
        /// @return NdArray of the decomposed L matrix
        ///
        template<typename dtype>
        NdArray<double> cholesky(const NdArray<dtype>& inMatrix)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            const auto shape = inMatrix.shape();
            if (!shape.issquare())
            {
                THROW_RUNTIME_ERROR("Input matrix should be square.");
            }

            auto lMatrix = inMatrix.template astype<double>();

            for (uint32 row = 0; row < shape.rows; ++row)
            {
                for (uint32 col = row + 1; col < shape.cols; ++col)
                {
                    lMatrix(row, col) = 0.0;
                }
            }

            for (uint32 k = 0; k < shape.cols; ++k)
            {
                const double& a_kk = lMatrix(k, k);

                if (a_kk > 0.0)
                {
                    lMatrix(k, k) = std::sqrt(a_kk);

                    for (uint32 i = k + 1; i < shape.rows; ++i)
                    {
                        lMatrix(i, k) /= lMatrix(k, k);

                        for (uint32 j = k + 1; j <= i; ++j)
                        {
                            lMatrix(i, j) -= lMatrix(i, k) * lMatrix(j, k);
                        }
                    }
                }
                else
                {
                    THROW_RUNTIME_ERROR("Matrix is not positive definite.");
                }
            }

            return lMatrix;
        }
    } // namespace linalg
} // namespace nc
