/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
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
/// linear least squares
///
#pragma once

#include <utility>

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Functions/eye.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/sqr.hpp"

namespace nc::linalg
{
    //============================================================================
    // Method Description:
    /// Compute the eigen values and eigen vectors of a real symmetric matrix.
    ///
    /// NumPy Reference:
    /// https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html#numpy.linalg.eig
    ///
    /// @param inA: Matrix for which the eigen values and eigen vectors will be computed, must be a real, symmetric MxM
    ///             array
    /// @param inTolerance (default 1e-12)
    ///
    /// @return std::pair<NdArray<double>, NdArray<double>> eigen values and eigen vectors
    ///
    template<typename dtype>
    std::pair<NdArray<double>, NdArray<double>> eig(const NdArray<dtype>& inA, double inTolerance = 1e-12)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (!inA.issquare())
        {
            THROW_INVALID_ARGUMENT_ERROR("Input array must be square.");
        }

        const auto n            = inA.numRows();
        auto       b            = inA.template astype<double>();
        auto       eigenVectors = eye<double>(n);
        auto       eigenVals    = NdArray<double>(1, n);

        constexpr auto MAX_ITERATIONS = 10000;
        for (auto iter = 0u; iter < MAX_ITERATIONS; ++iter)
        {
            auto max_off_diag = 0.;
            auto p            = 0u;
            auto q            = 1u;

            for (auto i = 0u; i < n; i++)
            {
                for (auto j = i + 1; j < n; j++)
                {
                    const auto val = std::fabs(b(i, j));
                    if (val > max_off_diag)
                    {
                        max_off_diag = val;
                        p            = i;
                        q            = j;
                    }
                }
            }

            if (max_off_diag < inTolerance)
            {
                break;
            }

            const auto app = b(p, p);
            const auto aqq = b(q, q);
            const auto apq = b(p, q);

            const auto theta           = (aqq - app) / (2. * apq);
            const auto onePlusThetaSqr = std::sqrt(1. + utils::sqr(theta));
            const auto t = (theta >= 0.) ? 1. / (theta + onePlusThetaSqr) : 1. / (theta - onePlusThetaSqr);
            const auto c = 1.0 / std::sqrt(1. + utils::sqr(t));
            const auto s = t * c;

            for (auto i = 0u; i < n; ++i)
            {
                if (i != p && i != q)
                {
                    const auto bip = b(i, p);
                    const auto biq = b(i, q);
                    b(i, p)        = c * bip - s * biq;
                    b(p, i)        = b(i, p);
                    b(i, q)        = s * bip + c * biq;
                    b(q, i)        = b(i, q);
                }
            }

            b(p, p) = c * c * app + s * s * aqq - 2. * c * s * apq;
            b(q, q) = s * s * app + c * c * aqq + 2. * c * s * apq;
            b(p, q) = 0.;
            b(q, p) = 0.;

            for (auto i = 0u; i < n; ++i)
            {
                const auto vip     = eigenVectors(i, p);
                const auto viq     = eigenVectors(i, q);
                eigenVectors(i, p) = c * vip - s * viq;
                eigenVectors(i, q) = s * vip + c * viq;
            }
        }

        for (auto i = 0u; i < n; ++i)
        {
            eigenVals[i] = b(i, i);
        }

        for (auto i = 0u; i < n - 1; ++i)
        {
            for (auto j = i + 1; j < n; ++j)
            {
                if (eigenVals[i] < eigenVals[j])
                {
                    std::swap(eigenVals[i], eigenVals[j]);

                    for (auto k = 0u; k < n; ++k)
                    {
                        std::swap(eigenVectors(k, i), eigenVectors(k, j));
                    }
                }
            }
        }

        return std::make_pair(eigenVals, eigenVectors);
    }
} // namespace nc::linalg
