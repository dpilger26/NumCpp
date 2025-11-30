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
/// Performs the singular value decomposition of a general matrix,
/// taken and adapted from Numerical Recipes Third Edition svd.h
///
#pragma once

#include <cmath>
#include <limits>
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/dot.hpp"
#include "NumCpp/Functions/norm.hpp"
#include "NumCpp/Functions/zeros.hpp"
#include "NumCpp/Linalg/eig.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::linalg
{
    // =============================================================================
    // Class Description:
    /// Performs the singular value decomposition of a general matrix
    template<typename dtype>
    class SVD
    {
    public:
        STATIC_ASSERT_ARITHMETIC(dtype);

        static constexpr auto TOLERANCE = 1e-12;

        // =============================================================================
        // Description:
        /// Constructor
        ///
        /// @param inMatrix: matrix to perform SVD on
        ///
        explicit SVD(const NdArray<dtype>& inMatrix) :
            m_{ inMatrix.shape().rows },
            n_{ inMatrix.shape().cols },
            s_(1, m_)
        {
            compute(inMatrix.template astype<double>());
        }

        // =============================================================================
        // Description:
        /// the resultant u matrix
        ///
        /// @return u matrix
        ///
        const NdArray<double>& u() const noexcept
        {
            return u_;
        }

        // =============================================================================
        // Description:
        /// the resultant v transpose matrix
        ///
        /// @return v matrix
        ///
        const NdArray<double>& v() const noexcept
        {
            return v_;
        }

        // =============================================================================
        // Description:
        /// the resultant w matrix
        ///
        /// @return s matrix
        ///
        const NdArray<double>& s() const noexcept
        {
            return s_;
        }

        // =============================================================================
        // Description:
        /// Returns the pseudo-inverse of the input matrix
        ///
        /// @return NdArray
        ///
        NdArray<double> pinv()
        {
            // lazy evaluation
            if (pinv_.isempty())
            {
                auto sInverse = nc::zeros<double>(n_, m_); // transpose
                for (auto i = 0u; i < std::min(m_, n_); ++i)
                {
                    if (s_[i] > TOLERANCE)
                    {
                        sInverse(i, i) = 1. / s_[i];
                    }
                }

                pinv_ = dot(v_, dot(sInverse, u_.transpose()));
            }

            return pinv_;
        }

        // =============================================================================
        // Description:
        /// solves the linear least squares problem
        ///
        /// @param inInput
        ///
        /// @return NdArray
        ///
        NdArray<double> lstsq(const NdArray<double>& inInput)
        {
            if (inInput.size() != m_)
            {
                THROW_INVALID_ARGUMENT_ERROR("Invalid matrix dimensions");
            }

            if (inInput.numCols() == 1)
            {
                return dot(pinv(), inInput);
            }
            else
            {
                const auto input = inInput.copy().reshape(inInput.size(), 1);
                return dot(pinv(), input);
            }
        }

    private:
        // =============================================================================
        // Description:
        /// Computes the SVD of the input matrix
        ///
        /// @param A: matrix to perform SVD on
        ///
        void compute(const NdArray<double>& A)
        {
            const auto At  = A.transpose();
            const auto AtA = dot(At, A);
            const auto AAt = dot(A, At);

            const auto& [sigmaSquaredU, U] = eig(AAt);
            const auto& [sigmaSquaredV, V] = eig(AtA);

            auto rank = 0u;
            for (auto i = 0u; i < std::min(m_, n_); ++i)
            {
                if (sigmaSquaredV[i] > TOLERANCE)
                {
                    s_[i] = std::sqrt(sigmaSquaredV[i]);
                    rank++;
                }
            }

            // std::cout << U.front() << ' ' << U.back() << '\n';
            // std::cout << V.front() << ' ' << V.back() << '\n';
            // std::cout << "hello world\n";

            u_ = std::move(U);
            v_ = std::move(V);

            auto Av = NdArray<double>(m_, 1);
            for (auto i = 0u; i < rank; ++i)
            {
                for (auto j = 0u; j < m_; ++j)
                {
                    auto sum = 0.;
                    for (auto k = 0u; k < n_; ++k)
                    {
                        sum += A(j, k) * v_(k, i);
                    }
                    Av[j] = sum;
                }

                const auto normalization = norm(Av).item();

                if (normalization > TOLERANCE)
                {
                    for (auto j = 0u; j < m_; ++j)
                    {
                        u_(j, i) = Av[j] / normalization;
                    }
                }
            }
        }

    private:
        // ===============================Attributes====================================
        const uint32    m_{};
        const uint32    n_{};
        NdArray<double> u_{};
        NdArray<double> v_{};
        NdArray<double> s_{};
        NdArray<double> pinv_{};
    };
} // namespace nc::linalg
