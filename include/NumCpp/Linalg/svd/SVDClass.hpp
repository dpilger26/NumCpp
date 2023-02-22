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
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"

namespace nc::linalg
{
    // =============================================================================
    // Class Description:
    /// performs the singular value decomposition of a general matrix,
    /// taken and adapted from Numerical Recipes Third Edition svd.h
    class SVD
    {
    public:
        // =============================================================================
        // Description:
        /// Constructor
        ///
        /// @param inMatrix: matrix to perform SVD on
        ///
        explicit SVD(const NdArray<double>& inMatrix) :
            m_(inMatrix.shape().rows),
            n_(inMatrix.shape().cols),
            u_(inMatrix),
            v_(n_, n_),
            s_(1, n_),
            eps_(std::numeric_limits<double>::epsilon())
        {
            decompose();
            reorder();
            tsh_ = 0.5 * std::sqrt(m_ + n_ + 1.) * s_.front() * eps_;
        }

        // =============================================================================
        // Description:
        /// the resultant u matrix
        ///
        /// @return u matrix
        ///
        const NdArray<double>& u() noexcept
        {
            return u_;
        }

        // =============================================================================
        // Description:
        /// the resultant v matrix
        ///
        /// @return v matrix
        ///
        const NdArray<double>& v() noexcept
        {
            return v_;
        }

        // =============================================================================
        // Description:
        /// the resultant w matrix
        ///
        /// @return s matrix
        ///
        const NdArray<double>& s() noexcept
        {
            return s_;
        }

        // =============================================================================
        // Description:
        /// solves the linear least squares problem
        ///
        /// @param inInput
        /// @param inThresh (default -1.)
        ///
        /// @return NdArray
        ///
        NdArray<double> solve(const NdArray<double>& inInput, double inThresh = -1.)
        {
            double ss{};

            if (inInput.size() != m_)
            {
                THROW_INVALID_ARGUMENT_ERROR("bad sizes.");
            }

            NdArray<double> returnArray(1, n_);

            NdArray<double> tmp(1, n_);

            tsh_ = (inThresh >= 0. ? inThresh : 0.5 * sqrt(m_ + n_ + 1.) * s_.front() * eps_);

            for (uint32 j = 0; j < n_; j++)
            {
                ss = 0.;
                if (s_[j] > tsh_)
                {
                    for (uint32 i = 0; i < m_; i++)
                    {
                        ss += u_(i, j) * inInput[i];
                    }
                    ss /= s_[j];
                }
                tmp[j] = ss;
            }

            for (uint32 j = 0; j < n_; j++)
            {
                ss = 0.;
                for (uint32 jj = 0; jj < n_; jj++)
                {
                    ss += v_(j, jj) * tmp[jj];
                }

                returnArray[j] = ss;
            }

            return returnArray;
        }

    private:
        // =============================================================================
        // Description:
        /// returns the SIGN of two values
        ///
        /// @param inA
        /// @param inB
        ///
        /// @return value
        ///
        static double SIGN(double inA, double inB) noexcept
        {
            return inB >= 0 ? (inA >= 0 ? inA : -inA) : (inA >= 0 ? -inA : inA);
        }

        // =============================================================================
        // Description:
        /// decomposes the input matrix
        ///
        void decompose()
        {
            bool   flag{};
            uint32 i{};
            uint32 its{};
            uint32 j{};
            uint32 jj{};
            uint32 k{};
            uint32 l{};
            uint32 nm{};

            double anorm{};
            double c{};
            double f{};
            double g{};
            double h{};
            double ss{};
            double scale{};
            double x{};
            double y{};
            double z{};

            NdArray<double> rv1(n_, 1);

            for (i = 0; i < n_; ++i)
            {
                l      = i + 2;
                rv1[i] = scale * g;
                g = ss = scale = 0.;

                if (i < m_)
                {
                    for (k = i; k < m_; ++k)
                    {
                        scale += std::abs(u_(k, i));
                    }

                    if (!utils::essentiallyEqual(scale, 0.))
                    {
                        for (k = i; k < m_; ++k)
                        {
                            u_(k, i) /= scale;
                            ss += u_(k, i) * u_(k, i);
                        }

                        f        = u_(i, i);
                        g        = -SIGN(std::sqrt(ss), f);
                        h        = f * g - ss;
                        u_(i, i) = f - g;

                        for (j = l - 1; j < n_; ++j)
                        {
                            for (ss = 0., k = i; k < m_; ++k)
                            {
                                ss += u_(k, i) * u_(k, j);
                            }

                            f = ss / h;

                            for (k = i; k < m_; ++k)
                            {
                                u_(k, j) += f * u_(k, i);
                            }
                        }

                        for (k = i; k < m_; ++k)
                        {
                            u_(k, i) *= scale;
                        }
                    }
                }

                s_[i] = scale * g;
                g = ss = scale = 0.;

                if (i + 1 <= m_ && i + 1 != n_)
                {
                    for (k = l - 1; k < n_; ++k)
                    {
                        scale += std::abs(u_(i, k));
                    }

                    if (!utils::essentiallyEqual(scale, 0.))
                    {
                        for (k = l - 1; k < n_; ++k)
                        {
                            u_(i, k) /= scale;
                            ss += u_(i, k) * u_(i, k);
                        }

                        f            = u_(i, l - 1);
                        g            = -SIGN(std::sqrt(ss), f);
                        h            = f * g - ss;
                        u_(i, l - 1) = f - g;

                        for (k = l - 1; k < n_; ++k)
                        {
                            rv1[k] = u_(i, k) / h;
                        }

                        for (j = l - 1; j < m_; ++j)
                        {
                            for (ss = 0., k = l - 1; k < n_; ++k)
                            {
                                ss += u_(j, k) * u_(i, k);
                            }

                            for (k = l - 1; k < n_; ++k)
                            {
                                u_(j, k) += ss * rv1[k];
                            }
                        }

                        for (k = l - 1; k < n_; ++k)
                        {
                            u_(i, k) *= scale;
                        }
                    }
                }

                anorm = std::max(anorm, (std::abs(s_[i]) + std::abs(rv1[i])));
            }

            for (i = n_ - 1; i != static_cast<uint32>(-1); --i)
            {
                if (i < n_ - 1)
                {
                    if (!utils::essentiallyEqual(g, 0.))
                    {
                        for (j = l; j < n_; ++j)
                        {
                            v_(j, i) = (u_(i, j) / u_(i, l)) / g;
                        }

                        for (j = l; j < n_; ++j)
                        {
                            for (ss = 0., k = l; k < n_; ++k)
                            {
                                ss += u_(i, k) * v_(k, j);
                            }

                            for (k = l; k < n_; ++k)
                            {
                                v_(k, j) += ss * v_(k, i);
                            }
                        }
                    }

                    for (j = l; j < n_; ++j)
                    {
                        v_(i, j) = v_(j, i) = 0.;
                    }
                }

                v_(i, i) = 1.;
                g        = rv1[i];
                l        = i;
            }

            for (i = std::min(m_, n_) - 1; i != static_cast<uint32>(-1); --i)
            {
                l = i + 1;
                g = s_[i];

                for (j = l; j < n_; ++j)
                {
                    u_(i, j) = 0.;
                }

                if (!utils::essentiallyEqual(g, 0.))
                {
                    g = 1. / g;

                    for (j = l; j < n_; ++j)
                    {
                        for (ss = 0., k = l; k < m_; ++k)
                        {
                            ss += u_(k, i) * u_(k, j);
                        }

                        f = (ss / u_(i, i)) * g;

                        for (k = i; k < m_; ++k)
                        {
                            u_(k, j) += f * u_(k, i);
                        }
                    }

                    for (j = i; j < m_; ++j)
                    {
                        u_(j, i) *= g;
                    }
                }
                else
                {
                    for (j = i; j < m_; ++j)
                    {
                        u_(j, i) = 0.;
                    }
                }

                ++u_(i, i);
            }

            for (k = n_ - 1; k != static_cast<uint32>(-1); --k)
            {
                for (its = 0; its < 30; ++its)
                {
                    flag = true;
                    for (l = k; l != static_cast<uint32>(-1); --l)
                    {
                        nm = l - 1;
                        if (l == 0 || std::abs(rv1[l]) <= eps_ * anorm)
                        {
                            flag = false;
                            break;
                        }

                        if (std::abs(s_[nm]) <= eps_ * anorm)
                        {
                            break;
                        }
                    }

                    if (flag)
                    {
                        c  = 0.;
                        ss = 1.;
                        for (i = l; i < k + 1; ++i)
                        {
                            f      = ss * rv1[i];
                            rv1[i] = c * rv1[i];

                            if (std::abs(f) <= eps_ * anorm)
                            {
                                break;
                            }

                            g     = s_[i];
                            h     = pythag(f, g);
                            s_[i] = h;
                            h     = 1. / h;
                            c     = g * h;
                            ss    = -f * h;

                            for (j = 0; j < m_; ++j)
                            {
                                y         = u_(j, nm);
                                z         = u_(j, i);
                                u_(j, nm) = y * c + z * ss;
                                u_(j, i)  = z * c - y * ss;
                            }
                        }
                    }

                    z = s_[k];
                    if (l == k)
                    {
                        if (z < 0.)
                        {
                            s_[k] = -z;
                            for (j = 0; j < n_; ++j)
                            {
                                v_(j, k) = -v_(j, k);
                            }
                        }
                        break;
                    }

                    if (its == 29)
                    {
                        THROW_INVALID_ARGUMENT_ERROR("no convergence in 30 svdcmp iterations");
                    }

                    x  = s_[l];
                    nm = k - 1;
                    y  = s_[nm];
                    g  = rv1[nm];
                    h  = rv1[k];
                    f  = ((y - z) * (y + z) + (g - h) * (g + h)) / (2. * h * y);
                    g  = pythag(f, 1.);
                    f  = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
                    c = ss = 1.;

                    for (j = l; j <= nm; j++)
                    {
                        i      = j + 1;
                        g      = rv1[i];
                        y      = s_[i];
                        h      = ss * g;
                        g      = c * g;
                        z      = pythag(f, h);
                        rv1[j] = z;
                        c      = f / z;
                        ss     = h / z;
                        f      = x * c + g * ss;
                        g      = g * c - x * ss;
                        h      = y * ss;
                        y *= c;

                        for (jj = 0; jj < n_; ++jj)
                        {
                            x         = v_(jj, j);
                            z         = v_(jj, i);
                            v_(jj, j) = x * c + z * ss;
                            v_(jj, i) = z * c - x * ss;
                        }

                        z     = pythag(f, h);
                        s_[j] = z;

                        if (!utils::essentiallyEqual(z, 0.))
                        {
                            z  = 1. / z;
                            c  = f * z;
                            ss = h * z;
                        }

                        f = c * g + ss * y;
                        x = c * y - ss * g;

                        for (jj = 0; jj < m_; ++jj)
                        {
                            y         = u_(jj, j);
                            z         = u_(jj, i);
                            u_(jj, j) = y * c + z * ss;
                            u_(jj, i) = z * c - y * ss;
                        }
                    }
                    rv1[l] = 0.;
                    rv1[k] = f;
                    s_[k]  = x;
                }
            }
        }

        // =============================================================================
        // Description:
        /// reorders the input matrix
        ///
        void reorder()
        {
            uint32 i   = 0;
            uint32 j   = 0;
            uint32 k   = 0;
            uint32 ss  = 0;
            uint32 inc = 1;

            double          sw{};
            NdArray<double> su(m_, 1);
            NdArray<double> sv(n_, 1);

            do
            {
                inc *= 3;
                ++inc;
            } while (inc <= n_);

            do
            {
                inc /= 3;
                for (i = inc; i < n_; ++i)
                {
                    sw = s_[i];
                    for (k = 0; k < m_; ++k)
                    {
                        su[k] = u_(k, i);
                    }

                    for (k = 0; k < n_; ++k)
                    {
                        sv[k] = v_(k, i);
                    }

                    j = i;
                    while (s_[j - inc] < sw)
                    {
                        s_[j] = s_[j - inc];

                        for (k = 0; k < m_; ++k)
                        {
                            u_(k, j) = u_(k, j - inc);
                        }

                        for (k = 0; k < n_; ++k)
                        {
                            v_(k, j) = v_(k, j - inc);
                        }

                        j -= inc;

                        if (j < inc)
                        {
                            break;
                        }
                    }

                    s_[j] = sw;

                    for (k = 0; k < m_; ++k)
                    {
                        u_(k, j) = su[k];
                    }

                    for (k = 0; k < n_; ++k)
                    {
                        v_(k, j) = sv[k];
                    }
                }
            } while (inc > 1);

            for (k = 0; k < n_; ++k)
            {
                ss = 0;

                for (i = 0; i < m_; i++)
                {
                    if (u_(i, k) < 0.)
                    {
                        ss++;
                    }
                }

                for (j = 0; j < n_; ++j)
                {
                    if (v_(j, k) < 0.)
                    {
                        ss++;
                    }
                }

                if (ss > (m_ + n_) / 2)
                {
                    for (i = 0; i < m_; ++i)
                    {
                        u_(i, k) = -u_(i, k);
                    }

                    for (j = 0; j < n_; ++j)
                    {
                        v_(j, k) = -v_(j, k);
                    }
                }
            }
        }

        // =============================================================================
        // Description:
        /// performs pythag of input values
        ///
        /// @param inA
        /// @param inB
        ///
        /// @return resultant value
        ///
        static double pythag(double inA, double inB) noexcept
        {
            const double absa = std::abs(inA);
            const double absb = std::abs(inB);
            return (absa > absb
                        ? absa * std::sqrt(1. + utils::sqr(absb / absa))
                        : (utils::essentiallyEqual(absb, 0.) ? 0. : absb * std::sqrt(1. + utils::sqr(absa / absb))));
        }

    private:
        // ===============================Attributes====================================
        const uint32    m_{};
        const uint32    n_{};
        NdArray<double> u_{};
        NdArray<double> v_{};
        NdArray<double> s_{};
        double          eps_{};
        double          tsh_{};
    };
} // namespace nc::linalg
