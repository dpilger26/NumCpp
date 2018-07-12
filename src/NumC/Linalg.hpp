/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// @version 1.0
///
/// @section LICENSE
/// Copyright 2018 David Pilger
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
/// @section DESCRIPTION
/// Class for doing linear algebra operations
///
#pragma once

#include"NumC/Methods.hpp"
#include"NumC/NdArray.hpp"
#include"NumC/Shape.hpp"
#include"NumC/Types.hpp"

#include<cmath>
#include<initializer_list>
#include<limits>
#include<stdexcept>
#include<utility>

namespace NumC
{
    //================================Linalg Class=============================
    /// Class for doing linear algebra operations
    template<typename dtype>
    class Linalg
    {
    public:
        //============================================================================
        // Method Description: 
        ///						matrix determinant.
        ///						NOTE: can get verrrrry slow for large matrices (order > 10)
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.det.html#scipy.linalg.det
        ///		
        /// @param
        ///				NdArray
        /// @return
        ///				dtype
        ///
        static dtype det(const NdArray<dtype>& inArray)
        {
            Shape inShape = inArray.shape();
            if (inShape.rows != inShape.cols)
            {
                throw std::runtime_error("ERROR: Linalg::determinant: input array must be square with size no larger than 3x3.");
            }

            if (inShape.rows == 1)
            {
                return inArray[0];
            }
            else if (inShape.rows == 2)
            {
                return inArray(0, 0) * inArray(1, 1) - inArray(0, 1) * inArray(1, 0);
            }
            else if (inShape.rows == 3)
            {
                dtype aei = inArray(0, 0) * inArray(1, 1) * inArray(2, 2);
                dtype bfg = inArray(0, 1) * inArray(1, 2) * inArray(2, 0);
                dtype cdh = inArray(0, 2) * inArray(1, 0) * inArray(2, 1);
                dtype ceg = inArray(0, 2) * inArray(1, 1) * inArray(2, 0);
                dtype bdi = inArray(0, 1) * inArray(1, 0) * inArray(2, 2);
                dtype afh = inArray(0, 0) * inArray(1, 2) * inArray(2, 1);

                return aei + bfg + cdh - ceg - bdi - afh;
            }
            else
            {
                dtype determinant = 0;
                NdArray<dtype> submat(inShape.rows - 1);

                for (uint32 c = 0; c < inShape.rows; ++c)
                {
                    uint32 subi = 0;
                    for (uint32 i = 1; i < inShape.rows; ++i)
                    {
                        uint32 subj = 0;
                        for (uint32 j = 0; j < inShape.rows; ++j)
                        {
                            if (j == c)
                            {
                                continue;
                            }

                            submat(subi, subj++) = inArray(i, j);
                        }
                        ++subi;
                    }
                    determinant += (std::pow(-1, c) * inArray(0, c) * det(submat));
                }

                return determinant;
            }
        }

        //============================================================================
        // Method Description: 
        ///						vector hat operator
        ///		
        /// @param			x
        /// @param			y
        /// @param			z
        /// @return
        ///				3x3 NdArray
        ///
        static NdArray<dtype> hat(dtype inX, dtype inY, dtype inZ)
        {
            NdArray<dtype> returnArray(3);
            returnArray(0, 0) = 0.0;
            returnArray(0, 1) = -inZ;
            returnArray(0, 2) = inY;
            returnArray(1, 0) = inZ;
            returnArray(1, 1) = 0.0;
            returnArray(1, 2) = -inX;
            returnArray(2, 0) = -inY;
            returnArray(2, 1) = inX;
            returnArray(2, 2) = 0.0;

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						vector hat operator
        ///		
        /// @param
        ///				NdArray 3x1, or 1x3 cartesian vector
        /// @return
        ///				3x3 NdArray
        ///
        static NdArray<dtype> hat(const NdArray<dtype>& inVec)
        {
            if (inVec.size() != 3)
            {
                throw std::invalid_argument("ERROR: Linalg::hat: input vector must be a length 3 cartesian vector.");
            }

            return std::move(hat(inVec[0], inVec[1], inVec[2]));
        }

        //============================================================================
        // Method Description: 
        ///						matrix inverse
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html#scipy.linalg.inv
        ///		
        /// @param
        ///				NdArray
        /// @return
        ///				NdArray
        ///
        static NdArray<double> inv(const NdArray<dtype>& inArray)
        {
            Shape inShape = inArray.shape();
            if (inShape.rows != inShape.cols)
            {
                throw std::runtime_error("ERROR: Linalg::inv: input array must be square.");
            }

            uint32 order = inShape.rows;

            Shape newShape(inShape);
            newShape.rows *= 2;
            newShape.cols *= 2;

            NdArray<double> tempArray(newShape);
            for (uint32 row = 0; row < order; ++row)
            {
                for (uint32 col = 0; col < order; ++col)
                {
                    tempArray(row, col) = static_cast<double>(inArray(row, col));
                }
            }

            for (uint32 row = 0; row < order; ++row)
            {
                for (uint32 col = order; col < 2 * order; ++col)
                {
                    if (row == col - order)
                    {
                        tempArray(row, col) = 1.0;
                    }
                    else
                    {
                        tempArray(row, col) = 0.0;
                    }
                }
            }

            for (uint32 row = 0; row < order; ++row)
            {
                double t = tempArray(row, row);
                for (uint32 col = row; col < 2 * order; ++col)
                {
                    tempArray(row, col) /= t;
                }

                for (uint32 col = 0; col < order; ++col)
                {
                    if (row != col)
                    {
                        t = tempArray(col, row);
                        for (uint32 k = 0; k < 2 * order; ++k)
                        {
                            tempArray(col, k) -= t * tempArray(row, k);
                        }
                    }
                }
            }

            NdArray<double> returnArray(inShape);
            for (uint32 row = 0; row < order; row++)
            {
                uint32 colCounter = 0;
                for (uint32 col = order; col < 2 * order; ++col)
                {
                    returnArray(row, colCounter++) = tempArray(row, col);
                }
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						Solves the equation a x = b by computing a vector x 
        ///						that minimizes the Euclidean 2-norm || b - a x ||^2. 
        ///						The equation may be under-, well-, or over- determined 
        ///						(i.e., the number of linearly independent rows of a can 
        ///						be less than, equal to, or greater than its number of 
        ///						linearly independent columns). If a is square and of 
        ///						full rank, then x (but for round-off error) is the 
        ///						"exact" solution of the equation.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html#scipy.linalg.lstsq
        ///		
        /// @param				NdArray, coefficient matrix
        /// @param  			NdArray, Ordinate or "dependent variable" values
        /// @param				double, tolerance
        ///
        /// @return
        ///				NdArray
        ///
        static NdArray<double> lstsq(const NdArray<dtype>& inA, const NdArray<dtype>& inB, double inTolerance = 1.e-12)
        {
            SVD svdSolver(inA);
            double threshold = inTolerance * svdSolver.s()[0];

            return std::move(svdSolver.solve(inB, threshold));
        }

        //============================================================================
        // Method Description: 
        ///						Raise a square matrix to the (integer) power n.
        ///
        ///						For positive integers n, the power is computed by repeated
        ///						matrix squarings and matrix multiplications.  If n == 0, 
        ///						the identity matrix of the same shape as M is returned.
        ///						If n < 0, the inverse is computed and then raised to the abs(n).
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.matrix_power.html#numpy.linalg.matrix_power
        ///		
        /// @param				NdArray
        /// @param				power
        ///
        /// @return
        ///				NdArray
        ///
        template<typename dtypeOut = double>
        static NdArray<dtypeOut> matrix_power(const NdArray<dtype>& inArray, int16 inPower)
        {
            Shape inShape = inArray.shape();
            if (inShape.rows != inShape.cols)
            {
                throw std::invalid_argument("ERROR: Linalg::matrix_power: input matrix must be square.");
            }

            if (inPower == 0)
            {
                return std::move(Methods<dtypeOut>::identity(inShape.rows));
            }
            else if (inPower == 1)
            {
                return std::move(inArray.template astype<dtypeOut>());
            }
            else if (inPower == -1)
            {
                return std::move(inv(inArray).template astype<dtypeOut>());
            }
            else if (inPower > 1)
            {
                NdArray<dtypeOut> returnArray = Methods<dtype>::template dot<dtypeOut>(inArray, inArray);
                for (int16 i = 2; i < inPower; ++i)
                {
                    returnArray = std::move(Methods<dtypeOut>::template dot<dtypeOut>(returnArray, inArray.template astype<dtypeOut>()));
                }
                return std::move(returnArray);
            }
            else
            {
                NdArray<double> inverse = inv(inArray);
                NdArray<double> returnArray = Methods<double>::dot<double>(inverse, inverse);
                for (int16 i = 2; i < std::abs(inPower); ++i)
                {
                    returnArray = std::move(Methods<double>::dot<double>(returnArray, inverse));
                }
                return std::move(returnArray.astype<dtypeOut>());
            }
        }

        //============================================================================
        // Method Description: 
        ///						Compute the dot product of two or more arrays in a single 
        ///						function call..
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.multi_dot.html#numpy.linalg.multi_dot
        ///		
        /// @param
        ///				initializer_list<NdArray<dtype> >, list of arrays
        ///
        /// @return
        ///				NdArray
        ///
        template<typename dtypeOut = double>
        static NdArray<dtypeOut> multi_dot(const std::initializer_list<NdArray<dtype> >& inList)
        {
            typename std::initializer_list<NdArray<dtype> >::iterator iter = inList.begin();

            if (inList.size() == 0)
            {
                throw std::invalid_argument("ERROR: Linalg::multi_dot: input empty list of arrays.");
            }
            else if (inList.size() == 1)
            {
                return std::move(iter->template astype<dtypeOut>());
            }

            NdArray<dtypeOut> returnArray = Methods<dtype>::template dot<dtypeOut>(*iter, *(iter + 1));
            iter += 2;
            for (; iter < inList.end(); ++iter)
            {
                returnArray = std::move(Methods<dtypeOut>::template dot<dtypeOut>(returnArray, iter->template astype<dtypeOut>()));
            }

            return std::move(returnArray);
        }

        //============================================================================
        // Method Description: 
        ///						matrix svd
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd
        ///		
        /// @param				NdArray to be SVDed
        /// @param				NdArray output U
        /// @param				NdArray output S
        /// @param				NdArray output V transpose
        /// @return
        ///				NdArray
        ///
        static void svd(const NdArray<dtype>& inArray, NdArray<double>& outU, NdArray<double>& outS, NdArray<double>& outVt)
        {
            SVD svdSolver(inArray);
            outU = std::move(svdSolver.u());
            outVt = std::move(svdSolver.v());

            NdArray<double> s = Methods<double>::diagflat(svdSolver.s());
            outS = std::move(s);
        }

    private:
        // =============================================================================
        // Class Description:
        ///              performs the singular value decomposition of a general matrix,
        ///              taken and adapted from Numerical Recipes Third Edition svd.h
        class SVD
        {
        private:
            // ===============================Attributes====================================
            const uint32		m_;
            const uint32		n_;
            NdArray<double>     u_;
            NdArray<double>     v_;
            NdArray<double>     s_;
            double				eps_;
            double				tsh_;

        public:
            // =============================================================================
            // Description:
            ///              Constructor
            /// 
            /// @param 
            ///              Matrix::Mat<double> matrix to perform SVD on
            /// 
            /// @return 
            ///              None
            ///
            SVD(const NdArray<double>& inMatrix) :
                m_(inMatrix.shape().rows),
                n_(inMatrix.shape().cols),
                u_(inMatrix),
                v_(n_, n_),
                s_(1, n_)
            {
                eps_ = std::numeric_limits<double>::epsilon();
                decompose();
                reorder();
                tsh_ = 0.5 * std::sqrt(m_ + n_ + 1.) * s_[0] * eps_;
            }

            // =============================================================================
            // Description:
            ///              the resultant u matrix
            /// 
            /// @param 
            ///              None
            /// 
            /// @return 
            ///              u matrix
            ///
            const NdArray<double>& u()
            {
                return u_;
            }

            // =============================================================================
            // Description:
            ///              the resultant v matrix
            /// 
            /// @param 
            ///              None
            /// 
            /// @return 
            ///              v matrix
            ///
            const NdArray<double>& v()
            {
                return v_;
            }

            // =============================================================================
            // Description:
            ///              the resultant w matrix
            /// 
            /// @param 
            ///              None
            /// 
            /// @return 
            ///              s matrix
            ///
            const NdArray<double>& s()
            {
                return s_;
            }

            // =============================================================================
            // Description:
            ///              solves the linear least squares problem
            /// 
            /// @param 
            ///              None
            /// 
            /// @return 
            ///              None
            ///
            NdArray<double> solve(const NdArray<double>& inInput, double inThresh = -1.0)
            {
                double s;

                if (inInput.size() != m_)
                {
                    throw std::runtime_error("ERROR: SVD::solve bad sizes.");
                }

                NdArray<double> returnArray(1, n_);

                NdArray<double> tmp(1, n_);

                tsh_ = (inThresh >= 0. ? inThresh : 0.5 * sqrt(m_ + n_ + 1.) * s_[0] * eps_);

                for (uint32 j = 0; j < n_; j++)
                {
                    s = 0.0;
                    if (s_[j] > tsh_)
                    {
                        for (uint32 i = 0; i < m_; i++)
                        {
                            s += u_(i, j) * inInput[i];
                        }
                        s /= s_[j];
                    }
                    tmp[j] = s;
                }

                for (uint32 j = 0; j < n_; j++)
                {
                    s = 0.0;
                    for (uint32 jj = 0; jj < n_; jj++)
                    {
                        s += v_(j, jj) * tmp[jj];
                    }

                    returnArray[j] = s;
                }

                return std::move(returnArray);
            }

        private:
            // =============================================================================
            // Description:
            ///              returns the SIGN of two values
            ///  
            /// @param              a,
            /// @param              b
            /// 
            /// @return 
            ///              value
            ///
            double SIGN(double a, double b)
            {
                return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
            }

            // =============================================================================
            // Description:
            ///              decomposes the input matrix
            /// 
            /// @param 
            ///              None
            /// 
            /// @return 
            ///              None
            ///
            void decompose()
            {
                bool    flag;
                uint32  i;
                uint32  its;
                uint32  j;
                uint32  jj;
                uint32  k;
                uint32  l = 0; // initialize to zero to get rid of compiler warning
                uint32  nm = 0; // initialize to zero to get rid of compiler warning

                double  anorm = 0.0;
                double  c;
                double  f;
                double  g = 0.0;
                double  h;
                double  s;
                double  scale = 0.0;
                double  x;
                double  y;
                double  z;

                NdArray<double> rv1(n_, 1);

                for (i = 0; i < n_; ++i)
                {
                    l = i + 2;
                    rv1[i] = scale * g;
                    g = s = scale = 0.0;

                    if (i < m_)
                    {
                        for (k = i; k < m_; ++k)
                        {
                            scale += std::abs(u_(k, i));
                        }

                        if (scale != 0.0)
                        {
                            for (k = i; k < m_; ++k)
                            {
                                u_(k, i) /= scale;
                                s += u_(k, i) * u_(k, i);
                            }

                            f = u_(i, i);
                            g = -SIGN(std::sqrt(s), f);
                            h = f * g - s;
                            u_(i, i) = f - g;

                            for (j = l - 1; j < n_; ++j)
                            {
                                for (s = 0.0, k = i; k < m_; ++k)
                                {
                                    s += u_(k, i) * u_(k, j);
                                }

                                f = s / h;

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
                    g = s = scale = 0.0;

                    if (i + 1 <= m_ && i + 1 != n_)
                    {
                        for (k = l - 1; k < n_; ++k)
                        {
                            scale += std::abs(u_(i, k));
                        }

                        if (scale != 0.0)
                        {
                            for (k = l - 1; k < n_; ++k)
                            {
                                u_(i, k) /= scale;
                                s += u_(i, k) * u_(i, k);
                            }

                            f = u_(i, l - 1);
                            g = -SIGN(std::sqrt(s), f);
                            h = f * g - s;
                            u_(i, l - 1) = f - g;

                            for (k = l - 1; k < n_; ++k)
                            {
                                rv1[k] = u_(i, k) / h;
                            }

                            for (j = l - 1; j < m_; ++j)
                            {
                                for (s = 0.0, k = l - 1; k < n_; ++k)
                                {
                                    s += u_(j, k) * u_(i, k);
                                }

                                for (k = l - 1; k < n_; ++k)
                                {
                                    u_(j, k) += s * rv1[k];
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
                        if (g != 0.0)
                        {
                            for (j = l; j < n_; ++j)
                            {
                                v_(j, i) = (u_(i, j) / u_(i, l)) / g;
                            }

                            for (j = l; j < n_; ++j)
                            {
                                for (s = 0.0, k = l; k < n_; ++k)
                                {
                                    s += u_(i, k) * v_(k, j);
                                }

                                for (k = l; k < n_; ++k)
                                {
                                    v_(k, j) += s * v_(k, i);
                                }
                            }
                        }

                        for (j = l; j < n_; ++j)
                        {
                            v_(i, j) = v_(j, i) = 0.0;
                        }
                    }

                    v_(i, i) = 1.0;
                    g = rv1[i];
                    l = i;
                }

                for (i = std::min(m_, n_) - 1; i != static_cast<uint32>(-1); --i)
                {
                    l = i + 1;
                    g = s_[i];

                    for (j = l; j < n_; ++j)
                    {
                        u_(i, j) = 0.0;
                    }

                    if (g != 0.0)
                    {
                        g = 1.0 / g;

                        for (j = l; j < n_; ++j)
                        {
                            for (s = 0.0, k = l; k < m_; ++k)
                            {
                                s += u_(k, i) * u_(k, j);
                            }

                            f = (s / u_(i, i)) * g;

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
                            u_(j, i) = 0.0;
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

                            if (abs(s_[nm]) <= eps_ * anorm)
                            {
                                break;
                            }
                        }

                        if (flag)
                        {
                            c = 0.0;
                            s = 1.0;
                            for (i = l; i < k + 1; ++i)
                            {
                                f = s * rv1[i];
                                rv1[i] = c * rv1[i];

                                if (abs(f) <= eps_ * anorm)
                                {
                                    break;
                                }

                                g = s_[i];
                                h = pythag(f, g);
                                s_[i] = h;
                                h = 1.0 / h;
                                c = g * h;
                                s = -f * h;

                                for (j = 0; j < m_; ++j)
                                {
                                    y = u_(j, nm);
                                    z = u_(j, i);
                                    u_(j, nm) = y * c + z * s;
                                    u_(j, i) = z * c - y * s;
                                }
                            }
                        }

                        z = s_[k];
                        if (l == k)
                        {
                            if (z < 0.0)
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
                            throw std::runtime_error("ERROR: no convergence in 30 svdcmp iterations");
                        }

                        x = s_[l];
                        nm = k - 1;
                        y = s_[nm];
                        g = rv1[nm];
                        h = rv1[k];
                        f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
                        g = pythag(f, 1.0);
                        f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
                        c = s = 1.0;

                        for (j = l; j <= nm; j++)
                        {
                            i = j + 1;
                            g = rv1[i];
                            y = s_[i];
                            h = s * g;
                            g = c * g;
                            z = pythag(f, h);
                            rv1[j] = z;
                            c = f / z;
                            s = h / z;
                            f = x * c + g * s;
                            g = g * c - x * s;
                            h = y * s;
                            y *= c;

                            for (jj = 0; jj < n_; ++jj)
                            {
                                x = v_(jj, j);
                                z = v_(jj, i);
                                v_(jj, j) = x * c + z * s;
                                v_(jj, i) = z * c - x * s;
                            }

                            z = pythag(f, h);
                            s_[j] = z;

                            if (z)
                            {
                                z = 1.0 / z;
                                c = f * z;
                                s = h * z;
                            }

                            f = c * g + s * y;
                            x = c * y - s * g;

                            for (jj = 0; jj < m_; ++jj)
                            {
                                y = u_(jj, j);
                                z = u_(jj, i);
                                u_(jj, j) = y * c + z * s;
                                u_(jj, i) = z * c - y * s;
                            }
                        }
                        rv1[l] = 0.0;
                        rv1[k] = f;
                        s_[k] = x;
                    }
                }
            }

            // =============================================================================
            // Description:
            ///              reorders the input matrix
            /// 
            /// @param 
            ///              None
            /// 
            /// @return 
            ///              None
            ///
            void reorder()
            {
                uint32  i;
                uint32  j;
                uint32  k;
                uint32  s;
                uint32  inc = 1;

                double			sw;
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
                    s = 0;

                    for (i = 0; i < m_; i++)
                    {
                        if (u_(i, k) < 0.)
                        {
                            s++;
                        }
                    }

                    for (j = 0; j < n_; ++j)
                    {
                        if (v_(j, k) < 0.)
                        {
                            s++;
                        }
                    }

                    if (s > (m_ + n_) / 2)
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
            ///              performs pythag of input values
            ///  
            /// @param              a
            /// @param              b
            /// 
            /// @return 
            ///              resultant value
            ///
            double pythag(double a, double b)
            {
                double absa = std::abs(a);
                double absb = std::abs(b);
                return (absa > absb ? absa * std::sqrt(1.0 + Utils<double>::sqr(absb / absa)) : (absb == 0.0 ? 0.0 : absb * std::sqrt(1.0 + Utils<double>::sqr(absa / absb))));
            }
        };
    };
}
