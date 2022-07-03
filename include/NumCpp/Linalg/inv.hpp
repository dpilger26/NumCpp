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
/// matrix inverse
///
#pragma once

#include <algorithm>
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/zeros.hpp"
#include "NumCpp/Linalg/det.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"

namespace nc
{
    namespace linalg
    {
        //============================================================================
        // Method Description:
        /// matrix inverse
        ///
        /// SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html#scipy.linalg.inv
        ///
        /// @param inArray
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<double> inv1(const NdArray<dtype>& inArray)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            const Shape inShape = inArray.shape();
            if (inShape.rows != inShape.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must be square.");
            }

            const uint32 order = inShape.rows;

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

            return returnArray;
        }

        namespace detail
        {
            //============================================================================
            // Method Description:
            // Function to get cofactor of inArray(p, q) in inTemp(). n is current
            // dimension of inArray()
            //
            /// @param inArray
            /// @param p
            /// @param q
            /// @param n
            /// @param outCofactor
            ///
            template<typename dtype>
            void getCofactor(const NdArray<dtype>& inArray, uint32 p, uint32 q, uint32 n, NdArray<dtype>& outCofactor)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                uint32 i = 0;
                uint32 j = 0;

                // Looping for each element of the matrix
                for (uint32 row = 0; row < n; ++row)
                {
                    for (uint32 col = 0; col < n; ++col)
                    {
                        //  Copying into temporary matrix only those element
                        //  which are not in given row and column
                        if (row == p || col == q)
                        {
                            continue;
                        }

                        outCofactor(i, j++) = inArray(row, col);

                        if (j == n - 1)
                        {
                            // Row is filled, so increase row index and
                            // reset col index
                            j = 0;
                            ++i;
                        }
                    }
                }
            }

            //============================================================================
            // Method Description:
            /// Matrix adjoint
            ///
            /// @param inArray
            /// @return NdArray
            ///
            template<typename dtype>
            NdArray<dtype> adjoint(const NdArray<dtype>& inArray)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                const auto inShape = inArray.shape();
                const auto order   = static_cast<uint32>(inShape.rows);
                if (order == 1)
                {
                    return { 1 };
                }

                int            sign = 1;
                NdArray<dtype> adj(inShape);
                NdArray<dtype> cofactor(inShape);

                for (uint32 i = 0; i < order; ++i)
                {
                    for (uint32 j = 0; j < order; ++j)
                    {
                        getCofactor(inArray, i, j, order, cofactor);

                        // sign of adj(i, j) positive if sum of row
                        // and column indexes is even.
                        sign = ((i + j) % 2 == 0) ? 1 : -1;

                        // Interchanging rows and columns to get the
                        // transpose of the cofactor matrix
                        adj(j, i) = sign * detail::det(cofactor, order - 1);
                    }
                }

                return adj;
            }
        } // namespace detail

        //============================================================================
        // Method Description:
        /// matrix inverse
        ///
        /// SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html#scipy.linalg.inv
        ///
        /// @param inArray
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<double> inv2(const NdArray<dtype>& inArray)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            const auto inShape = inArray.shape();
            if (!inShape.issquare())
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must be square.");
            }

            const auto order = static_cast<uint32>(inShape.rows);

            const auto det_ = static_cast<double>(detail::det(inArray, inShape.rows));
            if (det_ == 0)
            {
                THROW_RUNTIME_ERROR("Input array is singular.");
            }

            const auto adj = detail::adjoint(inArray).template astype<double>();

            NdArray<double> returnArray(inShape);
            for (uint32 i = 0; i < order; ++i)
            {
                for (uint32 j = 0; j < order; ++j)
                {
                    returnArray(i, j) = adj(i, j) / det_;
                }
            }

            return returnArray;
        }

        //============================================================================
        // Method Description:
        /// matrix inverse
        ///
        /// SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html#scipy.linalg.inv
        ///
        /// @param inArray
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<double> inv3(const NdArray<dtype>& inArray)
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const Shape inShape = inArray.shape();
            if (inShape.rows != inShape.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must be square.");
            }

            NdArray<double> inArrayDouble = inArray.template astype<double>();
            NdArray<int>    incidence     = nc::zeros<int>(inShape);

            for (uint32 k = 0; k < inShape.rows - 1; ++k)
            {
                if (utils::essentiallyEqual(inArrayDouble(k, k), 0.0))
                {
                    uint32 l = k;
                    while (l < inShape.cols && utils::essentiallyEqual(inArrayDouble(k, l), 0.0))
                    {
                        ++l;
                    }

                    inArrayDouble.swapRows(k, l);
                    incidence(k, k) = 1;
                    incidence(k, l) = 1;
                }
            }

            NdArray<double> result(inShape);

            for (uint32 k = 0; k < inShape.rows; ++k)
            {
                result(k, k) = -1.0 / inArrayDouble(k, k);
                for (uint32 i = 0; i < inShape.rows; ++i)
                {
                    for (uint32 j = 0; j < inShape.cols; ++j)
                    {
                        if ((i - k) && (j - k))
                        {
                            result(i, j) =
                                inArrayDouble(i, j) + inArrayDouble(k, j) * inArrayDouble(i, k) * result(k, k);
                        }
                        else if ((i - k) && !(j - k))
                        {
                            result(i, k) = inArrayDouble(i, k) * result(k, k);
                        }
                        else if (!(i - k) && (j - k))
                        {
                            result(k, j) = inArrayDouble(k, j) * result(k, k);
                        }
                    }
                }

                for (uint32 i = 0; i < inShape.rows; ++i)
                {
                    for (uint32 j = 0; j < inShape.cols; ++j)
                    {
                        inArrayDouble(i, j) = result(i, j);
                    }
                }
            }

            result *= -1.0;

            for (int i = static_cast<int>(inShape.rows) - 1; i >= 0; --i)
            {
                if (incidence(i, i) == 1)
                {
                    int k = 0;
                    for (;; ++k)
                    {
                        if ((k - i) && incidence(i, k) != 0)
                        {
                            break;
                        }
                    }

                    result.swapCols(i, k);
                }
            }

            return result;
        }

        //============================================================================
        // Method Description:
        /// matrix inverse
        ///
        /// SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html#scipy.linalg.inv
        ///
        /// @param inArray
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<double> inv(const NdArray<dtype>& inArray)
        {
            return inv1(inArray);
        }
    } // namespace linalg
} // namespace nc
