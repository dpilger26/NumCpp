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

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Linalg/svd/SVDClass.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::linalg
{
    //============================================================================
    // Method Description:
    /// Solves the equation a x = b by computing a vector x
    /// that minimizes the Euclidean 2-norm || b - a x ||^2.
    /// The equation may be under-, well-, or over- determined
    /// (i.e., the number of linearly independent rows of a can
    /// be less than, equal to, or greater than its number of
    /// linearly independent columns). If a is square and of
    /// full rank, then x (but for round-off error) is the
    /// "exact" solution of the equation.
    ///
    /// SciPy Reference:
    /// https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html#scipy.linalg.lstsq
    ///
    /// @param inA: coefficient matrix
    /// @param inB: Ordinate or "dependent variable" values. If b is two-dimensional, the least-squares solution is
    ///             calculated for each of the K columns of b.
    /// @param inTolerance (default 1e-12)
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> lstsq(const NdArray<dtype>& inA, const NdArray<dtype>& inB, double inTolerance = 1e-12)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto& aShape = inA.shape();
        const auto& bShape = inB.shape();

        const auto bIsFlat = inB.isflat();
        if (bIsFlat && bShape.size() != aShape.rows)
        {
            THROW_INVALID_ARGUMENT_ERROR("Invalid matrix dimensions");
        }
        else if (!bIsFlat && inA.shape().rows != bShape.rows)
        {
            THROW_INVALID_ARGUMENT_ERROR("Invalid matrix dimensions");
        }

        SVD          svdSolver(inA.template astype<double>());
        const double threshold = inTolerance * svdSolver.s().front();

        if (bIsFlat)
        {
            return svdSolver.solve(inB.template astype<double>(), threshold);
        }

        const auto bCast     = inB.template astype<double>();
        const auto bRowSlice = bCast.rSlice();

        auto       result         = NdArray<double>(aShape.cols, bShape.cols);
        const auto resultRowSlice = result.rSlice();

        for (uint32 col = 0; col < bShape.cols; ++col)
        {
            result.put(resultRowSlice, col, svdSolver.solve(bCast(bRowSlice, col), threshold));
        }

        return result;
    }
} // namespace nc::linalg
