/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.4.0
///
/// @section License
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
/// @section Description
/// linear least squares
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Linalg/svd/SVDClass.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    namespace linalg
    {
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
        /// @param				inA: coefficient matrix
        /// @param  			inB: Ordinate or "dependent variable" values
        /// @param				inTolerance (default 1e-12)
        ///
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<double> lstsq(const NdArray<dtype>& inA, const NdArray<dtype>& inB, double inTolerance = 1e-12) noexcept
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            SVD svdSolver(inA.template astype<double>());
            const double threshold = inTolerance * svdSolver.s().front();

            return svdSolver.solve(inB.template astype<double>(), threshold);
        }
    }
}
