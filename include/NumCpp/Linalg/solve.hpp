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
/// matrix svd
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Functions/dot.hpp"
#include "NumCpp/Linalg/inv.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    namespace linalg
    {
        //============================================================================
        // Method Description:
        /// Solve a linear matrix equation, or system of linear scalar equations.
        /// Computes the “exact” solution, x, of the well-determined, i.e., full rank,
        /// linear matrix equation ax = b.
        ///
        /// https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
        ///
        /// @param inA
        /// @param inB
        /// @return NdArray<double> Solution to the system a x = b. Returned shape is identical to b
        ///
        template<typename dtype>
        NdArray<double> solve(const NdArray<dtype>& inA, const NdArray<dtype>& inB)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            if (!inA.issquare())
            {
                THROW_INVALID_ARGUMENT_ERROR("input array a must be square.");
            }

            if (!inB.isflat())
            {
                THROW_INVALID_ARGUMENT_ERROR("input array b must be flat.");
            }

            if (inA.numCols() != inB.size())
            {
                THROW_INVALID_ARGUMENT_ERROR("input array b size must be the same as the square size of a.");
            }

            return dot(inv(inA), inB.template astype<double>().reshape(inB.size(), 1)).reshape(inB.shape());
        }
    } // namespace linalg
} // namespace nc
