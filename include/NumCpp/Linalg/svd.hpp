/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
///
/// @section License
/// Copyright 2019 David Pilger
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
/// Compute the dot product of two or more arrays in a single function call.
///
#pragma once

#include "NumCpp/Linalg/svd/SVDClass.hpp"
#include "NumCpp/Methods/diagflat.hpp"
#include "NumCpp/NdArray.hpp"

#include <utility>

namespace nc
{
    namespace linalg
    {
        //============================================================================
        // Method Description:
        ///						matrix svd
        ///
        ///                     NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd
        ///
        /// @param				inArray: NdArray to be SVDed
        /// @param				outU: NdArray output U
        /// @param				outS: NdArray output S
        /// @param				outVt: NdArray output V transpose
        ///
        template<typename dtype>
        void svd(const NdArray<dtype>& inArray, NdArray<double>& outU, NdArray<double>& outS, NdArray<double>& outVt) noexcept
        {
            SVD svdSolver(inArray.template astype<double>());
            outU = std::move(svdSolver.u());
            outVt = std::move(svdSolver.v().transpose());

            NdArray<double> s = diagflat(svdSolver.s());
            outS = std::move(s);
        }
    }
}
