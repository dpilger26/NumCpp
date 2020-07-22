/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.1.0
///
/// @section License
/// Copyright 2019 Benjamin Mahr
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
/// matrix pivot LU decomposition
///
/// Code modified under MIT license from https://github.com/Ben1980/linAlg
/// as posted in https://thoughts-on-coding.com/2019/06/12/numerical-methods-with-c-part-4-introduction-into-decomposition-methods-of-linear-equation-systems/
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/zeros_like.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"

#include <cmath>
#include <utility>

namespace nc
{
    namespace linalg
    {
        //============================================================================
        // Method Description:
        ///						matrix LU decomposition A = LU
        ///
        /// @param				inMatrix: NdArray to be decomposed
        ///
        /// @return             std::pair<NdArray, NdArray> of the decomposed L and U matrices
        ///
        template<typename dtype>
        std::pair<NdArray<double>, NdArray<double> > lu_decomposition(const NdArray<dtype>& inMatrix)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            const auto shape = inMatrix.shape();
            if(!shape.issquare()) 
            {
                THROW_RUNTIME_ERROR("Input matrix should be square.");
            }

            NdArray<double> lMatrix = zeros_like<double>(inMatrix);
            NdArray<double> uMatrix = inMatrix.template astype<double>();

            for(uint32 col = 0; col < shape.cols; ++col)
            {
                lMatrix(col, col) = 1;

                for(uint32 row = col + 1; row < shape.rows; ++row)
                {
                    const double& divisor = uMatrix(col, col);
                    if (utils::essentiallyEqual(divisor, double{0.0}))
                    {
                        THROW_RUNTIME_ERROR("Division by 0.");
                    }

                    lMatrix(row, col) = uMatrix(row, col) / divisor;

                    for(uint32 col2 = col; col2 < shape.cols; ++col2) 
                    {
                        uMatrix(row, col2) -= lMatrix(row, col) * uMatrix(col, col2);
                    }
                }
            }

            return std::make_pair(lMatrix, uMatrix);
        }
    }  // namespace linalg
}  // namespace nc
