/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.2
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
/// matrix pivot LU decomposition
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Error.hpp"
#include "NumCpp/Functions/zeros_like.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"

#include <cmath>
#include <tuple>

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
            if(!inMatrix.issquare()) 
            {
                THROW_RUNTIME_ERROR("Input matrix should be square.");
            }

            NdArray<double> lMatrix = zeros_like<double>(inMatrix);
            NdArray<double> uMatrix = inMatrix.astype<double>();

            auto shape = inMatrix.shape();

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
    }
}
