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
/// matrix determinant.
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include <cmath>
#include <string>

namespace nc
{
    namespace linalg
    {
        //============================================================================
        // Method Description:
        /// matrix determinant.
        /// NOTE: can get verrrrry slow for large matrices (order > 10)
        ///
        /// SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.det.html#scipy.linalg.det
        ///
        /// @param inArray
        /// @return matrix determinant
        ///
        template<typename dtype>
        dtype det(const NdArray<dtype>& inArray)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            const Shape inShape = inArray.shape();
            if (inShape.rows != inShape.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must be square with size no larger than 3x3.");
            }

            if (inShape.rows == 1)
            {
                return inArray.front();
            }
            
            if (inShape.rows == 2)
            {
                return inArray(0, 0) * inArray(1, 1) - inArray(0, 1) * inArray(1, 0);
            }

            if (inShape.rows == 3)
            {
                dtype aei = inArray(0, 0) * inArray(1, 1) * inArray(2, 2);
                dtype bfg = inArray(0, 1) * inArray(1, 2) * inArray(2, 0);
                dtype cdh = inArray(0, 2) * inArray(1, 0) * inArray(2, 1);
                dtype ceg = inArray(0, 2) * inArray(1, 1) * inArray(2, 0);
                dtype bdi = inArray(0, 1) * inArray(1, 0) * inArray(2, 2);
                dtype afh = inArray(0, 0) * inArray(1, 2) * inArray(2, 1);

                return aei + bfg + cdh - ceg - bdi - afh;
            }
            
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
                determinant += (static_cast<dtype>(std::pow(-1, c)) * inArray(0, c) * det(submat));
            }

            return determinant;
        }
    }  // namespace linalg
}  // namespace nc
