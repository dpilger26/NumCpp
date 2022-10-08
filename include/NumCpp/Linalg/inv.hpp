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
        NdArray<double> inv(const NdArray<dtype>& inArray)
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
                if (utils::essentiallyEqual(inArrayDouble(k, k), 0.))
                {
                    uint32 l = k;
                    while (l < inShape.cols && utils::essentiallyEqual(inArrayDouble(k, l), 0.))
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
                result(k, k) = -1. / inArrayDouble(k, k);
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

                inArrayDouble = result;
            }

            result *= -1.;

            for (int i = static_cast<int>(inShape.rows) - 1; i >= 0; --i)
            {
                if (incidence(i, i) != 1)
                {
                    continue;
                }

                int k = 0;
                for (; k < static_cast<int>(inShape.cols); ++k)
                {
                    if ((k - i) && incidence(i, k) != 0)
                    {
                        result.swapCols(i, k);
                        break;
                    }
                }
            }

            return result;
        }
    } // namespace linalg
} // namespace nc
