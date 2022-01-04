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
/// Raise a square matrix to the (integer) power n.
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/dot.hpp"
#include "NumCpp/Functions/identity.hpp"
#include "NumCpp/NdArray.hpp"

#include <string>

namespace nc
{
    namespace linalg
    {
        //============================================================================
        // Method Description:
        /// Raise a square matrix to the (integer) power n.
        ///
        /// For positive integers n, the power is computed by repeated
        /// matrix squarings and matrix multiplications.  If n == 0,
        /// the identity matrix of the same shape as M is returned.
        /// If n < 0, the inverse is computed and then raised to the abs(n).
        ///
        /// NumPy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.matrix_power.html#numpy.linalg.matrix_power
        ///
        /// @param				inArray
        /// @param				inPower
        ///
        /// @return
        /// NdArray
        ///
        template<typename dtype>
        NdArray<double> matrix_power(const NdArray<dtype>& inArray, int16 inPower)
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const Shape inShape = inArray.shape();
            if (inShape.rows != inShape.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input matrix must be square.");
            }

            if (inPower == 0)
            {
                return identity<double>(inShape.rows);
            }

            if (inPower == 1)
            {
                return inArray.template astype<double>();
            }

            if (inPower == -1)
            {
                return inv(inArray);
            }

            if (inPower > 1)
            {
                NdArray<double> inArrayDouble = inArray.template astype<double>();
                NdArray<double> returnArray = dot(inArrayDouble, inArrayDouble);
                for (int16 i = 2; i < inPower; ++i)
                {
                    returnArray = dot(returnArray, inArrayDouble);
                }
                return returnArray;
            }

            NdArray<double> inverse = inv(inArray);
            NdArray<double> returnArray = dot(inverse, inverse);
            inPower *= -1;
            for (int16 i = 2; i < inPower; ++i)
            {
                returnArray = dot(returnArray, inverse);
            }
            return returnArray;
        }
    } // namespace linalg
}  // namespace nc
