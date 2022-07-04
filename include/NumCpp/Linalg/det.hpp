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

#include <cmath>
#include <string>
#include <type_traits>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/TypeTraits.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    namespace linalg
    {
        namespace detail
        {
            //============================================================================
            // Method Description:
            /// matrix determinant.
            ///
            /// @param inArray
            /// @param order
            /// @return matrix determinant
            ///
            template<typename dtype>
            auto det(const NdArray<dtype>& inArray, uint32 order)
                -> std::conditional_t<is_integral_v<dtype>, int64, double>
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                using ReturnType = std::conditional_t<is_integral_v<dtype>, int64, double>;

                if (order == 1)
                {
                    return static_cast<ReturnType>(inArray.front());
                }

                if (order == 2)
                {
                    return static_cast<ReturnType>(inArray(0, 0)) * static_cast<ReturnType>(inArray(1, 1)) -
                           static_cast<ReturnType>(inArray(0, 1)) * static_cast<ReturnType>(inArray(1, 0));
                }

                if (order == 3)
                {
                    const auto aei = static_cast<ReturnType>(inArray(0, 0)) * static_cast<ReturnType>(inArray(1, 1)) *
                                     static_cast<ReturnType>(inArray(2, 2));
                    const auto bfg = static_cast<ReturnType>(inArray(0, 1)) * static_cast<ReturnType>(inArray(1, 2)) *
                                     static_cast<ReturnType>(inArray(2, 0));
                    const auto cdh = static_cast<ReturnType>(inArray(0, 2)) * static_cast<ReturnType>(inArray(1, 0)) *
                                     static_cast<ReturnType>(inArray(2, 1));
                    const auto ceg = static_cast<ReturnType>(inArray(0, 2)) * static_cast<ReturnType>(inArray(1, 1)) *
                                     static_cast<ReturnType>(inArray(2, 0));
                    const auto bdi = static_cast<ReturnType>(inArray(0, 1)) * static_cast<ReturnType>(inArray(1, 0)) *
                                     static_cast<ReturnType>(inArray(2, 2));
                    const auto afh = static_cast<ReturnType>(inArray(0, 0)) * static_cast<ReturnType>(inArray(1, 2)) *
                                     static_cast<ReturnType>(inArray(2, 1));

                    return aei + bfg + cdh - ceg - bdi - afh;
                }

                ReturnType     determinant = 0;
                ReturnType     sign        = 1;
                NdArray<dtype> submat(order - 1);

                for (uint32 c = 0; c < order; ++c)
                {
                    uint32 subi = 0;
                    for (uint32 i = 1; i < order; ++i)
                    {
                        uint32 subj = 0;
                        for (uint32 j = 0; j < order; ++j)
                        {
                            if (j == c)
                            {
                                continue;
                            }

                            submat(subi, subj++) = inArray(i, j);
                        }
                        ++subi;
                    }

                    determinant += (sign * static_cast<ReturnType>(inArray(0, c)) * det(submat, order - 1));
                    sign *= -1;
                }

                return determinant;
            }
        } // namespace detail

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
        auto det(const NdArray<dtype>& inArray)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            const Shape inShape = inArray.shape();
            if (!inShape.issquare())
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must be square.");
            }

            return detail::det(inArray, inShape.rows);
        }
    } // namespace linalg
} // namespace nc
