/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
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
/// In applied mathematics, Wahba's problem, first posed by Grace Wahba in 1965, seeks to
/// find a rotation matrix (special orthogonal matrix) between two coordinate systems from
/// a set of (weighted) vector observations. Solutions to Wahba's problem are often used in
/// satellite attitude determination utilising sensors such as magnetometers and multi-antenna
/// GPS receivers
/// https://en.wikipedia.org/wiki/Wahba%27s_problem
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Functions/dot.hpp"
#include "NumCpp/Functions/eye.hpp"
#include "NumCpp/Functions/ones.hpp"
#include "NumCpp/Functions/zeros.hpp"
#include "NumCpp/Linalg/det.hpp"
#include "NumCpp/Linalg/svd.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    namespace rotations
    {
        //============================================================================
        // Method Description:
        /// Finds a rotation matrix (special orthogonal matrix) between two coordinate
        /// systems from a set of (weighted) vector observations. Solutions to Wahba's
        /// problem are often used in satellite attitude determination utilising sensors
        /// such as magnetometers and multi-antenna GPS receivers
        /// https://en.wikipedia.org/wiki/Wahba%27s_problem
        ///
        /// @param wk: k-th 3-vector measurement in the reference frame (n x 3 matrix)
        /// @param vk: corresponding k-th 3-vector measurement in the body frame (n x 3 matrix)
        /// @param ak: set of weights for each observation (1 x n or n x 1 matrix)
        ///
        /// @return NdArray rotation matrix
        ///
        template<typename dtype>
        NdArray<double> wahbasProblem(const NdArray<dtype>& wk, const NdArray<dtype>& vk, const NdArray<dtype>& ak)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            const auto wkShape = wk.shape();
            if (wkShape.cols != 3)
            {
                THROW_INVALID_ARGUMENT_ERROR("wk matrix must be of shape [n, 3]");
            }

            const auto vkShape = vk.shape();
            if (vkShape.cols != 3)
            {
                THROW_INVALID_ARGUMENT_ERROR("vk matrix must be of shape [n, 3]");
            }

            if (wkShape.rows != vkShape.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("wk and vk matrices must have the same number of rows");
            }

            if (ak.size() != wkShape.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("ak matrix must have the same number of elements as wk and vk rows");
            }

            auto       b      = zeros<dtype>(3, 3);
            const auto cSlice = wk.cSlice();
            for (uint32 row = 0; row < wkShape.rows; ++row)
            {
                const auto wkVec = wk(row, cSlice);
                const auto vkVec = vk(row, cSlice);
                b += ak[row] * dot(wkVec.transpose(), vkVec);
            }

            NdArray<double> u;
            NdArray<double> s;
            NdArray<double> vt;

            linalg::svd(b, u, s, vt);

            auto m  = eye<double>(3, 3);
            m(0, 0) = 1.;
            m(1, 1) = 1.;
            m(2, 2) = linalg::det(u) * linalg::det(vt.transpose());

            return dot(u, dot(m, vt));
        }

        //============================================================================
        // Method Description:
        /// Finds a rotation matrix (special orthogonal matrix) between two coordinate
        /// systems from a set of (weighted) vector observations. Solutions to Wahba's
        /// problem are often used in satellite attitude determination utilising sensors
        /// such as magnetometers and multi-antenna GPS receivers
        /// https://en.wikipedia.org/wiki/Wahba%27s_problem
        ///
        /// @param wk: k-th 3-vector measurement in the reference frame
        /// @param vk: corresponding k-th 3-vector measurement in the body frame
        ///
        /// @return NdArray rotation matrix
        ///
        template<typename dtype>
        NdArray<double> wahbasProblem(const NdArray<dtype>& wk, const NdArray<dtype>& vk)
        {
            const auto ak = ones<dtype>({ 1, wk.shape().rows });
            return wahbasProblem(wk, vk, ak);
        }
    } // namespace rotations
} // namespace nc
