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
/// Such a distribution produces random numbers uniformly
/// distributed on the unit sphere of arbitrary dimension dim.
///
#pragma once

#ifndef NUMCPP_NO_USE_BOOST

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Random/generator.hpp"

#include "boost/random/uniform_on_sphere.hpp"

#include <algorithm>
#include <string>

namespace nc
{
    namespace random
    {
        //============================================================================
        // Method Description:
        /// Such a distribution produces random numbers uniformly
        /// distributed on the unit sphere of arbitrary dimension dim.
        /// NOTE: Use of this function requires using the Boost includes.
        ///
        /// @param inNumPoints
        /// @param inDims: dimension of the sphere (default 2)
        /// @return NdArray
        ///
        template<typename dtype>
        NdArray<dtype> uniformOnSphere(uint32 inNumPoints, uint32 inDims = 2)
        {
            STATIC_ASSERT_FLOAT(dtype);

            if (inNumPoints == 0)
            {
                return {};
            }

            boost::random::uniform_on_sphere<dtype> dist(static_cast<int>(inDims));

            NdArray<dtype> returnArray(inNumPoints, inDims);
            for (uint32 row = 0; row < inNumPoints; ++row)
            {
                const auto& point = dist(generator_);
                std::copy(point.begin(), point.end(), returnArray.begin(row));
            }

            return returnArray;
        }
    } // namespace random
} // namespace nc

#endif // #ifndef NUMCPP_NO_USE_BOOST
