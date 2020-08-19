/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.2.0
///
/// @section License
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
/// Special Functions
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/exp.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        ///	The softmax function transforms each element of a collection by computing 
        /// the exponential of each element divided by the sum of the exponentials of all
        /// the elements. That is, if x is a one-dimensional numpy array:
        /// softmax(x) = np.exp(x)/sum(np.exp(x))
        ///
        /// @param      inArray
        /// @param      inAxis (Optional, default NONE)
        /// @return     NdArray<double>
        ///
        template<typename dtype>
        NdArray<double> softmax(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE) 
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            switch (inAxis)
            {
                case Axis::NONE:
                {
                    auto returnArray = exp(inArray).template astype<double>();
                    returnArray /= static_cast<double>(returnArray.sum().item());
                    return returnArray;
                }
                case Axis::COL:
                {
                    auto returnArray = exp(inArray).template astype<double>();
                    auto expSums = returnArray.sum(inAxis);

                    for (uint32 row = 0; row < returnArray.shape().rows; ++row)
                    {
                        const auto rowExpSum = static_cast<double>(expSums[row]);
                        stl_algorithms::for_each(returnArray.begin(row), returnArray.end(row), 
                            [rowExpSum](double& value) { value /= rowExpSum; });
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    auto returnArray = exp(inArray.transpose()).template astype<double>();
                    auto expSums = returnArray.sum(Axis::COL);

                    for (uint32 row = 0; row < returnArray.shape().rows; ++row)
                    {
                        const auto rowExpSum = static_cast<double>(expSums[row]);
                        stl_algorithms::for_each(returnArray.begin(row), returnArray.end(row), 
                            [rowExpSum](double& value) { value /= rowExpSum; });
                    }

                    return returnArray.transpose();
                }
                default:
                {
                    THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                    return {}; // get rid of compiler warning
                }
            }
        }
    } // namespace special
}  // namespace nc
