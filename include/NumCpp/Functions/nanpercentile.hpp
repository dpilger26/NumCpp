/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2021 David Pilger
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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/argmin.hpp"
#include "NumCpp/Functions/clip.hpp"
#include "NumCpp/Functions/isnan.hpp"
#include "NumCpp/Functions/percentile.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Compute the qth percentile of the data along the specified axis, while ignoring nan values.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.nanpercentile.html
    ///
    /// @param				inArray
    /// @param              inPercentile
    /// @param				inAxis (Optional, default NONE)
    /// @param              inInterpMethod (default linear) choices = ['linear','lower','higher','nearest','midpoint']
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> nanpercentile(const NdArray<dtype>& inArray, double inPercentile,
        Axis inAxis = Axis::NONE, const std::string& inInterpMethod = "linear")
    {
        STATIC_ASSERT_FLOAT(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::vector<double> arrayCopy;
                arrayCopy.reserve(inArray.size());
                for (auto value : inArray)
                {
                    if (!isnan(value))
                    {
                        arrayCopy.push_back(static_cast<double>(value));
                    }
                }

                if (arrayCopy.empty())
                {
                    NdArray<double> returnArray = { constants::nan };
                    return returnArray;
                }

                return percentile(NdArray<double>(arrayCopy.data(), arrayCopy.size(), false), inPercentile, Axis::NONE, inInterpMethod);
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();

                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    NdArray<double> outValue = nanpercentile(NdArray<dtype>(&inArray.front(row), inShape.cols),
                        inPercentile, Axis::NONE, inInterpMethod);

                    if (outValue.size() == 1)
                    {
                        returnArray[row] = outValue.item();
                    }
                    else
                    {
                        returnArray[row] = constants::nan;
                    }
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> arrayTrans = inArray.transpose();
                const Shape inShape = arrayTrans.shape();

                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    NdArray<double> outValue = nanpercentile(NdArray<dtype>(&arrayTrans.front(row), inShape.cols, false),
                        inPercentile, Axis::NONE, inInterpMethod);

                    if (outValue.size() == 1)
                    {
                        returnArray[row] = outValue.item();
                    }
                    else
                    {
                        returnArray[row] = constants::nan;
                    }
                }

                return returnArray;
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }

        return {}; // get rid of compiler warning
    }
} // namespace nc
