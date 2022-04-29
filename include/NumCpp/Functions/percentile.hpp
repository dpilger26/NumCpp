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
/// Functions for working with NdArrays
///
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/argmin.hpp"
#include "NumCpp/Functions/clip.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Compute the qth percentile of the data along the specified axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.percentile.html
    ///
    /// @param inArray
    /// @param inPercentile: percentile must be in the range [0, 100]
    /// @param inAxis (Optional, default NONE)
    /// @param inInterpMethod (Optional) interpolation method
    /// linear: i + (j - i) * fraction, where fraction is the fractional part of the index surrounded by i and j.
    /// lower : i.
    /// higher : j.
    /// nearest : i or j, whichever is nearest.
    /// midpoint : (i + j) / 2.
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> percentile(const NdArray<dtype>& inArray,
                               double                inPercentile,
                               Axis                  inAxis         = Axis::NONE,
                               const std::string&    inInterpMethod = "linear")
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (inPercentile < 0.0 || inPercentile > 100.0)
        {
            THROW_INVALID_ARGUMENT_ERROR("input percentile value must be of the range [0, 100].");
        }

        if (inArray.isempty())
        {
            return {};
        }
        else if (inArray.size() == 1)
        {
            NdArray<double> returnArray = { static_cast<double>(inArray.front()) };
            return returnArray;
        }

        if (inInterpMethod != "linear" && inInterpMethod != "lower" && inInterpMethod != "higher" &&
            inInterpMethod != "nearest" && inInterpMethod != "midpoint")
        {
            std::string errStr = "input interpolation method is not a vaid option.\n";
            errStr += "\tValid options are 'linear', 'lower', 'higher', 'nearest', 'midpoint'.";
            THROW_INVALID_ARGUMENT_ERROR(errStr);
        }

        switch (inAxis)
        {
            case Axis::NONE:
            {
                NdArray<double> arrayCopy = inArray.template astype<double>();
                stl_algorithms::sort(arrayCopy.begin(), arrayCopy.end());

                if (utils::essentiallyEqual(inPercentile, 0.0))
                {
                    NdArray<double> returnArray = { arrayCopy.front() };
                    return returnArray;
                }
                if (utils::essentiallyEqual(inPercentile, 100.0))
                {
                    NdArray<double> returnArray = { arrayCopy.back() };
                    return returnArray;
                }

                const auto i =
                    static_cast<uint32>(std::floor(static_cast<double>(inArray.size() - 1) * inPercentile / 100.0));
                const auto indexLower = clip<uint32>(i, 0, inArray.size() - 2);

                if (inInterpMethod == "linear")
                {
                    const double percentI = static_cast<double>(indexLower) / static_cast<double>(inArray.size() - 1);
                    const double fraction =
                        (inPercentile / 100.0 - percentI) /
                        (static_cast<double>(indexLower + 1) / static_cast<double>(inArray.size() - 1) - percentI);

                    NdArray<double> returnArray = { arrayCopy[indexLower] +
                                                    (arrayCopy[indexLower + 1] - arrayCopy[indexLower]) * fraction };
                    return returnArray;
                }

                if (inInterpMethod == "lower")
                {
                    NdArray<double> returnArray = { arrayCopy[indexLower] };
                    return returnArray;
                }

                if (inInterpMethod == "higher")
                {
                    NdArray<double> returnArray = { arrayCopy[indexLower + 1] };
                    return returnArray;
                }

                if (inInterpMethod == "nearest")
                {
                    const double percent  = inPercentile / 100.0;
                    const double percent1 = static_cast<double>(indexLower) / static_cast<double>(inArray.size() - 1);
                    const double percent2 =
                        static_cast<double>(indexLower + 1) / static_cast<double>(inArray.size() - 1);
                    const double diff1 = percent - percent1;
                    const double diff2 = percent2 - percent;

                    switch (argmin<double>({ diff1, diff2 }).item())
                    {
                        case 0:
                        {
                            NdArray<double> returnArray = { arrayCopy[indexLower] };
                            return returnArray;
                        }
                        case 1:
                        {
                            NdArray<double> returnArray = { arrayCopy[indexLower + 1] };
                            return returnArray;
                        }
                    }
                }

                if (inInterpMethod == "midpoint")
                {
                    NdArray<double> returnArray = { (arrayCopy[indexLower] + arrayCopy[indexLower + 1]) / 2.0 };
                    return returnArray;
                }

                THROW_INVALID_ARGUMENT_ERROR("interpolation method has not been implemented: " + inInterpMethod);
                break; // get rid of compiler warning...
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();

                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray[row] = percentile(NdArray<dtype>(&inArray.front(row), inShape.cols),
                                                  inPercentile,
                                                  Axis::NONE,
                                                  inInterpMethod)
                                           .item();
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> arrayTrans = inArray.transpose();
                const Shape    inShape    = arrayTrans.shape();

                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray[row] = percentile(NdArray<dtype>(&arrayTrans.front(row), inShape.cols, false),
                                                  inPercentile,
                                                  Axis::NONE,
                                                  inInterpMethod)
                                           .item();
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
