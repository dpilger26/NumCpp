/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
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
/// Methods for working with NdArrays
///
#pragma once

#include"NumCpp/Core/Shape.hpp"
#include"NumCpp/Core/Types.hpp"
#include"NumCpp/Methods/argmin.hpp"
#include"NumCpp/Methods/clip.hpp"
#include"NumCpp/NdArray/NdArray.hpp"
#include"NumCpp/Utils/essentiallyEqual.hpp"

#include<algorithm>
#include<cmath>
#include<iostream>
#include<string>
#include<stdexcept>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Compute the qth percentile of the data along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.percentile.html
    ///
    /// @param				inArray
    /// @param				inPercentile: percentile must be in the range [0, 100]
    /// @param				inAxis (Optional, default NONE)
    /// @param				inInterpMethod (Optional) interpolation method
    ///					linear: i + (j - i) * fraction, where fraction is the fractional part of the index surrounded by i and j.
    ///					lower : i.
    ///					higher : j.
    ///					nearest : i or j, whichever is nearest.
    ///					midpoint : (i + j) / 2.
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> percentile(const NdArray<dtype>& inArray, double inPercentile,
        Axis inAxis = Axis::NONE, const std::string& inInterpMethod = "linear")
    {
        if (inPercentile < 0.0 || inPercentile > 100.0)
        {
            std::string errStr = "ERROR: percentile: input percentile value must be of the range [0, 100].";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        if (inInterpMethod.compare("linear") != 0 &&
            inInterpMethod.compare("lower") != 0 &&
            inInterpMethod.compare("higher") != 0 &&
            inInterpMethod.compare("nearest") != 0 &&
            inInterpMethod.compare("midpoint") != 0)
        {
            std::string errStr = "ERROR: percentile: input interpolation method is not a vaid option.\n";
            errStr += "\tValid options are 'linear', 'lower', 'higher', 'nearest', 'midpoint'.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (utils::essentiallyEqual(inPercentile, 0.0))
                {
                    NdArray<dtype> returnArray = { *inArray.cbegin() };
                    return returnArray;
                }
                else if (utils::essentiallyEqual(inPercentile, 100.0))
                {
                    NdArray<dtype> returnArray = { *inArray.cend() };
                    return returnArray;
                }

                const int32 i = static_cast<int32>(std::floor(static_cast<double>(inArray.size() - 1) * inPercentile / 100.0));
                const uint32 indexLower = static_cast<uint32>(clip<uint32>(i, 0, inArray.size() - 2));

                NdArray<double> arrayCopy = inArray.template astype<double>();
                std::sort(arrayCopy.begin(), arrayCopy.end());

                if (inInterpMethod.compare("linear") == 0)
                {
                    const double percentI = static_cast<double>(indexLower) / static_cast<double>(inArray.size() - 1);
                    const double fraction = (inPercentile / 100.0 - percentI) /
                        (static_cast<double>(indexLower + 1) / static_cast<double>(inArray.size() - 1) - percentI);

                    const double returnValue = arrayCopy[indexLower] + (arrayCopy[indexLower + 1] - arrayCopy[indexLower]) * fraction;
                    NdArray<dtype> returnArray = { returnValue };
                    return returnArray;
                }
                else if (inInterpMethod.compare("lower") == 0)
                {
                    NdArray<dtype> returnArray = { arrayCopy[indexLower] };
                    return returnArray;
                }
                else if (inInterpMethod.compare("higher") == 0)
                {
                    NdArray<dtype> returnArray = { arrayCopy[indexLower + 1] };
                    return returnArray;
                }
                else if (inInterpMethod.compare("nearest") == 0)
                {
                    const double percent = inPercentile / 100.0;
                    const double percent1 = static_cast<double>(indexLower) / static_cast<double>(inArray.size() - 1);
                    const double percent2 = static_cast<double>(indexLower + 1) / static_cast<double>(inArray.size() - 1);
                    const double diff1 = percent - percent1;
                    const double diff2 = percent2 - percent;

                    switch (argmin<double>({ diff1, diff2 }).item())
                    {
                        case 0:
                        {
                            NdArray<dtype> returnArray = { arrayCopy[indexLower] };
                            return returnArray;
                        }
                        case 1:
                        {
                            NdArray<dtype> returnArray = { arrayCopy[indexLower + 1] };
                            return returnArray;
                        }
                    }
                }
                else if (inInterpMethod.compare("midpoint") == 0)
                {
                    NdArray<dtype> returnArray = { static_cast<dtype>((arrayCopy[indexLower] + arrayCopy[indexLower + 1]) / 2.0) };
                    return returnArray;
                }
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();

                NdArray<dtype> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray[row] = percentile(NdArray<dtype>(inArray.cbegin(row), inArray.cend(row)),
                        inPercentile, Axis::NONE, inInterpMethod).item();
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> arrayTrans = inArray.transpose();
                const Shape inShape = arrayTrans.shape();

                NdArray<dtype> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray[row] = percentile(NdArray<dtype>(arrayTrans.cbegin(row), arrayTrans.cend(row)),
                        inPercentile, Axis::NONE, inInterpMethod).item();
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtype>(0);
            }
        }
    }
}
