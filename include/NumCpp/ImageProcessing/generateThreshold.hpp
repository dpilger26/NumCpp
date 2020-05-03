/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.3
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
/// Generates a threshold
///

#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Core/DtypeInfo.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"

#include <cmath>
#include <string>

namespace nc
{
    namespace imageProcessing
    {
        //============================================================================
        // Method Description:
        ///						Calculates a threshold such that the input rate of pixels
        ///						exceeds the threshold. Really should only be used for integer
        ///                      input array values. If using floating point data, user beware...
        ///
        /// @param				inImageArray
        /// @param				inRate
        /// @return
        ///				dtype
        ///
        template<typename dtype>
        dtype generateThreshold(const NdArray<dtype>& inImageArray, double inRate)
        {
            if (inRate < 0.0 || inRate > 1.0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input rate must be of the range [0, 1]");
            }

            // first build a histogram
            int32 minValue = static_cast<int32>(std::floor(inImageArray.min().item()));
            int32 maxValue = static_cast<int32>(std::floor(inImageArray.max().item()));

            if (utils::essentiallyEqual(inRate, 0.0))
            {
                return static_cast<dtype>(maxValue);
            }
            else if (utils::essentiallyEqual(inRate, 1.0))
            {
                if (DtypeInfo<dtype>::isSigned())
                {
                    return static_cast<dtype>(minValue - 1);
                }
                else
                {
                    return dtype{ 0 };
                }
            }

            const uint32 histSize = static_cast<uint32>(maxValue - minValue + 1);

            NdArray<double> histogram(1, histSize);
            histogram.zeros();
            for (auto intensity : inImageArray)
            {
                const uint32 bin = static_cast<uint32>(static_cast<int32>(std::floor(intensity)) - minValue);
                ++histogram[bin];
            }

            // integrate the normalized histogram from right to left to make a survival function (1 - CDF)
            const double dNumPixels = static_cast<double>(inImageArray.size());
            NdArray<double> survivalFunction(1, histSize + 1);
            survivalFunction[-1] = 0;
            for (int32 i = histSize - 1; i > -1; --i)
            {
                double histValue = histogram[i] / dNumPixels;
                survivalFunction[i] = survivalFunction[i + 1] + histValue;
            }

            // binary search through the survival function to find the rate
            uint32 indexLow = 0;
            uint32 indexHigh = histSize - 1;
            uint32 index = indexHigh / 2; // integer division

            const bool keepGoing = true;
            while (keepGoing)
            {
                const double value = survivalFunction[index];
                if (value < inRate)
                {
                    indexHigh = index;
                }
                else if (value > inRate)
                {
                    indexLow = index;
                }
                else
                {
                    const int32 thresh = static_cast<int32>(index) + minValue - 1;
                    if (DtypeInfo<dtype>::isSigned())
                    {
                        return static_cast<dtype>(thresh);
                    }
                    else
                    {
                        return thresh < 0 ? 0 : static_cast<dtype>(thresh);
                    }
                }

                if (indexHigh - indexLow < 2)
                {
                    return static_cast<dtype>(static_cast<int32>(indexHigh) + minValue - 1);
                }

                index = indexLow + (indexHigh - indexLow) / 2;
            }

            // shouldn't ever get here but stop the compiler from throwing a warning
            return static_cast<dtype>(histSize - 1);
        }

    }
}
