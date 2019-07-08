/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
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

#include "NumCpp/Core/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Methods/linspace.hpp"
#include "NumCpp/Methods/zeros.hpp"
#include "NumCpp/NdArray.hpp"

#include <string>
#include <utility>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Compute the histogram of a set of data.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.histogram.html
    ///
    ///
    /// @param				inArray
    /// @param				inNumBins( default 10)
    ///
    /// @return
    ///				std::pair of NdArrays; first is histogram counts, seconds is the bin edges
    ///
    template<typename dtype>
    std::pair<NdArray<uint32>, NdArray<double> > histogram(const NdArray<dtype>& inArray, uint32 inNumBins = 10)
    {
        if (inNumBins == 0)
        {
            std::string errStr = "ERROR: histogram: number of bins must be positive.";
            error::throwInvalidArgument(errStr);
        }

        NdArray<uint32> histo = zeros<uint32>(1, inNumBins);

        const bool useEndPoint = true;
        NdArray<double> binEdges = linspace(static_cast<double>(inArray.min().item()),
            static_cast<double>(inArray.max().item()), inNumBins + 1, useEndPoint);

        for (uint32 i = 0; i < inArray.size(); ++i)
        {
            // binary search to find the bin idx
            const bool keepSearching = true;
            uint32 lowIdx = 0;
            uint32 highIdx = binEdges.size() - 1;
            while (keepSearching)
            {
                const uint32 idx = (lowIdx + highIdx) / 2; // integer division
                if (lowIdx == highIdx || lowIdx == highIdx - 1)
                {
                    // we found the bin
                    ++histo[lowIdx];
                    break;
                }

                if (inArray[i] > binEdges[idx])
                {
                    lowIdx = idx;
                }
                else if (inArray[i] < binEdges[idx])
                {
                    highIdx = idx;
                }
                else
                {
                    // we found the bin
                    ++histo[idx];
                    break;
                }
            }
        }

        return std::make_pair(histo, binEdges);
    }
}
