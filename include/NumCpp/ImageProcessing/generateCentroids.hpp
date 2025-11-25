/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2026 David Pilger
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
/// Generates a list of centroids givin an input exceedance rate
///

#pragma once

#include <string>
#include <vector>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/ImageProcessing/applyThreshold.hpp"
#include "NumCpp/ImageProcessing/centroidClusters.hpp"
#include "NumCpp/ImageProcessing/clusterPixels.hpp"
#include "NumCpp/ImageProcessing/generateThreshold.hpp"
#include "NumCpp/ImageProcessing/windowExceedances.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::imageProcessing
{
    //============================================================================
    // Method Description:
    /// Generates a list of centroids givin an input exceedance
    /// rate
    ///
    /// @param inImageArray
    /// @param inRate: exceedance rate
    /// @param inWindowType: (string "pre", or "post" for where to apply the exceedance windowing)
    /// @param inBorderWidth: border to apply (default 0)
    /// @return std::vector<Centroid>
    ///
    template<typename dtype>
    std::vector<Centroid<dtype>> generateCentroids(const NdArray<dtype>& inImageArray,
                                                   double                inRate,
                                                   const std::string&    inWindowType,
                                                   uint8                 inBorderWidth = 0)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        uint8 borderWidthPre  = 0;
        uint8 borderWidthPost = 0;
        if (inWindowType == "pre")
        {
            borderWidthPre = inBorderWidth;
        }
        else if (inWindowType == "post")
        {
            borderWidthPost = inBorderWidth;
        }
        else
        {
            THROW_INVALID_ARGUMENT_ERROR("input window type options are ['pre', 'post']");
        }

        // generate the threshold
        dtype threshold = generateThreshold(inImageArray, inRate);

        // apply the threshold to get xcds
        NdArray<bool> xcds = applyThreshold(inImageArray, threshold);

        // window around the xcds
        if (borderWidthPre > 0)
        {
            xcds = windowExceedances(xcds, borderWidthPre);
        }

        // cluster the exceedances
        std::vector<Cluster<dtype>> clusters = clusterPixels(inImageArray, xcds, borderWidthPost);

        // centroid the clusters
        return centroidClusters(clusters);
    }
} // namespace nc::imageProcessing
