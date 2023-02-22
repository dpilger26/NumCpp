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
/// Center of Mass centroids clusters
///

#pragma once

#include <utility>
#include <vector>

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/ImageProcessing/Centroid.hpp"
#include "NumCpp/ImageProcessing/Cluster.hpp"

namespace nc::imageProcessing
{
    //============================================================================
    // Method Description:
    /// Center of Mass centroids clusters
    ///
    /// @param inClusters
    /// @return std::vector<Centroid>
    ///
    template<typename dtype>
    std::vector<Centroid<dtype>> centroidClusters(const std::vector<Cluster<dtype>>& inClusters)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::vector<Centroid<dtype>> centroids(inClusters.size());
        stl_algorithms::transform(inClusters.begin(),
                                  inClusters.end(),
                                  centroids.begin(),
                                  [](const auto& cluster) -> Centroid<dtype> { return Centroid<dtype>(cluster); });
        return centroids;
    }
} // namespace nc::imageProcessing
