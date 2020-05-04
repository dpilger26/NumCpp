/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.4
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
/// Clusters exceedance data into contiguous groups
///

#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/ImageProcessing/Cluster.hpp"
#include "NumCpp/NdArray.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace nc
{
    namespace imageProcessing
    {
        //=============================================================================
        // Class Description:
        ///              Clusters exceedance data into contiguous groups
        template<typename dtype>
        class ClusterMaker
        {
        public:
            //================================Typedefs=====================================
            using const_iterator = typename std::vector<Cluster<dtype> >::const_iterator;

            //=============================================================================
            // Description:
            ///              constructor
            ///
            /// @param              inXcdArrayPtr: pointer to exceedance array
            /// @param              inIntensityArrayPtr: pointer to intensity array
            /// @param				inBorderWidth: border to apply around exceedance pixels post clustering (default 0)
            ///
            /// @return
            ///              None
            ///
            ClusterMaker(const NdArray<bool>* const inXcdArrayPtr, const NdArray<dtype>* const inIntensityArrayPtr, uint8 inBorderWidth = 0) :
                xcds_(inXcdArrayPtr),
                intensities_(inIntensityArrayPtr)
            {
                if (xcds_->shape() != intensities_->shape())
                {
                    THROW_INVALID_ARGUMENT_ERROR("input xcd and intensity arrays must be the same shape.");
                }

                shape_ = xcds_->shape();

                // convert the NdArray of booleans to a vector of exceedances
                for (uint32 row = 0; row < shape_.rows; ++row)
                {
                    for (uint32 col = 0; col < shape_.cols; ++col)
                    {
                        if (xcds_->operator()(row, col))
                        {
                            const Pixel<dtype> thePixel(row, col, intensities_->operator()(row, col));
                            xcdsVec_.push_back(thePixel);
                        }
                    }
                }

                runClusterMaker();

                for (uint8 i = 0; i < inBorderWidth; ++i)
                {
                    expandClusters();
                }
            }

            //=============================================================================
            // Description:
            ///              returns the number of clusters in the frame
            ///
            /// @return
            ///              number of clusters
            ///
            uint32 size()
            {
                return static_cast<uint32>(clusters_.size());
            }

            //=============================================================================
            // Description:
            ///              access operator, no bounds checking
            ///
            /// @param
            ///              inIndex
            ///
            /// @return
            ///              Cluster
            ///
            const Cluster<dtype>& operator[](uint32 inIndex) const
            {
                return clusters_[inIndex];
            }

            //=============================================================================
            // Description:
            ///              access method with bounds checking
            ///
            /// @param
            ///              inIndex
            ///
            /// @return
            ///              Cluster
            ///
            const Cluster<dtype>& at(uint32 inIndex) const
            {
                if (inIndex >= clusters_.size())
                {
                    THROW_INVALID_ARGUMENT_ERROR("index exceeds cluster size.");
                }
                return clusters_[inIndex];
            }

            //=============================================================================
            // Description:
            ///              returns in iterator to the beginning cluster of the container
            ///
            /// @return
            ///              const_iterator
            ///
            const_iterator begin() const noexcept
            {
                return clusters_.cbegin();
            }

            //=============================================================================
            // Description:
            ///              returns in iterator to the 1 past the end cluster of the container
            ///
            /// @return
            ///              const_iterator
            ///
            const_iterator end() const noexcept
            {
                return clusters_.cend();
            }

        private:
            //==================================Attributes=================================
            const NdArray<bool>* const      xcds_;
            const NdArray<dtype>* const     intensities_;
            std::vector<Pixel<dtype> >      xcdsVec_{};

            Shape                           shape_{};

            std::vector<Cluster<dtype> >    clusters_{};

            //=============================================================================
            // Description:
            ///              checks that the input row and column have not fallen off of the edge
            ///
            /// @param              inRow
            /// @param              inCol
            ///
            /// @return
            ///              returns a pixel object clipped to the image boundaries
            ///
            Pixel<dtype> makePixel(int32 inRow, int32 inCol) noexcept
            {
                // Make sure that on the edges after i've added or subtracted 1 from the row and col that
                // i haven't gone over the edge
                const uint32 row = std::min(static_cast<uint32>(std::max<int32>(inRow, 0)), shape_.rows - 1);
                const uint32 col = std::min(static_cast<uint32>(std::max<int32>(inCol, 0)), shape_.cols - 1);
                const dtype intensity = intensities_->operator()(row, col);

                return Pixel<dtype>(row, col, intensity);
            }

            //=============================================================================
            // Description:
            ///              finds all of the neighboring pixels to the input pixel
            ///
            /// @param               inPixel
            /// @param               outNeighbors
            /// @return
            ///              None
            ///
            void findNeighbors(const Pixel<dtype>& inPixel, std::set<Pixel<dtype> >& outNeighbors)
            {
                // using a set will auto take care of adding duplicate pixels on the edges

                // the 8 surrounding neighbors
                const int32 row = static_cast<int32>(inPixel.row);
                const int32 col = static_cast<int32>(inPixel.col);

                outNeighbors.insert(outNeighbors.end(), makePixel(row - 1, col - 1));
                outNeighbors.insert(outNeighbors.end(), makePixel(row - 1, col));
                outNeighbors.insert(outNeighbors.end(), makePixel(row - 1, col + 1));
                outNeighbors.insert(outNeighbors.end(), makePixel(row, col - 1));
                outNeighbors.insert(outNeighbors.end(), makePixel(row, col + 1));
                outNeighbors.insert(outNeighbors.end(), makePixel(row + 1, col - 1));
                outNeighbors.insert(outNeighbors.end(), makePixel(row + 1, col));
                outNeighbors.insert(outNeighbors.end(), makePixel(row + 1, col + 1));
            }

            //=============================================================================
            // Description:
            ///              finds all of the neighboring pixels to the input pixel that are NOT exceedances
            ///
            /// @param       inPixel
            /// @param       outNeighbors
            ///
            /// @return
            ///              vector of non exceedance neighboring pixels
            ///
            void findNeighborNotXcds(const Pixel<dtype>& inPixel, std::vector<Pixel<dtype> >& outNeighbors)
            {
                std::set<Pixel<dtype> > neighbors;
                findNeighbors(inPixel, neighbors);

                // check if the neighboring pixels are exceedances and insert into the xcd vector
                for (auto& pixel : neighbors)
                {
                    if (!xcds_->operator()(pixel.row, pixel.col))
                    {
                        outNeighbors.push_back(pixel);
                    }
                }
            }

            //=============================================================================
            // Description:
            ///              finds the pixel index of neighboring pixels
            ///
            /// @param       inPixel
            /// @param       outNeighbors
            ///
            /// @return
            ///              vector of neighboring pixel indices
            ///
            void findNeighborXcds(const Pixel<dtype>& inPixel, std::vector<uint32>& outNeighbors)
            {
                std::set<Pixel<dtype> > neighbors;
                findNeighbors(inPixel, neighbors);
                std::vector<Pixel<dtype> > neighborXcds;

                // check if the neighboring pixels are exceedances and insert into the xcd vector
                for (auto& pixel : neighbors)
                {
                    if (xcds_->operator()(pixel.row, pixel.col))
                    {
                        neighborXcds.push_back(pixel);
                    }
                }

                // loop through the neighbors and find the cooresponding index into exceedances_
                for (auto& pixel : neighborXcds)
                {
                    auto theExceedanceIter = std::find(xcdsVec_.begin(), xcdsVec_.end(), pixel);
                    outNeighbors.push_back(static_cast<uint32>(theExceedanceIter - xcdsVec_.begin()));
                }
            }

            //=============================================================================
            // Description:
            ///              workhorse method that performs the clustering algorithm
            ///
            void runClusterMaker()
            {
                uint32 clusterId = 0;

                for (auto& currentPixel : xcdsVec_)
                {
                    // not already visited
                    if (currentPixel.clusterId == -1)
                    {
                        Cluster<dtype> newCluster(clusterId);    // a new cluster
                        currentPixel.clusterId = clusterId;
                        newCluster.addPixel(currentPixel);  // assign pixel to cluster

                        // get the neighbors
                        std::vector<uint32> neighborIds;
                        findNeighborXcds(currentPixel, neighborIds);
                        if (neighborIds.empty())
                        {
                            clusters_.push_back(newCluster);
                            ++clusterId;
                            continue;
                        }

                        // loop through the neighbors
                        for (uint32 neighborsIdx = 0; neighborsIdx < neighborIds.size(); ++neighborsIdx)
                        {
                            Pixel<dtype>& currentNeighborPixel = xcdsVec_[neighborIds[neighborsIdx]];

                            // go to neighbors
                            std::vector<uint32> newNeighborIds;
                            findNeighborXcds(currentNeighborPixel, newNeighborIds);

                            // loop through the new neighbors and add them to neighbors
                            for (auto newNeighborId : newNeighborIds)
                            {
                                // not already in neighbors
                                if (std::find(neighborIds.begin(), neighborIds.end(), newNeighborId) == neighborIds.end())
                                {
                                    neighborIds.push_back(newNeighborId);
                                }
                            }

                            // not already assigned to a cluster
                            if (currentNeighborPixel.clusterId == -1)
                            {
                                currentNeighborPixel.clusterId = clusterId;
                                newCluster.addPixel(currentNeighborPixel);
                            }
                        }

                        clusters_.push_back(std::move(newCluster));
                        ++clusterId;
                    }
                }
            }

            //=============================================================================
            // Description:
            ///              3x3 dialates the clusters
            ///
            void expandClusters()
            {
                // loop through the clusters
                for (auto& theCluster : clusters_)
                {
                    // loop through the pixels of the cluster
                    for (auto& thePixel : theCluster)
                    {
                        std::vector<Pixel<dtype> > neighborsNotXcds;
                        findNeighborNotXcds(thePixel, neighborsNotXcds);

                        // loop through the neighbors and if they haven't already been added to the cluster, add them
                        for (auto& newPixel : neighborsNotXcds)
                        {
                            if (std::find(theCluster.begin(), theCluster.end(), newPixel) == theCluster.end())
                            {
                                theCluster.addPixel(newPixel);
                            }
                        }
                    }
                }
            }
        };
    }
}
