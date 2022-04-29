#include "Clustering.h"

#include "Core/Compatibility.hpp"
#include "Core/Utils.hpp"

#include <algorithm>
#include <cmath>

//==================================================================================================

namespace ieom
{
    namespace fpga
    {
        namespace clustering
        {
            Cluster::Cluster() BOOST_NOEXCEPT : 
                id(-1),
                row(0.0),
                col(0.0),
                a(0.0),
                b(0.0),
                eccentricity(0.0),
                orientation(0.0),
                intensity(0),
                rowBoundMin(0),
                rowBoundMax(0),
                colBoundMin(0),
                colBoundMax(0),
                numPixels(0),
                numSaturatedPixels(0),
                minPixel(0),
                maxPixel(0)
            {}

            //===========================================================================================

            Clusterer::Clusterer() BOOST_NOEXCEPT : 
                minClusterSize_(0)
            {}

            //===========================================================================================

            bool Cluster::operator==(const Cluster& rhs) const BOOST_NOEXCEPT
            {
                return id == rhs.id &&
                    utils::essentiallyEqual(row, rhs.row) &&
                    utils::essentiallyEqual(col, rhs.col) &&
                    utils::essentiallyEqual(a, rhs.a) &&
                    utils::essentiallyEqual(b, rhs.b) &&
                    utils::essentiallyEqual(eccentricity, rhs.eccentricity) &&
                    utils::essentiallyEqual(orientation, rhs.orientation) &&
                    intensity == rhs.intensity &&
                    rowBoundMin == rhs.rowBoundMin &&
                    rowBoundMax == rhs.rowBoundMax &&
                    colBoundMin == rhs.colBoundMin &&
                    colBoundMax == rhs.colBoundMax &&
                    numPixels == rhs.numPixels &&
                    numSaturatedPixels == rhs.numSaturatedPixels &&
                    minPixel == rhs.minPixel &&
                    maxPixel == rhs.maxPixel;
            }

            //===========================================================================================

            bool Cluster::operator<(const Cluster& rhs) const BOOST_NOEXCEPT
            {
                return numPixels < rhs.numPixels;
            }

#ifdef DEBUG
            //===========================================================================================

            std::ostream& operator<<(std::ostream& stream, const Cluster& cluster)
            {
                stream << "Cluster:\n";
                stream << "\tid                 = " << cluster.id << '\n';
                stream << "\trow                = " << cluster.row << '\n';
                stream << "\tcol                = " << cluster.col << '\n';
                stream << "\ta                  = " << cluster.a << '\n';
                stream << "\tb                  = " << cluster.b << '\n';
                stream << "\teccentricity       = " << cluster.eccentricity << '\n';
                stream << "\torientation        = " << utils::rad2deg(cluster.orientation) << " degrees\n";
                stream << "\tintensity          = " << cluster.intensity << '\n';
                stream << "\trowBoundMin        = " << cluster.rowBoundMin << '\n';
                stream << "\trowBoundMax        = " << cluster.rowBoundMax << '\n';
                stream << "\tcolBoundMin        = " << cluster.colBoundMin << '\n';
                stream << "\tcolBoundMax        = " << cluster.colBoundMax << '\n';
                stream << "\tnumPixels          = " << cluster.numPixels << '\n';
                stream << "\tnumSaturatedPixels = " << cluster.numSaturatedPixels << '\n';
                stream << "\tminPixel           = " << cluster.minPixel << '\n';
                stream << "\tmaxPixel           = " << cluster.maxPixel << '\n';

                return stream;
            }
#endif
            //===============================================================================================

            class FlagClusterPixelsTooSmall
            {
            public:
                FlagClusterPixelsTooSmall(const thresholding::ExceedanceList& exceedances) BOOST_NOEXCEPT:
                    exceedances_(exceedances) 
                {}

                void operator()(uint32 clusterPixelIdx) BOOST_NOEXCEPT
                {
                    exceedances_[clusterPixelIdx].clusterId = CLUSTER_TOO_SMALL_ID;
                }

            private:
                const thresholding::ExceedanceList& exceedances_;
            };

            //===============================================================================================

            class ExceedanceGetter
            {
            public:
                ExceedanceGetter(const thresholding::ExceedanceList& exceedances) BOOST_NOEXCEPT:
                    exceedances_(exceedances)
                {}

                const thresholding::Exceedance& operator()(uint32 clusterPixelIdx) const BOOST_NOEXCEPT
                {
                    return exceedances_[clusterPixelIdx];
                }

            private:
                const thresholding::ExceedanceList& exceedances_;
            };

            //===============================================================================================

            ClusterList Clusterer::process(const thresholding::ExceedanceList& exceedances) const
            {
                FlagClusterPixelsTooSmall flagClusterPixelsTooSmall(exceedances);
                ExceedanceGetter exceedanceGetter(exceedances);

                ClusterList clusters;
                uint32 clusterId = 0;
                for (uint32 xcdIdx = 0; xcdIdx < exceedances.size(); ++xcdIdx)
                {
                    const thresholding::Exceedance& pixel = exceedances[xcdIdx];
                    if (pixel.clusterId != -1)
                    {
                        // already checked
                        continue;
                    }

                    // start a new cluster with the pixel as the seed
                    Cluster newCluster;
                    newCluster.id = clusterId;

                    pixel.clusterId = clusterId;

                    ClusterPixelIdxs clusterPixelIdxs;
                    clusterPixelIdxs.push_back(xcdIdx);

                    // get the neighbor pixels that are also exceedances
                    getNeighborExceedances(exceedances, pixel, xcdIdx + 1, clusterPixelIdxs);

                    // loop over the neighbors until the cluster is complete
                    for (uint32 nIdx = 1; nIdx < clusterPixelIdxs.size(); ++nIdx)
                    {
                        const uint32 exceedanceIdx = clusterPixelIdxs[nIdx];
                        const thresholding::Exceedance& neighborPixel = exceedances[exceedanceIdx];

                        neighborPixel.clusterId = clusterId;
                        getNeighborExceedances(exceedances, neighborPixel, xcdIdx + 1, clusterPixelIdxs);
                    }

                    // check the cluster size is the right size
                    if (clusterPixelIdxs.size() < minClusterSize_)
                    {
                        // too small, flag the pixels and move on
                        std::for_each(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
                            std::execution::par_unseq,
#endif
                            clusterPixelIdxs.begin(), clusterPixelIdxs.end(), flagClusterPixelsTooSmall);
                        continue;
                    }

                    ClusterPixels clusterPixels(clusterPixelIdxs.size());
                    std::transform(
#ifdef PARALLEL_ALGORITHMS_SUPPORTED
                        std::execution::par_unseq,
#endif
                        clusterPixelIdxs.begin(), clusterPixelIdxs.end(), clusterPixels.begin(), exceedanceGetter);

                    centroidClusters(clusterPixels, newCluster);
                    setEllipseProperties(clusterPixels, newCluster);

                    // add the cluster to the cluster vector and go to the next exceedance pixel
                    if (clusters.size() < clusters.capacity())
                    {
                        clusters.push_back(newCluster);
                        ++clusterId;
                    }
                    else
                    {
                        break;
                    }
                }

                return clusters;
            }

            //===============================================================================================

            void Clusterer::setMinClusterSize(const uint32 minClusterSize) BOOST_NOEXCEPT
            {
                minClusterSize_ = minClusterSize;
            }

            //===============================================================================================

            void Clusterer::centroidClusters(const ClusterPixels& clusterPixels,
                Cluster& outCluster) const BOOST_NOEXCEPT
            {
                float_t weightedRowSum = 0;
                float_t weightedColSum = 0;

                // initialize the cluster fields just in case
                outCluster.rowBoundMin = std::numeric_limits<uint32>::max();
                outCluster.colBoundMin = std::numeric_limits<uint32>::max();
                outCluster.numPixels = static_cast<uint32>(clusterPixels.size());
                outCluster.minPixel = std::numeric_limits<accumulator_t>::max();

                // loop over the cluster pixels and calculate the weighted row/col sums and the 
                // total intensity
                for (ClusterPixels::const_iterator iter = clusterPixels.begin(); iter != clusterPixels.end(); ++iter)
                {
                    const thresholding::Exceedance& pixel = *iter;
                    if (pixel.isSaturated)
                    {
                        ++outCluster.numSaturatedPixels;
                    }

                    outCluster.intensity += static_cast<uint64>(pixel.counts);
                    const float_t pixelCountsFloat = static_cast<float_t>(pixel.counts);

                    weightedRowSum += static_cast<float_t>(pixel.row) * pixelCountsFloat;
                    weightedColSum += static_cast<float_t>(pixel.col) * pixelCountsFloat;

                    outCluster.rowBoundMin = std::min(outCluster.rowBoundMin, pixel.row);
                    outCluster.colBoundMin = std::min(outCluster.colBoundMin, pixel.col);
                    outCluster.rowBoundMax = std::max(outCluster.rowBoundMax, pixel.row);
                    outCluster.colBoundMax = std::max(outCluster.colBoundMax, pixel.col);

                    outCluster.minPixel = std::min(outCluster.minPixel, pixel.counts);
                    outCluster.maxPixel = std::max(outCluster.maxPixel, pixel.counts);
                }

                const float_t clusterIntensityFloat = static_cast<float_t>(outCluster.intensity);
                outCluster.row = weightedRowSum / clusterIntensityFloat;
                outCluster.col = weightedColSum / clusterIntensityFloat;
            }

            //===============================================================================================

            void Clusterer::setEllipseProperties(const ClusterPixels& clusterPixels,
                Cluster& outCluster) const BOOST_NOEXCEPT
            {
                BOOST_CONSTEXPR_OR_CONST float_t two = static_cast<float_t>(2.0);

                float_t m20 = static_cast<float_t>(0.0);
                float_t m02 = static_cast<float_t>(0.0);
                float_t m11 = static_cast<float_t>(0.0);

                for (ClusterPixels::const_iterator iter = clusterPixels.begin(); iter != clusterPixels.end(); ++iter)
                {
                    const thresholding::Exceedance& pixel = *iter;
                    const float_t deltaX = pixel.col - outCluster.col;
                    const float_t deltaY = pixel.row - outCluster.row;

                    m11 += deltaX * deltaY;
                    m20 += utils::sqr(deltaX);
                    m02 += utils::sqr(deltaY);
                }

                const float_t numPixels = static_cast<float_t>(clusterPixels.size());
                m11 /= numPixels;
                m20 /= numPixels;
                m02 /= numPixels;

                float_t piece1 = m20 + m02;
                piece1 /= two;

                float_t piece2 = std::sqrt(static_cast<float_t>(4.0) * utils::sqr(m11) + utils::sqr(m20 - m02));
                piece2 /= two;

                const float_t lambda1 = piece1 - piece2;
                const float_t lambda2 = piece1 + piece2;

                outCluster.eccentricity = std::sqrt(static_cast<float_t>(1.0) - lambda1 / lambda2);
                outCluster.orientation = static_cast<float_t>(-0.5) * std::atan2(two * m11, m20 - m02);
                outCluster.a = two * std::sqrt(lambda2);
                outCluster.b = two * std::sqrt(lambda1);
            }

            //===============================================================================================

            void Clusterer::getNeighborExceedances(const thresholding::ExceedanceList& xcds,
                const thresholding::Exceedance& centerPixel,
                const uint32 xcdStartIdx,
                ClusterPixelIdxs& clusterPixelIdxs) const
            {
                for (uint32 i = xcdStartIdx; i < xcds.size(); ++i)
                {
                    const thresholding::Exceedance& xcdPixel = xcds[i];

                    if (xcdPixel.clusterId != -1)
                    {
                        // already checked
                        continue;
                    }

                    // DP NOTE: this check isn't necessary due to the row major readout order of 
                    // the xcds
                    //if (static_cast<int32>(xcdPixel.row) < static_cast<int32>(centerPixel.row) - 1)
                    //{
                    //    // not adjacent
                    //    continue;
                    //}

                    if (xcdPixel.row > centerPixel.row + 1)
                    {
                        // due to row major readout order of the xcds we no longer
                        // need to keep checking
                        break;
                    }

                    if (static_cast<int32>(xcdPixel.col) < static_cast<int32>(centerPixel.col) - 1)
                    {
                        // not adjacent
                        continue;
                    }

                    if (xcdPixel.col > centerPixel.col + 1)
                    {
                        // not adjacent
                        continue;
                    }

                    // if we've gotten this far than it must be a neighbor.
                    clusterPixelIdxs.push_back(i);
                    xcdPixel.clusterId = CLUSTER_IN_PROGRESS;
                }
            }
        }
    }
}
