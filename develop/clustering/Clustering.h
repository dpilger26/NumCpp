#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "Thresholding.h"

#include "Core/Constants.hpp"
#include "Core/Types.hpp"

#include "boost/config.hpp"
#include "etl/vector.h"

#ifdef DEBUG
#include <iostream>
#endif
#include <limits>
#include <vector>

//==================================================================================================

namespace ieom
{
    namespace fpga
    {
        namespace clustering
        {
            /// Special ID to represent a cluster that has been formed but is too small
            BOOST_CONSTEXPR_OR_CONST int32 CLUSTER_TOO_SMALL_ID = -2;
            /// Special ID for a pixel that has started a cluster
            BOOST_CONSTEXPR_OR_CONST int32 CLUSTER_IN_PROGRESS = -3;

            //======================================================================================
            /// Holds all of the information for a clustered set of Exceedances
            struct Cluster
            {
                /// The cluster id.
                int32           id;
                /// The centroid row of the cluster.
                float_t         row;
                /// The centroid column of the cluster.
                float_t         col;
                /// The ellipse semi-major axis a
                float_t         a;
                /// The ellipse semi-minor axis b
                float_t         b;
                /// The centriod eccentricity
                float_t         eccentricity;
                /// The centriod ellipse orientation in radians.  Measured counter-clockwise from +x axis
                float_t         orientation;
                /// The total summed intensity of the pixels in the cluster.
                uint64          intensity;
                /// The bounding box minimum row of the cluster.
                uint32          rowBoundMin;
                /// The bounding box maximum row of the cluster.
                uint32          rowBoundMax;
                /// The bounding box minimum column of the cluster.
                uint32          colBoundMin;
                /// The bounding box maximum column of the cluster.
                uint32          colBoundMax;
                /// The number of pixels that made up the cluster.
                uint32          numPixels;
                /// The number of saturated pixels that made of the cluster
                uint32          numSaturatedPixels;
                /// The minimum pixel count value of the cluster 
                accumulator_t   minPixel;
                /// The maximum pixel count value of the cluster 
                accumulator_t   maxPixel;

                //============================================================================
                // Method Description:
                /// Default Constructor
                ///
                Cluster() BOOST_NOEXCEPT;

                //===========================================================================================
                /// Equality operator, Apparently needed by the Arm Compiler for some reason...?
                ///
                /// @param rhs: the rhs cluster to compare
                /// @return bool
                bool operator==(const Cluster& rhs) const BOOST_NOEXCEPT;

                //===========================================================================================
                /// Less than operator, needed for sorting
                ///
                /// @param rhs: the rhs cluster to compare
                /// @return bool
                bool operator<(const Cluster& rhs) const BOOST_NOEXCEPT;

#ifdef DEBUG
                //===========================================================================================
                /// Stream output operator
                ///
                /// @param stream
                /// @param cluster
                /// @return ostream
                ///
                friend std::ostream& operator<<(std::ostream& stream, const Cluster& cluster);
#endif
            };

            //======================================================================================
            /// A list of Clusters
            typedef etl::vector<Cluster, constants::MAX_CLUSTERS_FROM_FPGA> ClusterList;

            //======================================================================================
            /// Performs clustering on an input set of Exceedances.
            class Clusterer
            {
            public:
                //===========================================================================================
                /// Default Constructor
                Clusterer() BOOST_NOEXCEPT;

                //===========================================================================================
                /// Clusters the exceedances output from a thresholding algorithm
                ///
                /// @param exceedances: The list of exceedances to cluster
                ///
                /// @return a list of clusters
                ///
                ClusterList process(const thresholding::ExceedanceList& exceedances) const;

                //===========================================================================================
                /// Sets the minimum number of pixels a cluster must contain to be considered valid
                ///
                /// @param minClusterSize: The mimimum cluster size
                ///
                void setMinClusterSize(const uint32 minClusterSize) BOOST_NOEXCEPT;

            private:
                /// The minimum cluster size
                uint32 minClusterSize_;

                // type aliasing
                typedef std::vector<uint32>             ClusterPixelIdxs;
                typedef thresholding::ExceedanceList    ClusterPixels;

                //==================================================================================
                /// Centroids the clustered pixels using center of mass (COM)
                ///
                /// @param clusterPixels: list of pixels to centroid
                /// @param outCluster: the cluster to add the centroid information to
                ///
                void centroidClusters(const ClusterPixels& clusterPixels,
                    Cluster& outCluster) const BOOST_NOEXCEPT;

                //==================================================================================
                /// Calculates the cluster eccentricity and orientation angle
                ///
                /// @param clusterPixels: list of pixels to centroid
                /// @param outCluster: the cluster to add the centroid information to
                ///
                void setEllipseProperties(const ClusterPixels& clusterPixels,
                    Cluster& outCluster) const BOOST_NOEXCEPT;

                //===========================================================================================
                /// Gets the neighbor exceedances from a center pixel
                ///
                /// @param xcds: vector of exceedance pixels. Assumes the pixels are in 
                ///              row major readout order
                /// @param centerPixel: the center pixel to get neighbors around
                /// @param xcdStartIdx: the index into xcds to start searching at
                /// @param clusterPixelIdxs: the indexes into xcds that are valid neighbors
                ///
                void getNeighborExceedances(const thresholding::ExceedanceList& xcds,
                    const thresholding::Exceedance& centerPixel,
                    const uint32 xcdStartIdx,
                    ClusterPixelIdxs& clusterPixelIdxs) const;
            };
        }
    }
}

#endif
