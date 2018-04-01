// Copyright 2018 David Pilger
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files(the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, 
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
// permit persons to whom the Software is furnished to do so, subject to the following 
// conditions :
//
// The above copyright notice and this permission notice shall be included in all copies 
// or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
// DEALINGS IN THE SOFTWARE.

#pragma once
#include<NdArray.hpp>
#include<Types.hpp>

#include<vector>

namespace NumC
{
    //================================ImageProcessing Namespace=============================
    namespace ImageProcessing
    {
        namespace Filters
        {
            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional complemenatry median filter.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            // Outputs:
            //				NdArray
            //
            template<typename dtype, typename dtypeOut>
            inline NdArray<dtypeOut> complementaryMedianFilter(const NdArray<dtype>& inImageArray, uint32 inSize)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional complemenatry median filter.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				axis, default row
            // Outputs:
            //				NdArray
            //
            template<typename dtype, typename dtypeOut>
            inline NdArray<dtypeOut> complementaryMedianFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, Axis::Type inAxis = Axis::ROW)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional kernel convolution.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				NdArray, weights
            // Outputs:
            //				NdArray
            //
            template<typename dtype, typename dtypeOut>
            inline NdArray<dtypeOut> convolve(const NdArray<dtype>& inImageArray, uint32 inSize, const NdArray<dtype>& inWeights)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a one-dimensional kernel convolution along the given axis.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				NdArray, weights
            //				axis, default row
            // Outputs:
            //				NdArray
            //
            template<typename dtype, typename dtypeOut>
            inline NdArray<dtypeOut> convolve1d(const NdArray<dtype>& inImageArray, uint32 inSize, const NdArray<dtype>& inWeights, Axis::Type inAxis = Axis::ROW)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional gaussian filter.
            //		
            // Inputs:
            //				NdArray
            //				double, Standard deviation for Gaussian kernel
            // Outputs:
            //				NdArray
            //
            template<typename dtype, typename dtypeOut>
            inline NdArray<dtypeOut> gaussianFilter(const NdArray<dtype>& inImageArray, double inSigma)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional gaussian filter.
            //		
            // Inputs:
            //				NdArray
            //				double, Standard deviation for Gaussian kernel
            //				axis, default row
            // Outputs:
            //				NdArray
            //
            template<typename dtype, typename dtypeOut>
            inline NdArray<dtypeOut> gaussianFilter1d(const NdArray<dtype>& inImageArray, double inSigma, Axis::Type inAxis = Axis::ROW)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional linear filter.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				NdArray, weights
            // Outputs:
            //				NdArray
            //
            template<typename dtype, typename dtypeOut>
            inline NdArray<dtypeOut> linearFilter(const NdArray<dtype>& inImageArray, uint32 inSize, const NdArray<dtype>& inWeights)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a one-dimensional linear filter along the given axis.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				NdArray, weights
            //				axis, default row
            // Outputs:
            //				NdArray
            //
            template<typename dtype, typename dtypeOut>
            inline NdArray<dtypeOut> linearFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, const NdArray<dtype>& inWeights, Axis::Type inAxis = Axis::ROW)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional maximum filter.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            // Outputs:
            //				NdArray
            //
            template<typename dtype>
            inline NdArray<dtype> maximumFilter(const NdArray<dtype>& inImageArray, uint32 inSize)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a one-dimensional maximum filter along the given axis.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				axis, default row
            // Outputs:
            //				NdArray
            //
            template<typename dtype>
            inline NdArray<dtype> maximumFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, Axis::Type inAxis = Axis::ROW)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional median filter.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            // Outputs:
            //				NdArray
            //
            template<typename dtype>
            inline NdArray<dtype> medianFilter(const NdArray<dtype>& inImageArray, uint32 inSize)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a one-dimensional median filter along the given axis.
            //		
            // Inputs:
            //				NdArray
            //				linear size of the kernel to apply
            //				axis, default row
            // Outputs:
            //				NdArray
            //
            template<typename dtype>
            inline NdArray<dtype> medianFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, Axis::Type inAxis = Axis::ROW)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional minumum filter.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            // Outputs:
            //				NdArray
            //
            template<typename dtype>
            inline NdArray<dtype> minumumFilter(const NdArray<dtype>& inImageArray, uint32 inSize)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a one-dimensional minumum filter along the given axis.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				axis, default row
            // Outputs:
            //				NdArray
            //
            template<typename dtype>
            inline NdArray<dtype> minumumFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, Axis::Type inAxis = Axis::ROW)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional percentile filter.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				percentile [0, 100]
            // Outputs:
            //				NdArray
            //
            template<typename dtype>
            inline NdArray<dtype> percentileFilter(const NdArray<dtype>& inImageArray, uint32 inSize, uint8 inPercentile)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a one-dimensional percentile filter along the given axis.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				percentile [0, 100]
            //				axis, default row
            // Outputs:
            //				NdArray
            //
            template<typename dtype>
            inline NdArray<dtype> percentile1d(const NdArray<dtype>& inImageArray, uint32 inSize, uint8 inPercentile, Axis::Type inAxis = Axis::ROW)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional rank filter.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				rank [0, 100]
            // Outputs:
            //				NdArray
            //
            template<typename dtype>
            inline NdArray<dtype> rankFilter(const NdArray<dtype>& inImageArray, uint32 inSize, uint8 inRank)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a one-dimensional rank filter along the given axis.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				rank [0, 100]
            //				axis, default row
            // Outputs:
            //				NdArray
            //
            template<typename dtype>
            inline NdArray<dtype> rank1d(const NdArray<dtype>& inImageArray, uint32 inSize, uint8 inRank, Axis::Type inAxis = Axis::ROW)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional uniform filter.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            // Outputs:
            //				NdArray
            //
            template<typename dtype, typename dtypeOut>
            inline NdArray<dtypeOut> uniformFilter(const NdArray<dtype>& inImageArray, uint32 inSize)
            {

            }

            //============================================================================
            // Method Description: 
            //						Calculates a one-dimensional uniform filter along the given axis.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            //				axis, default row
            // Outputs:
            //				NdArray
            //
            template<typename dtype, typename dtypeOut>
            inline NdArray<dtypeOut> uniformFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, Axis::Type inAxis = Axis::ROW)
            {

            }
        }
    }

    //================================================================================
    // Class Description:
    //						holds the information for a single pixel
    //
    template<typename dtype>
    class Pixel
    {
    private:
        //==================================Attributes================================//
        int32	clusterId_;
        uint32	row_;
        uint32	col_;
        dtype	intensity_;

    public:
        // =============================================================================
        // Description:
        //              defualt constructor needed by containers
        // 
        // Inputs:
        //              None
        // 
        // Outputs:
        //              None
        //
        Pixel() :
            clusterId_(-1),
            row_(0),
            col_(0),
            intensity_(0)
        {};

        // =============================================================================
        // Description:
        //              constructor
        // 
        // Inputs:
        //              pixel row,
        //              pixel column,
        //              pixel intensity
        // 
        // Outputs:
        //              None
        //
        Pixel(uint32 inRow, uint32 inCol, type inIntensity) :
            clusterId_(-1),
            row_(inRow),
            col_(inCol),
            intensity_(inIntensity)
        {};

        // =============================================================================
        // Description:
        //              returns the cluster id that this pixel belongs to
        // 
        // Inputs:
        //              None
        // 
        // Outputs:
        //              cluster id
        //
        int32 clusterId() const
        {
            return clusterId_;
        }

        // =============================================================================
        // Description:
        //              sets the cluster id that this pixel belongs to
        // 
        // Inputs:
        //              cluster id
        // 
        // Outputs:
        //              None
        //
        void setClusterId(int32 inClusterId)
        {
            if (inClusterId < 0)
            {
                throw std::invalid_argument("ERROR: ImageProcessing::Pixel::setClusterId: input cluster id must be greater than or equal to 0.");
            }

            clusterId_ = inClusterId;
        }

        // =============================================================================
        // Description:
        //              returns the pixel row
        // 
        // Inputs:
        //              None
        // 
        // Outputs:
        //              row
        //
        uint32 row() const
        {
            return row_;
        }

        // =============================================================================
        // Description:
        //              returns the pixel column
        // 
        // Inputs:
        //              None
        // 
        // Outputs:
        //              column
        //
        uint32 col() const
        {
            return col_;
        }

        // =============================================================================
        // Description:
        //              returns the pixel intensity
        // 
        // Inputs:
        //              None
        // 
        // Outputs:
        //              intensity
        //
        dtype intensity() const
        {
            return intensity_;
        }
    };

    //================================================================================
    // Class Description:
    //						holds the information for a cluster of pixels
    //
    template<typename dtype>
    class Cluster
    {
    private:
        // ================================Attributes===============================
        uint16				id_;
        std::vector<Pixel>  pixels_;

        uint16				rowMin_;
        uint16				rowMax_;
        uint16				colMin_;
        uint16				colMax_;

        uint16              peakPixelIntensity_;
        uint32				intensity_;

        double              eod_;

        uint16              streakScalar_;
        double              streakRatio_;

    public:
        // =============================================================================
        // Description:
        //              default constructor needed by containers
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              None
        //
        Cluster() {};

        // =============================================================================
        // Description:
        //              constructor
        // 
        // Parameter(s): 
        //              cluster id
        // 
        // Return: 
        //              None
        //
        Cluster(uint16 inClusterId);
        {

        }

        // =============================================================================
        // Description:
        //              equality operator
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              None
        //
        bool operator==(const Cluster& rhs) const
        {

        }

        // =============================================================================
        // Description:
        //              sets the streak scalar value
        // 
        // Parameter(s): 
        //              scalar value
        // 
        // Return: 
        //              None
        //
        void setStreakScalar(uint16 inScalar);
        {

        }

        // =============================================================================
        // Description:
        //              sets the streak ratio value
        // 
        // Parameter(s): 
        //              ratio value
        // 
        // Return: 
        //              None
        //
        void setStreakRatio(double inRatio);
        {

        }

        // =============================================================================
        // Description:
        //              return the cluster id
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              cluster id
        //
        uint16 id() const;
        {

        }

        // =============================================================================
        // Description:
        //              returns the number of pixels in the cluster
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              number of pixels in the cluster
        //
        uint16 numPixels() const;
        {

        }

        // =============================================================================
        // Description:
        //              returns the number of columns the cluster spans
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              number of columns
        //
        uint16 width() const;
        {

        }

        // =============================================================================
        // Description:
        //              returns the number of rows the cluster spans
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              number of rows
        //
        uint16 height() const;
        {

        }

        // =============================================================================
        // Description:
        //              returns the intensity of the peak pixel in the cluster
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              peak pixel intensity
        //
        uint16 peakPixelIntensity() const;
        {

        }

        // =============================================================================
        // Description:
        //              returns the summed intensity of the cluster
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              summed cluster intensity
        //
        uint32 intensity() const;
        {

        }

        // =============================================================================
        // Description:
        //              returns the cluster estimated energy on detector (EOD)
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              eod
        //
        double eod() const;
        {

        }

        // =============================================================================
        // Description:
        //              returns the minimum row number of the cluster
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              minimum row number of the cluster
        //
        uint16 rowMin();
        {

        }

        // =============================================================================
        // Description:
        //              returns the maximum row number of the cluster
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              maximum row number of the cluster
        //
        uint16 rowMax();
        {

        }

        // =============================================================================
        // Description:
        //              returns the minimum column number of the cluster
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              minimum column number of the cluster
        //
        uint16 colMin();
        {

        }

        // =============================================================================
        // Description:
        //              returns the maximum column number of the cluster
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              maximum column number of the cluster
        //
        uint16 colMax();
        {

        }

        // =============================================================================
        // Description:
        //              returns whether or not the cluster is a streak
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              bool
        //
        bool isStreak();
        {

        }

        // =============================================================================
        // Description:
        //              returns a vector of the pixels in the cluster
        // 
        // Parameter(s): 
        //              None
        // 
        // Return: 
        //              vector of the pixels in the cluster
        //
        std::vector<Pixel>& pixels();
        {

        }

        // =============================================================================
        // Description:
        //              adds a pixel to the cluster
        // 
        // Parameter(s): 
        //              pixel
        // 
        // Return: 
        //              None
        //
        void addPixel(Pixel& inPixel)
        {

        }
    };

    //================================================================================
    // Class Description:
    //						holds the information for a centroid
    //
    template<typename dtype>
    class Centroid
    {

    };

    //============================================================================
    // Method Description: 
    //						Applies a threshold to an image
    //		
    // Inputs:
    //				NdArray
    //				threshold value
    // Outputs:
    //				NdArray of booleans of pixels that exceeded the threshold
    //
    template<typename dtype>
    inline NdArray<bool> applyThreshold(const NdArray<dtype>& inImageArray, dtype inThreshold)
    {

    }

    //============================================================================
    // Method Description: 
    //						Center of Mass centroids clusters
    //		
    // Inputs:
    //				NdArray
    //				threshold value
    // Outputs:
    //				std::vector<Centroid>
    //
    inline std::vector<Centroid> centroidClusters(const std::vector<Cluster>& inClusters)
    {

    }

    //============================================================================
    // Method Description: 
    //						Clusters exceedance pixels from an image
    //		
    // Inputs:
    //				NdArray
    //				NdArray of exceedances
    //				border to apply around exceedance pixels prior to clustering, default 0
    // Outputs:
    //				NdArray of booleans of pixels that exceeded the threshold
    //
    template<typename dtype>
    inline std::vector<Cluster> clusterPixels(const NdArray<dtype>& inImageArray, const NdArray<bool>& inExceedances, uint8 inBorderWidth = 0)
    {

    }

    //============================================================================
    // Method Description: 
    //						Calculates a threshold such that the input rate of pixels
    //						exceeds the threshold
    //		
    // Inputs:
    //				NdArray
    //				exceedance rate
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline dtype generateThreshold(const NdArray<dtype>& inImageArray, double inRate)
    {

    }

    //============================================================================
    // Method Description: 
    //						Generates a list of centroids givin an input exceedance
    //						rate
    //		
    // Inputs:
    //				NdArray
    //				exceedance rate
    // Outputs:
    //				std::vector<Centroid>
    //
    template<typename dtype>
    inline std::vector<Centroid> generateCentroids(const NdArray<dtype>& inImageArray, double inRate)
    {

    }

    //============================================================================
    // Method Description: 
    //						Window expand around clusters
    //		
    // Inputs:
    //				NdArray
    //				std::vector<Centroid>
    //				border width
    // Outputs:
    //				std::vector<Centroid>
    //
    template<typename dtype>
    inline std::vector<Cluster> windowClusters(const NdArray<dtype>& inImageArray, const std::vector<Cluster>& inClusters, uint8 inBorderWidth)
    {

    }

    //============================================================================
    // Method Description: 
    //						Window expand around exceedance pixels
    //		
    // Inputs:
    //				NdArray
    //				border width
    // Outputs:
    //				std::vector<Centroid>
    //
    template<typename dtype>
    inline NdArray<bool> windowExceedances(const NdArray<bool>& inExceedances, uint8 inBorderWidth)
    {

    }
}
