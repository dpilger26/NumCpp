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
#include<Methods.hpp>
#include<Types.hpp>
#include<Utils.hpp>

#include<cmath>
#include<iostream>
#include<limits>
#include<set>
#include<string>
#include<utility>
#include<vector>

namespace NumC
{
    //================================ImageProcessing Namespace=============================
    namespace ImageProcessing
    {
        //================================================================================
        // Class Description:
        //						class for performing many types of image filtering
        //
        template<typename dtype>
        class Filter
        {
        private:

        public:
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
            template<typename dtypeOut>
            static NdArray<dtypeOut> complementaryMedianFilter(const NdArray<dtype>& inImageArray, uint32 inSize)
            {


                return std::move(NdArray<dtypeOut>(0));
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
            template<typename dtypeOut>
            static NdArray<dtypeOut> complementaryMedianFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, Axis::Type inAxis = Axis::ROW)
            {


                return std::move(NdArray<dtypeOut>(0));
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
            template<typename dtypeOut>
            static NdArray<dtypeOut> convolve(const NdArray<dtype>& inImageArray, uint32 inSize, const NdArray<dtype>& inWeights)
            {


                return std::move(NdArray<dtypeOut>(0));
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
            template<typename dtypeOut>
            static NdArray<dtypeOut> convolve1d(const NdArray<dtype>& inImageArray, uint32 inSize, const NdArray<dtype>& inWeights, Axis::Type inAxis = Axis::ROW)
            {


                return std::move(NdArray<dtypeOut>(0));
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
            template<typename dtypeOut>
            static NdArray<dtypeOut> gaussianFilter(const NdArray<dtype>& inImageArray, double inSigma)
            {


                return std::move(NdArray<dtypeOut>(0));
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
            template<typename dtypeOut>
            static NdArray<dtypeOut> gaussianFilter1d(const NdArray<dtype>& inImageArray, double inSigma, Axis::Type inAxis = Axis::ROW)
            {


                return std::move(NdArray<dtypeOut>(0));
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
            template<typename dtypeOut>
            static NdArray<dtypeOut> linearFilter(const NdArray<dtype>& inImageArray, uint32 inSize, const NdArray<dtype>& inWeights)
            {


                return std::move(NdArray<dtypeOut>(0));
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
            template<typename dtypeOut>
            static NdArray<dtypeOut> linearFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, const NdArray<dtype>& inWeights, Axis::Type inAxis = Axis::ROW)
            {


                return std::move(NdArray<dtypeOut>(0));
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
            static NdArray<dtype> maximumFilter(const NdArray<dtype>& inImageArray, uint32 inSize)
            {


                return std::move(NdArray<dtype>(0));
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
            static NdArray<dtype> maximumFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, Axis::Type inAxis = Axis::ROW)
            {


                return std::move(NdArray<dtype>(0));
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
            static NdArray<dtype> medianFilter(const NdArray<dtype>& inImageArray, uint32 inSize)
            {


                return std::move(NdArray<dtype>(0));
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
            static NdArray<dtype> medianFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, Axis::Type inAxis = Axis::ROW)
            {


                return std::move(NdArray<dtype>(0));
            }

            //============================================================================
            // Method Description: 
            //						Calculates a multidimensional minimum filter.
            //		
            // Inputs:
            //				NdArray
            //				square size of the kernel to apply
            // Outputs:
            //				NdArray
            //
            static NdArray<dtype> minimumFilter(const NdArray<dtype>& inImageArray, uint32 inSize)
            {


                return std::move(NdArray<dtype>(0));
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
            static NdArray<dtype> minumumFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, Axis::Type inAxis = Axis::ROW)
            {


                return std::move(NdArray<dtype>(0));
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
            static NdArray<dtype> percentileFilter(const NdArray<dtype>& inImageArray, uint32 inSize, uint8 inPercentile)
            {


                return std::move(NdArray<dtype>(0));
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
            static NdArray<dtype> percentile1d(const NdArray<dtype>& inImageArray, uint32 inSize, uint8 inPercentile, Axis::Type inAxis = Axis::ROW)
            {


                return std::move(NdArray<dtype>(0));
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
            static NdArray<dtype> rankFilter(const NdArray<dtype>& inImageArray, uint32 inSize, uint8 inRank)
            {


                return std::move(NdArray<dtype>(0));
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
            static NdArray<dtype> rankFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, uint8 inRank, Axis::Type inAxis = Axis::ROW)
            {


                return std::move(NdArray<dtype>(0));
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
            template<typename dtypeOut>
            static NdArray<dtypeOut> uniformFilter(const NdArray<dtype>& inImageArray, uint32 inSize)
            {


                return std::move(NdArray<dtypeOut>(0));
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
            template<typename dtypeOut>
            static NdArray<dtypeOut> uniformFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, Axis::Type inAxis = Axis::ROW)
            {


                return std::move(NdArray<dtypeOut>(0));
            }
        };

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
            Pixel(uint32 inRow, uint32 inCol, dtype inIntensity) :
                clusterId_(-1),
                row_(inRow),
                col_(inCol),
                intensity_(inIntensity)
            {};

            // =============================================================================
            // Description:
            //              equality operator
            // 
            // Inputs:
            //              None
            // 
            // Outputs:
            //              bool
            //
            bool operator==(const Pixel<dtype>& rhs) const
            {
                return row_ == rhs.row_ && col_ == rhs.col_ && intensity_ == rhs.intensity_;
            }

            // =============================================================================
            // Description:
            //              not equality operator
            // 
            // Inputs:
            //              None
            // 
            // Outputs:
            //              bool
            //
            bool operator!=(const Pixel<dtype>& rhs) const
            {
                return !(*this == rhs);
            }

            // =============================================================================
            // Description:
            //              less than operator for std::sort algorithm and std::set<>;
            //              NOTE: std::sort sorts in ascending order. Since I want to sort 
            //              the centroids in descensing order, I am purposefully defining
            //              this operator backwards!
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              None
            //
            bool operator<(const Pixel<dtype>& rhs) const
            {
                if (row_ < rhs.row_)
                {
                    return true;
                }
                else if (row_ == rhs.row_)
                {
                    if (col_ < rhs.col_)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }

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
            std::string str() const
            {
                std::string out = "row = " + Utils::num2str(row_) + " col = " + Utils::num2str(col_);
                out += " intensity = " + Utils::num2str(intensity_) + "\n";
                return out;
            }

            //============================================================================
            // Method Description: 
            //						prints the Pixel object to the console
            //		
            // Inputs:
            //				None
            // Outputs:
            //				None
            //
            void print() const
            {
                std::cout << *this;
            }

            // =============================================================================
            // Description:
            //              osstream operator
            // 
            // Inputs:
            //              None
            // 
            // Outputs:
            //              None
            //
            friend std::ostream& operator<<(std::ostream& inStream, const Pixel<dtype>& inPixel)
            {
                inStream << inPixel.str();
                return inStream;
            }
        };

        //================================================================================
        // Class Description:
        //						holds the information for a cluster of pixels
        //
        template<typename dtype>
        class Cluster
        {
        public:
            // ================================Typedefs===============================
            typedef typename std::vector<Pixel<dtype> >::const_iterator    const_iterator;

        private:
            // ================================Attributes===============================
            uint32                      clusterId_;
            std::vector<Pixel<dtype> >  pixels_;

            uint32				        rowMin_;
            uint32				        rowMax_;
            uint32				        colMin_;
            uint32				        colMax_;

            dtype				        intensity_;
            dtype                       peakPixelIntensity_;

            double                      eod_;

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
            Cluster(uint32 inClusterId) :
                clusterId_(inClusterId),
                rowMin_(std::numeric_limits<uint32>::max()), // largest possible number
                rowMax_(0),
                colMin_(std::numeric_limits<uint32>::max()), // largest possible number
                colMax_(0),
                intensity_(0),
                peakPixelIntensity_(0),
                eod_(1.0)
            {};

            // =============================================================================
            // Description:
            //              equality operator
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              bool
            //
            bool operator==(const Cluster<dtype>& rhs) const
            {
                if (pixels_.size() != rhs.pixels_.size())
                {
                    return false;
                }

                for (uint32 i = 0; i < pixels_.size(); ++i)
                {
                    if (pixels_[i] != rhs.pixels_[i])
                    {
                        return false;
                    }
                }

                return true;
            }

            // =============================================================================
            // Description:
            //              not equality operator
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              bool
            //
            bool operator!=(const Cluster<dtype>& rhs) const
            {
                return !(*this == rhs);
            }

            // =============================================================================
            // Description:
            //              access operator, no bounds checking
            // 
            // Parameter(s): 
            //              index
            // 
            // Return: 
            //              Pixel
            //
            const Pixel<dtype>& operator[](uint32 inIndex) const
            {
                return pixels_[inIndex];
            }

            // =============================================================================
            // Description:
            //              access method with bounds checking
            // 
            // Parameter(s): 
            //              index
            // 
            // Return: 
            //              Pixel
            //
            const Pixel<dtype>& at(uint32 inIndex) const
            {
                if (inIndex >= pixels_.size())
                {
                    throw std::invalid_argument("ERROR: ImageProcessing::Cluster::at: index exceeds cluster size.");
                }
                return pixels_[inIndex];
            }

            // =============================================================================
            // Description:
            //              returns in iterator to the beginning pixel of the cluster
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              const_iterator
            //
            const_iterator begin() const
            {
                return pixels_.cbegin();
            }

            // =============================================================================
            // Description:
            //              returns in iterator to the 1 past the end pixel of the cluster
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              const_iterator
            //
            const_iterator end() const
            {
                return pixels_.cend();
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
            uint32 size() const
            {
                return static_cast<uint32>(pixels_.size());
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
            uint32 clusterId() const
            {
                return clusterId_;
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
            uint32 rowMin() const
            {
                return rowMin_;
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
            uint32 rowMax() const
            {
                return rowMax_;
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
            uint32 colMin() const
            {
                return colMin_;
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
            uint32 colMax() const
            {
                return colMax_;
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
            uint32 height() const
            {
                return rowMax_ - rowMin_ + 1;
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
            uint32 width() const
            {
                return colMax_ - colMin_ + 1;
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
            dtype intensity() const
            {
                return intensity_;
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
            dtype peakPixelIntensity() const
            {
                return peakPixelIntensity_;
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
            double eod() const
            {
                return eod_;
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
            void addPixel(const Pixel<dtype>& inPixel)
            {
                pixels_.push_back(inPixel);
                intensity_ += inPixel.intensity();

                // adjust the cluster bounds
                uint32 row = inPixel.row();
                uint32 col = inPixel.col();
                if (row < rowMin_)
                {
                    rowMin_ = row;
                }
                if (row > rowMax_)
                {
                    rowMax_ = row;
                }
                if (col < colMin_)
                {
                    colMin_ = col;
                }
                if (col > colMax_)
                {
                    colMax_ = col;
                }

                // adjust he peak pixel intensity
                if (inPixel.intensity() > peakPixelIntensity_)
                {
                    peakPixelIntensity_ = inPixel.intensity();
                }

                // calculate the energy on detector estimate
                eod_ = static_cast<double>(peakPixelIntensity_) / static_cast<double>(intensity_);
            }

            // =============================================================================
            // Description:
            //              returns a string representation of the cluster
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              string
            //
            std::string str() const
            {
                std::string out;
                for (uint32 i = 0; i < size(); ++i)
                {
                    out += "Pixel " + Utils::num2str(i) + ":" + pixels_[i].str();
                }

                return out;
            }

            //============================================================================
            // Method Description: 
            //						prints the Cluster object to the console
            //		
            // Inputs:
            //				None
            // Outputs:
            //				None
            //
            void print() const
            {
                std::cout << *this;
            }

            // =============================================================================
            // Description:
            //              osstream operator
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              None
            //
            friend std::ostream& operator<<(std::ostream& inStream, const Cluster<dtype>& inCluster)
            {
                inStream << inCluster.str();
                return inStream;
            }
        };

        // =============================================================================
        // Class Description:
        //              Clusters exceedance data into contiguous groups
        //
        template<class dtype>
        class ClusterMaker
        {
        private:
            // ==================================Attributes=================================
            const NdArray<bool>* const      xcds_;
            const NdArray<dtype>* const     intensities_;
            std::vector<Pixel<dtype> >      xcdsVec_;

            Shape                           shape_;

            std::vector<Cluster<dtype> >    clusters_;

            // =============================================================================
            // Description:
            //              checks that the input row and column have not fallen off of the edge
            // 
            // Parameter(s): 
            //              pixel row, pixel column
            // 
            // Return: 
            //              returns a pixel object clipped to the image boundaries
            //
            Pixel<dtype> makePixel(int32 inRow, int32 inCol)
            {
                // Make sure that on the edges after i've added or subtracted 1 from the row and col that 
                // i haven't gone over the edge
                uint32 row = std::min(static_cast<uint32>(std::max<int32>(inRow, 0)), shape_.rows - 1);
                uint32 col = std::min(static_cast<uint32>(std::max<int32>(inCol, 0)), shape_.cols - 1);
                dtype intensity = intensities_->operator()(row, col);

                return Pixel<dtype>(row, col, intensity);
            }

            // =============================================================================
            // Description:
            //              finds all of the neighboring pixels to the input pixel 
            // 
            // Parameter(s): 
            //              pixel object
            // 
            // Return: 
            //              set of pixels that neighbor the input pixel
            //
            void findNeighbors(const Pixel<dtype>& inPixel, std::set<Pixel<dtype> >& outNeighbors)
            {
                // using a set will auto take care of adding duplicate pixels on the edges

                // the 8 surrounding neighbors
                int32 row = static_cast<int32>(inPixel.row());
                int32 col = static_cast<int32>(inPixel.col());

                outNeighbors.insert(outNeighbors.end(), makePixel(row - 1, col - 1));
                outNeighbors.insert(outNeighbors.end(), makePixel(row - 1, col));
                outNeighbors.insert(outNeighbors.end(), makePixel(row - 1, col + 1));
                outNeighbors.insert(outNeighbors.end(), makePixel(row, col - 1));
                outNeighbors.insert(outNeighbors.end(), makePixel(row, col + 1));
                outNeighbors.insert(outNeighbors.end(), makePixel(row + 1, col - 1));
                outNeighbors.insert(outNeighbors.end(), makePixel(row + 1, col));
                outNeighbors.insert(outNeighbors.end(), makePixel(row + 1, col + 1));
            }

            // =============================================================================
            // Description:
            //              finds all of the neighboring pixels to the input pixel that are NOT exceedances
            // 
            // Parameter(s): 
            //              pixel object
            // 
            // Return: 
            //              vector of non exceedance neighboring pixels
            //
            void findNeighborNotXcds(const Pixel<dtype>& inPixel, std::vector<Pixel<dtype> >& outNeighbors)
            {
                std::set<Pixel<dtype> > neighbors;
                findNeighbors(inPixel, neighbors);

                // check if the neighboring pixels are exceedances and insert into the xcd vector
                for (auto pixelIter = neighbors.begin(); pixelIter != neighbors.end(); ++pixelIter)
                {
                    if (!xcds_->operator()(pixelIter->row(), pixelIter->col()))
                    {
                        outNeighbors.push_back(*pixelIter);
                    }
                }
            }

            // =============================================================================
            // Description:
            //              finds the pixel index of neighboring pixels
            // 
            // Parameter(s): 
            //              pixel object
            // 
            // Return: 
            //              vector of neighboring pixel indices
            //
            void findNeighborXcds(const Pixel<dtype>& inPixel, std::vector<uint32>& outNeighbors)
            {
                std::set<Pixel<dtype> > neighbors;
                findNeighbors(inPixel, neighbors);
                std::vector<Pixel<dtype> > neighborXcds;

                // check if the neighboring pixels are exceedances and insert into the xcd vector
                for (auto pixelIter = neighbors.begin(); pixelIter != neighbors.end(); ++pixelIter)
                {
                    if (xcds_->operator()(pixelIter->row(), pixelIter->col()))
                    {
                        neighborXcds.push_back(*pixelIter);
                    }
                }

                // loop through the neighbors and find the cooresponding index into exceedances_
                for (auto pixelIter = neighborXcds.begin(); pixelIter < neighborXcds.end(); ++pixelIter)
                {
                    auto theExceedanceIter = find(xcdsVec_.begin(), xcdsVec_.end(), *pixelIter);
                    outNeighbors.push_back(static_cast<uint32>(theExceedanceIter - xcdsVec_.begin()));
                }
            }

            // =============================================================================
            // Description:
            //              workhorse method that performs the clustering algorithm
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              None
            //
            void runClusterMaker()
            {
                uint32 clusterId = 0;

                for (uint32 xcdIdx = 0; xcdIdx < xcdsVec_.size(); ++xcdIdx)
                {
                    Pixel<dtype>& currentPixel = xcdsVec_[xcdIdx];

                    // not already visited
                    if (currentPixel.clusterId() == -1)
                    {
                        Cluster<dtype> newCluster(clusterId);    // a new cluster
                        currentPixel.setClusterId(clusterId);
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
                            for (uint32 newNeighborsIdx = 0; newNeighborsIdx < newNeighborIds.size(); ++newNeighborsIdx)
                            {
                                // not already in neighbors
                                if (find(neighborIds.begin(), neighborIds.end(), newNeighborIds[newNeighborsIdx]) == neighborIds.end())
                                {
                                    neighborIds.push_back(newNeighborIds[newNeighborsIdx]);
                                }
                            }

                            // not already assigned to a cluster
                            if (currentNeighborPixel.clusterId() == -1)
                            {
                                currentNeighborPixel.setClusterId(clusterId);
                                newCluster.addPixel(currentNeighborPixel);
                            }
                        }

                        clusters_.push_back(newCluster);
                        ++clusterId;
                    }
                }
            }

            // =============================================================================
            // Description:
            //              3x3 dialates the clusters
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              None
            //
            void expandClusters()
            {
                // loop through the clusters 
                for (auto clusterIter = clusters_.begin(); clusterIter < clusters_.end(); ++clusterIter)
                {
                    // loop through the pixels of the cluster 
                    Cluster<dtype>& theCluster = *clusterIter;
                    uint32 clusterSize = static_cast<uint32>(theCluster.size());
                    for (uint32 iPixel = 0; iPixel < clusterSize; ++iPixel)
                    {
                        const Pixel<dtype>& thePixel = theCluster[iPixel];
                        std::vector<Pixel<dtype> > neighborsNotXcds;
                        findNeighborNotXcds(thePixel, neighborsNotXcds);

                        // loop through the neighbors and if they haven't already been added to the cluster, add them
                        for (auto newPixelIter = neighborsNotXcds.begin(); newPixelIter < neighborsNotXcds.end(); ++newPixelIter)
                        {
                            if (find(theCluster.begin(), theCluster.end(), *newPixelIter) == theCluster.end())
                            {
                                theCluster.addPixel(*newPixelIter);
                            }
                        }
                    }
                }
            }

        public:
            // ================================Typedefs=====================================
            typedef typename std::vector<Cluster<dtype> >::const_iterator   const_iterator;

            // =============================================================================
            // Description:
            //              constructor
            // 
            // Parameter(s): 
            //              NdArray<bool>*, pointer to exceedance array
            //              NdArray<dtype>*, pointer to intensity array
            //				border to apply around exceedance pixels post clustering, default 0
            // 
            // Return: 
            //              None
            //
            ClusterMaker(const NdArray<bool>* const inXcdArrayPtr, const NdArray<dtype>* const inIntensityArrayPtr, uint8 inBorderWidth = 0) :
                xcds_(inXcdArrayPtr),
                intensities_(inIntensityArrayPtr)
            {
                if (xcds_->shape() != intensities_->shape())
                {
                    throw std::invalid_argument("ERROR: ImageProcessing::ClusterMaker(): input xcd and intensity arrays must be the same shape.");
                }

                shape_ = xcds_->shape();

                // convert the NdArray of booleans to a vector of exceedances
                for (uint32 row = 0; row < shape_.rows; ++row)
                {
                    for (uint32 col = 0; col < shape_.cols; ++col)
                    {
                        if (xcds_->operator()(row, col))
                        {
                            Pixel<dtype> thePixel(row, col, intensities_->operator()(row, col));
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

            // =============================================================================
            // Description:
            //              returns the number of clusters in the frame
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              number of clusters
            //
            uint32 size()
            {
                return static_cast<uint32>(clusters_.size());
            }

            // =============================================================================
            // Description:
            //              access operator, no bounds checking
            // 
            // Parameter(s): 
            //              index
            // 
            // Return: 
            //              None
            //
            const Cluster<dtype>& operator[](uint32 inIndex) const
            {
                return clusters_[inIndex];
            }

            // =============================================================================
            // Description:
            //              access method with bounds checking
            // 
            // Parameter(s): 
            //              index
            // 
            // Return: 
            //              None
            //
            const Cluster<dtype>& at(uint32 inIndex) const
            {
                if (inIndex >= clusters_.size())
                {
                    throw std::invalid_argument("ERROR: ImageProcessing::ClusterMaker::at: index exceeds cluster size.");
                }
                return clusters_[inIndex];
            }

            // =============================================================================
            // Description:
            //              returns in iterator to the beginning cluster of the container
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              const_iterator
            //
            const_iterator begin() const
            {
                return clusters_.cbegin();
            }

            // =============================================================================
            // Description:
            //              returns in iterator to the 1 past the end cluster of the container
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              const_iterator
            //
            const_iterator end() const
            {
                return clusters_.cend();
            }
        };

        //================================================================================
        // Class Description:
        //						holds the information for a centroid
        //
        template<typename dtype>
        class Centroid
        {
            //==================================Attributes================================//
            double          row_;
            double          col_;
            dtype           intensity_;
            double          eod_;

            // =============================================================================
            // Description:
            //              center of mass algorithm;
            //              WARNING: if both positive and negative values are present in the cluster,
            //              it can lead to an undefined COM.
            // 
            // Parameter(s): 
            //              cluster
            // 
            // Return: 
            //              None
            //
            void centerOfMass(const Cluster<dtype>& inCluster)
            {
                Shape clusterShape(inCluster.height(), inCluster.width());
                NdArray<dtype> clusterArray(clusterShape);
                clusterArray.zeros();

                uint32 rowMin = inCluster.rowMin();
                uint32 colMin = inCluster.colMin();
                dtype intensity = inCluster.intensity();

                auto iter = inCluster.begin();
                for (; iter < inCluster.end(); ++iter)
                {
                    clusterArray(iter->row() - rowMin, iter->col() - colMin) = iter->intensity();
                }

                // first get the row center
                row_ = 0;
                uint32 row = rowMin;
                for (uint32 rowIdx = 0; rowIdx < clusterShape.rows; ++rowIdx)
                {
                    double rowSum = 0;
                    for (uint32 colIdx = 0; colIdx < clusterShape.cols; ++colIdx)
                    {
                        rowSum += static_cast<double>(clusterArray(rowIdx, colIdx));
                    }
                    row_ += rowSum * static_cast<double>(row++);
                }

                row_ /= static_cast<double>(intensity);

                // then get the column center
                col_ = 0;
                uint32 col = colMin;
                for (uint32 colIdx = 0; colIdx < clusterShape.cols; ++colIdx)
                {
                    double colSum = 0;
                    for (uint32 rowIdx = 0; rowIdx < clusterShape.rows; ++rowIdx)
                    {
                        colSum += static_cast<double>(clusterArray(rowIdx, colIdx));
                    }
                    col_ += colSum * static_cast<double>(col++);
                }

                col_ /= static_cast<double>(intensity);
            }

        public:
            // =============================================================================
            // Description:
            //              defualt constructor needed by containers
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              None
            //
            Centroid() :
                row_(0),
                col_(0),
                intensity_(0),
                eod_(0)
            {};

            // =============================================================================
            // Description:
            //              constructor
            // 
            // Parameter(s): 
            //              centroid id,
            //              FP row,
            //              FP column,
            //              centroid intensity
            //              cluster EOD
            //              cluster number of pixels
            // 
            // Return: 
            //              None
            //
            Centroid(const Cluster<dtype>& inCluster) :
                intensity_(inCluster.intensity()),
                eod_(inCluster.eod())
            {
                centerOfMass(inCluster);
            }

            // =============================================================================
            // Description:
            //              gets the centroid row
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              centroid row
            //
            double row() const
            {
                return row_;
            }

            // =============================================================================
            // Description:
            //              gets the centroid col
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              centroid col
            //
            double col() const
            {
                return col_;
            }

            // =============================================================================
            // Description:
            //              gets the centroid intensity
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              centroid intensity
            //
            dtype intensity() const
            {
                return intensity_;
            }

            // =============================================================================
            // Description:
            //              returns the estimated eod of the centroid
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              star id
            //
            double eod() const
            {
                return eod_;
            }

            // =============================================================================
            // Description:
            //              returns the centroid as a string representation
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              string
            //
            std::string str() const
            {
                std::string out;
                out += "row = " + Utils::num2str(row_) + " col = " + Utils::num2str(col_);
                out += " intensity = " + Utils::num2str(intensity_) + " eod = " + Utils::num2str(eod_) + "\n";

                return out;
            }

            //============================================================================
            // Method Description: 
            //						prints the Centroid object to the console
            //		
            // Inputs:
            //				None
            // Outputs:
            //				None
            //
            void print() const
            {
                std::cout << *this;
            }

            // =============================================================================
            // Description:
            //              equality operator
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              bool
            //
            bool operator==(const Centroid<dtype>& rhs) const
            {
                return row_ == rhs.row_ && col_ == rhs.col_ && intensity_ == rhs.intensity_ && eod_ == rhs.eod_;
            }

            // =============================================================================
            // Description:
            //              not equality operator
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              bool
            //
            bool operator!=(const Centroid<dtype>& rhs) const
            {
                return !(*this == rhs);
            }

            // =============================================================================
            // Description:
            //              less than operator for std::sort algorithm;
            //              NOTE: std::sort sorts in ascending order. Since I want to sort 
            //              the centroids in descensing order, I am purposefully defining
            //              this operator backwards!
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              None
            //
            bool operator<(const Centroid<dtype>& rhs) const
            {
                return intensity_ < rhs.intensity_ ? false : true;
            }

            // =============================================================================
            // Description:
            //              ostream operator
            // 
            // Parameter(s): 
            //              None
            // 
            // Return: 
            //              None
            //
            friend std::ostream& operator<<(std::ostream& inStream, const Centroid<dtype>& inCentriod)
            {
                inStream << inCentriod.str();
                return inStream;
            }
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
            return std::move(inImageArray > inThreshold);
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
        template<typename dtype>
        inline std::vector<Centroid<dtype> > centroidClusters(const std::vector<Cluster<dtype> >& inClusters)
        {
            std::vector<Centroid<dtype> > centroids(inClusters.size());

            for (uint32 i = 0; i < inClusters.size(); ++i)
            {
                centroids[i] = std::move(Centroid<dtype>(inClusters[i]));
            }

            return std::move(centroids);
        }

        //============================================================================
        // Method Description: 
        //						Clusters exceedance pixels from an image
        //		
        // Inputs:
        //				NdArray
        //				NdArray of exceedances
        //				border to apply around exceedance pixels post clustering, default 0
        // Outputs:
        //				std::vector<Cluster>
        //
        template<typename dtype>
        inline std::vector<Cluster<dtype> > clusterPixels(const NdArray<dtype>& inImageArray, const NdArray<bool>& inExceedances, uint8 inBorderWidth = 0)
        {
            ClusterMaker<dtype> clusterMaker(&inExceedances, &inImageArray, inBorderWidth);
            return std::move(std::vector<Cluster<dtype> >(clusterMaker.begin(), clusterMaker.end()));
        }

        //============================================================================
        // Method Description: 
        //						Generates a list of centroids givin an input exceedance
        //						rate
        //		
        // Inputs:
        //				NdArray
        //				exceedance rate
        //              string "pre", or "post" for where to apply the exceedance windowing
        //				border to apply, default 0
        // Outputs:
        //				std::vector<Centroid>
        //
        template<typename dtype>
        inline std::vector<Centroid<dtype> > generateCentroids(const NdArray<dtype>& inImageArray, double inRate, const std::string inWindowType, uint8 inBorderWidth = 0)
        {
            uint8 borderWidthPre = 0;
            uint8 borderWidthPost = 0;
            if (inWindowType.compare("pre") == 0)
            {
                borderWidthPre = inBorderWidth;
            }
            else if (inWindowType.compare("post") == 0)
            {
                borderWidthPost = inBorderWidth;
            }
            else
            {
                throw std::invalid_argument("ERROR ImageProcessing::generateCentroids: input window type options are ['pre', 'post']");
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
            std::vector<Cluster<dtype> > clusters = clusterPixels(inImageArray, xcds, borderWidthPost);

            // centroid the clusters
            return std::move(centroidClusters(clusters));
        }

        //============================================================================
        // Method Description: 
        //						Calculates a threshold such that the input rate of pixels
        //						exceeds the threshold. Really should only be used for integer
        //                      input array values. If using floating point data, user beware...
        //		
        // Inputs:
        //				NdArray
        //				exceedance rate
        // Outputs:
        //				dtype
        //
        template<typename dtype>
        inline dtype generateThreshold(const NdArray<dtype>& inImageArray, double inRate)
        {
            if (inRate < 0 || inRate > 1)
            {
                throw std::invalid_argument("ERROR: ImageProcessing::generateThreshold: input rate must be of the range [0, 1]");
            }

            // first build a histogram
            int32 minValue = static_cast<int32>(std::floor(inImageArray.min().item()));
            int32 maxValue = static_cast<int32>(std::floor(inImageArray.max().item()));

            if (inRate == 0)
            {
                return static_cast<dtype>(maxValue);
            }
            else if (inRate == 1)
            {
                if (DtypeInfo<dtype>::isSigned())
                {
                    return static_cast<dtype>(minValue - 1);
                }
                else
                {
                    return static_cast<dtype>(0);
                }
            }

            uint32 histSize = static_cast<uint32>(maxValue - minValue + 1);

            NdArray<double> histogram(1, histSize);
            histogram.zeros();
            uint32 numPixels = inImageArray.size();
            for (uint32 i = 0; i < numPixels; ++i)
            {
                uint32 bin = static_cast<uint32>(static_cast<int32>(std::floor(inImageArray[i])) - minValue);
                ++histogram[bin];
            }

            // integrate the normalized histogram from right to left to make a survival function (1 - CDF)
            double dNumPixels = static_cast<double>(numPixels);
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

            bool keepGoing = true;
            while (keepGoing)
            {
                double value = survivalFunction[index];
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
                    int32 thresh = static_cast<int32>(index) + minValue - 1;
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

        //============================================================================
        // Method Description: 
        //						Window expand around exceedance pixels
        //		
        // Inputs:
        //				NdArray<bool>
        //				border width
        // Outputs:
        //				NdArray<bool>
        //
        inline NdArray<bool> windowExceedances(const NdArray<bool>& inExceedances, uint8 inBorderWidth)
        {
            // not the most efficient way to do things, but the easist...
            NdArray<bool> xcds(inExceedances);
            Shape inShape = xcds.shape();
            for (uint8 border = 0; border < inBorderWidth; ++border)
            {
                for (int32 row = 0; row < static_cast<int32>(inShape.rows); ++row)
                {
                    for (int32 col = 0; col < static_cast<int32>(inShape.cols); ++col)
                    {
                        if (inExceedances(row, col))
                        {
                            xcds(std::max(row - 1, 0), std::max(col - 1, 0)) = true;
                            xcds(std::max(row - 1, 0), col) = true;
                            xcds(std::max(row - 1, 0), std::min<int32>(col + 1, inShape.cols - 1)) = true;

                            xcds(row, std::max<int32>(col - 1, 0)) = true;
                            xcds(row, std::min<int32>(col + 1, inShape.cols - 1)) = true;

                            xcds(std::min<int32>(row + 1, inShape.rows - 1), std::max(col - 1, 0)) = true;
                            xcds(std::min<int32>(row + 1, inShape.rows - 1), col) = true;
                            xcds(std::min<int32>(row + 1, inShape.rows - 1), std::min<int32>(col + 1, inShape.cols - 1)) = true;
                        }
                    }
                }
            }

            return std::move(xcds);
        }
    }
}
