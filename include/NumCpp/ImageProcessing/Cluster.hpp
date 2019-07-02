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
/// A module for basic image processing
///

#pragma once

#include"NumCpp/Core/Types.hpp"
#include"NumCpp/Utils/num2str.hpp"

#include<algorithm>
#include<iostream>
#include<limits>
#include<stdexcept>
#include<string>
#include<utility>
#include<vector>

namespace nc
{
    //================================================================================
    // Class Description:
    ///						Class for basic image processing
    namespace imageProcessing
    {
        //================================================================================
        // Class Description:
        ///						Holds the information for a cluster of pixels
        template<typename dtype>
        class Cluster
        {
        public:
            //================================Typedefs===============================
            typedef typename std::vector<Pixel<dtype> >::const_iterator    const_iterator;

        private:
            //================================Attributes===============================
            uint32                      clusterId_;
            std::vector<Pixel<dtype> >  pixels_;

            uint32                      rowMin_{ std::numeric_limits<uint32>::max() }; // largest possible number
            uint32                      rowMax_{ 0 };
            uint32                      colMin_{ std::numeric_limits<uint32>::max() }; // largest possible number
            uint32                      colMax_{ 0 };

            dtype                       intensity_{ 0 };
            dtype                       peakPixelIntensity_{ 0 };

            double                      eod_{ 1.0 };

        public:
            //=============================================================================
            // Description:
            ///              default constructor needed by containers
            ///
            /// @param
            ///              inClusterId
            ///
            Cluster(uint32 inClusterId) :
                clusterId_(-1)
            {}

            //=============================================================================
            // Description:
            ///              constructor
            ///
            /// @param
            ///              inClusterId
            ///
            Cluster(uint32 inClusterId) :
                clusterId_(inClusterId)
            {}

            //=============================================================================
            // Description:
            ///              equality operator
            ///
            /// @param
            ///              rhs
            ///
            /// @return
            ///              bool
            ///
            bool operator==(const Cluster<dtype>& rhs) const
            {
                if (pixels_.size() != rhs.pixels_.size())
                {
                    return false;
                }

                return std::equal(begin(), end(), rhs.begin());
            }

            //=============================================================================
            // Description:
            ///              not equality operator
            ///
            /// @param
            ///              rhs
            ///
            /// @return
            ///              bool
            ///
            bool operator!=(const Cluster<dtype>& rhs) const
            {
                return !(*this == rhs);
            }

            //=============================================================================
            // Description:
            ///              access operator, no bounds checking
            ///
            /// @param
            ///              inIndex
            ///
            /// @return
            ///              Pixel
            ///
            const Pixel<dtype>& operator[](uint32 inIndex) const
            {
                return pixels_[inIndex];
            }

            //=============================================================================
            // Description:
            ///              access method with bounds checking
            ///
            /// @param
            ///              inIndex
            ///
            /// @return
            ///              Pixel
            ///
            const Pixel<dtype>& at(uint32 inIndex) const
            {
                if (inIndex >= pixels_.size())
                {
                    std::string errStr = "ERROR: imageProcessing::Cluster::at: index exceeds cluster size.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }
                return pixels_[inIndex];
            }

            //=============================================================================
            // Description:
            ///              returns in iterator to the beginning pixel of the cluster
            ///
            /// @return
            ///              const_iterator
            ///
            const_iterator begin() const noexcept
            {
                return pixels_.cbegin();
            }

            //=============================================================================
            // Description:
            ///              returns in iterator to the 1 past the end pixel of the cluster
            ///
            /// @return
            ///              const_iterator
            ///
            const_iterator end() const noexcept
            {
                return pixels_.cend();
            }

            //=============================================================================
            // Description:
            ///              returns the number of pixels in the cluster
            ///
            /// @return
            ///              number of pixels in the cluster
            ///
            uint32 size() const noexcept
            {
                return static_cast<uint32>(pixels_.size());
            }

            //=============================================================================
            // Description:
            ///              returns the minimum row number of the cluster
            ///
            /// @return
            ///              minimum row number of the cluster
            ///
            uint32 clusterId() const noexcept
            {
                return clusterId_;
            }

            //=============================================================================
            // Description:
            ///              returns the minimum row number of the cluster
            ///
            /// @return
            ///              minimum row number of the cluster
            ///
            uint32 rowMin() const noexcept
            {
                return rowMin_;
            }

            //=============================================================================
            // Description:
            ///              returns the maximum row number of the cluster
            ///
            /// @return
            ///              maximum row number of the cluster
            ///
            uint32 rowMax() const noexcept
            {
                return rowMax_;
            }

            //=============================================================================
            // Description:
            ///              returns the minimum column number of the cluster
            ///
            /// @return
            ///              minimum column number of the cluster
            ///
            uint32 colMin() const noexcept
            {
                return colMin_;
            }

            //=============================================================================
            // Description:
            ///              returns the maximum column number of the cluster
            ///
            /// @return
            ///              maximum column number of the cluster
            ///
            uint32 colMax() const noexcept
            {
                return colMax_;
            }

            //=============================================================================
            // Description:
            ///              returns the number of rows the cluster spans
            ///
            /// @return
            ///              number of rows
            ///
            uint32 height() const noexcept
            {
                return rowMax_ - rowMin_ + 1;
            }

            //=============================================================================
            // Description:
            ///              returns the number of columns the cluster spans
            ///
            /// @return
            ///              number of columns
            ///
            uint32 width() const noexcept
            {
                return colMax_ - colMin_ + 1;
            }

            //=============================================================================
            // Description:
            ///              returns the summed intensity of the cluster
            ///
            /// @return
            ///              summed cluster intensity
            ///
            dtype intensity() const noexcept
            {
                return intensity_;
            }

            //=============================================================================
            // Description:
            ///              returns the intensity of the peak pixel in the cluster
            ///
            /// @return
            ///              peak pixel intensity
            ///
            dtype peakPixelIntensity() const noexcept
            {
                return peakPixelIntensity_;
            }

            //=============================================================================
            // Description:
            ///              returns the cluster estimated energy on detector (EOD)
            ///
            /// @return
            ///              eod
            ///
            double eod() const noexcept
            {
                return eod_;
            }

            //=============================================================================
            // Description:
            ///              adds a pixel to the cluster
            ///
            /// @param
            ///              inPixel
            ///
            void addPixel(const Pixel<dtype>& inPixel)
            {
                pixels_.push_back(inPixel);
                intensity_ += inPixel.intensity();

                // adjust the cluster bounds
                const uint32 row = inPixel.row();
                const uint32 col = inPixel.col();
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

            //=============================================================================
            // Description:
            ///              returns a string representation of the cluster
            ///
            /// @return
            ///              string
            ///
            std::string str() const
            {
                std::string out;
                uint32 counter = 0;
                std::for_each(begin(), end(), 
                    [&](const Pixel<dtype>& pixel) { out += "Pixel " + utils::num2str(counter++) + ":" + pixel.str(); });

                return out;
            }

            //============================================================================
            /// Method Description:
            ///						prints the Cluster object to the console
            ///
            void print() const
            {
                std::cout << *this;
            }

            //=============================================================================
            // Description:
            ///              osstream operator
            ///
            /// @param               inStream
            /// @param               inCluster
            /// @return
            ///              std::ostream
            ///
            friend std::ostream& operator<<(std::ostream& inStream, const Cluster<dtype>& inCluster)
            {
                inStream << inCluster.str();
                return inStream;
            }
        };
    }
}
