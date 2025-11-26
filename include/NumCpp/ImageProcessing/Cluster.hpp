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
/// Holds the information for a cluster of pixels
///

#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/ImageProcessing/Pixel.hpp"
#include "NumCpp/Utils/num2str.hpp"

namespace nc::imageProcessing
{
    //================================================================================
    // Class Description:
    /// Holds the information for a cluster of pixels
    template<typename dtype>
    class Cluster
    {
    private:
        STATIC_ASSERT_ARITHMETIC(dtype);

    public:
        //================================Typedefs===============================
        using const_iterator = typename std::vector<Pixel<dtype>>::const_iterator;
        using accumulator_t  = typename std::conditional<std::is_integral<dtype>::value, int64, double>::type;

        //=============================================================================
        // Description:
        /// default constructor needed by containers
        ///
        Cluster() = default;

        //=============================================================================
        // Description:
        /// constructor
        ///
        /// @param inClusterId
        ///
        explicit Cluster(uint32 inClusterId) noexcept :
            clusterId_(inClusterId)
        {
        }

        //=============================================================================
        // Description:
        /// equality operator
        ///
        /// @param rhs
        ///
        /// @return bool
        ///
        bool operator==(const Cluster<dtype>& rhs) const noexcept
        {
            if (pixels_.size() != rhs.pixels_.size())
            {
                return false;
            }

            return stl_algorithms::equal(begin(), end(), rhs.begin());
        }

        //=============================================================================
        // Description:
        /// not equality operator
        ///
        /// @param rhs
        ///
        /// @return bool
        ///
        bool operator!=(const Cluster<dtype>& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        //=============================================================================
        // Description:
        /// access operator, no bounds checking
        ///
        /// @param inIndex
        ///
        /// @return Pixel
        ///
        const Pixel<dtype>& operator[](uint32 inIndex) const noexcept
        {
            return pixels_[inIndex];
        }

        //=============================================================================
        // Description:
        /// access method with bounds checking
        ///
        /// @param inIndex
        ///
        /// @return Pixel
        ///
        [[nodiscard]] const Pixel<dtype>& at(uint32 inIndex) const
        {
            if (inIndex >= pixels_.size())
            {
                THROW_INVALID_ARGUMENT_ERROR("index exceeds cluster size.");
            }
            return pixels_[inIndex];
        }

        //=============================================================================
        // Description:
        /// returns in iterator to the beginning pixel of the cluster
        ///
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator begin() const noexcept
        {
            return pixels_.cbegin();
        }

        //=============================================================================
        // Description:
        /// returns in iterator to the 1 past the end pixel of the cluster
        ///
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator end() const noexcept
        {
            return pixels_.cend();
        }

        //=============================================================================
        // Description:
        /// returns the number of pixels in the cluster
        ///
        /// @return number of pixels in the cluster
        ///
        [[nodiscard]] uint32 size() const noexcept
        {
            return static_cast<uint32>(pixels_.size());
        }

        //=============================================================================
        // Description:
        /// returns the minimum row number of the cluster
        ///
        /// @return minimum row number of the cluster
        ///
        [[nodiscard]] uint32 clusterId() const noexcept
        {
            return clusterId_;
        }

        //=============================================================================
        // Description:
        /// returns the minimum row number of the cluster
        ///
        /// @return minimum row number of the cluster
        ///
        [[nodiscard]] uint32 rowMin() const noexcept
        {
            return rowMin_;
        }

        //=============================================================================
        // Description:
        /// returns the maximum row number of the cluster
        ///
        /// @return maximum row number of the cluster
        ///
        [[nodiscard]] uint32 rowMax() const noexcept
        {
            return rowMax_;
        }

        //=============================================================================
        // Description:
        /// returns the minimum column number of the cluster
        ///
        /// @return minimum column number of the cluster
        ///
        [[nodiscard]] uint32 colMin() const noexcept
        {
            return colMin_;
        }

        //=============================================================================
        // Description:
        /// returns the maximum column number of the cluster
        ///
        /// @return maximum column number of the cluster
        ///
        [[nodiscard]] uint32 colMax() const noexcept
        {
            return colMax_;
        }

        //=============================================================================
        // Description:
        /// returns the number of rows the cluster spans
        ///
        /// @return number of rows
        ///
        [[nodiscard]] uint32 height() const noexcept
        {
            return rowMax_ - rowMin_ + 1;
        }

        //=============================================================================
        // Description:
        /// returns the number of columns the cluster spans
        ///
        /// @return number of columns
        ///
        [[nodiscard]] uint32 width() const noexcept
        {
            return colMax_ - colMin_ + 1;
        }

        //=============================================================================
        // Description:
        /// returns the summed intensity of the cluster
        ///
        /// @return summed cluster intensity
        ///
        [[nodiscard]] accumulator_t intensity() const noexcept
        {
            return intensity_;
        }

        //=============================================================================
        // Description:
        /// returns the intensity of the peak pixel in the cluster
        ///
        /// @return peak pixel intensity
        ///
        [[nodiscard]] dtype peakPixelIntensity() const noexcept
        {
            return peakPixelIntensity_;
        }

        //=============================================================================
        // Description:
        /// returns the cluster estimated energy on detector (EOD)
        ///
        /// @return eod
        ///
        [[nodiscard]] double eod() const noexcept
        {
            return eod_;
        }

        //=============================================================================
        // Description:
        /// adds a pixel to the cluster
        ///
        /// @param inPixel
        ///
        void addPixel(const Pixel<dtype>& inPixel)
        {
            pixels_.push_back(inPixel);
            intensity_ += static_cast<accumulator_t>(inPixel.intensity);

            // adjust the cluster bounds
            rowMin_             = std::min(rowMin_, inPixel.row);
            rowMax_             = std::max(rowMax_, inPixel.row);
            colMin_             = std::min(colMin_, inPixel.col);
            colMax_             = std::max(colMax_, inPixel.col);
            peakPixelIntensity_ = std::max(peakPixelIntensity_, inPixel.intensity);

            // calculate the energy on detector estimate
            eod_ = static_cast<double>(peakPixelIntensity_) / static_cast<double>(intensity_);
        }

        //=============================================================================
        // Description:
        /// returns a string representation of the cluster
        ///
        /// @return string
        ///
        [[nodiscard]] std::string str() const
        {
            std::string out;
            uint32      counter = 0;
            std::for_each(begin(),
                          end(),
                          [&](const Pixel<dtype>& pixel)
                          { out += "Pixel " + utils::num2str(counter++) + ":" + pixel.str(); });

            return out;
        }

        //============================================================================
        /// Method Description:
        /// prints the Cluster object to the console
        ///
        void print() const
        {
            std::cout << *this;
        }

        //=============================================================================
        // Description:
        /// osstream operator
        ///
        /// @param inStream
        /// @param inCluster
        /// @return std::ostream
        ///
        friend std::ostream& operator<<(std::ostream& inStream, const Cluster<dtype>& inCluster)
        {
            inStream << inCluster.str();
            return inStream;
        }

    private:
        //================================Attributes===============================
        /// The cluster id
        int32 clusterId_{ -1 };
        /// The pixels that make up the cluster
        std::vector<Pixel<dtype>> pixels_{};
        /// The bounding box minimum row of the cluster.
        uint32 rowMin_{ std::numeric_limits<uint32>::max() }; // largest possible number
        /// The bounding box maximum row of the cluster.
        uint32 rowMax_{ 0 };
        /// The bounding box minimum col of the cluster.
        uint32 colMin_{ std::numeric_limits<uint32>::max() }; // largest possible number
        /// The bounding box maximum row of the cluster.
        uint32 colMax_{ 0 };
        /// The total summed intensity of the pixels in the cluster.
        accumulator_t intensity_{ 0 };
        /// The peak pixel intensity of the cluster
        dtype peakPixelIntensity_{ 0 };
        /// The minimum pixel count value of the cluster
        dtype minPixel{};
        /// The maximum pixel count value of the cluster
        dtype maxPixel{};
        /// The cluster energy on detector
        double eod_{ 1. };
    };
} // namespace nc::imageProcessing
