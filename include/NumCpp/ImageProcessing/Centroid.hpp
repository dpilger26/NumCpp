/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.4.0
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
/// holds the information for a centroid
///
#pragma once

#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Functions/centerOfMass.hpp"
#include "NumCpp/ImageProcessing/Cluster.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/num2str.hpp"

#include <iostream>
#include <string>

namespace nc
{
    namespace imageProcessing
    {
        //================================================================================
        // Class Description:
        ///						holds the information for a centroid
        template<typename dtype>
        class Centroid
        {
        private:
            STATIC_ASSERT_ARITHMETIC(dtype);

        public:
            //=============================================================================
            // Description:
            ///              defualt constructor needed by containers
            ///
            Centroid() noexcept = default;

            //=============================================================================
            // Description:
            ///              constructor
            ///
            /// @param               inCluster
            ///
            Centroid(const Cluster<dtype>& inCluster) noexcept :
                intensity_(inCluster.intensity()),
                eod_(inCluster.eod())
            {
                centerOfMass(inCluster);
            }

            //=============================================================================
            // Description:
            ///              gets the centroid row
            ///
            /// @return
            ///              centroid row
            ///
            double row() const noexcept
            {
                return row_;
            }

            //=============================================================================
            // Description:
            ///              gets the centroid col
            ///
            /// @return
            ///              centroid col
            ///
            double col() const noexcept
            {
                return col_;
            }

            //=============================================================================
            // Description:
            ///              gets the centroid intensity
            ///
            /// @return
            ///              centroid intensity
            ///
            dtype intensity() const noexcept
            {
                return intensity_;
            }

            //=============================================================================
            // Description:
            ///              returns the estimated eod of the centroid
            ///
            /// @return
            ///              star id
            ///
            double eod() const noexcept
            {
                return eod_;
            }

            //=============================================================================
            // Description:
            ///              returns the centroid as a string representation
            ///
            /// @return
            ///              std::string
            ///
            std::string str() const
            {
                std::string out;
                out += "row = " + utils::num2str(row_) + " col = " + utils::num2str(col_);
                out += " intensity = " + utils::num2str(intensity_) + " eod = " + utils::num2str(eod_) + "\n";

                return out;
            }

            //============================================================================
            /// Method Description:
            ///						prints the Centroid object to the console
            ///
            void print() const
            {
                std::cout << *this;
            }

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
            bool operator==(const Centroid<dtype>& rhs) const noexcept
            {
                return row_ == rhs.row_ && col_ == rhs.col_ && intensity_ == rhs.intensity_ && eod_ == rhs.eod_;
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
            bool operator!=(const Centroid<dtype>& rhs) const noexcept
            {
                return !(*this == rhs);
            }

            //=============================================================================
            // Description:
            ///              less than operator for std::sort algorithm;
            ///              NOTE: std::sort sorts in ascending order. Since I want to sort
            ///              the centroids in descensing order, I am purposefully defining
            ///              this operator backwards!
            ///
            /// @param
            ///              rhs
            ///
            /// @return
            ///              bool
            ///
            bool operator<(const Centroid<dtype>& rhs) const noexcept
            {
                return intensity_ < rhs.intensity_ ? false : true;
            }

            //=============================================================================
            // Description:
            ///              ostream operator
            ///
            /// @param              inStream
            /// @param              inCentriod
            /// @return
            ///              std::ostream
            ///
            friend std::ostream& operator<<(std::ostream& inStream, const Centroid<dtype>& inCentriod)
            {
                inStream << inCentriod.str();
                return inStream;
            }

        private:
            //==================================Attributes================================///
            double          row_{ 0.0 };
            double          col_{ 0.0 };
            dtype           intensity_{ 0 };
            double          eod_{ 0.0 };

            //=============================================================================
            // Description:
            ///              center of mass algorithm;
            ///              WARNING: if both positive and negative values are present in the cluster,
            ///              it can lead to an undefined COM.
            ///
            /// @param
            ///              inCluster
            ///
            void centerOfMass(const Cluster<dtype>& inCluster) noexcept
            {
                const Shape clusterShape(inCluster.height(), inCluster.width());
                NdArray<dtype> clusterArray(clusterShape);
                clusterArray.zeros();

                const uint32 rowMin = inCluster.rowMin();
                const uint32 colMin = inCluster.colMin();

                for (auto& pixel : inCluster)
                {
                    clusterArray(pixel.row - rowMin, pixel.col - colMin) = pixel.intensity;
                }

                const auto rowCol = nc::centerOfMass(clusterArray);
                row_ = rowCol.front() + rowMin;
                col_ = rowCol.back() + colMin;
            }
        };
    }
}
