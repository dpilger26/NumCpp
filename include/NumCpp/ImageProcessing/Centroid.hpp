/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2025 David Pilger
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
/// holds the information for a centroid
///
#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/centerOfMass.hpp"
#include "NumCpp/ImageProcessing/Cluster.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/essentiallyEqual.hpp"
#include "NumCpp/Utils/num2str.hpp"

namespace nc::imageProcessing
{
    //================================================================================
    // Class Description:
    /// holds the information for a centroid
    template<typename dtype>
    class Centroid
    {
    private:
        STATIC_ASSERT_ARITHMETIC(dtype);

    public:
        using accumulator_t = typename std::conditional<std::is_integral<dtype>::value, int64, double>::type;

        //=============================================================================
        // Description:
        /// defualt constructor needed by containers
        ///
        Centroid() = default;

        //=============================================================================
        // Description:
        /// constructor
        ///
        /// @param inCluster
        ///
        explicit Centroid(const Cluster<dtype>& inCluster) :
            intensity_(inCluster.intensity()),
            eod_(inCluster.eod())
        {
            centerOfMass(inCluster);
            setEllipseProperties(inCluster);
        }

        //=============================================================================
        // Description:
        /// gets the centroid row
        ///
        /// @return centroid row
        ///
        [[nodiscard]] double row() const noexcept
        {
            return row_;
        }

        //=============================================================================
        // Description:
        /// gets the centroid col
        ///
        /// @return centroid col
        ///
        [[nodiscard]] double col() const noexcept
        {
            return col_;
        }

        //=============================================================================
        // Description:
        /// gets the centroid intensity
        ///
        /// @return centroid intensity
        ///
        [[nodiscard]] accumulator_t intensity() const noexcept
        {
            return intensity_;
        }

        //=============================================================================
        // Description:
        /// returns the estimated eod of the centroid
        ///
        /// @return star id
        ///
        [[nodiscard]] double eod() const noexcept
        {
            return eod_;
        }

        //=============================================================================
        // Description:
        /// returns the ellipse semi-major axis a
        ///
        /// @return a
        ///
        [[nodiscard]] double a() const noexcept
        {
            return a_;
        }

        //=============================================================================

        // Description:
        /// returns the ellipse semi-minor axis b
        ///
        /// @return b
        ///
        [[nodiscard]] double b() const noexcept
        {
            return b_;
        }

        //=============================================================================
        // Description:
        /// returns the ellipse eccentricity
        ///
        /// @return eccentricity
        ///
        [[nodiscard]] double eccentricity() const noexcept
        {
            return eccentricity_;
        }

        //=============================================================================

        // Description:
        /// returns the ellipse semi-minor axis orientation
        ///
        /// @return orientation
        ///
        [[nodiscard]] double orientation() const noexcept
        {
            return orientation_;
        }

        //=============================================================================
        // Description:
        /// returns the centroid as a string representation
        ///
        /// @return std::string
        ///
        [[nodiscard]] std::string str() const
        {
            std::string out = "row = " + utils::num2str(row_) + " col = " + utils::num2str(col_) +
                              " intensity = " + utils::num2str(intensity_) + " eod = " + utils::num2str(eod_) +
                              " a = " + utils::num2str(a_) + " b = " + utils::num2str(b_) +
                              " eccentricity = " + utils::num2str(eccentricity_) +
                              " orientation = " + utils::num2str(orientation_) + '\n';

            return out;
        }

        //============================================================================
        /// Method Description:
        /// prints the Centroid object to the console
        ///
        void print() const
        {
            std::cout << *this;
        }

        //=============================================================================
        // Description:
        /// equality operator
        ///
        /// @param rhs
        ///
        /// @return bool
        ///
        bool operator==(const Centroid<dtype>& rhs) const noexcept
        {
            return (utils::essentiallyEqual(row_, rhs.row_) && utils::essentiallyEqual(col_, rhs.col_) &&
                    utils::essentiallyEqual(intensity_, rhs.intensity_) && utils::essentiallyEqual(eod_, rhs.eod_) &&
                    utils::essentiallyEqual(a_, rhs.a_) && utils::essentiallyEqual(b_, rhs.b_) &&
                    utils::essentiallyEqual(eccentricity_, rhs.eccentricity_) &&
                    utils::essentiallyEqual(orientation_, rhs.orientation_));
        }

        //=============================================================================
        // Description:
        /// not equality operator
        ///
        /// @param rhs
        ///
        /// @return bool
        ///
        bool operator!=(const Centroid<dtype>& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        //=============================================================================
        // Description:
        /// less than operator for std::sort algorithm;
        /// NOTE: std::sort sorts in ascending order. Since I want to sort
        /// the centroids in descensing order, I am purposefully defining
        /// this operator backwards!
        ///
        /// @param rhs
        ///
        /// @return bool
        ///
        bool operator<(const Centroid<dtype>& rhs) const noexcept
        {
            return intensity_ < rhs.intensity_ ? false : true;
        }

        //=============================================================================
        // Description:
        /// ostream operator
        ///
        /// @param inStream
        /// @param inCentriod
        /// @return std::ostream
        ///
        friend std::ostream& operator<<(std::ostream& inStream, const Centroid<dtype>& inCentriod)
        {
            inStream << inCentriod.str();
            return inStream;
        }

    private:
        //==================================Attributes================================///
        double        row_{ 0. };
        double        col_{ 0. };
        accumulator_t intensity_{ 0 };
        double        eod_{ 0. };
        /// The ellipse semi-major axis a
        double a_{};
        /// The ellipse semi-minor axis b
        double b_{};
        /// The centriod eccentricity
        double eccentricity_{};
        /// The centriod ellipse orientation in radians.  Measured counter-clockwise from +x axis
        double orientation_{};

        //=============================================================================
        // Description:
        /// center of mass algorithm;
        /// WARNING: if both positive and negative values are present in the cluster,
        /// it can lead to an undefined COM.
        ///
        /// @param inCluster
        ///
        void centerOfMass(const Cluster<dtype>& inCluster)
        {
            const Shape    clusterShape(inCluster.height(), inCluster.width());
            NdArray<dtype> clusterArray(clusterShape);
            clusterArray.zeros();

            const uint32 rowMin = inCluster.rowMin();
            const uint32 colMin = inCluster.colMin();

            for (auto& pixel : inCluster)
            {
                clusterArray(pixel.row - rowMin, pixel.col - colMin) = pixel.intensity;
            }

            const auto rowCol = nc::centerOfMass(clusterArray);
            row_              = rowCol.front() + rowMin;
            col_              = rowCol.back() + colMin;
        }

        //=============================================================================
        // Description:
        /// Sets the cluster ellipse properties
        ///
        /// @param inCluster
        ///
        void setEllipseProperties(const Cluster<dtype>& inCluster) noexcept
        {
            constexpr auto two = static_cast<double>(2.);

            auto m20 = static_cast<double>(0.);
            auto m02 = static_cast<double>(0.);
            auto m11 = static_cast<double>(0.);

            for (typename Cluster<dtype>::const_iterator iter = inCluster.begin(); iter != inCluster.end(); ++iter)
            {
                const auto&  pixel  = *iter;
                const double deltaX = pixel.col - col_;
                const double deltaY = pixel.row - row_;

                m11 += deltaX * deltaY;
                m20 += utils::sqr(deltaX);
                m02 += utils::sqr(deltaY);
            }

            const auto numPixels = static_cast<double>(inCluster.size());
            m11 /= numPixels;
            m20 /= numPixels;
            m02 /= numPixels;

            double piece1 = m20 + m02;
            piece1 /= two;

            double piece2 = std::sqrt(static_cast<double>(4.) * utils::sqr(m11) + utils::sqr(m20 - m02));
            piece2 /= two;

            const double lambda1 = piece1 - piece2;
            const double lambda2 = piece1 + piece2;

            eccentricity_ = std::sqrt(static_cast<double>(1.) - lambda1 / lambda2);
            orientation_  = static_cast<double>(-0.5) * std::atan2(two * m11, m20 - m02);
            a_            = two * std::sqrt(lambda2);
            b_            = two * std::sqrt(lambda1);
        }
    };
} // namespace nc::imageProcessing
