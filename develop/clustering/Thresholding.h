#ifndef THRESHOLDING_H
#define THRESHOLDING_H

#include "Core/Types.hpp"
#include "Sensor/ImageArray.hpp"

#include "boost/config.hpp"

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
        namespace thresholding
        {
            /// Holds all of the information for an exceedance pixel.
            struct Exceedance
            {
                /// The pixel row on the FP
                uint32 row;
                /// The pixel column on the FP
                uint32 col;
                /// The pixel counts
                accumulator_t counts;
                /// Whether or not the pixel is saturated
                bool isSaturated;
                /// cluster id that this pixel is associated with.
                mutable int32 clusterId;

                //===========================================================================================
                /// Default Constructor
                Exceedance() BOOST_NOEXCEPT;

#ifdef DEBUG
                //===========================================================================================
                /// Stream output operator
                ///
                /// @param stream
                /// @param xcd
                /// @return ostream
                ///
                friend std::ostream& operator<<(std::ostream& stream, const Exceedance& xcd);
#endif
            };

            //======================================================================================
            /// A list of Exceedances
            typedef std::vector<Exceedance> ExceedanceList;

            //======================================================================================
            /// Generates exceedances
            class Thresholder
            {
            public:
                //===========================================================================================
                /// Default Constructor
                Thresholder() BOOST_NOEXCEPT;

                //==================================================================================
                /// Generates exceedances
                ///
                /// @param image: The image to process
                /// @param threshold: the threshold value to use
                ///
                /// @return The method results
                ///
                ExceedanceList process(const sensor::ImageArray<accumulator_t>& image, const accumulator_t threshold) const;

                //==================================================================================
                /// Sets the bad pixel map
                ///
                /// @param bpmPtr: Pointer to a bad pixel map
                ///
                void setBadPixelMap(const bool* const bpmPtr) BOOST_NOEXCEPT;

            private:
                /// Pointer to a bad pixel map
                const bool* bpmPtr_;
            };
        }
    }
}

#endif
