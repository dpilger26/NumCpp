#include "Thresholding.h"

#include "Sensor/SensorConfig.hpp"

#include <cmath>
#include <cstdlib>
#ifdef DEBUG
#include <iostream>
#endif
#include <limits>

//==================================================================================================

namespace ieom
{
    namespace fpga
    {
        namespace thresholding
        {
            Exceedance::Exceedance() BOOST_NOEXCEPT :
                row(0),
                col(0),
                counts(0),
                isSaturated(false),
                clusterId(-1)
            {}

            //==================================================================================================

#ifdef DEBUG
            std::ostream& operator<<(std::ostream& stream, const Exceedance& xcd)
            {
                stream << "Exceedance:\n";
                stream << "row         = " << xcd.row << '\n';
                stream << "col         = " << xcd.col << '\n';
                stream << "counts      = " << xcd.counts << '\n';
                stream << "isSaturated = " << xcd.isSaturated << '\n';
                stream << "clusterId   = " << xcd.clusterId << '\n';

                return stream;
            }
#endif
            //==================================================================================================

            Thresholder::Thresholder() BOOST_NOEXCEPT :
                bpmPtr_(NULL)
            {}

            //==================================================================================================

            ExceedanceList Thresholder::process(const sensor::ImageArray<accumulator_t>& image, const accumulator_t threshold) const
            {
                const bool bpmPtrNotNull = bpmPtr_ != NULL;
                const uint32 saturationLimit = sensor::config::saturatedCounts(image.numCoaddedFrames());

                ExceedanceList exceedanceList;
                for (uint32 row = 0; row < image.numRows(); ++row)
                {
                    for (uint32 col = 0; col < image.numCols(); ++col)
                    {
                        if (bpmPtrNotNull && bpmPtr_[row * image.numCols() + col])
                        {
                            // bad pixel
                            continue;
                        }

                        const accumulator_t counts = image(row, col);
                        if (counts > threshold)
                        {
                            Exceedance exceedance;
                            exceedance.row = row;
                            exceedance.col = col;
                            exceedance.counts = counts;
                            exceedance.isSaturated = counts >= saturationLimit ? true : false;

                            exceedanceList.push_back(exceedance);
                        }
                    }
                }

                return exceedanceList;
            }

            //======================================================================================

            void Thresholder::setBadPixelMap(const bool* const bpmPtr) BOOST_NOEXCEPT
            {
                bpmPtr_ = bpmPtr;
            }
        }
    }
}
