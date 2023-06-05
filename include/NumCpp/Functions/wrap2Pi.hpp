#pragma once

#include <cmath>

#include "NumCpp/Core/Constants.hpp"

namespace nc
{
    /**
     * @brief Wrap the input angle to [0, 2*pi]
     *
     * @params: inAngle: in radians
     * @returns Wrapped angle
     */
    template<typename dtype>
    double wrap2Pi(dtype inAngle) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        auto angle = std::fmod(static_cast<double>(inAngle), constants::twoPi);
        if (angle < 0.)
        {
            angle += constants::twoPi;
        }

        return angle;
    }

    /**
     * @brief Wrap the input angle to [0, 2*pi]
     *
     * @params: inAngles: in radians
     * @returns Wrapped angles
     */
    template<typename dtype>
    NdArray<double> wrap2Pi(const NdArray<dtype>& inAngles) noexcept
    {
        NdArray<double> returnArray(inAngles.size());
        stl_algorithms::transform(inAngles.begin(),
                                  inAngles.end(),
                                  returnArray.begin(),
                                  [](const auto angle) noexcept -> double { return wrap2Pi(angle); });
        return returnArray;
    }
} // namespace nc
