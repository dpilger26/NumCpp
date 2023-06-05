#pragma once

#include <cmath>

#include "NumCpp/Core/Constants.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"

namespace nc
{
    /**
     * @brief Wrap the input angle to [-pi, pi]
     *
     * @params: inAngle: in radians
     * @returns Wrapped angle
     */
    template<typename dtype>
    double wrap(dtype inAngle) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        auto angle = std::fmod(static_cast<double>(inAngle) + constants::pi, constants::twoPi);
        if (angle < 0.)
        {
            angle += constants::twoPi;
        }

        return angle - constants::pi;
    }

    /**
     * @brief Wrap the input angle to [-pi, pi]
     *
     * @params: inAngles: in radians
     * @returns Wrapped angles
     */
    template<typename dtype>
    NdArray<double> wrap(const NdArray<dtype>& inAngles) noexcept
    {
        NdArray<double> returnArray(inAngles.size());
        stl_algorithms::transform(inAngles.begin(),
                                  inAngles.end(),
                                  returnArray.begin(),
                                  [](const auto angle) noexcept -> double { return wrap(angle); });
        return returnArray;
    }
} // namespace nc
