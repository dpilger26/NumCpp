/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
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
/// Coordinate Transforms
///
#pragma once

#include <cmath>
#include <iostream>

#include "NumCpp/Coordinates/Euler.h"
#include "NumCpp/Coordinates/Orientation.h"
#include "NumCpp/Coordinates/ReferenceFrames/AzEl.hpp"
#include "NumCpp/Coordinates/ReferenceFrames/Cartesian.hpp"
#include "NumCpp/Core/Constants.hpp"
#include "NumCpp/Functions/sign.hpp"
#include "NumCpp/Functions/wrap.hpp"
#include "NumCpp/Functions/wrap2pi.hpp"
#include "NumCpp/Rotations/Quaternion.hpp"
#include "NumCpp/Utils/sqr.hpp"
#include "NumCpp/Vector/Vec3.hpp"

namespace nc::coordinates::transforms
{
    /**
     * @brief Converts the LLA coordinates to Az El with geodedic up
     *        https://geospace-code.github.io/matmap3d/enu2aer.html
     *
     * @param target: the target of interest
     * @param referencePoint: the referencePoint
     * @returns AzEl
     */
    [[nodiscard]] inline reference_frames::AzEl LLAtoAzElGeodetic(const reference_frames::LLA& target,
                                                                  const reference_frames::LLA& referencePoint) noexcept
    {
        return ECEFtoAzElGeodetic(LLAtoECEF(target), referencePoint);
    }

    /**
     * @brief Converts the LLA coordinates to Az El with geodedic up
     *        https://geospace-code.github.io/matmap3d/enu2aer.html
     *
     * @param target: the target of interest
     * @param referencePoint: the referencePoint
     * @returns AzEl
     */
    [[nodiscard]] inline reference_frames::AzEl LLAtoAzElGeodetic(const reference_frames::LLA&  target,
                                                                  const reference_frames::ECEF& referencePoint) noexcept
    {
    }
} // namespace nc::coordinates::transforms
