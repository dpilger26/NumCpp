/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.4
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
/// Calculate a one-dimensional gaussian filter.
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Filter/Boundaries/Boundary.hpp"
#include "NumCpp/Filter/Filters/Filters1d/convolve1d.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/gaussian1d.hpp"

#include <cmath>
#include <string>

namespace nc
{
    namespace filter
    {
        //============================================================================
        // Method Description:
        ///						Calculate a one-dimensional gaussian filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter1d.html#scipy.ndimage.generic_filter1d
        ///
        /// @param				inImageArray
        /// @param				inSigma: Standard deviation for Gaussian kernel
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> gaussianFilter1d(const NdArray<dtype>& inImageArray, double inSigma,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            if (inSigma <= 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input sigma value must be greater than zero.");
            }

            // calculate the kernel size based off of the input sigma value
            const uint32 MIN_KERNEL_SIZE = 5;
            uint32 kernelSize = std::max(static_cast<uint32>(std::ceil(inSigma * 2.0 * 4.0)), MIN_KERNEL_SIZE); // 4 standard deviations
            if (kernelSize % 2 == 0)
            {
                ++kernelSize; // make sure the kernel is an odd size
            }

            const double kernalHalfSize = static_cast<double>(kernelSize / 2); // integer division

            // calculate the gaussian kernel
            NdArray<double> kernel(1, kernelSize);
            for (double i = 0; i < kernelSize; ++i)
            {
                kernel[static_cast<uint32>(i)] = utils::gaussian1d(i - kernalHalfSize, 0.0, inSigma);
            }

            // normalize the kernel
            kernel /= kernel.sum().item();

            // perform the convolution
            NdArray<dtype> output = convolve1d(inImageArray.template astype<double>(),
                kernel,
                inBoundaryType,
                inConstantValue).template astype<dtype>();

            return output;
        }
    }
}
