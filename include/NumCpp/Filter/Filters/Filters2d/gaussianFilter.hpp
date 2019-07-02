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
/// Image and signal filtering
///
#pragma once

#include"NumCpp/Core/Types.hpp"
#include"NumCpp/Filter/Filters/Filters2d/convolve.hpp"
#include"NumCpp/NdArray/NdArray.hpp"
#include"NumCpp/Utils/gaussian.hpp"

#include<cmath>
#include<iostream>
#include<string>
#include<utility>

namespace nc
{
    //================================================================================
    ///						Image and signal filtering
    namespace filter
    {
        //============================================================================
        // Method Description:
        ///						Calculates a multidimensional gaussian filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter
        ///
        /// @param				inImageArray
        /// @param				inSigma: Standard deviation for Gaussian kernel
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> gaussianFilter(const NdArray<dtype>& inImageArray, double inSigma,
            Boundary inBoundaryType, dtype inConstantValue)
        {
            if (inSigma <= 0)
            {
                std::string errStr = "ERROR: NC::Filters::gaussianFilter: input sigma value must be greater than zero.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
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
            NdArray<double> kernel(kernelSize);
            for (double row = 0; row < kernelSize; ++row)
            {
                for (double col = 0; col < kernelSize; ++col)
                {
                    kernel(static_cast<uint32>(row), static_cast<uint32>(col)) =
                        gaussian(row - kernalHalfSize, col - kernalHalfSize, inSigma);
                }
            }

            // normalize the kernel
            kernel /= kernel.sum().item();

            // perform the convolution
            NdArray<dtype> output = convolve(inImageArray.template astype<double>(),
                kernelSize,
                kernel,
                inBoundaryType,
                inConstantValue).template astype<dtype>();

            return output;
        }
    }
}
