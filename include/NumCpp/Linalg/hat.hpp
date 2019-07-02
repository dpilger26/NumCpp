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
/// vector hat operator
///
#pragma once

#include"NumCpp/Methods/Methods.hpp"
#include"NumCpp/NdArray/NdArray.hpp"
#include"NumCpp/Vector/Vec3.hpp"

#include<iostream>
#include<stdexcept>
#include<string>

namespace nc
{
    namespace linalg
    {
        //============================================================================
        // Method Description:
        ///						vector hat operator
        ///
        /// @param			inX
        /// @param			inY
        /// @param			inZ
        /// @return
        ///				3x3 NdArray
        ///
        template<typename dtype>
        NdArray<dtype> hat(dtype inX, dtype inY, dtype inZ) noexcept
        {
            NdArray<dtype> returnArray(3);
            returnArray(0, 0) = 0.0;
            returnArray(0, 1) = -inZ;
            returnArray(0, 2) = inY;
            returnArray(1, 0) = inZ;
            returnArray(1, 1) = 0.0;
            returnArray(1, 2) = -inX;
            returnArray(2, 0) = -inY;
            returnArray(2, 1) = inX;
            returnArray(2, 2) = 0.0;

            return returnArray;
        }

        //============================================================================
        // Method Description:
        ///						vector hat operator
        ///
        /// @param
        ///				inVec (3x1, or 1x3 cartesian vector)
        /// @return
        ///				3x3 NdArray
        ///
        template<typename dtype>
        NdArray<dtype> hat(const NdArray<dtype>& inVec)
        {
            if (inVec.size() != 3)
            {
                std::string errStr = "ERROR: linalg::hat: input vector must be a length 3 cartesian vector.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            return hat(inVec[0], inVec[1], inVec[2]);
        }

        //============================================================================
        // Method Description:
        ///						vector hat operator
        ///
        /// @param
        ///				inVec
        /// @return
        ///				3x3 NdArray
        ///
        inline NdArray<double> hat(const Vec3& inVec)
        {
            return hat(inVec.x, inVec.y, inVec.z);
        }
    }
}
