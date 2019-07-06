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
/// Methods for working with NdArrays
///
#pragma once

#include"NumCpp/NdArray/NdArray.hpp"
#include"NumCpp/Utils/interp.hpp"

#include<iostream>
#include<string>
#include<stdexcept>

namespace nc
{
    //============================================================================
    ///						Returns the linear interpolation between two points
    ///
    /// @param      inValue1
    /// @param      inValue2
    /// @param      inPercent
    ///
    /// @return     linear interpolated point
    ///
    template<typename dtype>
    double interp(dtype inValue1, dtype inValue2, double inPercent) noexcept
    {
        return utils::interp(inValue1, inValue2, inPercent);
    }

    //============================================================================
    // Method Description:
    ///						One-dimensional linear interpolation.
    ///
    ///                     Returns the one - dimensional piecewise linear interpolant
    ///                     to a function with given values at discrete data - points.
    ///                     If input arrays are not one dimensional they will be
    ///                     internally flattened.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.interp.html
    ///
    /// @param				inX: The x-coordinates at which to evaluate the interpolated values.
    /// @param              inXp: The x-coordinates of the data points, must be increasing. Otherwise, xp is internally sorted.
    /// @param				inFp: The y-coordinates of the data points, same length as inXp.
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> interp(const NdArray<dtype>& inX, const NdArray<dtype>& inXp, const NdArray<dtype>& inFp)
    {
        // do some error checking first
        if (inXp.size() != inFp.size())
        {
            std::string errStr = "ERROR: interp: inXp and inFp need to be the same size().";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        if (inX.min().item() < inXp.min().item() || inX.max().item() > inXp.max().item())
        {
            std::string errStr = "ERROR: interp: endpoints of inX should be contained within inXp.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        // sort the input inXp and inFp data
        NdArray<uint32> sortedXpIdxs = argsort(inXp);
        NdArray<dtype> sortedXp(1, inFp.size());
        NdArray<dtype> sortedFp(1, inFp.size());
        uint32 counter = 0;
        for (auto sortedXpIdx : sortedXpIdxs)
        {
            sortedXp[counter] = inXp[sortedXpIdx];
            sortedFp[counter++] = inFp[sortedXpIdx];
        }

        // sort the input inX array
        NdArray<dtype> sortedX = sort(inX);

        NdArray<dtype> returnArray(1, inX.size());

        uint32 currXpIdx = 0;
        uint32 currXidx = 0;
        while (currXidx < sortedX.size())
        {
            if (sortedXp[currXpIdx] <= sortedX[currXidx] && sortedX[currXidx] <= sortedXp[currXpIdx + 1])
            {
                const double percent = static_cast<double>(sortedX[currXidx] - sortedXp[currXpIdx]) /
                    static_cast<double>(sortedXp[currXpIdx + 1] - sortedXp[currXpIdx]);
                returnArray[currXidx++] = utils::interp(sortedFp[currXpIdx], sortedFp[currXpIdx + 1], percent);
            }
            else
            {
                ++currXpIdx;
            }
        }

        return returnArray;
    }
}
