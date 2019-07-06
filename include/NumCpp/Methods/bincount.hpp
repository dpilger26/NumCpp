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

#include"NumCpp/Core/DtypeInfo.hpp"
#include"NumCpp/Core/Types.hpp"
#include"NumCpp/NdArray/NdArray.hpp"

#include<algorithm>
#include<iostream>
#include<string>
#include<stdexcept>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Count number of occurrences of each value in array of non-negative ints.
    ///						Negative values will be counted in the zero bin.
    ///
    ///						The number of bins(of size 1) is one larger than the largest value in x.
    ///						If minlength is specified, there will be at least this number of bins in
    ///						the output array(though it will be longer if necessary, depending on the
    ///						contents of x).Each bin gives the number of occurrences of its index value
    ///						in x.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.bincount.html
    ///
    /// @param				inArray
    /// @param				inMinLength
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> bincount(const NdArray<dtype>& inArray, uint16 inMinLength)
    {
        // only works with integer input types
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: bincount: can only use with integer types.");

        dtype maxValue = inArray.max().item();
        if (maxValue < 0)
        {
            // no positive values so just return an empty array
            return NdArray<dtype>(0);
        }

        if (maxValue + 1 > DtypeInfo<dtype>::max())
        {
            std::string errStr = "Error: bincount: array values too large, will result in gigantic array that will take up alot of memory...";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        const uint16 outArraySize = std::max(static_cast<uint16>(maxValue + 1), inMinLength);
        NdArray<dtype> clippedArray = inArray.clip(0, maxValue);

        NdArray<dtype> outArray(1, outArraySize);
        outArray.zeros();
        std::for_each(clippedArray.cbegin(), clippedArray.cend(),
            [&outArray](dtype value) noexcept -> void
            { ++outArray[value]; });

        return outArray;
    }

        //============================================================================
    // Method Description:
    ///						Count number of occurrences of each value in array of non-negative ints.
    ///						Negative values will be counted in the zero bin.
    ///
    ///						The number of bins(of size 1) is one larger than the largest value in x.
    ///						If minlength is specified, there will be at least this number of bins in
    ///						the output array(though it will be longer if necessary, depending on the
    ///						contents of x).Each bin gives the number of occurrences of its index value
    ///						in x.If weights is specified the input array is weighted by it, i.e. if a
    ///						value n is found at position i, out[n] += weight[i] instead of out[n] += 1.
    ///						Weights array shall be of the same shape as inArray.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.bincount.html
    ///
    /// @param				inArray
    /// @param				inWeights
    /// @param				inMinLength
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> bincount(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength)
    {
        // only works with integer input types
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: bincount: can only use with integer types.");

        if (inArray.shape() != inWeights.shape())
        {
            std::string errStr = "ERROR: bincount: weights array must be the same shape as the input array.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        dtype maxValue = inArray.max().item();
        if (maxValue < 0)
        {
            // no positive values so just return an empty array
            return NdArray<dtype>(0);
        }

        if (maxValue + 1 > DtypeInfo<dtype>::max())
        {
            std::string errStr = "Error: bincount: array values too large, will result in gigantic array that will take up alot of memory...";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        const uint16 outArraySize = std::max(static_cast<uint16>(maxValue + 1), inMinLength);
        NdArray<dtype> clippedArray = inArray.clip(0, maxValue);

        NdArray<dtype> outArray(1, outArraySize);
        outArray.zeros();
        uint32 counter = 0;
        std::for_each(clippedArray.cbegin(), clippedArray.cend(),
            [&outArray, &inWeights, &counter](dtype value) noexcept -> void
            { outArray[value] += inWeights[counter++]; });

        return outArray;
    }
}
