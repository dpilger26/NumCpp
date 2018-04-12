// Copyright 2018 David Pilger
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files(the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, 
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
// permit persons to whom the Software is furnished to do so, subject to the following 
// conditions :
//
// The above copyright notice and this permission notice shall be included in all copies 
// or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
// DEALINGS IN THE SOFTWARE.

#pragma once

#include"NdArray.hpp"
#include"Types.hpp"

#include<boost/filesystem.hpp>

#include<algorithm>
#include<cmath>
#include<fstream>
#include<initializer_list>
#include<iostream>
#include<set>
#include<stdexcept>
#include<string>
#include<utility>
#include<vector>

namespace NumC
{
    //============================================================================
    // Method Description: 
    //						Calculate the absolute value.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype abs(dtype inValue)
    {
        return std::abs(inValue);
    }

    //============================================================================
    // Method Description: 
    //						Calculate the absolute value element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> abs(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::abs(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Add arguments element-wise.
    //		
    // Inputs:
    //				NdArray1
    //				NdArray2
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> add(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1.astype<dtypeOut>() + inArray2.astype<dtypeOut>());
    }

    //============================================================================
    // Method Description: 
    //						Return the length of the first dimension of the input array.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				length uint16
    //
    template<typename dtype>
    inline uint32 alen(const NdArray<dtype>& inArray)
    {
        return inArray.shape().rows;
    }

    //============================================================================
    // Method Description: 
    //						Test whether all array elements along a given axis evaluate to True.
    //		
    // Inputs:
    //				NdArray
    //				Axis
    // Outputs:
    //				bool
    //
    template<typename dtype>
    inline NdArray<bool> all(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.all(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Returns True if two arrays are element-wise equal within a tolerance.
    //						inTolerance must be a positive number
    //		
    // Inputs:
    //				NdArray1
    //				NdArray2
    //				(Optional) tolerance
    // Outputs:
    //				bool
    //
    template<typename dtype>
    inline bool allclose(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, double inTolerance = 1e-5)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: allclose: input array dimensions are not consistant.");
        }

        for (uint32 i = 0; i < inArray1.size(); ++i)
        {
            if (std::abs(inArray1[i] - inArray2[i]) > inTolerance)
            {
                return false;
            }
        }

        return true;
    }

    //============================================================================
    // Method Description: 
    //						Return the maximum of an array or maximum along an axis.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    // Outputs:
    //				max value
    //
    template<typename dtype>
    inline NdArray<dtype> amax(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.max(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Return the minimum of an array or minimum along an axis.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    // Outputs:
    //				min value
    //
    template<typename dtype>
    inline NdArray<dtype> amin(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.min(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Test whether any array element along a given axis evaluates to True.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> any(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.any(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Append values to the end of an array.
    //		
    // Inputs:
    //				NdArray
    //				NdArray append values
    //				(Optional) axis -	The axis along which values are appended. 
    //									If axis is not given, both inArray and inAppendValues 
    //									are flattened before use.	
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> append(const NdArray<dtype>& inArray, const NdArray<dtype>& inAppendValues, Axis::Type inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                NdArray<dtype> returnArray(1, inArray.size() + inAppendValues.size());
                std::copy(inArray.cbegin(), inArray.cend(), returnArray.begin());
                std::copy(inAppendValues.cbegin(), inAppendValues.cend(), returnArray.begin() + inArray.size());

                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                Shape inShape = inArray.shape();
                Shape appendShape = inAppendValues.shape();
                if (inShape.cols != appendShape.cols)
                {
                    throw std::invalid_argument("ERROR: append: all the input array dimensions except for the concatenation axis must match exactly");
                }

                NdArray<dtype> returnArray(inShape.rows + appendShape.rows, inShape.cols);
                std::copy(inArray.cbegin(), inArray.cend(), returnArray.begin());
                std::copy(inAppendValues.cbegin(), inAppendValues.cend(), returnArray.begin() + inArray.size());

                return std::move(returnArray);
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();
                Shape appendShape = inAppendValues.shape();
                if (inShape.rows != appendShape.rows)
                {
                    throw std::invalid_argument("ERROR: append: all the input array dimensions except for the concatenation axis must match exactly");
                }

                NdArray<dtype> returnArray(inShape.rows, inShape.cols + appendShape.cols);
                for (uint32 row = 0; row < returnArray.shape().rows; ++row)
                {
                    std::copy(inArray.cbegin(row), inArray.cend(row), returnArray.begin(row));
                    std::copy(inAppendValues.cbegin(row), inAppendValues.cend(row), returnArray.begin(row) + inShape.cols);
                }

                return std::move(returnArray);
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<dtype>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Return evenly spaced values within a given interval.
    //
    //						Values are generated within the half - open interval[start, stop)
    //						(in other words, the interval including start but excluding stop).
    //						For integer arguments the function is equivalent to the Python built - in
    //						range function, but returns an ndarray rather than a list.
    //
    //						When using a non - integer step, such as 0.1, the results will often 
    //						not be consistent.It is better to use linspace for these cases.
    //		
    // Inputs:
    //				start value,
    //				stop value, 
    //				(Optional) step value, defaults to 1
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> arange(dtype inStart, dtype inStop, dtype inStep = 1)
    {
        if (inStep > 0 && inStop < inStart)
        {
            throw std::invalid_argument("ERROR: arange: stop value must be larger than the start value for positive step.");
        }

        if (inStep < 0 && inStop > inStart)
        {
            throw std::invalid_argument("ERROR: arange: start value must be larger than the stop value for negative step.");
        }

        std::vector<dtype> values;

        dtype theValue = inStart;

        if (inStep > 0)
        {
            while (theValue < inStop)
            {
                values.push_back(theValue);
                theValue += static_cast<dtype>(inStep);
            }
        }
        else
        {
            while (theValue > inStop)
            {
                values.push_back(theValue);
                theValue += static_cast<dtype>(inStep);
            }
        }

        return std::move(NdArray<dtype>(values));
    }

    //============================================================================
    // Method Description: 
    //						Return evenly spaced values within a given interval.
    //
    //						Values are generated within the half - open interval[start, stop)
    //						(in other words, the interval including start but excluding stop).
    //						For integer arguments the function is equivalent to the Python built - in
    //						range function, but returns an ndarray rather than a list.
    //
    //						When using a non - integer step, such as 0.1, the results will often 
    //						not be consistent.It is better to use linspace for these cases.
    //		
    // Inputs:
    //				stop value, start is 0 and step is 1
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> arange(dtype inStop)
    {
        if (inStop <= 0)
        {
            throw std::invalid_argument("ERROR: arange: stop value must ge greater than 0.");
        }

        return std::move(arange<dtype>(0, inStop, 1));
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse cosine
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double arccos(dtype inValue)
    {
        return std::acos(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse cosine, element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> arccos(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::acos(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse hyperbolic cosine.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double arccosh(dtype inValue)
    {
        return std::acosh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse hyperbolic cosine, element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> arccosh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::acosh(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse sine.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double arcsin(dtype inValue)
    {
        return std::asin(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse sine, element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> arcsin(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::asin(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse hyperbolic sine.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double arcsinh(dtype inValue)
    {
        return std::asinh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse hyperbolic sine, element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> arcsinh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::asinh(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse tangent.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double arctan(dtype inValue)
    {
        return std::atan(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse tangent, element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> arctan(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::atan(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse tangent.
    //		
    // Inputs:
    //				Y
    //				x
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double arctan2(dtype inY, dtype inX)
    {
        return std::atan2(static_cast<double>(inY), static_cast<double>(inX));
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse tangent, element-wise.
    //		
    // Inputs:
    //				NdArray y
    //				NdArray x
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> arctan2(const NdArray<dtype>& inY, const NdArray<dtype>& inX)
    {
        if (inX.shape() != inY.shape())
        {
            throw std::invalid_argument("Error: arctan2: input array shapes are not consistant.");
        }

        NdArray<double> returnArray(inY.shape());
        std::transform(inY.cbegin(), inY.cend(), inX.cbegin(), returnArray.begin(),
            [](dtype y, dtype x) { return std::atan2(static_cast<double>(y), static_cast<double>(x)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse hyperbolic tangent.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double arctanh(dtype inValue)
    {
        return std::atanh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric inverse hyperbolic tangent, element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> arctanh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::atanh(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Returns the indices of the maximum values along an axis.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<uint32> argmax(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.argmax(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Returns the indices of the minimum values along an axis.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<uint32> argmin(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.argmin(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Returns the indices that would sort an array.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<uint32> argsort(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.argsort(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Returns the indices that would sort an array.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<uint32> argwhere(const NdArray<dtype>& inArray)
    {
        return std::move(inArray.nonzero());
    }

    //============================================================================
    // Method Description: 
    //						Evenly round to the given number of decimals.
    //		
    // Inputs:
    //				value
    //				(Optional) decimals, default = 0
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype around(dtype inValue, uint8 inNumDecimals = 0)
    {
        NdArray<dtype> value = { inValue };
        return value.round(inNumDecimals).item();
    }

    //============================================================================
    // Method Description: 
    //						Evenly round to the given number of decimals.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) decimals, default = 0
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> around(const NdArray<dtype>& inArray, uint8 inNumDecimals = 0)
    {
        return std::move(inArray.round(inNumDecimals));
    }

    //============================================================================
    // Method Description: 
    //						True if two arrays have the same shape and elements, False otherwise.
    //		
    // Inputs:
    //				NdArray1,
    //				NdArray2
    //				
    // Outputs:
    //				bool
    //
    template<typename dtype>
    inline bool array_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            return false;
        }

        NdArray<bool> equal = inArray1 == inArray2;
        return equal.all().item();
    }

    //============================================================================
    // Method Description: 
    //						Returns True if input arrays are shape consistent and all elements equal.
    //
    //						Shape consistent means they are either the same shape, or one input array 
    //						can be broadcasted to create the same shape as the other one.
    //		
    // Inputs:
    //				NdArray1,
    //				NdArray2
    //				
    // Outputs:
    //				bool
    //
    template<typename dtype>
    inline bool array_equiv(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.size() != inArray2.size())
        {
            return false;
        }

        for (uint32 i = 0; i < inArray1.size(); ++i)
        {
            if (inArray1[i] != inArray2[i])
            {
                return false;
            }
        }

        return true;
    }

    //============================================================================
    // Method Description: 
    //						Convert the vector to an array.
    //		
    // Inputs:
    //				std::vector
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> asarray(const std::vector<dtype>& inVector)
    {
        return std::move(NdArray<dtype>(inVector));
    }

    //============================================================================
    // Method Description: 
    //						Convert the list initializer to an array.
    //						eg: NdArray<int> myArray = NumC::asarray<int>({1,2,3});
    //		
    // Inputs:
    //				std::vector
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> asarray(std::initializer_list<dtype>& inList)
    {
        return std::move(NdArray<dtype>(inList));
    }

    //============================================================================
    // Method Description: 
    //						Returns a copy of the array, cast to a specified type.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> astype(const NdArray<dtype> inArray)
    {
        return std::move(inArray.astype<dtypeOut>());
    }

    //============================================================================
    // Method Description: 
    //						Compute the average along the specified axis.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> average(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.mean(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Compute the weighted average along the specified axis.
    //		
    // Inputs:
    //				NdArray
    //				NdArray of weights, otherwise all weights = 1
    //				(Optional) axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> average(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, Axis::Type inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inWeights.shape() != inArray.shape())
                {
                    throw std::invalid_argument("ERROR: input array and weight values are not consistant.");
                }

                NdArray<double> weightedArray(inArray.shape());
                std::transform(inArray.cbegin(), inArray.cend(), inWeights.cbegin(), weightedArray.begin(), std::multiplies<double>());

                double sum = static_cast<double>(std::accumulate(weightedArray.begin(), weightedArray.end(), 0.0));
                NdArray<double> returnArray = { sum /= inWeights.sum<double>().item() };

                return std::move(returnArray);
            }
            case Axis::COL:
            {
                Shape arrayShape = inArray.shape();
                if (inWeights.size() != arrayShape.cols)
                {
                    throw std::invalid_argument("ERROR: input array and weights value are not consistant.");
                }

                double weightSum = inWeights.sum<double>().item();
                NdArray<double> returnArray(1, arrayShape.rows);
                for (uint32 row = 0; row < arrayShape.rows; ++row)
                {
                    NdArray<double> weightedArray(1, arrayShape.cols);
                    std::transform(inArray.cbegin(row), inArray.cend(row), inWeights.cbegin(), weightedArray.begin(), std::multiplies<double>());

                    double sum = static_cast<double>(std::accumulate(weightedArray.begin(), weightedArray.end(), 0.0));
                    returnArray(0, row) = sum / weightSum;
                }

                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                if (inWeights.size() != inArray.shape().rows)
                {
                    throw std::invalid_argument("ERROR: input array and weight values are not consistant.");
                }

                NdArray<dtype> transposedArray = inArray.transpose();

                Shape transShape = transposedArray.shape();
                double weightSum = inWeights.sum<double>().item();
                NdArray<double> returnArray(1, transShape.rows);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    NdArray<double> weightedArray(1, transShape.cols);
                    std::transform(transposedArray.cbegin(row), transposedArray.cend(row), inWeights.cbegin(), weightedArray.begin(), std::multiplies<double>());

                    double sum = static_cast<double>(std::accumulate(weightedArray.begin(), weightedArray.end(), 0.0));
                    returnArray(0, row) = sum / weightSum;
                }

                return std::move(returnArray);
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<double>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Count number of occurrences of each value in array of non-negative ints.
    //						Negative values will be counted in the zero bin.
    //
    //						The number of bins(of size 1) is one larger than the largest value in x.
    //						If minlength is specified, there will be at least this number of bins in 
    //						the output array(though it will be longer if necessary, depending on the 
    //						contents of x).Each bin gives the number of occurrences of its index value 
    //						in x.
    //		
    // Inputs:
    //				NdArray
    //				min bin length
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> bincount(const NdArray<dtype>& inArray, uint16 inMinLength = 0)
    {
        // only works with integer input types
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: bincount: can only use with integer types.");

        dtype maxValue = inArray.max().item();
        if (maxValue < 0)
        {
            // no positive values so just return an empty array
            return std::move(NdArray<dtype>(0));
        }

        if (maxValue + 1 > DtypeInfo<dtype>::max())
        {
            throw std::runtime_error("Error: bincount: array values too large, will result in gigantic array that will take up alot of memory...");
        }

        uint16 outArraySize = std::max(static_cast<uint16>(maxValue + 1), inMinLength);
        NdArray<dtype> clippedArray = inArray.clip(0, maxValue);

        NdArray<dtype> outArray(1, outArraySize);
        outArray.zeros();
        for (uint32 i = 0; i < inArray.size(); ++i)
        {
            ++outArray[clippedArray[i]];
        }

        return std::move(outArray);
    }

    //============================================================================
    // Method Description: 
    //						Count number of occurrences of each value in array of non-negative ints.
    //						Negative values will be counted in the zero bin.
    //
    //						The number of bins(of size 1) is one larger than the largest value in x.
    //						If minlength is specified, there will be at least this number of bins in 
    //						the output array(though it will be longer if necessary, depending on the 
    //						contents of x).Each bin gives the number of occurrences of its index value 
    //						in x.If weights is specified the input array is weighted by it, i.e. if a 
    //						value n is found at position i, out[n] += weight[i] instead of out[n] += 1.
    //						Weights array shall be of the same shape as inArray.
    //		
    // Inputs:
    //				NdArray
    //				NdArray weights
    //				min bin length
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> bincount(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0)
    {
        // only works with integer input types
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: bincount: can only use with integer types.");

        if (inArray.shape() != inWeights.shape())
        {
            throw std::invalid_argument("ERROR: bincount: weights array must be the same shape as the input array.");
        }

        dtype maxValue = inArray.max().item();
        if (maxValue < 0)
        {
            // no positive values so just return an empty array
            return std::move(NdArray<dtype>(0));
        }

        if (maxValue + 1 > DtypeInfo<dtype>::max())
        {
            throw std::runtime_error("Error: bincount: array values too large, will result in gigantic array that will take up alot of memory...");
        }

        uint16 outArraySize = std::max(static_cast<uint16>(maxValue + 1), inMinLength);
        NdArray<dtype> clippedArray = inArray.clip(0, maxValue);

        NdArray<dtype> outArray(1, outArraySize);
        outArray.zeros();
        for (uint32 i = 0; i < inArray.size(); ++i)
        {
            outArray[clippedArray[i]] += inWeights[i];
        }

        return std::move(outArray);
    }

    //============================================================================
    // Method Description: 
    //						Compute the bit-wise AND of two arrays element-wise.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> bitwise_and(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1 & inArray2);
    }

    //============================================================================
    // Method Description: 
    //						Compute the bit-wise NOT the input array element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> bitwise_not(const NdArray<dtype>& inArray)
    {
        return std::move(~inArray);
    }

    //============================================================================
    // Method Description: 
    //						Compute the bit-wise OR of two arrays element-wise.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> bitwise_or(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1 | inArray2);
    }

    //============================================================================
    // Method Description: 
    //						Compute the bit-wise XOR of two arrays element-wise.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> bitwise_xor(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1 ^ inArray2);
    }

    //============================================================================
    // Method Description: 
    //						Return a new array with the bytes of the array elements
    //						swapped.
    //		
    // Inputs:
    //				NdArray 
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> byteswap(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray);
        returnArray.byteswap();
        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the cube-root of an array. Not super usefull 
    //						if not using a floating point type
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double cbrt(dtype inValue)
    {
        return std::cbrt(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Return the cube-root of an array, element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> cbrt(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::cbrt(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the ceiling of the input.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype ceil(dtype inValue)
    {
        return std::ceil(inValue);
    }

    //============================================================================
    // Method Description: 
    //						Return the ceiling of the input, element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> ceil(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::ceil(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Clip (limit) the value.
    //		
    // Inputs:
    //				value
    //				min Value
    //				max Value
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline dtype clip(dtype inValue, dtype inMinValue, dtype inMaxValue)
    {
        NdArray<dtype> value = { inValue };
        return std::move(value.clip(inMinValue, inMaxValue).item());
    }

    //============================================================================
    // Method Description: 
    //						Clip (limit) the values in an array.
    //		
    // Inputs:
    //				NdArray
    //				min Value
    //				max Value
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> clip(const NdArray<dtype>& inArray, dtype inMinValue, dtype inMaxValue)
    {
        return std::move(inArray.clip(inMinValue, inMaxValue));
    }

    //============================================================================
    // Method Description: 
    //						Stack 1-D arrays as columns into a 2-D array.
    //		
    // Inputs:
    //				{list} of arrays to stack
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> column_stack(const std::initializer_list<NdArray<dtype> >& inArrayList)
    {
        // first loop through to calculate the final size of the array
        typename std::initializer_list<NdArray<dtype> >::iterator iter;
        Shape finalShape;
        for (iter = inArrayList.begin(); iter < inArrayList.end(); ++iter)
        {
            if (finalShape.isnull())
            {
                finalShape = iter->shape();
            }
            else if (iter->shape().rows != finalShape.rows)
            {
                throw std::invalid_argument("ERROR: column_stack: input arrays must have the same number of rows.");
            }
            else
            {
                finalShape.cols += iter->shape().cols;
            }
        }

        // now that we know the final size, contruct the output array
        NdArray<dtype> returnArray(finalShape);
        uint32 colStart = 0;
        for (iter = inArrayList.begin(); iter < inArrayList.end(); ++iter)
        {
            Shape theShape = iter->shape();
            for (uint32 row = 0; row < theShape.rows; ++row)
            {
                for (uint32 col = 0; col < theShape.cols; ++col)
                {
                    returnArray(row, colStart + col) = iter->operator()(row, col);
                }
            }
            colStart += theShape.cols;
        }

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Join a sequence of arrays along an existing axis.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				(Optional) Axis (Default NONE)
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> concatenate(const std::initializer_list<NdArray<dtype> >& inArrayList, Axis::Type inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                uint32 finalSize = 0;
                std::initializer_list<NdArray<dtype> >::iterator iter;
                for (iter = inArrayList.begin(); iter < inArrayList.end(); ++iter)
                {
                    finalSize += iter->size();
                }

                NdArray<dtype> returnArray(1, finalSize);
                uint32 offset = 0;
                for (iter = inArrayList.begin(); iter < inArrayList.end(); ++iter)
                {
                    std::copy(iter->cbegin(), iter->cend(), returnArray.begin() + offset);
                    offset += iter->size();
                }

                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                return std::move(row_stack(inArrayList));
            }
            case Axis::COL:
            {
                return std::move(column_stack(inArrayList));
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<dtype>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						returns whether or not a value is included the array
    //		
    // Inputs:
    //				NdArray 
    //				value
    //				(Optional) axis
    // Outputs:
    //				bool
    //
    template<typename dtype>
    inline NdArray<bool> contains(const NdArray<dtype>& inArray, dtype inValue, Axis::Type inAxis = Axis::NONE)
    {
        return inArray.contains(inValue, inAxis);
    }

    //============================================================================
    // Method Description: 
    //						Return an array copy of the given object.
    //		
    // Inputs:
    //				NdArray 
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> copy(const NdArray<dtype>& inArray)
    {
        return std::move(NdArray<dtype>(inArray));
    }

    //============================================================================
    // Method Description: 
    //						Change the sign of x1 to that of x2, element-wise.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> copySign(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: copysign: input arrays are not consistant.");
        }

        NdArray<dtype> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) { return inValue2 < 0 ? std::abs(inValue1) * -1 : std::abs(inValue1); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Copies values from one array to another
    //		
    // Inputs:
    //				NdArray destination
    //				NdArray source
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline void copyto(NdArray<dtype>& inDestArray, const NdArray<dtype>& inSrcArray)
    {
        inDestArray = inSrcArray;
    }

    //============================================================================
    // Method Description: 
    //						Cosine .
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double cos(dtype inValue)
    {
        return std::cos(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Cosine element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> cos(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::cos(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Hyperbolic Cosine.
    //		
    // Inputs:
    //				Value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double cosh(dtype inValue)
    {
        return std::cosh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Hyperbolic Cosine element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> cosh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::cosh(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Counts the number of non-zero values in the array.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<uint32> count_nonzero(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                NdArray<uint32> count = { inArray.nonzero().size() };
                return count;
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();

                NdArray<uint32> returnArray(1, inShape.rows);
                returnArray.zeros();
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (inArray(row, col) != static_cast<dtype>(0))
                        {
                            ++returnArray(0, row);
                        }
                    }
                }

                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                Shape inShape = inArray.shape();

                NdArray<uint32> returnArray(1, inShape.cols);
                returnArray.zeros();
                for (uint32 col = 0; col < inShape.cols; ++col)
                {
                    for (uint32 row = 0; row < inShape.rows; ++row)
                    {
                        if (inArray(row, col) != static_cast<dtype>(0))
                        {
                            ++returnArray(0, col);
                        }
                    }
                }

                return std::move(returnArray);
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<uint32>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Return the cross product of two (arrays of) vectors.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				(Optional) Axis - default = row
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> cross(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, Axis::Type inAxis = Axis::NONE)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: cross: the input array dimensions are not consistant.");
        }

        switch (inAxis)
        {
            case Axis::NONE:
            {
                uint32 arraySize = inArray1.size();
                if (arraySize != inArray2.size() || arraySize < 2 || arraySize > 3)
                {
                    throw std::invalid_argument("ERROR: cross: incompatible dimensions for cross product (dimension must be 2 or 3)");
                }

                NdArray<dtype> in1 = inArray1.flatten();
                NdArray<dtype> in2 = inArray2.flatten();

                switch (arraySize)
                {
                    case 2:
                    {
                        NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(in1[0]) * static_cast<dtypeOut>(in2[1])
                            - static_cast<dtypeOut>(in1[1]) * static_cast<dtypeOut>(in2[0]) };
                        return std::move(returnArray);
                    }
                    case 3:
                    {
                        dtypeOut i = static_cast<dtypeOut>(in1[1]) * static_cast<dtypeOut>(in2[2])
                            - static_cast<dtypeOut>(in1[2]) * static_cast<dtypeOut>(in2[1]);
                        dtypeOut j = -(static_cast<dtypeOut>(in1[0]) * static_cast<dtypeOut>(in2[2])
                            - static_cast<dtypeOut>(in1[2]) * static_cast<dtypeOut>(in2[0]));
                        dtypeOut k = (static_cast<dtypeOut>(in1[0]) * static_cast<dtypeOut>(in2[1])
                            - static_cast<dtypeOut>(in1[1]) * static_cast<dtypeOut>(in2[0]));

                        NdArray<dtypeOut> returnArray = { i, j, k };
                        return std::move(returnArray);
                    }
                    default:
                    {
                        // this isn't actually possible, just putting this here to get rid
                        // of the compiler warning.
                        return std::move(NdArray<dtypeOut>(0));
                    }
                }
            }
            case Axis::ROW:
            {
                Shape arrayShape = inArray1.shape();
                if (arrayShape != inArray2.shape() || arrayShape.rows < 2 || arrayShape.rows > 3)
                {
                    throw std::invalid_argument("ERROR: cross: incompatible dimensions for cross product (dimension must be 2 or 3)");
                }

                Shape returnArrayShape;
                if (arrayShape.rows == 2)
                {
                    returnArrayShape = Shape(1, arrayShape.cols);
                }
                else
                {
                    returnArrayShape = Shape(3, arrayShape.cols);
                }

                NdArray<dtypeOut> returnArray(returnArrayShape);
                for (uint32 col = 0; col < arrayShape.cols; ++col)
                {
                    int32 theCol = static_cast<int32>(col);
                    NdArray<dtype> vec1 = inArray1({ 0, static_cast<int32>(arrayShape.rows) }, { theCol, theCol + 1 });
                    NdArray<dtype> vec2 = inArray2({ 0, static_cast<int32>(arrayShape.rows) }, { theCol, theCol + 1 });
                    NdArray<dtypeOut> vecCross = cross<dtype, dtypeOut>(vec1, vec2, Axis::NONE);

                    returnArray.put({ 0, static_cast<int32>(returnArrayShape.rows) }, { theCol, theCol + 1 }, vecCross);
                }

                return std::move(returnArray);
            }
            case Axis::COL:
            {
                Shape arrayShape = inArray1.shape();
                if (arrayShape != inArray2.shape() || arrayShape.cols < 2 || arrayShape.cols > 3)
                {
                    throw std::invalid_argument("ERROR: cross: incompatible dimensions for cross product (dimension must be 2 or 3)");
                }

                Shape returnArrayShape;
                if (arrayShape.cols == 2)
                {
                    returnArrayShape = Shape(arrayShape.rows, 1);
                }
                else
                {
                    returnArrayShape = Shape(arrayShape.rows, 3);
                }

                NdArray<dtypeOut> returnArray(returnArrayShape);
                for (uint32 row = 0; row < arrayShape.rows; ++row)
                {
                    int32 theRow = static_cast<int32>(row);
                    NdArray<dtype> vec1 = inArray1({ theRow, theRow + 1 }, { 0, static_cast<int32>(arrayShape.cols) });
                    NdArray<dtype> vec2 = inArray2({ theRow, theRow + 1 }, { 0, static_cast<int32>(arrayShape.cols) });
                    NdArray<dtypeOut> vecCross = cross<dtype, dtypeOut>(vec1, vec2, Axis::NONE);

                    returnArray.put({ theRow, theRow + 1 }, { 0, static_cast<int32>(returnArrayShape.cols) }, vecCross);
                }

                return std::move(returnArray);
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<dtypeOut>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Cubes the elements of the array
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> cube(const NdArray<dtype>& inArray)
    {
        NdArray<dtypeOut> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return Utils::cube(static_cast<dtypeOut>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the cumulative product of elements along a given axis.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> cumprod(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.cumprod<dtypeOut>(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Return the cumulative sum of the elements along a given axis.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> cumsum(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.cumsum<dtypeOut>(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Convert angles from degrees to radians.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double deg2rad(dtype inValue)
    {
        return inValue * Constants::pi / 180.0;
    }

    //============================================================================
    // Method Description: 
    //						Convert angles from degrees to radians.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> deg2rad(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return deg2rad(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return a new array with sub-arrays along an axis deleted.
    //		
    // Inputs:
    //				NdArray
    //				NdArray indices to delete
    //				(Optional) Axis, if none the indices will be applied to the flattened array
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, const NdArray<uint32>& inArrayIdxs, Axis::Type inAxis = Axis::NONE)
    {
        // make sure that the indices are unique first
        NdArray<uint32> indices = unique(inArrayIdxs);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::vector<dtype> values;
                for (uint32 i = 0; i < inArray.size(); ++i)
                {
                    if (indices.contains(i).item())
                    {
                        continue;
                    }

                    values.push_back(inArray[i]);
                }

                return std::move(NdArray<dtype>(values));
            }
            case Axis::ROW:
            {
                Shape inShape = inArray.shape();
                if (indices.max().item() >= inShape.rows)
                {
                    throw std::runtime_error("ERROR: deleteIndices: input index value is greater than the number of rows in the array.");
                }

                uint32 numNewRows = inShape.rows - indices.size();
                NdArray<dtype> returnArray(numNewRows, inShape.cols);

                uint32 rowCounter = 0;
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    if (indices.contains(row).item())
                    {
                        continue;
                    }

                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        returnArray(rowCounter, col) = inArray(row, col);
                    }
                    ++rowCounter;
                }

                return std::move(returnArray);
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();
                if (indices.max().item() >= inShape.cols)
                {
                    throw std::runtime_error("ERROR: deleteIndices: input index value is greater than the number of cols in the array.");
                }

                uint32 numNewCols = inShape.cols - indices.size();
                NdArray<dtype> returnArray(inShape.rows, numNewCols);

                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    uint32 colCounter = 0;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (indices.contains(col).item())
                        {
                            continue;
                        }

                        returnArray(row, colCounter++) = inArray(row, col);
                    }
                }

                return std::move(returnArray);


            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<dtype>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Return a new array with sub-arrays along an axis deleted.
    //		
    // Inputs:
    //				NdArray
    //				inIndex to delete
    //				(Optional) Axis, if none the indices will be applied to the flattened array
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, const Slice& inIndicesSlice, Axis::Type inAxis = Axis::NONE)
    {
        Slice sliceCopy(inIndicesSlice);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                sliceCopy.makePositiveAndValidate(inArray.size());
                break;
            }
            case Axis::ROW:
            {
                sliceCopy.makePositiveAndValidate(inArray.shape().cols);
                break;
            }
            case Axis::COL:
            {
                sliceCopy.makePositiveAndValidate(inArray.shape().rows);
                break;
            }
        }

        std::vector<uint32> indices;
        for (uint32 i = static_cast<uint32>(sliceCopy.start); i < static_cast<uint32>(sliceCopy.stop); i += sliceCopy.step)
        {
            indices.push_back(i);
        }

        return std::move(deleteIndices(inArray, NdArray<uint32>(indices), inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Return a new array with sub-arrays along an axis deleted.
    //		
    // Inputs:
    //				NdArray
    //				inIndex to delete
    //				(Optional) Axis, if none the indices will be applied to the flattened array
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, uint32 inIndex, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<uint32> inIndices = { inIndex };
        return std::move(deleteIndices(inArray, inIndices, inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Create a two-dimensional array with the flattened input as a diagonal.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> diagflat(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray.size());
        returnArray.zeros();
        for (uint32 i = 0; i < inArray.size(); ++i)
        {
            returnArray(i, i) = inArray[i];
        }

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return specified diagonals.
    //		
    // Inputs:
    //				NdArray
    //				Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0. 
    //				(Optional) axis the offset is applied to
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> diagonal(const NdArray<dtype>& inArray, uint32 inOffset = 0, Axis::Type inAxis = Axis::ROW)
    {
        return std::move(inArray.diagonal(inOffset, inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Calculate the n-th discrete difference along given axis.
    //						Unsigned dtypes will give you weird results...obviously.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> diff(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        Shape inShape = inArray.shape();

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inArray.size() < 2)
                {
                    return std::move(NdArray<dtype>(0));
                }

                NdArray<dtype> returnArray(1, inArray.size() - 1);
                std::transform(inArray.cbegin(), inArray.cend() - 1, inArray.cbegin() + 1, returnArray.begin(),
                    [](dtype inValue1, dtype inValue2) { return inValue2 - inValue1; });

                return std::move(returnArray);
            }
            case Axis::COL:
            {
                if (inShape.cols < 2)
                {
                    return std::move(NdArray<dtype>(0));
                }

                NdArray<dtype> returnArray(inShape.rows, inShape.cols - 1);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    std::transform(inArray.cbegin(row), inArray.cend(row) - 1, inArray.cbegin(row) + 1, returnArray.begin(row),
                        [](dtype inValue1, dtype inValue2) { return inValue2 - inValue1; });
                }

                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                if (inShape.rows < 2)
                {
                    return std::move(NdArray<dtype>(0));
                }

                NdArray<dtype> transArray = inArray.transpose();
                Shape transShape = transArray.shape();
                NdArray<dtype> returnArray(transShape.rows, transShape.cols - 1);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    std::transform(transArray.cbegin(row), transArray.cend(row) - 1, transArray.cbegin(row) + 1, returnArray.begin(row),
                        [](dtype inValue1, dtype inValue2) { return inValue2 - inValue1; });
                }

                return std::move(returnArray.transpose());
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<dtype>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Returns a true division of the inputs, element-wise.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> divide(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1.astype<dtypeOut>() / inArray2.astype<dtypeOut>());
    }

    //============================================================================
    // Method Description: 
    //						Dot product of two arrays.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> dot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1.dot<dtypeOut>(inArray2));
    }

    //============================================================================
    // Method Description: 
    //						Dump a binary file of the array to the specified file. 
    //						The array can be read back with or NumC::load.
    //		
    // Inputs:
    //				NdArray
    //				string filename
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline void dump(const NdArray<dtype>& inArray, const std::string& inFilename)
    {
        inArray.dump(inFilename);
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, without initializing entries.
    //		
    // Inputs:
    //				inNumRows
    //				inNumCols
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> empty(uint32 inNumRows, uint32 inNumCols)
    {
        return std::move(NdArray<dtype>(inNumRows, inNumCols));
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, without initializing entries.
    //		
    // Inputs:
    //				Shape
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> empty(const Shape& inShape)
    {
        return std::move(NdArray<dtype>(inShape));
    }

    //============================================================================
    // Method Description: 
    //						Return a new array with the same shape as a given array.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> empty_like(const NdArray<dtype>& inArray)
    {
        return std::move(NdArray<dtypeOut>(inArray.shape()));
    }

    //============================================================================
    // Method Description: 
    //						Return the endianess of the array values.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				Endian::Type
    //
    template<typename dtype>
    inline Endian::Type endianess(const NdArray<dtype>& inArray)
    {
        return inArray.endianess();
    }

    //============================================================================
    // Method Description: 
    //						Return (x1 == x2) element-wise.
    //		
    // Inputs:
    //				NdArray1
    //				NdArray2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1 == inArray2);
    }

    //============================================================================
    // Method Description: 
    //						Calculate the exponential of the input value.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double exp(dtype inValue)
    {
        return std::exp(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Calculate the exponential of all elements in the input array.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> exp(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return exp(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Calculate 2**p for all p in the input value.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double exp2(dtype inValue)
    {
        return std::exp2(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Calculate 2**p for all p in the input array.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> exp2(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return exp2(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Calculate exp(x) - 1 for the input value.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double expm1(dtype inValue)
    {
        return std::exp(static_cast<double>(inValue)) - 1.0;
    }

    //============================================================================
    // Method Description: 
    //						Calculate exp(x) - 1 for all elements in the array.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> expm1(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return expm1(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return a 2-D array with ones on the diagonal and zeros elsewhere.
    //		
    // Inputs:
    //				number of rows and columns (N)
    //				K - Index of the diagonal: 0 (the default) refers to the main diagonal,
    //				a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> eye(uint32 inN, int32 inK = 0)
    {
        return std::move(eye<dtype>(inN, inN, inK));
    }

    //============================================================================
    // Method Description: 
    //						Return a 2-D array with ones on the diagonal and zeros elsewhere.
    //		
    // Inputs:
    //				number of rows (N)
    //				number of columns (M)
    //				K - Index of the diagonal: 0 (the default) refers to the main diagonal,
    //				a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> eye(uint32 inN, uint32 inM, int32 inK = 0)
    {
        NdArray<dtype> returnArray(inN, inM);
        returnArray.zeros();

        if (inK < 0)
        {
            uint32 col = 0;
            for (uint32 row = inK; row < inN; ++row)
            {
                if (col >= inM)
                {
                    break;
                }

                returnArray(row, col++) = 1;
            }
        }
        else
        {
            uint32 row = 0;
            for (uint32 col = inK; col < inM; ++col)
            {
                if (row >= inN)
                {
                    break;
                }

                returnArray(row++, col) = 1;
            }
        }

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return a 2-D array with ones on the diagonal and zeros elsewhere.
    //		
    // Inputs:
    //				Shape
    //				K - Index of the diagonal: 0 (the default) refers to the main diagonal,
    //				a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> eye(const Shape& inShape, int32 inK = 0)
    {
        return std::move(eye<dtype>(inShape.rows, inShape.cols, inK));
    }

    //============================================================================
    // Method Description: 
    //						Round to nearest integer towards zero.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype fix(dtype inValue)
    {
        return inValue > 0 ? std::floor(inValue) : std::ceil(inValue);
    }

    //============================================================================
    // Method Description: 
    //						Round to nearest integer towards zero.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> fix(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return fix(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return a copy of the array collapsed into one dimension.
    //		
    // Inputs:
    //				NdArray
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> flatten(const NdArray<dtype>& inArray)
    {
        return std::move(inArray.flatten());
    }

    //============================================================================
    // Method Description: 
    //						Return indices that are non-zero in the flattened version of a.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<uint32> flatnonzero(const NdArray<dtype>& inArray)
    {
        return std::move(inArray.flatten().nonzero());
    }

    //============================================================================
    // Method Description: 
    //						Reverse the order of elements in an array along the given axis.
    //		
    // Inputs:
    //				NdArray
    //				axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> flip(const NdArray<dtype>& inArray, Axis::Type inAxis)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                NdArray<dtype> returnArray(inArray);
                std::reverse(returnArray.begin(), returnArray.end());
                return std::move(returnArray);
            }
            case Axis::COL:
            {
                NdArray<dtype> returnArray(inArray);
                for (uint32 row = 0; row < inArray.shape().rows; ++row)
                {
                    std::reverse(returnArray.begin(row), returnArray.end(row));
                }
                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                NdArray<dtype> returnArray = inArray.transpose();
                for (uint32 row = 0; row < returnArray.shape().rows; ++row)
                {
                    std::reverse(returnArray.begin(row), returnArray.end(row));
                }
                return std::move(returnArray.transpose());
            }
            default:
            {
                return std::move(NdArray<dtype>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Flip array in the left/right direction.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> fliplr(const NdArray<dtype>& inArray)
    {
        return std::move(flip(inArray, Axis::COL));
    }

    //============================================================================
    // Method Description: 
    //						Flip array in the up/down direction.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> flipud(const NdArray<dtype>& inArray)
    {
        return std::move(flip(inArray, Axis::ROW));
    }

    //============================================================================
    // Method Description: 
    //						Return the floor of the input.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype floor(dtype inValue)
    {
        return std::floor(inValue);
    }

    //============================================================================
    // Method Description: 
    //						Return the floor of the input, element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> floor(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());

        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return floor(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the largest integer smaller or equal to the division of the inputs.
    //		
    // Inputs:
    //				value 1
    //				value 2
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype floor_divide(dtype inValue1, dtype inValue2)
    {
        return std::floor(inValue1 / inValue2);
    }

    //============================================================================
    // Method Description: 
    //						Return the largest integer smaller or equal to the division of the inputs.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> floor_divide(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(floor(inArray1 / inArray2));
    }

    //============================================================================
    // Method Description: 
    //						maximum of inputs.
    //
    //						Compare two value and returns a value containing the 
    //						maxima
    //		
    // Inputs:
    //				value 1
    //				value 2
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype fmax(dtype inValue1, dtype inValue2)
    {
        return std::max(inValue1, inValue2);
    }

    //============================================================================
    // Method Description: 
    //						Element-wise maximum of array elements.
    //
    //						Compare two arrays and returns a new array containing the 
    //						element - wise maxima
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> fmax(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: fmax: input array shapes are not consistant.");
        }

        NdArray<double> returnArray(inArray1.shape());

        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) { return std::max(inValue1, inValue2); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						minimum of inputs.
    //
    //						Compare two value and returns a value containing the 
    //						minima
    //		
    // Inputs:
    //				value 1
    //				value 2
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype fmin(dtype inValue1, dtype inValue2)
    {
        return std::min(inValue1, inValue2);
    }

    //============================================================================
    // Method Description: 
    //						Element-wise minimum of array elements.
    //
    //						Compare two arrays and returns a new array containing the 
    //						element - wise minima
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> fmin(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: fmax: input array shapes are not consistant.");
        }

        NdArray<double> returnArray(inArray1.shape());

        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) { return std::min(inValue1, inValue2); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the remainder of division.
    //
    //		
    // Inputs:
    //				value 1
    //				value 2
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype fmod(dtype inValue1, dtype inValue2)
    {
        // can only be called on integer types
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: % operator can only be compiled with integer types.");

        return inValue1 % inValue2;
    }

    //============================================================================
    // Method Description: 
    //						Return the element-wise remainder of division.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> fmod(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        // can only be called on integer types
        static_assert(DtypeInfo<dtype>::isInteger(), "ERROR: % operator can only be compiled with integer types.");

        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: fmax: input array shapes are not consistant.");
        }

        NdArray<dtype> returnArray(inArray1.shape());

        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) { return inValue1 % inValue2; });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Construct an array from data in a text or binary file.
    //		
    // Inputs:
    //				filename
    //				seperator, Separator between items if file is a text file. Empty (“”) 
    //							separator means the file should be treated as binary.
    //							Right now the only supported seperators are " ", "\t", "\n"
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> fromfile(const std::string& inFilename, const std::string& inSep = "")
    {
        boost::filesystem::path p(inFilename);
        if (!boost::filesystem::exists(inFilename))
        {
            throw std::invalid_argument("ERROR: fromfile: input filename does not exist.\n\t" + inFilename);
        }

        if (inSep.compare("") == 0)
        {
            // read in as binary file
            std::ifstream in(inFilename.c_str(), std::ios::in | std::ios::binary);
            in.seekg(0, in.end);
            uint32 fileSize = static_cast<uint32>(in.tellg());

            FILE* filePtr;
            fopen_s(&filePtr, inFilename.c_str(), "rb");
            if (filePtr == NULL)
            {
                throw std::runtime_error("ERROR: fromfile: unable to open the file.");
            }

            char* fileBuffer = new char[fileSize];
            fread(fileBuffer, sizeof(char), fileSize, filePtr);
            fclose(filePtr);

            NdArray<dtype> returnArray(reinterpret_cast<dtype*>(fileBuffer), fileSize);
            delete[] fileBuffer;

            return std::move(returnArray);
        }
        else
        {
            // read in as txt file
            if (!(inSep.compare(" ") == 0 || inSep.compare("\t") == 0 || inSep.compare("\n") == 0))
            {
                throw std::invalid_argument("ERROR: fromfile: only [' ', '\\t', '\\n'] seperators are supported");
            }

            std::vector<dtype> values;

            std::ifstream file(inFilename.c_str());
            if (file.is_open())
            {
                while (!file.eof())
                {
                    std::string line;
                    std::getline(file, line);

                    std::istringstream iss(line);
                    do
                    {
                        std::string sub;
                        iss >> sub;

                        try
                        {
                            values.push_back(static_cast<dtype>(std::stod(sub)));
                        }
                        catch (const std::invalid_argument& /*ia*/)
                        {
                            //std::cout << "Warning: fromfile: " << ia.what() << std::endl;
                        }
                    } while (iss);
                }
                file.close();
            }
            else
            {
                throw std::runtime_error("ERROR: fromfile: unable to open file.");
            }

            return std::move(NdArray<dtype>(values));
        }
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, filled with inFillValue
    //		
    // Inputs:
    //				square size
    //				fill value
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> full(uint32 inSquareSize, dtype inFillValue)
    {
        NdArray<dtype> returnArray(inSquareSize, inSquareSize);
        returnArray.fill(inFillValue);
        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, filled with inFillValue
    //		
    // Inputs:
    //				numRows
    //				numCols
    //				fill value
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> full(uint32 inNumRows, uint32 inNumCols, dtype inFillValue)
    {
        NdArray<dtype> returnArray(inNumRows, inNumCols);
        returnArray.fill(inFillValue);
        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, filled with inFillValue
    //		
    // Inputs:
    //				Shape
    //				fill value
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> full(const Shape& inShape, dtype inFillValue)
    {
        return std::move(full(inShape.rows, inShape.cols, inFillValue));
    }

    //============================================================================
    // Method Description: 
    //						Return a full array with the same shape and type as a given array.
    //		
    // Inputs:
    //				NdArray
    //				fill value
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> full_like(const NdArray<dtype>& inArray, dtype inFillValue)
    {
        return std::move(full(inArray.shape(), static_cast<dtypeOut>(inFillValue)));
    }

    //============================================================================
    // Method Description: 
    //						Return the truth value of (x1 > x2) element-wise.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> greater(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1 > inArray2);
    }

    //============================================================================
    // Method Description: 
    //						Return the truth value of (x1 >= x2) element-wise.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> greater_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1 >= inArray2);
    }

    //============================================================================
    // Method Description: 
    //						Compute the histogram of a set of data.
    //
    //		
    // Inputs:
    //				NdArray 
    //				number of bins, default 10
    //				
    // Outputs:
    //				pair of NdArrays; first is histogram counts, seconds is the bin edges
    //
    template<typename dtype>
    inline std::pair<NdArray<dtype>, NdArray<dtype> > histogram(const NdArray<dtype>& inArray, uint32 inNumBins = 10)
    {

    }

    //============================================================================
    // Method Description: 
    //						Stack arrays in sequence horizontally (column wise).
    //
    //		
    // Inputs:
    //				{list} of arrays to stack
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> hstack(const std::initializer_list<NdArray<dtype> >& inArrayList)
    {
        return std::move(column_stack(inArrayList));
    }

    //============================================================================
    // Method Description: 
    //						Given the “legs” of a right triangle, return its hypotenuse.
    //
    //						Equivalent to sqrt(x1**2 + x2 * *2), element - wise.
    //
    //		
    // Inputs:
    //				value 1
    //				value 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline dtypeOut hypot(dtype inValue1, dtype inValue2)
    {
        return std::hypot(static_cast<dtypeOut>(inValue1), static_cast<dtypeOut>(inValue2));
    }

    //============================================================================
    // Method Description: 
    //						Given the “legs” of a right triangle, return its hypotenuse.
    //
    //						Equivalent to sqrt(x1**2 + x2 * *2), element - wise.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> hypot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: hypot: input array shapes are not consistant.");
        }

        NdArray<dtypeOut> returnArray(inArray1.shape());

        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) { return std::hypot(static_cast<dtypeOut>(inValue1), static_cast<dtypeOut>(inValue2)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the identity array.
    //
    //						The identity array is a square array with ones on the main diagonal.
    //
    //		
    // Inputs:
    //				matrix square size
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> identity(uint32 inSquareSize)
    {
        NdArray<dtype> returnArray(inSquareSize);
        returnArray.zeros();
        for (uint32 i = 0; i < inSquareSize; ++i)
        {
            returnArray(i, i) = 1;
        }

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Insert values along the given axis before the given indices.
    //	
    // Inputs:
    //				
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> insert(const NdArray<dtype>& inArray, const NdArray<uint32>& inIndices, const NdArray<dtype>& inValues, Axis::Type inAxis = Axis::ROW)
    {
        //NdArray<uint32>& indices = np.unique(inIndices);

        //switch (inAxis)
        //{
        //	case Axis::NONE:
        //	{
        //		if (indices.size() != inValues.size())
        //		{
        //			throw std::invalid_argument("ERROR: insert: input indices and value arrays size are not consistant.");
        //		}

        //		if (indices.max() > inArray.size())
        //		{
        //			throw std::invalid_argument("ERROR: insert: input index in greater than the size of the array.");
        //		}

        //		NdArray<dtype> returnArray(1, inArray.size(), inIndices.size());
        //		uint32 doneCounter = 0;
        //		for (uint32 i = 0; i < returnArray.size(); ++i)
        //		{
        //			if (indices.contains(i))
        //			{
        //				returnArray[i] = inValues[doneCounter++];
        //			}
        //			else
        //			{
        //				returnArray[i] = inArray[i];
        //			}
        //		}
        //			
        //		return std::move(returnArray);
        //	}
        //	case Axis::COL:
        //	{
        //		Shape inShape = inArray.shape();
        //		Shape valuesShape = inValues.shape();

        //		if (indices.size() != inValues.size() && indices.size() != valuesShape.cols)
        //		{
        //			throw std::invalid_argument("ERROR: insert: input indices and value array sizes are not consistant.");
        //		}

        //		if (indices.max() > inShape.cols)
        //		{
        //			throw std::invalid_argument("ERROR: insert: input index in greater than the size of the array.");
        //		}

        //		if (indices.size() == inValues.size())
        //		{
        //			NdArray<dtype> returnArray(inShape.rows, inShape.cols + indices.size());

        //			for (uint32 i)
        //			{

        //			}
        //		}
        //		else if (indices.size() == valuesShape.cols)
        //		{

        //		}
        //		else
        //		{
        //			throw std::runtime_error("ERROR: insert: I've made a mistake somewhere in my logic, please investigate this!");
        //		}

        //	}
        //	case Axis::ROW:
        //	{

        //	}
        //	default:
        //	{
        //		// this isn't actually possible, just putting this here to get rid
        //		// of the compiler warning.
        //		return std::move(NdArray<dtype>(0));
        //	}
        //}
    }

    //============================================================================
    // Method Description: 
    //						Find the intersection of two arrays.
    //
    //						Return the sorted, unique values that are in both of the input arrays.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> intersect1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        std::vector<dtype> res(inArray1.size() + inArray2.size());
        std::set<dtype> in1(inArray1.cbegin(), inArray1.cend());
        std::set<dtype> in2(inArray2.cbegin(), inArray2.cend());

        std::vector<dtype>::iterator iter = std::set_intersection(in1.begin(), in1.end(),
            in2.begin(), in2.end(), res.begin());
        res.resize(iter - res.begin());
        return std::move(NdArray<dtype>(res));
    }

    //============================================================================
    // Method Description: 
    //						Compute bit-wise inversion, or bit-wise NOT, element-wise.
    //
    //		
    // Inputs:
    //				NdArray
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> invert(const NdArray<dtype>& inArray)
    {
        return std::move(~inArray);
    }

    //============================================================================
    // Method Description: 
    //						Returns a boolean array where two arrays are element-wise 
    //						equal within a tolerance.
    //
    //						For finite values, isclose uses the following equation to test whether two floating point values are equivalent.
    //						absolute(a - b) <= (atol + rtol * absolute(b))
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				relative tolerance
    //				absolute tolerance
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> isclose(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, double inRtol = 1e-05, double inAtol = 1e-08)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: isclose: input array shapes are not consistant.");
        }

        NdArray<bool> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [inRtol, inAtol](dtype inValueA, dtype inValueB) { return std::abs(inValueA - inValueB) <= (inAtol + inRtol * std::abs(inValueB)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Test for NaN and return result as a boolean.
    //
    //		
    // Inputs:
    //				value
    //				
    // Outputs:
    //				bool
    //
    template<typename dtype>
    inline bool isnan(dtype inValue)
    {
        return std::isnan(inValue);
    }

    //============================================================================
    // Method Description: 
    //						Test element-wise for NaN and return result as a boolean array.
    //
    //		
    // Inputs:
    //				NdArray
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> isnan(const NdArray<dtype>& inArray)
    {
        NdArray<bool> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return std::isnan(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Returns x1 * 2^x2.
    //
    //		
    // Inputs:
    //				value 1
    //				value 2
    //				
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype ldexp(dtype inValue1, uint8 inValue2)
    {
        return static_cast<dtype>(std::ldexp(static_cast<double>(inValue1), inValue2));
    }

    //============================================================================
    // Method Description: 
    //						Returns x1 * 2^x2, element-wise.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> ldexp(const NdArray<dtype>& inArray1, const NdArray<uint8>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: ldexp: input array shapes are not consistant.");
        }

        NdArray<dtype> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, uint8 inValue2) { return static_cast<dtype>(std::ldexp(static_cast<double>(inValue1), inValue2)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Shift the bits of an integer to the left.
    //
    //		
    // Inputs:
    //				NdArray 
    //				number of bits to sift
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> left_shift(const NdArray<dtype>& inArray, uint8 inNumBits)
    {
        return std::move(inArray << inNumBits);
    }

    //============================================================================
    // Method Description: 
    //						Return the truth value of (x1 < x2) element-wise.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> less(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1 < inArray2);
    }

    //============================================================================
    // Method Description: 
    //						Return the truth value of (x1 <= x2) element-wise.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> less_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1 <= inArray2);
    }

    //============================================================================
    // Method Description: 
    //						Return evenly spaced numbers over a specified interval.
    //
    //						Returns num evenly spaced samples, calculated over the 
    //						interval[start, stop].
    //
    //						The endpoint of the interval can optionally be excluded.
    //
    //						Mostly only usefull if called with a floating point type 
    //						for the template argument.
    // Inputs:
    //				start point
    //				end point
    //				number of points, default = 50
    //				include endPoint, default = true
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> linspace(dtype inStart, dtype inStop, uint32 inNum = 50, bool endPoint = true)
    {
        if (inNum == 0)
        {
            return std::move(NdArray<dtype>(0));
        }
        else if (inNum == 1)
        {
            NdArray<dtype> returnArray = { inStart };
            return std::move(returnArray);
        }

        if (inStop <= inStart)
        {
            throw std::invalid_argument("ERROR: linspace: stop value must be greater than the start value.");
        }

        if (endPoint)
        {
            if (inNum == 2)
            {
                NdArray<dtype> returnArray = { inStart, inStop };
                return std::move(returnArray);
            }
            else
            {
                NdArray<dtype> returnArray(1, inNum);
                returnArray[0] = inStart;
                returnArray[inNum - 1] = inStop;

                dtype step = (inStop - inStart) / (inNum - 1);
                for (uint32 i = 1; i < inNum - 1; ++i)
                {
                    returnArray[i] = returnArray[i - 1] + step;
                }

                return std::move(returnArray);
            }
        }
        else
        {
            if (inNum == 2)
            {
                dtype step = (inStop - inStart) / (inNum);
                NdArray<dtype> returnArray = { inStart, inStart + step };
                return std::move(returnArray);
            }
            else
            {
                NdArray<dtype> returnArray(1, inNum);
                returnArray[0] = inStart;

                dtype step = (inStop - inStart) / inNum;
                for (uint32 i = 1; i < inNum; ++i)
                {
                    returnArray[i] = returnArray[i - 1] + step;
                }

                return std::move(returnArray);
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						loads a .bin file from the dump() method into an NdArray
    //
    //		
    // Inputs:
    //				string filename
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> load(const std::string& inFilename)
    {
        return std::move(fromfile<dtype>(inFilename, ""));
    }

    //============================================================================
    // Method Description: 
    //						Natural logarithm.
    //
    //		
    // Inputs:
    //				value
    //				
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double log(dtype inValue)
    {
        return std::log(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Natural logarithm, element-wise.
    //
    //		
    // Inputs:
    //				NdArray
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> log(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return std::log(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the base 10 logarithm of the input array.
    //
    //		
    // Inputs:
    //				value
    //				
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double log10(dtype inValue)
    {
        return std::log10(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Return the base 10 logarithm of the input array, element-wise.
    //
    //		
    // Inputs:
    //				NdArray
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> log10(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return std::log10(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the natural logarithm of one plus the input array.
    //
    //						Calculates log(1 + x).
    //
    //		
    // Inputs:
    //				value
    //				
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double log1p(dtype inValue)
    {
        return std::log1p(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Return the natural logarithm of one plus the input array, element-wise.
    //
    //						Calculates log(1 + x).
    //
    //		
    // Inputs:
    //				NdArray
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> log1p(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return std::log1p(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Base-2 logarithm of x.
    //
    //		
    // Inputs:
    //				value
    //				
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double log2(dtype inValue)
    {
        return std::log2(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Base-2 logarithm of x.
    //
    //		
    // Inputs:
    //				NdArray
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> log2(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return std::log2(static_cast<double>(inValue)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Compute the truth value of x1 AND x2 element-wise.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> logical_and(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: logical_and: input array shapes are not consistant.");

        }

        NdArray<bool> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) { return (inValue1 != 0) && (inValue2 != 0); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Compute the truth value of NOT x element-wise.
    //
    //		
    // Inputs:
    //				NdArray 
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> logical_not(const NdArray<dtype>& inArray)
    {
        NdArray<bool> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return inValue == 0; });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Compute the truth value of x1 OR x2 element-wise.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> logical_or(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: logical_or: input array shapes are not consistant.");

        }

        NdArray<bool> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) { return (inValue1 != 0) || (inValue2 != 0); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Compute the truth value of x1 XOR x2 element-wise.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> logical_xor(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: logical_xor: input array shapes are not consistant.");

        }

        NdArray<bool> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) { return (inValue1 != 0) != (inValue2 != 0); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Matrix product of two arrays.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> matmul(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1.dot<dtypeOut>(inArray2));
    }

    //============================================================================
    // Method Description: 
    //						Return the maximum of an array or maximum along an axis.
    //
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> max(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.max(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Element-wise maximum of array elements.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> maximum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: maximum: input array shapes are not consistant.");

        }

        NdArray<dtype> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) { return std::max(inValue1, inValue2); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Compute the mean along the specified axis.
    //
    //		
    // Inputs:
    //				NdArray 
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> mean(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.mean(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Compute the median along the specified axis.
    //
    //		
    // Inputs:
    //				NdArray 
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> median(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.median(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Return the minimum of an array or maximum along an axis.
    //
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> min(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.min(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Element-wise minimum of array elements.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> minimum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: minimum: input array shapes are not consistant.");

        }

        NdArray<dtype> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) { return std::min(inValue1, inValue2); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return element-wise remainder of division.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> mod(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1 % inArray2);
    }

    //============================================================================
    // Method Description: 
    //						Multiply arguments element-wise.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> multiply(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1 * inArray2);
    }

    //============================================================================
    // Method Description: 
    //						Returns the indices of the maximum values along an axis ignoring NaNs.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<uint32> nanargmax(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (uint32 i = 0; i < arrayCopy.size(); ++i)
        {
            if (std::isnan(arrayCopy[i]))
            {
                arrayCopy[i] = DtypeInfo<dtype>::min();
            }
        }

        return std::move(argmax(arrayCopy, inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Returns the indices of the minimum values along an axis ignoring NaNs.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<uint32> nanargmin(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (uint32 i = 0; i < arrayCopy.size(); ++i)
        {
            if (std::isnan(arrayCopy[i]))
            {
                arrayCopy[i] = DtypeInfo<dtype>::max();
            }
        }

        return std::move(argmin(arrayCopy, inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Return the cumulative product of elements along a given axis ignoring NaNs.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> nancumprod(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (uint32 i = 0; i < arrayCopy.size(); ++i)
        {
            if (std::isnan(arrayCopy[i]))
            {
                arrayCopy[i] = 1;
            }
        }

        return std::move(cumprod<dtype, dtypeOut>(arrayCopy, inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Return the cumulative sum of the elements along a given axis ignoring NaNs.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> nancumsum(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (uint32 i = 0; i < arrayCopy.size(); ++i)
        {
            if (std::isnan(arrayCopy[i]))
            {
                arrayCopy[i] = 0;
            }
        }

        return std::move(cumsum<dtype, dtypeOut>(arrayCopy, inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Return the maximum of an array or maximum along an axis ignoring NaNs.
    //
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> nanmax(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (uint32 i = 0; i < arrayCopy.size(); ++i)
        {
            if (std::isnan(arrayCopy[i]))
            {
                arrayCopy[i] = DtypeInfo<dtype>::min();
            }
        }

        return std::move(max(arrayCopy, inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Compute the mean along the specified axis ignoring NaNs.
    //
    //		
    // Inputs:
    //				NdArray 
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> nanmean(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                double sum = static_cast<double>(std::accumulate(inArray.cbegin(), inArray.cend(), 0.0,
                    [](dtype inValue1, dtype inValue2) { return std::isnan(inValue2) ? inValue1 : inValue1 + inValue2; }));

                double numberNonNan = static_cast<double>(std::accumulate(inArray.cbegin(), inArray.cend(), 0.0,
                    [](dtype inValue1, dtype inValue2) { return std::isnan(inValue2) ? inValue1 : inValue1 + 1; }));

                NdArray<double> returnArray = { sum /= numberNonNan };

                return std::move(returnArray);
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();
                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    double sum = static_cast<double>(std::accumulate(inArray.cbegin(row), inArray.cend(row), 0.0,
                        [](dtype inValue1, dtype inValue2) { return std::isnan(inValue2) ? inValue1 : inValue1 + inValue2; }));

                    double numberNonNan = static_cast<double>(std::accumulate(inArray.cbegin(row), inArray.cend(row), 0.0,
                        [](dtype inValue1, dtype inValue2) { return std::isnan(inValue2) ? inValue1 : inValue1 + 1; }));

                    returnArray(0, row) = sum / numberNonNan;
                }

                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                NdArray<dtype> transposedArray = inArray.transpose();
                Shape transShape = transposedArray.shape();
                NdArray<double> returnArray(1, transShape.rows);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    double sum = static_cast<double>(std::accumulate(transposedArray.cbegin(row), transposedArray.cend(row), 0.0,
                        [](dtype inValue1, dtype inValue2) { return std::isnan(inValue2) ? inValue1 : inValue1 + inValue2; }));

                    double numberNonNan = static_cast<double>(std::accumulate(transposedArray.cbegin(row), transposedArray.cend(row), 0.0,
                        [](dtype inValue1, dtype inValue2) { return std::isnan(inValue2) ? inValue1 : inValue1 + 1; }));

                    returnArray(0, row) = sum / numberNonNan;
                }

                return std::move(returnArray);
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<double>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Compute the median along the specified axis ignoring NaNs.
    //
    //		
    // Inputs:
    //				NdArray 
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> nanmedian(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::vector<dtype> values;
                for (uint32 i = 0; i < inArray.size(); ++i)
                {
                    if (!std::isnan(inArray[i]))
                    {
                        values.push_back(inArray[i]);
                    }
                }

                uint32 middle = static_cast<uint32>(values.size()) / 2;
                std::nth_element(values.begin(), values.begin() + middle, values.end());
                NdArray<dtype> returnArray = { values[middle] };

                return std::move(returnArray);
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();
                NdArray<dtype> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    std::vector<dtype> values;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (!std::isnan(inArray(row, col)))
                        {
                            values.push_back(inArray(row, col));
                        }
                    }

                    uint32 middle = static_cast<uint32>(values.size()) / 2;
                    std::nth_element(values.begin(), values.begin() + middle, values.end());
                    returnArray(0, row) = values[middle];
                }

                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                NdArray<dtype> transposedArray = inArray.transpose();
                Shape inShape = transposedArray.shape();
                NdArray<dtype> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    std::vector<dtype> values;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (!std::isnan(transposedArray(row, col)))
                        {
                            values.push_back(transposedArray(row, col));
                        }
                    }

                    uint32 middle = static_cast<uint32>(values.size()) / 2;
                    std::nth_element(values.begin(), values.begin() + middle, values.end());
                    returnArray(0, row) = values[middle];
                }

                return std::move(returnArray);
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<dtype>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Return the minimum of an array or maximum along an axis ignoring NaNs.
    //
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> nanmin(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (uint32 i = 0; i < arrayCopy.size(); ++i)
        {
            if (std::isnan(arrayCopy[i]))
            {
                arrayCopy[i] = DtypeInfo<dtype>::max();
            }
        }

        return std::move(min(arrayCopy, inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Compute the qth percentile of the data along the specified axis, while ignoring nan values.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<double> nanpercentile(const NdArray<dtype>& inArray, double inPercentile, Axis::Type inAxis = Axis::NONE, const std::string& inInterpMethod = "linear")
    {
        if (inPercentile < 0 || inPercentile > 100)
        {
            throw std::invalid_argument("ERROR: percentile: input percentile value must be of the range [0, 100].");
        }

        if (inInterpMethod.compare("linear") != 0 &&
            inInterpMethod.compare("lower") != 0 &&
            inInterpMethod.compare("higher") != 0 &&
            inInterpMethod.compare("nearest") != 0 &&
            inInterpMethod.compare("midpoint") != 0)
        {
            std::string errStr = "ERROR: percentile: input interpolation method is not a vaid option.\n";
            errStr += "\tValid options are 'linear', 'lower', 'higher', 'nearest', 'midpoint'.";
            throw std::invalid_argument(errStr);
        }

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inPercentile == 0)
                {
                    for (uint32 i = 0; i < inArray.size(); ++i)
                    {
                        if (!isnan(inArray[i]))
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(inArray[i]) };
                            return std::move(returnArray);
                        }
                    }
                    return std::move(NdArray<dtypeOut>(0));
                }
                else if (inPercentile == 1)
                {
                    for (int32 i = static_cast<int32>(inArray.size()) - 1; i > -1; --i)
                    {
                        if (!isnan(inArray[i]))
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(inArray[i]) };
                            return std::move(returnArray);
                        }
                    }
                    return std::move(NdArray<dtypeOut>(0));
                }

                std::vector<double> arrayCopy;
                uint32 numNonNan = 0;
                for (uint32 j = 0; j < inArray.size(); ++j)
                {
                    if (!isnan(inArray[j]))
                    {
                        arrayCopy.push_back(inArray[j]);
                        ++numNonNan;
                    }
                }

                if (arrayCopy.size() < 2)
                {
                    return std::move(NdArray<dtypeOut>(0));
                }

                int32 i = static_cast<int32>(std::floor(static_cast<double>(numNonNan - 1) * inPercentile / 100.0));
                uint32 indexLower = static_cast<uint32>(clip<int32>(i, 0, numNonNan - 2));

                std::sort(arrayCopy.begin(), arrayCopy.end());

                if (inInterpMethod.compare("linear") == 0)
                {
                    double percentI = static_cast<double>(indexLower) / static_cast<double>(numNonNan - 1);
                    double fraction = (inPercentile / 100.0 - percentI) /
                        (static_cast<double>(indexLower + 1) / static_cast<double>(numNonNan - 1) - percentI);

                    double returnValue = arrayCopy[indexLower] + (arrayCopy[indexLower + 1] - arrayCopy[indexLower]) * fraction;
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(returnValue) };
                    return std::move(returnArray);
                }
                else if (inInterpMethod.compare("lower") == 0)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower]) };
                    return std::move(returnArray);
                }
                else if (inInterpMethod.compare("higher") == 0)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower + 1]) };
                    return std::move(returnArray);
                }
                else if (inInterpMethod.compare("nearest") == 0)
                {
                    double percent = inPercentile / 100.0;
                    double percent1 = static_cast<double>(indexLower) / static_cast<double>(numNonNan - 1);
                    double percent2 = static_cast<double>(indexLower + 1) / static_cast<double>(numNonNan - 1);
                    double diff1 = percent - percent1;
                    double diff2 = percent2 - percent;

                    switch (argmin<double>({ diff1, diff2 }).item())
                    {
                        case 0:
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower]) };
                            return std::move(returnArray);
                        }
                        case 1:
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower + 1]) };
                            return std::move(returnArray);
                        }
                    }
                }
                else if (inInterpMethod.compare("midpoint") == 0)
                {
                    NdArray<dtypeOut> returnArray = { (arrayCopy[indexLower] + arrayCopy[indexLower + 1]) / 2.0 };
                    return std::move(returnArray);
                }
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();

                NdArray<dtypeOut> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    NdArray<dtypeOut> outValue = nanpercentile<dtype, dtypeOut>(NdArray<dtype>(inArray.cbegin(row), inArray.cend(row)),
                        inPercentile, Axis::NONE, inInterpMethod);

                    if (outValue.size() == 1)
                    {
                        returnArray[row] = outValue.item();
                    }
                    else
                    {
                        returnArray[row] = Constants::nan;
                    }
                }

                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                NdArray<dtype> arrayTrans = inArray.transpose();
                Shape inShape = arrayTrans.shape();

                NdArray<dtypeOut> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    NdArray<dtypeOut> outValue = nanpercentile<dtype, dtypeOut>(NdArray<dtype>(arrayTrans.cbegin(row), arrayTrans.cend(row)),
                        inPercentile, Axis::NONE, inInterpMethod);

                    if (outValue.size() == 1)
                    {
                        returnArray[row] = outValue.item();
                    }
                    else
                    {
                        returnArray[row] = Constants::nan;
                    }
                }

                return std::move(returnArray);
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<dtypeOut>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.
    //
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> nanprod(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (uint32 i = 0; i < arrayCopy.size(); ++i)
        {
            if (std::isnan(arrayCopy[i]))
            {
                arrayCopy[i] = 1;
            }
        }

        return std::move(prod<dtype, dtypeOut>(arrayCopy, inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Compute the standard deviation along the specified axis, while ignoring NaNs.
    //
    //		
    // Inputs:
    //				NdArray 
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> nanstd(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                double meanValue = nanmean(inArray, inAxis).item();
                double sum = 0;
                double counter = 0;
                for (uint32 i = 0; i < inArray.size(); ++i)
                {
                    if (std::isnan(inArray[i]))
                    {
                        continue;
                    }

                    sum += Utils::sqr(static_cast<double>(inArray[i]) - meanValue);
                    ++counter;
                }
                NdArray<double> returnArray = { std::sqrt(sum / counter) };
                return std::move(returnArray);
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();
                NdArray<double> meanValue = nanmean(inArray, inAxis);
                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    double sum = 0;
                    double counter = 0;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (std::isnan(inArray(row, col)))
                        {
                            continue;
                        }

                        sum += Utils::sqr(static_cast<double>(inArray(row, col)) - meanValue[row]);
                        ++counter;
                    }
                    returnArray(0, row) = std::sqrt(sum / counter);
                }

                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                NdArray<double> meanValue = nanmean(inArray, inAxis);
                NdArray<dtype> transposedArray = inArray.transpose();
                Shape inShape = transposedArray.shape();
                NdArray<double> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    double sum = 0;
                    double counter = 0;
                    for (uint32 col = 0; col < inShape.cols; ++col)
                    {
                        if (std::isnan(transposedArray(row, col)))
                        {
                            continue;
                        }

                        sum += Utils::sqr(static_cast<double>(transposedArray(row, col)) - meanValue[row]);
                        ++counter;
                    }
                    returnArray(0, row) = std::sqrt(sum / counter);
                }

                return std::move(returnArray);
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<double>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
    //
    //		
    // Inputs:
    //				NdArray 
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> nansum(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<dtype> arrayCopy(inArray);
        for (uint32 i = 0; i < arrayCopy.size(); ++i)
        {
            if (std::isnan(arrayCopy[i]))
            {
                arrayCopy[i] = 0;
            }
        }

        return std::move(sum<dtype, dtypeOut>(arrayCopy, inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Compute the variance along the specified axis, while ignoring NaNs.
    //
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> nanvar(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<double> stdValues = nanstd(inArray, inAxis);
        for (uint32 i = 0; i < stdValues.size(); ++i)
        {
            stdValues[i] *= stdValues[i];
        }
        return std::move(stdValues);
    }

    //============================================================================
    // Method Description: 
    //						Returns the number of bytes held by the array
    //	
    // Inputs:
    //				None
    // Outputs:
    //				number of bytes
    //
    template<typename dtype>
    inline uint64 nbytes(const NdArray<dtype>& inArray)
    {
        return inArray.nbytes();
    }

    //============================================================================
    // Method Description: 
    //						Return the array with the same data viewed with a 
    //						different byte order. only works for integer types, 
    //						floating point types will not compile and you will
    //						be confused as to why...
    //
    //		
    // Inputs:
    //				inValue
    //				Endianess
    //				
    // Outputs:
    //				inValue
    //
    template<typename dtype>
    inline dtype newbyteorder(dtype inValue, Endian::Type inEndianess)
    {
        NdArray<dtype> valueArray = { inValue };
        return valueArray.newbyteorder(inEndianess).item();
    }

    //============================================================================
    // Method Description: 
    //						Return the array with the same data viewed with a 
    //						different byte order. only works for integer types, 
    //						floating point types will not compile and you will
    //						be confused as to why...
    //
    //		
    // Inputs:
    //				NdArray
    //				Endianess
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> newbyteorder(const NdArray<dtype>& inArray, Endian::Type inEndianess)
    {
        return std::move(inArray.newbyteorder(inEndianess));
    }

    //============================================================================
    // Method Description: 
    //						Numerical negative, element-wise.
    //
    //		
    // Inputs:
    //				NdArray
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> negative(const NdArray<dtype>& inArray)
    {
        NdArray<dtypeOut> returnArray = inArray.astype<dtypeOut>();
        return std::move(returnArray *= -1);
    }

    //============================================================================
    // Method Description: 
    //						Return the indices of the flattened array of the 
    //						elements that are non-zero.
    //
    //		
    // Inputs:
    //				NdArray
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<uint32> nonzero(const NdArray<dtype>& inArray)
    {
        return std::move(inArray.nonzero());
    }

    //============================================================================
    // Method Description: 
    //						Matrix or vector norm.
    //
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> norm(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.norm<dtypeOut>(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Return (x1 != x2) element-wise.
    //
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> not_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return std::move(inArray1 != inArray2);
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, filled with ones.
    //		
    // Inputs:
    //				square size
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> ones(uint32 inSquareSize)
    {
        return std::move(full(inSquareSize, static_cast<dtype>(1)));
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, filled with ones.
    //		
    // Inputs:
    //				numRows
    //				numCols
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> ones(uint32 inNumRows, uint32 inNumCols)
    {
        return std::move(full(inNumRows, inNumCols, static_cast<dtype>(1)));
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, filled with ones.
    //		
    // Inputs:
    //				Shape
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> ones(const Shape& inShape)
    {
        return std::move(full(inShape, static_cast<dtype>(1)));
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, filled with ones.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> ones_like(const NdArray<dtype>& inArray)
    {
        NdArray<dtypeOut> returnArray(inArray.shape());
        returnArray.ones();
        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Pads an array.
    //		
    // Inputs:
    //				NdArray
    //				pad width
    //				pad value
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> pad(const NdArray<dtype>& inArray, uint16 inPadWidth, dtype inPadValue)
    {
        Shape inShape = inArray.shape();
        Shape outShape(inShape);
        outShape.rows += 2 * inPadWidth;
        outShape.cols += 2 * inPadWidth;

        NdArray<dtype> returnArray(outShape);
        returnArray.fill(inPadValue);
        returnArray.put(Slice(inPadWidth, inPadWidth + inShape.rows), Slice(inPadWidth, inPadWidth + inShape.cols), inArray);

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Rearranges the elements in the array in such a way that 
    //						value of the element in kth position is in the position it 
    //						would be in a sorted array. All elements smaller than the kth 
    //						element are moved before this element and all equal or greater 
    //						are moved behind it. The ordering of the elements in the two 
    //						partitions is undefined.
    //		
    // Inputs:
    //				kth element
    //				(Optional) Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> partition(const NdArray<dtype>& inArray, uint32 inKth, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<dtype> returnArray(inArray);
        returnArray.partition(inKth, inAxis);
        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Compute the qth percentile of the data along the specified axis.
    //		
    // Inputs:
    //				NdArray
    //				percentile, must be in the range [0, 100]
    //				(Optional) axis
    //				(Optional) interpolation method
    //					linear: i + (j - i) * fraction, where fraction is the fractional part of the index surrounded by i and j.
    //					lower : i.
    //					higher : j.
    //					nearest : i or j, whichever is nearest.
    //					midpoint : (i + j) / 2.
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> percentile(const NdArray<dtype>& inArray, double inPercentile, Axis::Type inAxis = Axis::NONE, const std::string& inInterpMethod = "linear")
    {
        if (inPercentile < 0 || inPercentile > 100)
        {
            throw std::invalid_argument("ERROR: percentile: input percentile value must be of the range [0, 100].");
        }

        if (inInterpMethod.compare("linear") != 0 &&
            inInterpMethod.compare("lower") != 0 &&
            inInterpMethod.compare("higher") != 0 &&
            inInterpMethod.compare("nearest") != 0 &&
            inInterpMethod.compare("midpoint") != 0)
        {
            std::string errStr = "ERROR: percentile: input interpolation method is not a vaid option.\n";
            errStr += "\tValid options are 'linear', 'lower', 'higher', 'nearest', 'midpoint'.";
            throw std::invalid_argument(errStr);
        }

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inPercentile == 0)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(*inArray.cbegin()) };
                    return std::move(returnArray);
                }
                else if (inPercentile == 1)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(*inArray.cend()) };
                    return std::move(returnArray);
                }

                int32 i = static_cast<int32>(std::floor(static_cast<double>(inArray.size() - 1) * inPercentile / 100.0));
                uint32 indexLower = static_cast<uint32>(clip<int32>(i, 0, inArray.size() - 2));

                NdArray<double> arrayCopy = inArray.astype<double>();
                std::sort(arrayCopy.begin(), arrayCopy.end());

                if (inInterpMethod.compare("linear") == 0)
                {
                    double percentI = static_cast<double>(indexLower) / static_cast<double>(inArray.size() - 1);
                    double fraction = (inPercentile / 100.0 - percentI) /
                        (static_cast<double>(indexLower + 1) / static_cast<double>(inArray.size() - 1) - percentI);

                    double returnValue = arrayCopy[indexLower] + (arrayCopy[indexLower + 1] - arrayCopy[indexLower]) * fraction;
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(returnValue) };
                    return std::move(returnArray);
                }
                else if (inInterpMethod.compare("lower") == 0)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower]) };
                    return std::move(returnArray);
                }
                else if (inInterpMethod.compare("higher") == 0)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower + 1]) };
                    return std::move(returnArray);
                }
                else if (inInterpMethod.compare("nearest") == 0)
                {
                    double percent = inPercentile / 100.0;
                    double percent1 = static_cast<double>(indexLower) / static_cast<double>(inArray.size() - 1);
                    double percent2 = static_cast<double>(indexLower + 1) / static_cast<double>(inArray.size() - 1);
                    double diff1 = percent - percent1;
                    double diff2 = percent2 - percent;

                    switch (argmin<double>({ diff1, diff2 }).item())
                    {
                        case 0:
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower]) };
                            return std::move(returnArray);
                        }
                        case 1:
                        {
                            NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>(arrayCopy[indexLower + 1]) };
                            return std::move(returnArray);
                        }
                    }
                }
                else if (inInterpMethod.compare("midpoint") == 0)
                {
                    NdArray<dtypeOut> returnArray = { static_cast<dtypeOut>((arrayCopy[indexLower] + arrayCopy[indexLower + 1]) / 2.0) };
                    return std::move(returnArray);
                }
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();

                NdArray<dtypeOut> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray[row] = percentile<dtype, dtypeOut>(NdArray<dtype>(inArray.cbegin(row), inArray.cend(row)),
                        inPercentile, Axis::NONE, inInterpMethod).item();
                }

                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                NdArray<dtype> arrayTrans = inArray.transpose();
                Shape inShape = arrayTrans.shape();

                NdArray<dtypeOut> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray[row] = percentile<dtype, dtypeOut>(NdArray<dtype>(arrayTrans.cbegin(row), arrayTrans.cend(row)),
                        inPercentile, Axis::NONE, inInterpMethod).item();
                }

                return std::move(returnArray);
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<dtypeOut>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Raises the elements of the array to the input power
    //		
    // Inputs:
    //				NdArray
    //				exponent
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> power(const NdArray<dtype>& inArray, uint8 inExponent)
    {
        NdArray<dtypeOut> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [inExponent](dtype inValue) { return Utils::power(static_cast<dtypeOut>(inValue), inExponent); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Raises the elements of the array to the input powers
    //		
    // Inputs:
    //				NdArray
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> power(const NdArray<dtype>& inArray, const NdArray<uint8>& inExponents)
    {
        if (inArray.shape() != inExponents.shape())
        {
            throw std::invalid_argument("ERROR: power: input array shapes are not consistant.");
        }

        NdArray<dtypeOut> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), inExponents.cbegin(), returnArray.begin(),
            [](dtype inValue, uint8 inExponent) { return Utils::power(static_cast<dtypeOut>(inValue), inExponent); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Prints the array to the console.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				None
    //
    template<typename dtype>
    inline void print(const NdArray<dtype>& inArray)
    {
        std::cout << inArray;
    }

    //============================================================================
    // Method Description: 
    //						Return the product of array elements over a given axis.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> prod(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.prod<dtypeOut>(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Range of values (maximum - minimum) along an axis.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> ptp(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.ptp(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Replaces specified elements of an array with given values.
    //						The indexing works on the flattened target array
    //		
    // Inputs:
    //				NdArray
    //				NdArray of indices
    //				NdArray of values to put
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline void put(NdArray<dtype>& inArray, const NdArray<uint32>& inIndices, const NdArray<dtype>& inValues)
    {
        inArray.put(inIndices, inValues);
    }

    //============================================================================
    // Method Description: 
    //						Changes elements of an array based on conditional and input values.
    //
    //						Sets a.flat[n] = values[n] for each n where mask.flat[n] == True.
    //
    //						If values is not the same size as a and mask then it will repeat.
    //		
    // Inputs:
    //				NdArray
    //				NdArray mask
    //				NdArray of values to put
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> put_mask(const NdArray<dtype>& inArray, const NdArray<bool>& inMask, const NdArray<dtype>& inValues)
    {

    }

    //============================================================================
    // Method Description: 
    //						Convert angles from radians to degrees.
    //		
    // Inputs:
    //				value
    //
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double rad2deg(dtype inValue)
    {
        return inValue * 180.0 / Constants::pi;
    }

    //============================================================================
    // Method Description: 
    //						Convert angles from radians to degrees.
    //		
    // Inputs:
    //				NdArray
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> rad2deg(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return rad2deg(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the reciprocal of the argument, element-wise.
    //
    //						Calculates 1 / x.
    //		
    // Inputs:
    //				NdArray
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> reciprocal(const NdArray<dtype>& inArray)
    {
        NdArray<dtypeOut> returnArray(inArray.shape());
        for (uint32 i = 0; i < returnArray.size(); ++i)
        {
            returnArray[i] = static_cast<dtypeOut>(1.0 / static_cast<double>(inArray[i]));
        }

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return remainder of division.
    //		
    // Inputs:
    //				value 1
    //				value 2
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline dtypeOut remainder(dtype inValue1, dtype inValue2)
    {
        return std::remainder(static_cast<dtypeOut>(inValue1), static_cast<dtypeOut>(inValue2));
    }

    //============================================================================
    // Method Description: 
    //						Return element-wise remainder of division.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> remainder(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: remainder: input array shapes are not consistant.");
        }

        NdArray<dtypeOut> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) { return std::remainder(static_cast<dtypeOut>(inValue1), static_cast<dtypeOut>(inValue2)); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Repeat elements of an array.
    //		
    // Inputs:
    //				numRows
    //				numCols
    //				Shape
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> repeat(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        return std::move(inArray.repeat(inNumRows, inNumCols));
    }

    //============================================================================
    // Method Description: 
    //						Repeat elements of an array.
    //		
    // Inputs:
    //				NdArray
    //				Shape
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> repeat(const NdArray<dtype>& inArray, const Shape& inRepeatShape)
    {
        return std::move(inArray.repeat(inRepeatShape));
    }

    //============================================================================
    // Method Description: 
    //						Gives a new shape to an array without changing its data.
    //		
    // Inputs:
    //				numRows
    //				numCols
    //				Shape, new Shape
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline void reshape(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        inArray.reshape(inNumRows, inNumCols);
    }

    //============================================================================
    // Method Description: 
    //						Gives a new shape to an array without changing its data.
    //		
    // Inputs:
    //				NdArray
    //				Shape, new Shape
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline void reshape(const NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        inArray.reshape(inNewShape);
    }

    //============================================================================
    // Method Description: 
    //						Change shape and size of array in-place. All previous
    //						data of the array is lost.
    //		
    // Inputs:
    //				NdArray
    //				numRows
    //				numCols
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline void resizeFast(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        inArray.resizeFast(inNumRows, inNumCols);
    }

    //============================================================================
    // Method Description: 
    //						Change shape and size of array in-place. All previous
    //						data of the array is lost.
    //		
    // Inputs:
    //				NdArray 
    //				Shape, new Shape
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline void resizeFast(const NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        inArray.resizeFast(inNewShape);
    }

    //============================================================================
    // Method Description: 
    //						Return a new array with the specified shape. If new shape
    //						is larger than old shape then array will be padded with zeros.
    //						If new shape is smaller than the old shape then the data will
    //						be discarded.
    //		
    // Inputs:
    //				NdArray
    //				numRows
    //				numCols
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline void resizeSlow(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        inArray.resizeSlow(inNumRows, inNumCols);
    }

    //============================================================================
    // Method Description: 
    //						Return a new array with the specified shape. If new shape
    //						is larger than old shape then array will be padded with zeros.
    //						If new shape is smaller than the old shape then the data will
    //						be discarded.
    //		
    // Inputs:
    //				NdArray 
    //				Shape, new Shape
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline void resizeSlow(const NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        inArray.resizeSlow(inNewShape);
    }

    //============================================================================
    // Method Description: 
    //						Shift the bits of an integer to the right.
    //
    //		
    // Inputs:
    //				NdArray 
    //				number of bits to sift
    //				
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> right_shift(const NdArray<dtype>& inArray, uint8 inNumBits)
    {
        return std::move(inArray >> inNumBits);
    }

    //============================================================================
    // Method Description: 
    //						Round value to the nearest integer.
    //		
    // Inputs:
    //				value 
    //
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype rint(dtype inValue)
    {
        return std::rint(inValue);
    }

    //============================================================================
    // Method Description: 
    //						Round elements of the array to the nearest integer.
    //		
    // Inputs:
    //				NdArray 
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> rint(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::rint(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Roll array elements along a given axis.
    //		
    // Inputs:
    //				NdArray 
    //				elements to shift, positive means forward, negative means backwards
    //				(Optional) axis
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> roll(const NdArray<dtype>& inArray, int32 inShift, Axis::Type inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                uint32 shift = std::abs(inShift) % inArray.size();
                if (inShift > 0)
                {
                    shift = inArray.size() - shift;
                }

                NdArray<dtype> returnArray(inArray);
                std::rotate(returnArray.begin(), returnArray.begin() + shift, returnArray.end());

                return std::move(returnArray);
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();

                uint32 shift = std::abs(inShift) % inShape.cols;
                if (inShift > 0)
                {
                    shift = inShape.cols - shift;
                }

                NdArray<dtype> returnArray(inArray);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    std::rotate(returnArray.begin(row), returnArray.begin(row) + shift, returnArray.end(row));
                }

                return std::move(returnArray);
            }
            case Axis::ROW:
            {
                Shape inShape = inArray.shape();

                uint32 shift = std::abs(inShift) % inShape.rows;
                if (inShift > 0)
                {
                    shift = inShape.rows - shift;
                }

                NdArray<dtype> returnArray = inArray.transpose();
                for (uint32 row = 0; row < inShape.cols; ++row)
                {
                    std::rotate(returnArray.begin(row), returnArray.begin(row) + shift, returnArray.end(row));
                }

                return std::move(returnArray.transpose());
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<dtype>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Rotate an array by 90 degrees counter clockwise in the plane.
    //		
    // Inputs:
    //				NdArray 
    //				the number of times to rotate 90 degrees
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> rot90(const NdArray<dtype>& inArray, uint8 inK)
    {
        inK %= 4;
        switch (inK)
        {
            case 1:
            {
                return std::move(flipud(inArray.transpose()));
            }
            case 2:
            {
                return std::move(flip(inArray, Axis::NONE));
            }
            case 3:
            {
                return std::move(fliplr(inArray.transpose()));
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return std::move(NdArray<dtype>(0));
            }
        }
    }

    //============================================================================
    // Method Description: 
    //						Round value to the given number of decimals.
    //		
    // Inputs:
    //				value 
    //				the number of decimals
    //
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype round(dtype inValue, uint8 inDecimals)
    {
        NdArray<dtype> input = { inValue };
        return input.round(inDecimals).item();
    }

    //============================================================================
    // Method Description: 
    //						Round an array to the given number of decimals.
    //		
    // Inputs:
    //				NdArray 
    //				the number of decimals
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> round(const NdArray<dtype>& inArray, uint8 inDecimals)
    {
        return std::move(inArray.round(inDecimals));
    }

    //============================================================================
    // Method Description: 
    //						Stack arrays in sequence vertically (row wise).
    //
    // Inputs:
    //				{list} of arrays to stack
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> row_stack(const std::initializer_list<NdArray<dtype> >& inArrayList)
    {
        // first loop through to calculate the final size of the array
        typename std::initializer_list<NdArray<dtype> >::iterator iter;
        Shape finalShape;
        for (iter = inArrayList.begin(); iter < inArrayList.end(); ++iter)
        {
            if (finalShape.isnull())
            {
                finalShape = iter->shape();
            }
            else if (iter->shape().cols != finalShape.cols)
            {
                throw std::invalid_argument("ERROR: row_stack: input arrays must have the same number of columns.");
            }
            else
            {
                finalShape.rows += iter->shape().rows;
            }
        }

        // now that we know the final size, contruct the output array
        NdArray<dtype> returnArray(finalShape);
        uint32 rowStart = 0;
        for (iter = inArrayList.begin(); iter < inArrayList.end(); ++iter)
        {
            Shape theShape = iter->shape();
            for (uint32 row = 0; row < theShape.rows; ++row)
            {
                for (uint32 col = 0; col < theShape.cols; ++col)
                {
                    returnArray(rowStart + row, col) = iter->operator()(row, col);
                }
            }
            rowStart += theShape.rows;
        }

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Find the set difference of two arrays.
    //
    //						Return the sorted, unique values in ar1 that are not in ar2.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> setdiff1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        std::vector<dtype> res(inArray1.size() + inArray2.size());
        std::set<dtype> in1(inArray1.cbegin(), inArray1.cend());
        std::set<dtype> in2(inArray2.cbegin(), inArray2.cend());

        std::vector<dtype>::iterator iter = std::set_difference(in1.begin(), in1.end(),
            in2.begin(), in2.end(), res.begin());
        res.resize(iter - res.begin());
        return std::move(NdArray<dtype>(res));
    }

    //============================================================================
    // Method Description: 
    //						Return the shape of the array
    //		
    // Inputs:
    //				NdArray 
    // Outputs:
    //				Shape
    //
    template<typename dtype>
    inline Shape shape(const NdArray<dtype>& inArray)
    {
        return inArray.shape();
    }

    //============================================================================
    // Method Description: 
    //						Returns an element-wise indication of the sign of a number.
    //
    //						The sign function returns - 1 if x < 0, 0 if x == 0, 1 if x > 0. 
    //						nan is returned for nan inputs.
    //		
    // Inputs:
    //				NdArray 
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline int8 sign(dtype inValue)
    {
        if (inValue < 0)
        {
            return -1;
        }
        else if (inValue > 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    //============================================================================
    // Method Description: 
    //						Returns an element-wise indication of the sign of a number.
    //
    //						The sign function returns - 1 if x < 0, 0 if x == 0, 1 if x > 0. 
    //						nan is returned for nan inputs.
    //		
    // Inputs:
    //				NdArray 
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<int8> sign(const NdArray<dtype>& inArray)
    {
        NdArray<int8> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return sign(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Returns element-wise True where signbit is set (less than zero).
    //		
    // Inputs:
    //				NdArray 
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline bool signbit(dtype inValue)
    {
        return inValue < 0 ? true : false;
    }

    //============================================================================
    // Method Description: 
    //						Returns element-wise True where signbit is set (less than zero).
    //		
    // Inputs:
    //				NdArray 
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<bool> signbit(const NdArray<dtype>& inArray)
    {
        NdArray<bool> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return signbit(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric sine.
    //		
    // Inputs:
    //				value 
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double sin(dtype inValue)
    {
        return std::sin(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Trigonometric sine, element-wise.
    //		
    // Inputs:
    //				NdArray 
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> sin(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return sin(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the sinc function.
    //
    //						The sinc function is sin(pi*x) / (pi*x).
    //		
    // Inputs:
    //				value 
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double sinc(dtype inValue)
    {
        double input = static_cast<double>(inValue);
        return std::sin(Constants::pi * input) / (Constants::pi * input);
    }

    //============================================================================
    // Method Description: 
    //						Return the sinc function.
    //
    //						The sinc function is sin(pi*x) / (pi*x).
    //		
    // Inputs:
    //				NdArray 
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> sinc(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return sinc(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Hyperbolic sine,.
    //		
    // Inputs:
    //				value 
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double sinh(dtype inValue)
    {
        return std::sinh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Hyperbolic sine, element-wise.
    //		
    // Inputs:
    //				NdArray 
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> sinh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return sinh(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the number of elements.
    //		
    // Inputs:
    //				uint32 
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline uint32 size(const NdArray<dtype>& inArray)
    {
        return inArray.size();
    }

    //============================================================================
    // Method Description: 
    //						Return a sorted copy of an array.
    //		
    // Inputs:
    //				NdArray 
    //				(Optional) Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> sort(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        NdArray<dtype> returnArray(inArray);
        returnArray.sort(inAxis);
        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the positive square-root of a value.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double sqrt(dtype inValue)
    {
        return std::sqrt(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Return the positive square-root of an array, element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> sqrt(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return sqrt(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Return the square of an array.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype square(dtype inValue)
    {
        return Utils::sqr(inValue);
    }

    //============================================================================
    // Method Description: 
    //						Return the square of an array, element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> square(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return Utils::sqr(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Compute the standard deviation along the specified axis.
    //		
    // Inputs:
    //				NdArray
    //				Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> std(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.std(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Sum of array elements over a given axis.
    //		
    // Inputs:
    //				NdArray
    //				Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> sum(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.sum<dtypeOut>(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Interchange two axes of an array.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> swapaxes(const NdArray<dtype>& inArray)
    {
        return std::move(inArray.swapaxes());
    }

    //============================================================================
    // Method Description: 
    //						Compute tangent.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double tan(dtype inValue)
    {
        return std::tan(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Compute tangent element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> tan(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return tan(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Compute hyperbolic tangent.
    //		
    // Inputs:
    //				value
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline double tanh(dtype inValue)
    {
        return std::tanh(static_cast<double>(inValue));
    }

    //============================================================================
    // Method Description: 
    //						Compute hyperbolic tangent element-wise.
    //		
    // Inputs:
    //				NdArray
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> tanh(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return tanh(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Construct an array by repeating A the number of times given by reps.
    //		
    // Inputs:
    //				NdArray
    //				numRows
    //				numCols
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> tile(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
    {
        return std::move(inArray.repeat(inNumRows, inNumCols));
    }

    //============================================================================
    // Method Description: 
    //						Construct an array by repeating A the number of times given by reps.
    //		
    // Inputs:
    //				NdArray
    //				Shape
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> tile(const NdArray<dtype>& inArray, const Shape& inReps)
    {
        return std::move(inArray.repeat(inReps));
    }

    //============================================================================
    // Method Description: 
    //						Write array to a file as text or binary (default)..
    //						The data produced by this method can be recovered 
    //						using the function fromfile().
    //		
    // Inputs:
    //				NdArray
    //				filename
    //				Separator between array items for text output. If “” (empty), a binary file is written 
    // Outputs:
    //				None
    //
    template<typename dtype>
    inline void tofile(const NdArray<dtype>& inArray, const std::string& inFilename, const std::string& inSep = "")
    {
        return inArray.tofile(inFilename, inSep);
    }

    //============================================================================
    // Method Description: 
    //						Write flattened array to an STL vector
    //		
    // Inputs:
    //				NdArray 
    // Outputs:
    //				std::vector
    //
    template<typename dtype>
    inline std::vector<dtype> toStlVector(const NdArray<dtype>& inArray)
    {
        return std::move(inArray.toStlVector());
    }

    //============================================================================
    // Method Description: 
    //						Return the sum along diagonals of the array.
    //		
    // Inputs:
    //				NdArray
    //				Offset from main diaganol, default = 0, negative=above, positve=below
    //				Axis
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline dtypeOut trace(const NdArray<dtype>& inArray, uint16 inOffset = 0, Axis::Type inAxis = Axis::ROW)
    {
        return std::move(inArray.trace<dtypeOut>(inOffset, inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Permute the dimensions of an array.
    //		
    // Inputs:
    //				NdArray
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> transpose(const NdArray<dtype>& inArray)
    {
        return std::move(inArray.transpose());
    }

    //============================================================================
    // Method Description: 
    //						Integrate along the given axis using the composite trapezoidal rule.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) Axis
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> trapz(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {

    }

    //============================================================================
    // Method Description: 
    //						Integrate along the given axis using the composite trapezoidal rule.
    //		
    // Inputs:
    //				NdArray x values
    //				NdArray y values
    //				(Optional) Axis
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype, typename dtypeOut = double>
    inline NdArray<dtypeOut> trapz(const NdArray<dtype>& inArrayX, const NdArray<dtype>& inArrayY, Axis::Type inAxis = Axis::NONE)
    {

    }

    //============================================================================
    // Method Description: 
    //						An array with ones at and below the given diagonal and zeros elsewhere.
    //		
    // Inputs:
    //				N, number of rows and cols
    //				Offset, the sub-diagonal at and below which the array is filled. 
    //						k = 0 is the main diagonal, while k < 0 is below it, 
    //						and k > 0 is above. The default is 0.
    //				
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> tri(uint32 inN, int32 inOffset = 0)
    {
        uint32 rowStart = 0;
        uint32 colStart = 0;
        if (inOffset > 0)
        {
            colStart = inOffset;
        }
        else
        {
            rowStart = inOffset * -1;
        }

        NdArray<dtype> returnArray(inN);
        returnArray.zeros();
        for (uint32 row = rowStart; row < inN; ++row)
        {
            for (uint32 col = 0; col < row + colStart + 1 - rowStart; ++col)
            {
                if (col == inN)
                {
                    break;
                }

                returnArray(row, col) = static_cast<dtype>(1);
            }
        }

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						An array with ones at and below the given diagonal and zeros elsewhere.
    //		
    // Inputs:
    //				N, number of rows
    //				M, number of columns
    //				Offset, the sub-diagonal at and below which the array is filled. 
    //						k = 0 is the main diagonal, while k < 0 is below it, 
    //						and k > 0 is above. The default is 0.
    //				
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> tri(uint32 inN, uint32 inM, int32 inOffset = 0)
    {
        uint32 rowStart = 0;
        uint32 colStart = 0;
        if (inOffset > 0)
        {
            colStart = inOffset;
        }
        else if (inOffset < 0)
        {
            rowStart = inOffset * -1;
        }

        NdArray<dtype> returnArray(inN, inM);
        returnArray.zeros();
        for (uint32 row = rowStart; row < inN; ++row)
        {
            for (uint32 col = 0; col < row + colStart + 1 - rowStart; ++col)
            {
                if (col == inM)
                {
                    break;
                }

                returnArray(row, col) = static_cast<dtype>(1);
            }
        }

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Lower triangle of an array.
    //
    //						Return a copy of an array with elements above the k-th diagonal zeroed.
    //		
    // Inputs:
    //				NdArray
    //				Offset, the sub-diagonal at and below which the array is filled. 
    //						k = 0 is the main diagonal, while k < 0 is below it, 
    //						and k > 0 is above. The default is 0.
    //				
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> tril(const NdArray<dtype>& inArray, int32 inOffset = 0)
    {

    }

    //============================================================================
    // Method Description: 
    //						Upper triangle of an array.
    //
    //						Return a copy of an array with elements below the k-th diagonal zeroed.
    //		
    // Inputs:
    //				NdArray
    //				Offset, the sub-diagonal at and below which the array is filled. 
    //						k = 0 is the main diagonal, while k < 0 is below it, 
    //						and k > 0 is above. The default is 0.
    //				
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> triu(const NdArray<dtype>& inArray, int32 inOffset = 0)
    {


    }

    //============================================================================
    // Method Description: 
    //						Trim the leading and/or trailing zeros from a 1-D array or sequence.
    //		
    // Inputs:
    //				NdArray
    //				string, "f" = front, "b" = back, "fb" = front and back
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> trim_zeros(const NdArray<dtype>& inArray, const std::string inTrim = "fb")
    {
        if (inTrim == "f")
        {
            uint32 place = 0;
            for (uint32 i = 0; i < inArray.size(); ++i)
            {
                if (inArray[i] != static_cast<dtype>(0))
                {
                    break;
                }
                else
                {
                    ++place;
                }
            }

            if (place == inArray.size())
            {
                return std::move(NdArray<dtype>(0));
            }

            NdArray<dtype> returnArray(1, inArray.size() - place);
            std::copy(inArray.cbegin() + place, inArray.cend(), returnArray.begin());

            return std::move(returnArray);
        }
        else if (inTrim == "b")
        {
            uint32 place = inArray.size();
            for (uint32 i = inArray.size() - 1; i > 0; --i)
            {
                if (inArray[i] != static_cast<dtype>(0))
                {
                    break;
                }
                else
                {
                    --place;
                }
            }

            if (place == 0 || (place == 1 && inArray[0] == 0))
            {
                return std::move(NdArray<dtype>(0));
            }

            NdArray<dtype> returnArray(1, place);
            std::copy(inArray.cbegin(), inArray.cbegin() + place, returnArray.begin());

            return std::move(returnArray);
        }
        else if (inTrim == "fb")
        {
            uint32 placeBegin = 0;
            for (uint32 i = 0; i < inArray.size(); ++i)
            {
                if (inArray[i] != static_cast<dtype>(0))
                {
                    break;
                }
                else
                {
                    ++placeBegin;
                }
            }

            if (placeBegin == inArray.size())
            {
                return std::move(NdArray<dtype>(0));
            }

            uint32 placeEnd = inArray.size();
            for (uint32 i = inArray.size() - 1; i > 0; --i)
            {
                if (inArray[i] != static_cast<dtype>(0))
                {
                    break;
                }
                else
                {
                    --placeEnd;
                }
            }

            if (placeEnd == 0 || (placeEnd == 1 && inArray[0] == 0))
            {
                return std::move(NdArray<dtype>(0));
            }

            NdArray<dtype> returnArray(1, placeEnd - placeBegin);
            std::copy(inArray.cbegin() + placeBegin, inArray.cbegin() + placeEnd, returnArray.begin());

            return std::move(returnArray);
        }
        else
        {
            throw std::invalid_argument("ERROR: trim_zeros: trim options are 'f' = front, 'b' = back, 'fb' = front and back.");
        }
    }

    //============================================================================
    // Method Description: 
    //						Return the truncated value of the input.
    //		
    // Inputs:
    //				value 
    //
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype trunc(dtype inValue)
    {
        return std::trunc(inValue);
    }

    //============================================================================
    // Method Description: 
    //						Return the truncated value of the input, element-wise.
    //		
    // Inputs:
    //				NdArray 
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> trunc(const NdArray<dtype>& inArray)
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::trunc(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Find the union of two arrays.
    //
    //						Return the unique, sorted array of values that are in 
    //						either of the two input arrays.
    //		
    // Inputs:
    //				NdArray 1
    //				NdArray 2
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> union1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            throw std::invalid_argument("ERROR: union1d: input array shapes are not consistant.");
        }

        std::set<dtype> theSet(inArray1.cbegin(), inArray1.cend());
        theSet.insert(inArray2.cbegin(), inArray2.cend());
        return std::move(NdArray<dtype>(theSet));
    }

    //============================================================================
    // Method Description: 
    //						Find the unique elements of an array.
    //
    //						Returns the sorted unique elements of an array.
    //		
    // Inputs:
    //				NdArray 
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> unique(const NdArray<dtype>& inArray)
    {
        std::set<dtype> theSet(inArray.cbegin(), inArray.cend());
        return std::move(NdArray<dtype>(theSet));
    }

    //============================================================================
    // Method Description: 
    //						Unwrap by changing deltas between values to 2*pi complement.
    //		
    // Inputs:
    //				value 
    //
    // Outputs:
    //				value
    //
    template<typename dtype>
    inline dtype unwrap(dtype inValue)
    {
        if (inValue < 0)
        {
            return inValue + 2 * Constants::pi;
        }
        else if (inValue >= 2 * Constants::pi)
        {
            return inValue - 2 * Constants::pi;
        }
        else
        {
            return inValue;
        }
    }

    //============================================================================
    // Method Description: 
    //						Unwrap by changing deltas between values to 2*pi complement.
    //		
    // Inputs:
    //				NdArray 
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> unwrap(const NdArray<dtype>& inArray)
    {
        NdArray<dtype> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) { return unwrap(inValue); });

        return std::move(returnArray);
    }

    //============================================================================
    // Method Description: 
    //						Compute the variance along the specified axis.
    //		
    // Inputs:
    //				NdArray
    //				(Optional) axis
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<double> var(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
    {
        return std::move(inArray.var(inAxis));
    }

    //============================================================================
    // Method Description: 
    //						Compute the variance along the specified axis.
    //		
    // Inputs:
    //				{list} of arrays to stack
    //
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> vstack(const std::initializer_list<NdArray<dtype> >& inArrayList)
    {
        return std::move(row_stack(inArrayList));
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, filled with zeros.
    //		
    // Inputs:
    //				square size
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> zeros(uint32 inSquareSize)
    {
        return std::move(full(inSquareSize, static_cast<dtype>(0)));
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, filled with zeros.
    //		
    // Inputs:
    //				numRows
    //				numCols
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> zeros(uint32 inNumRows, uint32 inNumCols)
    {
        return std::move(full(inNumRows, inNumCols, static_cast<dtype>(0)));
    }

    //============================================================================
    // Method Description: 
    //						Return a new array of given shape and type, filled with zeros.
    //		
    // Inputs:
    //				Shape
    // Outputs:
    //				NdArray
    //
    template<typename dtype>
    inline NdArray<dtype> zeros(const Shape& inShape)
    {
        return std::move(full(inShape, static_cast<dtype>(0)));
    }
}
