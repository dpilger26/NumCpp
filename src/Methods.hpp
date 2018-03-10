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

#include"Types.hpp"
#include"NdArray.hpp"

#include<boost/filesystem.hpp>

#include<cmath>
#include<initializer_list>
#include<stdexcept>
#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<utility>

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
	dtype abs(dtype inValue)
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
	NdArray<dtype> abs(const NdArray<dtype>& inArray)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> add(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return std::move(inArray1 + inArray2);
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
	uint32 alen(const NdArray<dtype>& inArray)
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
	NdArray<bool> all(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	bool allclose(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, double inTolerance = 1e-5)
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
	NdArray<dtype> amax(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtype> amin(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<bool> any(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtype> append(const NdArray<dtype>& inArray, const NdArray<dtype>& inAppendValues, Axis::Type inAxis = Axis::NONE)
	{

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
	NdArray<dtype> arange(dtype inStart, dtype inStop, dtype inStep = 1)
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
	//						Trigonometric inverse cosine
	//		
	// Inputs:
	//				value
	// Outputs:
	//				value
	//
	template<typename dtype>
	double arccos(dtype inValue)
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
	NdArray<double> arccos(const NdArray<dtype>& inArray)
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
	double arccosh(dtype inValue)
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
	NdArray<double> arccosh(const NdArray<dtype>& inArray)
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
	double arcsin(dtype inValue)
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
	NdArray<double> arcsin(const NdArray<dtype>& inArray)
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
	double arcsinh(dtype inValue)
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
	NdArray<double> arcsinh(const NdArray<dtype>& inArray)
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
	double arctan(dtype inValue)
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
	NdArray<double> arctan(const NdArray<dtype>& inArray)
	{
		NdArray<double> returnArray(inArray.shape());
		std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return std::atan(static_cast<double>(inValue)); });

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
	double arctanh(dtype inValue)
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
	NdArray<double> arctanh(const NdArray<dtype>& inArray)
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
	NdArray<uint32> argmax(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<uint32> argmin(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<uint32> argsort(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return std::move(inArray.argsort(inAxis));
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
	dtype around(dtype inValue, uint8 inNumDecimals = 0)
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
	NdArray<dtype> around(const NdArray<dtype>& inArray, uint8 inNumDecimals = 0)
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
	bool array_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
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
	bool array_equiv(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
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
	NdArray<dtype> asarray(const std::vector<dtype>& inVector)
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
	NdArray<dtype> asarray(std::initializer_list<dtype>& inList)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> astype(const NdArray<dtype> inArray)
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
	NdArray<double> average(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<double> average(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtype> bincount(const NdArray<dtype>& inArray, uint16 inMinLength = 0)
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
	NdArray<dtype> bincount(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0)
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
	NdArray<dtype> bitwise_and(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
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
	NdArray<dtype> bitwise_not(const NdArray<dtype>& inArray)
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
	NdArray<dtype> bitwise_or(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
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
	NdArray<dtype> bitwise_xor(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
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
	NdArray<dtype> byteswap(const NdArray<dtype>& inArray)
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
	double cbrt(dtype inValue)
	{
		return static_cast<double>(std::pow(static_cast<double>(inValue), 1. / 3.));
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
	NdArray<double> cbrt(const NdArray<dtype>& inArray)
	{
		NdArray<double> returnArray(inArray.shape());
		std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return cbrt(static_cast<double>(inValue)); });

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
	dtype ceil(dtype inValue)
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
	NdArray<dtype> ceil(const NdArray<dtype>& inArray)
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
	dtype clip(dtype inValue, dtype inMinValue, dtype inMaxValue)
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
	NdArray<dtype> clip(const NdArray<dtype>& inArray, dtype inMinValue, dtype inMaxValue)
	{
		return std::move(inArray.clip(inMinValue, inMaxValue));
	}

	//============================================================================
	// Method Description: 
	//						Stack 1-D arrays as columns into a 2-D array.
	//		
	// Inputs:
	//				NdArray 1
	//				NdArray 2
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> column_stack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

	}

	//============================================================================
	// Method Description: 
	//						Join a sequence of arrays along an existing axis.
	//		
	// Inputs:
	//				NdArray 1
	//				NdArray 2
	//				(Optional) Axis (Default row)
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> concatenate(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, Axis::Type inAxis = Axis::ROW)
	{

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
	NdArray<dtype> copy(const NdArray<dtype>& inArray)
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
	NdArray<dtype> copySign(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
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
	void copyto(NdArray<dtype>& inDestArray, const NdArray<dtype>& inSrcArray)
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
	double cos(dtype inValue)
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
	NdArray<double> cos(const NdArray<dtype>& inArray)
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
	double cosh(dtype inValue)
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
	NdArray<double> cosh(const NdArray<dtype>& inArray)
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
	NdArray<uint32> count_nonzero(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::ROW)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> cross(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, Axis::Type inAxis = Axis::ROW)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> cube(const NdArray<dtype>& inArray)
	{
		NdArray<dtypeOut> returnArray(inArray.shape());
		std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return cube(static_cast<dtypeOut>(inValue)); });

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> cumprod(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> cumsum(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	double deg2rad(dtype inValue)
	{
		return inValue * pi / 180.0;
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
	NdArray<double> deg2rad(const NdArray<dtype>& inArray)
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
	NdArray<dtype> deleteIndices(const NdArray<dtype>& inArray, const NdArray<uint32>& inArrayIdxs, Axis::Type inAxis = Axis::NONE)
	{

	}

	//============================================================================
	// Method Description: 
	//						Return the indices to access the main diagonal of an array.
	//		
	// Inputs:
	//				NdArray
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<uint32> diag_indices(const NdArray<dtype>& inArray)
	{

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
	NdArray<dtype> diagflat(const NdArray<dtype>& inArray)
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
	NdArray<dtype> diagonal(const NdArray<dtype>& inArray, uint32 inOffset = 0, Axis::Type inAxis = Axis::ROW)
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
	NdArray<dtype> diff(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> divide(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> dot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
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
	void dump(const NdArray<dtype>& inArray, const std::string& inFilename)
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
	NdArray<dtype> empty(uint32 inNumRows, uint32 inNumCols)
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
	NdArray<dtype> empty(const Shape& inShape)
	{
		return std::move(NdArray<dtype>(inShape));
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
	NdArray<dtype> empty(std::initializer_list<uint32>& inShapeList)
	{
		return std::move(NdArray<dtype>(inShapeList));
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> empty_like(const NdArray<dtype>& inArray)
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
	Endian::Type endianess(const NdArray<dtype>& inArray)
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
	NdArray<bool> equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
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
	double exp(dtype inValue)
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
	NdArray<double> exp(const NdArray<dtype>& inArray)
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
	double exp2(dtype inValue)
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
	NdArray<double> exp2(const NdArray<dtype>& inArray)
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
	double expm1(dtype inValue)
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
	NdArray<double> expm1(const NdArray<dtype>& inArray)
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
	NdArray<dtype> eye(uint32 inN, int32 inK = 0)
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
	NdArray<dtype> eye(uint32 inN, uint32 inM, int32 inK = 0)
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
	NdArray<dtype> eye(const Shape& inShape, int32 inK = 0)
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
	dtype fix(dtype inValue)
	{

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
	NdArray<dtype> fix(const NdArray<dtype>& inArray)
	{

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
	NdArray<dtype> flatten(const NdArray<dtype>& inArray)
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
	NdArray<uint16> flatnonzero(const NdArray<dtype>& inArray)
	{

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
	NdArray<dtype> flip(const NdArray<dtype>& inArray, Axis::Type)
	{

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
	NdArray<dtype> fliplr(const NdArray<dtype>& inArray)
	{

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
	NdArray<dtype> flipud(const NdArray<dtype>& inArray)
	{

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
	dtype floor(dtype inValue)
	{

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
	NdArray<dtype> floor(const NdArray<dtype>& inArray)
	{

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
	dtype floor_divide(dtype inValue1, dtype inValue2)
	{

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
	NdArray<dtype> floor_divide(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	dtype fmax(dtype inValue1, dtype inValue2)
	{

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
	NdArray<dtype> fmax(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	dtype fmin(dtype inValue1, dtype inValue2)
	{

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
	NdArray<dtype> fmin(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	dtype fmod(dtype iValue1, dtype inValue2)
	{

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
	NdArray<dtype> fmod(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

	}

	//============================================================================
	// Method Description: 
	//						Construct an array from data in a text or binary file.
	//		
	// Inputs:
	//				filename
	//				seperator, Separator between items if file is a text file. Empty (“”) 
	//							separator means the file should be treated as binary.
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> fromfile(const std::string& inFilename, const std::string& inSep = "")
	{

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
	NdArray<dtype> full(uint32 inNumRows, uint32 inNumCols, dtype inFillValue)
	{
		return std::move(NdArray<dtype>(inNumRows, inNumCols).fill(inFillValue));
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
	NdArray<dtype> full(const Shape& inShape, dtype inFillValue)
	{
		return std::move(NdArray<dtype>(inShape).fill(inFillValue));
	}

	//============================================================================
	// Method Description: 
	//						Return a new array of given shape and type, filled with inFillValue
	//		
	// Inputs:
	//				initializer_list
	//				fill value
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> full(std::initializer_list<uint32>& inShapeList, dtype inFillValue)
	{
		return std::move(NdArray<dtype>(inShapeList).fill(inFillValue));
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> full_like(const NdArray<dtype>& inArray, dtype inFillValue)
	{
		return std::move(NdArray<dtype>(inArray, shape()).fill(inFillValue));
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
	NdArray<dtype> greater(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> greater_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> histogram(const NdArray<dtype>& inArray, uint32 inNumBins = 10)
	{

	}

	//============================================================================
	// Method Description: 
	//						Stack arrays in sequence horizontally (column wise).
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
	NdArray<dtype> hstack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	template<typename dtype>
	dtype hypot(dtype inValue1, dtype inValue2)
	{

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
	template<typename dtype>
	NdArray<dtype> hypot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> identity(uint16 inSquareSize)
	{

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
	NdArray<dtype> intersect1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> invert(const NdArray<dtype>& inArray)
	{

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
	//				relative tolerance
	//				absolute tolerance
	//				
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<bool> isclose(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, double inRtol = 1e-05, double inAtol = 1e-08)
	{

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
	bool isnan(dtype inValue)
	{

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
	NdArray<bool> isnan(const NdArray<dtype>& inArray)
	{

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
	dtype ldexp(dtype inValue1, dtype inValue2)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> ldexp(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> left_shift(const NdArray<dtype>& inArray, uint8 inNumBits)
	{

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
	NdArray<bool> less(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<bool> less_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	//		
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
	NdArray<dtype> linspace(dtype inStart, dtype inStop, uint16 inNum = 50, bool endPoint = true)
	{

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
	NdArray<dtype> load(const std::string& inFilename)
	{

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
	double log(dtype inValue)
	{

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
	NdArray<double> log(const NdArray<dtype>& inArray)
	{

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
	double log10(dtype inValue)
	{

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
	NdArray<double> log10(const NdArray<dtype>& inArray)
	{

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
	double log1p(dtype inValue)
	{

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
	NdArray<double> log1p(const NdArray<dtype>& inArray)
	{

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
	double log2(dtype inValue)
	{

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
	NdArray<double> log2(const NdArray<dtype>& inArray)
	{

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
	NdArray<bool> logical_and(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<bool> logical_not(const NdArray<dtype>& inArray)
	{

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
	NdArray<bool> logical_or(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<bool> logical_xor(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> matmult(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> max(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtype> maximum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<double> mean(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtype> median(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtype> min(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtype> minimum(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> mod(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> multiply(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<uint16> nanargmax(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	NdArray<uint16> nanargmin(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> nancumprod(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> nancumsum(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	NdArray<dtype> nanmax(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	NdArray<double> nanmean(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	NdArray<dtype> nanmedian(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	NdArray<dtype> nanmin(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<double> nanpercentile(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> nanprod(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	NdArray<double> nanstd(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> nansum(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	NdArray<double> nanvar(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	uint64 nbytes(const NdArray<dtype>& inArray)
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
	dtype newbyteorder(dtype inValue, Endian::Type inEndiness)
	{

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
	NdArray<dtype> newbyteorder(const NdArray<dtype>& inArray, Endian::Type inEndiness)
	{
		return std::move(inArray.newbyteorder(inEndiness));
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> negative(const NdArray<dtype>& inArray)
	{

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
	NdArray<uint32> nonzero(const NdArray<dtype>& inArray)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> norm(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return std::move(inArray.norm(inAxis));
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
	NdArray<bool> not_equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> ones(uint32 inNumRows, uint32 inNumCols)
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
	NdArray<dtype> ones(const Shape& inShape)
	{
		return std::move(full(inShape, static_cast<dtype>(1)));
	}

	//============================================================================
	// Method Description: 
	//						Return a new array of given shape and type, filled with ones.
	//		
	// Inputs:
	//				initializer_list
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> ones(std::initializer_list<uint32>& inShapeList)
	{
		return std::move(full(inShapeList, static_cast<dtype>(1)));
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> ones_like(const NdArray<dtype>& inArray)
	{
		return std::move(NdArray<dtype>(inArray.shape()).ones());
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
	NdArray<dtype> pad(const NdArray<dtype>& inArray, uint16 inPadWidth, dtype inPadValue)
	{

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
	NdArray<dtype> partition(const NdArray<dtype>& inArray, uint32 inKth, Axis::Type inAxis = Axis::NONE)
	{
		NdArray<dtype> returnArray(inArray);
		returnArray.partition(inKth, inAxis);
		return std::move(returnArray);
	}

	//============================================================================
	// Method Description: 
	//						Pads an array.
	//		
	// Inputs:
	//				NdArray
	//				percentile, must be in the range [0, 100]
	//				(Optional) axis
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<double> percentile(const NdArray<dtype>& inArray, double inPercentile, Axis::Type inAxis = Axis::NONE)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> power(const NdArray<dtype>& inArray, uint8 inExponent)
	{
		NdArray<dtypeOut> returnArray(inArray.shape());
		std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), 
			[inExponent](dtype inValue) { return power(static_cast<dtypeOut>(inValue), inExponent); });

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> power(const NdArray<dtype>& inArray, const NdArray<uint8>& inExponents)
	{
		NdArray<dtypeOut> returnArray(inArray.shape());
		std::transform(inArray.cbegin(), inArray.cend(), inExponents.cbegin(), returnArray.begin(), 
			[](dtype inValue, uint8 inExponent) { return power(static_cast<dtypeOut>(inValue), inExponent); });

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
	void print(const NdArray<dtype>& inArray)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> prod(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtype> ptp(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtype> put(const NdArray<dtype>& inArray, const NdArray<dtype>& inIndices, const NdArray<dtype>& inValues)
	{

	}

	//============================================================================
	// Method Description: 
	//						Changes elements of an array based on conditional and input values.
	//
	//						Sets a.flat[n] = values[n] for each n where mask.flat[n] == True.

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
	NdArray<dtype> put_mask(const NdArray<dtype>& inArray, const NdArray<bool>& inMask, const NdArray<dtype>& inValues)
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
	double rad2deg(dtype inValue)
	{
		return inValue * 180.0 / pi;
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
	NdArray<double> rad2deg(const NdArray<dtype>& inArray)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> reciprocal(const NdArray<dtype>& inArray)
	{

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
	template<typename dtype>
	NdArray<dtype> remainder(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> repeat(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
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
	NdArray<dtype> repeat(const NdArray<dtype>& inArray, const Shape& inRepeatShape)
	{
		return std::move(inArray.repeat(inRepeatShape));
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
	NdArray<dtype> repeat(const NdArray<dtype>& inArray, std::initializer_list<uint32>& inRepeatShapeList)
	{
		return std::move(inArray.repeat(inRepeatShapeList));
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
	void reshape(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
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
	void reshape(const NdArray<dtype>& inArray, const Shape& inNewShape)
	{
		inArray.reshape(inNewShape);
	}

	//============================================================================
	// Method Description: 
	//						Gives a new shape to an array without changing its data.
	//		
	// Inputs:
	//				NdArray
	//				initializer_list
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	void reshape(const NdArray<dtype>& inArray, std::initializer_list<uint32>& inNewShapeList)
	{
		inArray.resizeFast(inNewShapeList);
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
	void resizeFast(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
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
	void resizeFast(const NdArray<dtype>& inArray, const Shape& inNewShape)
	{
		inArray.resizeFast(inNewShape);
	}

	//============================================================================
	// Method Description: 
	//						Change shape and size of array in-place. All previous
	//						data of the array is lost.
	//		
	// Inputs:
	//				NdArray
	//				initializer_list 
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	void resizeFast(const NdArray<dtype>& inArray, std::initializer_list<uint32>& inNewShapeList)
	{
		inArray.resize(inNewShapeList);
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
	void resizeSlow(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
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
	void resizeSlow(const NdArray<dtype>& inArray, const Shape& inNewShape)
	{
		inArray.resizeSlow(inNewShape);
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
	//				initializer_list 
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	void resizeSlow(const NdArray<dtype>& inArray, std::initializer_list<uint32>& inNewShapeList)
	{
		inArray.resizeSlow(inNewShapeList);
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
	dtype rint(dtype inValue)
	{

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
	NdArray<dtype> rint(const NdArray<dtype>& inArray)
	{

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
	NdArray<dtype> roll(const NdArray<dtype>& inArray, int32 inShift, Axis::Type inAxis = Axis::NONE)
	{

	}

	//============================================================================
	// Method Description: 
	//						Rotate an array by 90 degrees in the plane.
	//		
	// Inputs:
	//				NdArray 
	//				the number of times to rotate 90 degrees
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> rot90(const NdArray<dtype>& inArray, uint8 inK)
	{

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
	dtype round(dtype inValue, uint8 inDecimals)
	{

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
	NdArray<dtype> round(const NdArray<dtype>& inArray, uint8 inDecimals)
	{
		return std::move(inArray.round(inDecimals));
	}

	//============================================================================
	// Method Description: 
	//						Stack arrays in sequence vertically (row wise).
	//
	// Inputs:
	//				NdArray 1
	//				NdArray 2
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> row_stack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> setdiff1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArra2)
	{

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
	Shape shape(const NdArray<dtype>& inArray)
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
	NdArray<dtype> sign(const NdArray<dtype>& inArray)
	{

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
	NdArray<dtype> signbit(const NdArray<dtype>& inArray)
	{

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
	double sin(dtype inValue)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<double> sin(const NdArray<dtype>& inArray)
	{

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
	double sinc(dtype inValue)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<double> sinc(const NdArray<dtype>& inArray)
	{

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
	double sinh(dtype inValue)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<double> sinh(const NdArray<dtype>& inArray)
	{

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
	uint32 size(const NdArray<dtype>& inArray)
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
	NdArray<dtype> sort(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		NdArray<dtype> returnArray(inArray);
		returnArray.sort(inAxis);
		return std::move(returnArray);
	}

	//============================================================================
	// Method Description: 
	//						squares the elements of the array
	//		
	// Inputs:
	//				NdArray
	// Outputs:
	//				NdArray
	//
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> sqr(const NdArray<dtype>& inArray)
	{
		NdArray<dtypeOut> returnArray(inArray.shape());
		std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), [](dtype inValue) { return sqr(static_cast<dtypeOut>(inValue)); });

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
	double sqrt(dtype inValue)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<double> sqrt(const NdArray<dtype>& inArray)
	{

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
	template<typename dtype, typename dtypeOut>
	dtype square(dtype inValue)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> square(const NdArray<dtype>& inArray)
	{

	}

	//============================================================================
	// Method Description: 
	//						Join a sequence of arrays along a new axis.
	//		
	// Inputs:
	//				NdArray
	//				Axis
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> stack(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::ROW)
	{

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
	NdArray<double> std(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> sum(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtype> swapaxes(const NdArray<dtype>& inArray)
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
	double tan(dtype inValue)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<double> tan(const NdArray<dtype>& inArray)
	{

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
	double tanh(dtype inValue)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<double> tanh(const NdArray<dtype>& inArray)
	{

	}

	//============================================================================
	// Method Description: 
	//						Construct an array by repeating A the number of times given by reps.
	//		
	// Inputs:
	//				initializer_list
	//				numRows
	//				numCols
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> tile(const NdArray<dtype>& inArray, uint32 inNumRows, uint32 inNumCols)
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
	NdArray<dtype> tile(const NdArray<dtype>& inArray, const Shape& inReps)
	{
		return std::move(inArray.repeat(inReps));
	}

	//============================================================================
	// Method Description: 
	//						Construct an array by repeating A the number of times given by reps.
	//		
	// Inputs:
	//				NdArray
	//				initializer_list
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> tile(const NdArray<dtype>& inArray, std::initializer_list<uint32>& inRepsList)
	{
		return std::move(inArray.repeat(inRepsList));
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
	void tofile(const NdArray<dtype>& inArray, const std::string& inFilename, const std::string& inSep = "")
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
	std::vector<dtype> toStlVector(const NdArray<dtype>& inArray)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> trace(const NdArray<dtype>& inArray, uint16 inOffset = 0, Axis::Type inAxis = Axis::ROW)
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
	NdArray<dtype> tranpose(const NdArray<dtype>& inArray)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> trapz(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> trapz(const NdArray<dtype>& inArrayX, const NdArray<dtype>& inArrayY, Axis::Type inAxis = Axis::NONE)
	{

	}

	//============================================================================
	// Method Description: 
	//						An array with ones at and below the given diagonal and zeros elsewhere.
	//		
	// Inputs:
	//				N, number of rows and cols
	//				Offset from main diaganol, default = 0, negaive=above, positve=below
	//				
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> tri(uint16 inN, int16 inOffset = 0)
	{

	}

	//============================================================================
	// Method Description: 
	//						An array with ones at and below the given diagonal and zeros elsewhere.
	//		
	// Inputs:
	//				N, number of rows
	//				M, number of columns
	//				Offset from main diaganol, default = 0, negative=above, positve=below
	//				
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> tri(uint16 inN, uint16 inM, int16 inOffset = 0)
	{

	}

	//============================================================================
	// Method Description: 
	//						Lower triangle of an array.
	//		
	// Inputs:
	//				N, number of rows and cols
	//				Offset from main diaganol, default = 0, negative=above, positve=below
	//				
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> tril(uint16 inN, int16 inOffset = 0)
	{

	}

	//============================================================================
	// Method Description: 
	//						Lower triangle of an array.
	//		
	// Inputs:
	//				N, number of rows
	//				M, number of columns
	//				Offset from main diaganol, default = 0, negative=above, positve=below
	//				
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> tril(uint16 inN, uint16 inM, int16 inOffset = 0)
	{

	}

	//============================================================================
	// Method Description: 
	//						Upper triangle of an array.
	//		
	// Inputs:
	//				N, number of rows and cols
	//				Offset from main diaganol, default = 0, negative=above, positve=below
	//				
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> triu(uint16 inN, int16 inOffset = 0)
	{


	}

	//============================================================================
	// Method Description: 
	//						Upper triangle of an array.
	//		
	// Inputs:
	//				N, number of rows
	//				M, number of columns
	//				Offset from main diaganol, default = 0, negative=above, positve=below
	//				
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> triu(uint16 inN, uint16 inM, int16 inOffset = 0)
	{

	}

	//============================================================================
	// Method Description: 
	//						Trim the leading and/or trailing zeros from a value.
	//		
	// Inputs:
	//				value
	//				string, "f" = front, "b" = back, "fb" = front and back
	//
	// Outputs:
	//				value
	//
	template<typename dtype>
	dtype trimZeros(dtype inValue, const std::string inTrim)
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
	NdArray<dtype> trimZeros(const NdArray<dtype>& inArray1, const std::string inTrim)
	{

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
	dtype trunc(dtype inValue)
	{


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
	NdArray<dtype> trunc(const NdArray<dtype>& inArray)
	{

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
	NdArray<dtype> union1d(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> unique(const NdArray<dtype>& inArray)
	{

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
	dtype unwrap(dtype inValue)
	{

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
	NdArray<dtype> unwrap(const NdArray<dtype>& inArray)
	{

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
	NdArray<double> var(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return std::move(inArray.var(inAxis));
	}

	//============================================================================
	// Method Description: 
	//						Compute the variance along the specified axis.
	//		
	// Inputs:
	//				NdArray 1
	//				NdArray 2
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> vstack(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	NdArray<dtype> zeros(uint32 inNumRows, uint32 inNumCols)
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
	NdArray<dtype> zeros(const Shape& inShape)
	{
		return std::move(full(inShape, static_cast<dtype>(0)));
	}

	//============================================================================
	// Method Description: 
	//						Return a new array of given shape and type, filled with zeros.
	//		
	// Inputs:
	//				initializer_list
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> zeros(std::initializer_list<uint32>& inShapeList)
	{
		return std::move(full(inShapeList, static_cast<dtype>(0)));
	}
}