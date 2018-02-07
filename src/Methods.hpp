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

namespace NumC
{
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
	uint16 alen(const NdArray<dtype>& inArray)
	{

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
	//				stop value, start is assumed to be zero
	//				(Optional) step value, defaults to 1
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> arange(dtype inStop, double inStep = 1.0)
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
	NdArray<dtype> arange(dtype inStart, dtype inStop, double inStep = 1.0)
	{

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
	NdArray<double> arcstanh(const NdArray<dtype>& inArray)
	{

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
	NdArray<uint16> argmax(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	NdArray<uint16> argmin(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	NdArray<uint16> argsort(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
		return NdArray<dtype>(inVector);
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
	NdArray<dtype> asarray(std::initializer_list<dtype> inList)
	{
		return NdArray<dtype>(inList);
	}

	//============================================================================
	// Method Description: 
	//						Convert an array of size 1 to its scalar equivalent.
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

	}

	//============================================================================
	// Method Description: 
	//						Convert an array of size 1 to its scalar equivalent.
	//		
	// Inputs:
	//				NdArray
	//				(Optional) NdArray of weights, otherwise all weights = 1
	//				(Optional) axis
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<double> average(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, Axis::Type inAxis = Axis::NONE)
	{

	}

	//============================================================================
	// Method Description: 
	//						Count number of occurrences of each value in array of non-negative ints.
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
	NdArray<uint32> bincount(const NdArray<dtype>& inArray, uint16 inMinLength = 0)
	{

	}

	//============================================================================
	// Method Description: 
	//						Count number of occurrences of each value in array of non-negative ints.
	//
	//						The number of bins(of size 1) is one larger than the largest value in x.
	//						If minlength is specified, there will be at least this number of bins in 
	//						the output array(though it will be longer if necessary, depending on the 
	//						contents of x).Each bin gives the number of occurrences of its index value 
	//						in x.If weights is specified the input array is weighted by it, i.e. if a 
	//						value n is found at position i, out[n] += weight[i] instead of out[n] += 1.
	//		
	// Inputs:
	//				NdArray
	//				NdArray weights
	//				min bin length
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<uint32> bincount(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0)
	{

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
	NdArray<dtype> bitwise_and(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0)
	{

	}

	//============================================================================
	// Method Description: 
	//						Compute the bit-wise NOT of two arrays element-wise.
	//		
	// Inputs:
	//				NdArray 1
	//				NdArray 2
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> bitwise_not(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0)
	{

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
	NdArray<dtype> bitwise_or(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0)
	{

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
	NdArray<dtype> bitwise_xor(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0)
	{

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
	NdArray<double> cbr(const NdArray<dtype>& inArray)
	{

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
	NdArray<dtype> copysign(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> cos(const NdArray<dtype>& inArray)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> cosh(const NdArray<dtype>& inArray)
	{

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
	NdArray<uint16> count_nonzero(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::ROW)
	{

	}

	//============================================================================
	// Method Description: 
	//						Counts the number of non-zero values in the array.
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
	NdArray<uint32> diagflat(const NdArray<dtype>& inArray)
	{

	}

	//============================================================================
	// Method Description: 
	//						Calculate the n-th discrete difference along given axis.
	//		
	// Inputs:
	//				NdArray
	//				(Optional) Axis
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<uint32> diff(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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

	}

	//============================================================================
	// Method Description: 
	//						Dot product of two arrays.
	//		
	// Inputs:
	//				NdArray 1
	//				NdArray 2
	//				(Optional) Axis
	// Outputs:
	//				NdArray
	//
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> dot(const NdArray<dtype>& inArray1, const NdArray<uint32>& inArray2, Axis::Type inAxis = Axis::NONE)
	{

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
		return NdArray<dtype>(inShape);
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
		return NdArray<dtypeOut>(inArray.shape());
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> exp(const NdArray<dtype>& inArray)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> exp2(const NdArray<dtype>& inArray)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> expm1(const NdArray<dtype>& inArray)
	{

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
	NdArray<dtype> eye(uint16 inN, int32 inK = 0)
	{

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
	NdArray<dtype> eye(uint16 inN, uint16 inM, int32 inK = 0)
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
	//				Shape
	//				fill value
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> full(const Shape& inShape, dtype inFillValue)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> mean(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> median(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> nanmean(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> nanmedian(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtypeOut> nanpercentile(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> nanstd(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> nanvar(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	template<typename dtype>
	NdArray<dtype> negative(const NdArray<dtype>& inArray)
	{

	}

	//============================================================================
	// Method Description: 
	//						Return the indices of the elements that are non-zero.
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
	//						Converts the number into a string
	//		
	// Inputs:
	//				number
	// Outputs:
	//				string
	//
	template<typename dtype>
	std::string num2str(dtype inNumber)
	{
		std::ostringstream ss;
		ss << inNumber;
		return ss.str();
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
	NdArray<dtype> percentile(const NdArray<dtype>& inArray, double inPercentile, Axis::Type inAxis = Axis::NONE)
	{

	}

	//============================================================================
	// Method Description: 
	//						Raises the elements of the array to the input power
	//		
	// Inputs:
	//				NdArray
	//				power
	// Outputs:
	//				NdArray
	//
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> power(const NdArray<dtype>& inArray, double inPower)
	{

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

	}

	//============================================================================
	// Method Description: 
	//						Prints the array
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
	//				NdArray
	//
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> rad2deg(const NdArray<dtype>& inArray)
	{

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
	NdArray<dtype> reshape(const NdArray<dtype>& inArray, const Shape& inNewShape)
	{

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
	NdArray<dtype> resize(const NdArray<dtype>& inArray, const Shape& inNewShape)
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
	//						Trigonometric sine, element-wise.
	//		
	// Inputs:
	//				NdArray 
	// Outputs:
	//				NdArray
	//
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> sin(const NdArray<dtype>& inArray)
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
	NdArray<dtypeOut> sinc(const NdArray<dtype>& inArray)
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
	NdArray<dtypeOut> sinh(const NdArray<dtype>& inArray)
	{

	}

	//============================================================================
	// Method Description: 
	//						Return the number of elements along a given axis.
	//		
	// Inputs:
	//				NdArray 
	//				(Optional) Axis
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<uint32> size(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	NdArray<dtypeOut> sqrt(const NdArray<dtype>& inArray)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> std(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> swapaxes(const NdArray<dtype>& inArray)
	{

	}

	//============================================================================
	// Method Description: 
	//						Take elements from an array along an axis.
	//		
	// Inputs:
	//				NdArray
	//				NdArray indices
	//				(optional) Axis
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> take(const NdArray<dtype>& inArray, const NdArray<uint16>& inIndices, Axis::Type inAxis = Axis::NONE)
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
	NdArray<dtypeOut> tan(const NdArray<dtype>& inArray)
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
	NdArray<dtypeOut> tanh(const NdArray<dtype>& inArray)
	{

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

	}

	//============================================================================
	// Method Description: 
	//						Return the sum along diagonals of the array.
	//		
	// Inputs:
	//				NdArray
	//				Offset from main diaganol, default = 0, negative=above, positve=below
	// Outputs:
	//				NdArray
	//
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> trace(const NdArray<dtype>& inArray, int16 inOffset=0)
	{

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
	NdArray<dtype> tri(uint16 inN, int16 inOffset=0)
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
	template<typename dtype, typename dtypeOut>
	NdArray<dtypeOut> var(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{

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
	//				Shape
	// Outputs:
	//				NdArray
	//
	template<typename dtype>
	NdArray<dtype> zeros(const Shape& inShape)
	{
		NdArray<double> returnArray(inShape);
		returnArray.zeros();
		return returnArray;
	}
}