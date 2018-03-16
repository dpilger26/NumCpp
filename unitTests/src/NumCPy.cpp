#include"NumC.hpp"

#include<string>
#include<iostream>

#ifndef BOOST_PYTHON_STATIC_LIB
#define BOOST_PYTHON_STATIC_LIB    
#endif

#ifndef BOOST_NUMPY_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB    
#endif

#include "boost/python.hpp"
#include "boost/python/suite/indexing/vector_indexing_suite.hpp" // needed for returning a std::vector directly
#include "boost/python/return_internal_reference.hpp" // needed for returning references and pointers
#include "boost/python/numpy.hpp" // needed for working with numpy 
// i don't know why, but google said these are needed to fix a linker error i was running into for numpy. 
#define BOOST_LIB_NAME "boost_numpy"
#include "boost/config/auto_link.hpp"

namespace bp = boost::python;
namespace np = boost::python::numpy;

using namespace NumC;

//================================================================================

namespace ShapeInterface
{
	bool testListContructor()
	{
		Shape test = { 357, 666 };
		if (test.rows == 357 && test.cols == 666)
		{
			return true;
		}

		return false;
	}
}

//================================================================================

namespace SliceInterface
{
	bool testListContructor()
	{
		Slice test1 = { 666 };
		if (!(test1.start == 0 && test1.stop == 666 && test1.step == 1))
		{
			return false;
		}

		Slice test2 = { 357, 777 };
		if (!(test2.start == 357 && test2.stop == 777 && test2.step == 1))
		{
			return false;
		}

		Slice test3 = { 7, 45, 10 };
		if (!(test3.start == 7 && test3.stop == 45 && test3.step == 10))
		{
			return false;
		}

		return true;
	}
}

//================================================================================

namespace NdArrayInterface
{
	template<typename dtype>
	bool test1DListContructor()
	{
		NdArray<dtype> test = { 1,2,3,4,666,357,314159 };
		if (test.size() != 7)
		{
			return false;
		}

		if (test.shape().rows != 1 || test.shape().cols != test.size())
		{
			return false;
		}

		return test[0] == 1 && test[1] == 2 && test[2] == 3 && test[3] == 4 && test[4] == 666 && test[5] == 357 && test[6] == 314159;
	}

	//================================================================================

	template<typename dtype>
	bool test2DListContructor()
	{
		NdArray<dtype> test = { {1,2}, {4,666}, {314159, 9}, {0, 8} };
		if (test.size() != 8)
		{
			return false;
		}

		if (test.shape().rows != 4 || test.shape().cols != 2)
		{
			return false;
		}

		return test[0] == 1 && test[1] == 2 && test[2] == 4 && test[3] == 666 && test[4] == 314159 && test[5] == 9 && test[6] == 0 && test[7] == 8;
	}

	//================================================================================

	template<typename dtype>
	np::ndarray getNumpyArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(inArray);
	}

	//================================================================================

	template<typename dtype>
	void setArray(NdArray<dtype>& self, np::ndarray& inBoostArray)
	{
		BoostNdarrayHelper newNdArrayHelper(&inBoostArray);
		uint8 numDims = newNdArrayHelper.numDimensions();
		if (numDims > 2)
		{
			std::string errorString = "ERROR: Input array can only have up to 2 dimensions!";
			PyErr_SetString(PyExc_RuntimeError, errorString.c_str());
		}

		self = boostToNumC<dtype>(inBoostArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray all(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.all(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray any(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.any(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray argmax(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.argmax(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray argmin(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.argmin(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray argsort(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.argsort(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray clip(NdArray<dtype>& self, dtype inMin, dtype inMax)
	{
		return numCToBoost(self.clip(inMin, inMax));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray copy(NdArray<dtype>& self)
	{
		return numCToBoost(self.copy());
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray cumprod(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.cumprod<dtypeOut>(inAxis));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray cumsum(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.cumsum<dtypeOut>(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray diagonal(NdArray<dtype>& self, uint32 inOffset = 0, Axis::Type inAxis = Axis::ROW)
	{
		return numCToBoost(self.diagonal(inOffset, inAxis));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray dot(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self.dot<dtypeOut>(inOtherArray));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray fill(NdArray<dtype>& self, dtype inFillValue)
	{
		self.fill(inFillValue);
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray flatten(NdArray<dtype>& self)
	{
		return numCToBoost(self.flatten());
	}

	//================================================================================

	template<typename dtype>
	dtype getValueFlat(NdArray<dtype>& self, int32 inIndex)
	{
		return self.at(inIndex);
	}

	//================================================================================

	template<typename dtype>
	dtype getValueRowCol(NdArray<dtype>& self, int32 inRow, int32 inCol)
	{
		return self.at(inRow, inCol);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray getSlice1D(NdArray<dtype>& self, const Slice& inSlice)
	{
		return numCToBoost(self.at(inSlice));
	}

	//================================================================================

	template<typename dtype>
	bool testGetSlice1DList()
	{
		NdArray<dtype> test = {9,8,7,6,5,4,3,2,1,0};
		NdArray<dtype> slice = test.at({0,10,2});

		if (slice.size() != 5 || !(slice.shape().rows == 1 && slice.shape().cols == slice.size()))
		{
			return false;
		}

		return slice[0] == 9 && slice[1] == 7 && slice[2] == 5 && slice[3] == 3 && slice[4] == 1;
	}

	//================================================================================

	template<typename dtype>
	np::ndarray getSlice2D(NdArray<dtype>& self, const Slice& inRowSlice, const Slice& inColSlice)
	{
		return numCToBoost(self.at(inRowSlice, inColSlice));
	}

	//================================================================================

	template<typename dtype>
	bool testGetSlice2DList()
	{
		NdArray<dtype> test = { {9,8},{7,6},{5,4},{3,2},{1,0} };
		NdArray<dtype> slice = test.at({ 1,3,1 }, {0, 1, 2});

		if (slice.size() != 2 || !(slice.shape().rows == 2 && slice.shape().cols == 1))
		{
			return false;
		}

		return slice[0] == 7 && slice[1] == 5;
	}

	//================================================================================

	template<typename dtype>
	np::ndarray max(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.max(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray min(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.min(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray mean(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.mean(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray median(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.median(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray newbyteorder(NdArray<dtype>& self, Endian::Type inEndiness = Endian::NATIVE)
	{
		return numCToBoost(self.newbyteorder(inEndiness));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray nonzero(NdArray<dtype>& self)
	{
		return numCToBoost(self.nonzero());
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray norm(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost<dtypeOut>(self.norm<dtypeOut>(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray ones(NdArray<dtype>& self)
	{
		self.ones();
		return numCToBoost<dtype>(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray partition(NdArray<dtype>& self, uint32 inKth, Axis::Type inAxis = Axis::NONE)
	{
		self.partition(inKth, inAxis);
		return numCToBoost<dtype>(self);
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray prod(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost<dtypeOut>(self.prod<dtypeOut>(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray ptp(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.ptp(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putFlat(NdArray<dtype>& self, int32 inIndex, dtype inValue)
	{
		self.put(inIndex, inValue);
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putRowCol(NdArray<dtype>& self, int32 inRow, int32 inCol, dtype inValue)
	{
		self.put(inRow, inCol, inValue);
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putSlice1DValue(NdArray<dtype>& self, const Slice& inSlice, dtype inValue)
	{
		self.put(inSlice, inValue);
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putSlice1DValues(NdArray<dtype>& self, const Slice& inSlice, np::ndarray& inArrayValues)
	{
		NdArray<dtype> inValues = boostToNumC<dtype>(inArrayValues);
		self.put(inSlice, inValues);
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putSlice2DValue(NdArray<dtype>& self, const Slice& inSliceRow, const Slice& inSliceCol, dtype inValue)
	{
		self.put(inSliceRow, inSliceCol, inValue);
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putSlice2DValues(NdArray<dtype>& self, const Slice& inSliceRow, const Slice& inSliceCol, np::ndarray& inArrayValues)
	{
		NdArray<dtype> inValues = boostToNumC<dtype>(inArrayValues);
		self.put(inSliceRow, inSliceCol, inValues);
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	bool testPutSlice1DValueList()
	{
		NdArray<dtype> test = { 9,8,7,6,5,4,3,2,1,0 };
		test.put({ 0,10,2 }, 666);

		for (uint32 i = 0; i < 10; i += 2)
		{
			if (test[i] != 666)
			{
				return false;
			}
		}

		return true;
	}

	//================================================================================

	template<typename dtype>
	bool testPutSlice1DValuesList()
	{
		NdArray<dtype> test = { 9,8,7,6,5,4,3,2,1,0 };
		Slice theSlice = { 0,10,2 };
		std::vector<dtype> values;
		for (uint32 i = 0; i < theSlice.numElements(test.size()); ++i)
		{
			values.push_back(666);
		}
		test.put({ 0,10,2 }, NdArray<dtype>(values));

		for (int32 i = theSlice.start; i < theSlice.stop; i += theSlice.step)
		{
			if (test[i] != 666)
			{
				return false;
			}
		}

		return true;
	}

	//================================================================================

	template<typename dtype>
	bool testPutSlice2DValueList()
	{
		NdArray<dtype> test = { { 1,1,1,1,1 },{ 2,2,2,2,2 },{ 3,3,3,3,3 },{ 4,4,4,4,4 },{ 5,5,5,5,5 } };
		test.put({ 0,5,2 }, { 0,5,2 }, 666);

		for (uint32 row = 0; row < 5; row += 2)
		{
			for (uint32 col = 0; col < 5; col += 2)
			{
				if (test(row, col) != 666)
				{
					return false;
				}
			}
		}

		return true;
	}

	//================================================================================

	template<typename dtype>
	bool testPutSlice2DValuesList()
	{
		NdArray<dtype> test = { { 1,1,1,1,1 },{ 2,2,2,2,2 },{ 3,3,3,3,3 },{ 4,4,4,4,4 },{ 5,5,5,5,5 } };
		Slice theSlice = { 0,5,2 };
		std::vector<dtype> values;
		for (uint32 i = 0; i < sqr(theSlice.numElements(test.shape().rows)); ++i)
		{
			values.push_back(666);
		}
		test.put({ 0,5,2 }, { 0,5,2 }, NdArray<dtype>(values));

		for (uint32 row = 0; row < 5; row += 2)
		{
			for (uint32 col = 0; col < 5; col += 2)
			{
				if (test(row, col) != 666)
				{
					return false;
				}
			}
		}

		return true;
	}

	//================================================================================

	template<typename dtype>
	np::ndarray repeat(NdArray<dtype>& self, const Shape& inRepeatShape)
	{
		return numCToBoost(self.repeat(inRepeatShape));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray reshape(NdArray<dtype>& self, const Shape& inShape)
	{
		self.reshape(inShape);
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray reshapeList(NdArray<dtype>& self, const Shape& inShape)
	{
		self.reshape({ inShape.rows, inShape.cols });
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray resizeFast(NdArray<dtype>& self, const Shape& inShape)
	{
		self.resizeFast(inShape);
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray resizeFastList(NdArray<dtype>& self, const Shape& inShape)
	{
		self.resizeFast({ inShape.rows, inShape.cols });
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray resizeSlow(NdArray<dtype>& self, const Shape& inShape)
	{
		self.resizeSlow(inShape);
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray resizeSlowList(NdArray<dtype>& self, const Shape& inShape)
	{
		self.resizeSlow({ inShape.rows, inShape.cols });
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray round(NdArray<dtype>& self, uint8 inNumDecimals)
	{
		return numCToBoost(self.round(inNumDecimals));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray sort(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		self.sort(inAxis);
		return numCToBoost(self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray std(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.std(inAxis));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray sum(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.sum<dtypeOut>(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray swapaxes(NdArray<dtype>& self)
	{
		return numCToBoost(self.swapaxes());
	}

	//================================================================================

	template<typename dtype>
	np::ndarray transpose(NdArray<dtype>& self)
	{
		return numCToBoost(self.transpose());
	}

	//================================================================================

	template<typename dtype>
	np::ndarray var(NdArray<dtype>& self, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(self.var(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorPlusScalar(NdArray<dtype>& self, dtype inScalar)
	{
		return numCToBoost(self + inScalar);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorPlusArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self + inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorMinusScalar(NdArray<dtype>& self, dtype inScalar)
	{
		return numCToBoost(self - inScalar);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorMinusArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self - inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorMultiplyScalar(NdArray<dtype>& self, dtype inScalar)
	{
		return numCToBoost(self * inScalar);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorMultiplyArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self * inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorDivideScalar(NdArray<dtype>& self, dtype inScalar)
	{
		return numCToBoost(self / inScalar);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorDivideArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self / inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorModulusScalar(NdArray<dtype>& self, dtype inScalar)
	{
		return numCToBoost(self % inScalar);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorModulusArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self % inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorBitwiseOrScalar(NdArray<dtype>& self, dtype inScalar)
	{
		return numCToBoost(self | inScalar);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorBitwiseOrArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self | inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorBitwiseAndScalar(NdArray<dtype>& self, dtype inScalar)
	{
		return numCToBoost(self & inScalar);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorBitwiseAndArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self & inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorBitwiseXorScalar(NdArray<dtype>& self, dtype inScalar)
	{
		return numCToBoost(self ^ inScalar);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorBitwiseXorArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self ^ inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorBitwiseNot(NdArray<dtype>& self)
	{
		return numCToBoost(~self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorEqualityScalar(NdArray<dtype>& self, dtype inValue)
	{
		return numCToBoost(self == inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorEqualityArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self == inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorLessScalar(NdArray<dtype>& self, dtype inValue)
	{
		return numCToBoost(self < inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorLessArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self < inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorGreaterScalar(NdArray<dtype>& self, dtype inValue)
	{
		return numCToBoost(self > inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorGreaterArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self > inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorLessEqualScalar(NdArray<dtype>& self, dtype inValue)
	{
		return numCToBoost(self <= inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorLessEqualArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self <= inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorGreaterEqualScalar(NdArray<dtype>& self, dtype inValue)
	{
		return numCToBoost(self >= inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorGreaterEqualArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self >= inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorNotEqualityScalar(NdArray<dtype>& self, dtype inValue)
	{
		return numCToBoost(self != inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorNotEqualityArray(NdArray<dtype>& self, NdArray<dtype>& inOtherArray)
	{
		return numCToBoost(self != inOtherArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorBitshiftLeft(NdArray<dtype>& self, uint8 inNumBits)
	{
		return numCToBoost(self << inNumBits);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorBitshiftRight(NdArray<dtype>& self, uint8 inNumBits)
	{
		return numCToBoost(self >> inNumBits);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorPrePlusPlus(NdArray<dtype>& self)
	{
		return numCToBoost(++self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorPostPlusPlus(NdArray<dtype>& self)
	{
		return numCToBoost(self++);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorPreMinusMinus(NdArray<dtype>& self)
	{
		return numCToBoost(--self);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray operatorPostMinusMinus(NdArray<dtype>& self)
	{
		return numCToBoost(self--);
	}
}

//================================================================================

namespace MethodsInterface
{
	template<typename dtype>
	dtype absScalar(dtype inValue)
	{
		return abs(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray absArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(abs(inArray));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray addArrays(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(add<dtype, dtypeOut>(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray allArray(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(all(inArray, inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray anyArray(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(any(inArray, inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray argmaxArray(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(argmax(inArray, inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray argminArray(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(argmin(inArray, inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray argsortArray(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(argsort(inArray, inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray amaxArray(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(amax(inArray, inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray aminArray(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(amin(inArray, inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray arangeArray(dtype inStart, dtype inStop, dtype inStep)
	{
		return numCToBoost(arange<dtype>(inStart, inStop, inStep));
	}

	//================================================================================

	template<typename dtype>
	dtype arccosScalar(dtype inValue)
	{
		return arccos(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray arccosArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(arccos(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype arccoshScalar(dtype inValue)
	{
		return arccosh(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray arccoshArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(arccosh(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype arcsinScalar(dtype inValue)
	{
		return arcsin(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray arcsinArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(arcsin(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype arcsinhScalar(dtype inValue)
	{
		return arcsinh(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray arcsinhArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(arcsinh(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype arctanScalar(dtype inValue)
	{
		return arctan(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray arctanArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(arctan(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype arctan2Scalar(dtype inY, dtype inX)
	{
		return arctan2(inY, inX);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray arctan2Array(const NdArray<dtype>& inY, const NdArray<dtype>& inX)
	{
		return numCToBoost(arctan2(inY, inX));
	}

	//================================================================================

	template<typename dtype>
	dtype arctanhScalar(dtype inValue)
	{
		return arctanh(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray arctanhArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(arctanh(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype aroundScalar(dtype inValue, uint8 inNumDecimals)
	{
		return around(inValue, inNumDecimals);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray aroundArray(const NdArray<dtype>& inArray, uint8 inNumDecimals)
	{
		return numCToBoost(around(inArray, inNumDecimals));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray average(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.mean(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray averageWeighted(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(NumC::average(inArray, inWeights, inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray bincount(const NdArray<dtype>& inArray, uint16 inMinLength = 0)
	{
		return numCToBoost(NumC::bincount(inArray, inMinLength));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray bincountWeighted(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, uint16 inMinLength = 0)
	{
		return numCToBoost(NumC::bincount(inArray, inWeights, inMinLength));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray bitwise_and(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::bitwise_and(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray bitwise_not(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::bitwise_not(inArray));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray bitwise_or(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::bitwise_or(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray bitwise_xor(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::bitwise_xor(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray byteswap(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::byteswap(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype cbrtScalar(dtype inValue)
	{
		return cbrt(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray cbrtArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(cbrt(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype ceilScalar(dtype inValue)
	{
		return ceil(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray ceilArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(ceil(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype clipScalar(dtype inValue, dtype inMinValue, dtype inMaxValue)
	{
		return clip(inValue, inMinValue, inMaxValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray clipArray(const NdArray<dtype>& inArray, dtype inMinValue, dtype inMaxValue)
	{
		return numCToBoost(clip(inArray, inMinValue, inMaxValue));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray copy(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::copy(inArray));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray copySign(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::copySign(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray copyto(NdArray<dtype>& inArrayDest, const NdArray<dtype>& inArraySrc)
	{
		NumC::copyto(inArrayDest, inArraySrc);
		return numCToBoost(inArrayDest);
	}

	//================================================================================

	template<typename dtype>
	dtype cosScalar(dtype inValue)
	{
		return NumC::cos(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray cosArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::cos(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype coshScalar(dtype inValue)
	{
		return NumC::cosh(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray coshArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::cosh(inArray));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray count_nonzero(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::ROW)
	{
		return numCToBoost(NumC::count_nonzero(inArray, inAxis));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray cubeArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::cube<dtype, dtypeOut>(inArray));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray cumprodArray(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(cumprod<dtype, dtypeOut>(inArray, inAxis));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray cumsumArray(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(cumsum<dtype, dtypeOut>(inArray, inAxis));
	}

	//================================================================================

	template<typename dtype>
	dtype deg2radScalar(dtype inValue)
	{
		return NumC::deg2rad(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray deg2radArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::deg2rad(inArray));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray diagflat(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::diagflat(inArray));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray diagonal(const NdArray<dtype>& inArray, uint32 inOffset = 0, Axis::Type inAxis = Axis::ROW)
	{
		return numCToBoost(NumC::diagonal(inArray, inOffset, inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray diff(const NdArray<dtype>& inArray, Axis::Type inAxis = Axis::ROW)
	{
		return numCToBoost(NumC::diff(inArray, inAxis));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray divide(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::divide<dtype, dtypeOut>(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray dot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::dot<dtype, dtypeOut>(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray emptyRowCol(uint32 inNumRows, uint32 inNumCols)
	{
		return numCToBoost(NumC::empty<dtype>(inNumRows, inNumCols));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray emptyShape(const Shape& inShape)
	{
		return numCToBoost(NumC::empty<dtype>(inShape));
	}

	//================================================================================

	template<typename dtype>
	bool testEmptyList()
	{
		uint32 numRows = 4;
		uint32 numCols = 11;
		
		NdArray<dtype> theArray = NumC::empty<dtype>({ numRows, numCols });

		Shape theShape = theArray.shape();
		if (theShape.rows == numRows && theShape.cols == numCols && theArray.size() == numRows * numCols)
		{
			return true;
		}

		return false;
	}

	//================================================================================

	template<typename dtype>
	np::ndarray equal(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::equal(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype>
	dtype expScalar(dtype inValue)
	{
		return NumC::exp(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray expArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::exp(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype exp2Scalar(dtype inValue)
	{
		return NumC::exp2(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray exp2Array(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::exp2(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype expm1Scalar(dtype inValue)
	{
		return NumC::expm1(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray expm1Array(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::expm1(inArray));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray eye1D(uint32 inN, int32 inK)
	{
		return numCToBoost(NumC::eye<dtype>(inN, inK));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray eye2D(uint32 inN, uint32 inM, int32 inK)
	{
		return numCToBoost(NumC::eye<dtype>(inN, inM, inK));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray eyeShape(const Shape& inShape, int32 inK)
	{
		return numCToBoost(NumC::eye<dtype>(inShape, inK));
	}

	//================================================================================

	template<typename dtype>
	dtype fixScalar(dtype inValue)
	{
		return NumC::fix(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray fixArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::fix(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype floorScalar(dtype inValue)
	{
		return NumC::floor(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray floorArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::floor(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype floor_divideScalar(dtype inValue1, dtype inValue2)
	{
		return NumC::floor_divide(inValue1, inValue2);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray floor_divideArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::floor_divide(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype>
	dtype fmaxScalar(dtype inValue1, dtype inValue2)
	{
		return NumC::fmax(inValue1, inValue2);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray fmaxArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::fmax(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype>
	dtype fminScalar(dtype inValue1, dtype inValue2)
	{
		return NumC::fmin(inValue1, inValue2);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray fminArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::fmin(inArray1, inArray2));
	}

	template<typename dtype>
	dtype fmodScalar(dtype inValue1, dtype inValue2)
	{
		return NumC::fmod(inValue1, inValue2);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray fmodArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::fmod(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray fullRowCol(uint32 inNumRows, uint32 inNumCols, dtype inValue)
	{
		return numCToBoost(NumC::full(inNumRows, inNumCols, inValue));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray fullShape(const Shape& inShape, dtype inValue)
	{
		return numCToBoost(NumC::full(inShape, inValue));
	}

	//================================================================================

	template<typename dtype>
	bool testFullList(dtype inValue)
	{
		uint32 numRows = 4;
		uint32 numCols = 11;

		NdArray<dtype> theArray = NumC::full({ numRows, numCols }, inValue);

		Shape theShape = theArray.shape();
		if (theShape.rows == numRows && theShape.cols == numCols && theArray.size() == numRows * numCols && all(theArray == inValue).item())
		{
			return true;
		}

		return false;
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	dtype hypotScalar(dtype inValue1, dtype inValue2)
	{
		return NumC::hypot<dtype, dtypeOut>(inValue1, inValue2);
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray hypotArray(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
	{
		return numCToBoost(NumC::hypot<dtype, dtypeOut>(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype>
	bool isnanScalar(dtype inValue)
	{
		return NumC::isnan(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray isnanArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::isnan(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype ldexpScalar(dtype inValue1, uint8 inValue2)
	{
		return NumC::ldexp(inValue1, inValue2);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray ldexpArray(const NdArray<dtype>& inArray1, const NdArray<uint8>& inArray2)
	{
		return numCToBoost(NumC::ldexp(inArray1, inArray2));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray negative(const NdArray<dtypeOut> inArray)
	{
		return numCToBoost(NumC::negative<dtype, dtypeOut>(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype logScalar(dtype inValue)
	{
		return NumC::log(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray logArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::log(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype log10Scalar(dtype inValue)
	{
		return NumC::log10(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray log10Array(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::log10(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype log1pScalar(dtype inValue)
	{
		return NumC::log1p(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray log1pArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::log1p(inArray));
	}

	//================================================================================

	template<typename dtype>
	dtype log2Scalar(dtype inValue)
	{
		return NumC::log2(inValue);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray log2Array(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::log2(inArray));
	}

	template<typename dtype>
	dtype newbyteorderScalar(dtype inValue, Endian::Type inEndianess)
	{
		return NumC::newbyteorder(inValue, inEndianess);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray newbyteorderArray(const NdArray<dtype>& inArray, Endian::Type inEndianess)
	{
		return numCToBoost(NumC::newbyteorder(inArray, inEndianess));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray sqrArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(NumC::sqr<dtype, dtypeOut>(inArray));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray powerArrayScalar(const NdArray<dtype>& inArray, uint8 inExponent)
	{
		return numCToBoost(NumC::power<dtype, dtypeOut>(inArray, inExponent));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray powerArrayArray(const NdArray<dtype>& inArray, const NdArray<uint8>& inExponents)
	{
		return numCToBoost(NumC::power<dtype, dtypeOut>(inArray, inExponents));
	}


}

namespace RandomInterface
{
	template<typename dtype>
	np::ndarray permutationScalar(dtype inValue)
	{
		return numCToBoost(Random::permutation<dtype>(inValue));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray permutationArray(const NdArray<dtype>& inArray)
	{
		return numCToBoost(Random::permutation<dtype>(inArray));
	}
}

//================================================================================

BOOST_PYTHON_MODULE(NumC)
{
	Py_Initialize();
	np::initialize(); // needs to be called first thing in the BOOST_PYTHON_MODULE for numpy

	//http://www.boost.org/doc/libs/1_60_0/libs/python/doc/html/tutorial/tutorial/exposing.html

	bp::class_<std::vector<double> >("double_vector")
		.def(bp::vector_indexing_suite<std::vector<double> >());

	// Constants.hpp
	bp::scope().attr("e") = NumC::Constants::e;
	bp::scope().attr("pi") = NumC::Constants::pi;
	bp::scope().attr("nan") = NumC::Constants::nan;
	bp::scope().attr("version") = NumC::Constants::version;

	// DtypeInfo.hpp
	typedef DtypeInfo<uint32> DtypeInfoUint32;
	bp::class_<DtypeInfoUint32>
		("DtypeIntoUint32", bp::init<>())
		.def("bits", &DtypeInfoUint32::bits).staticmethod("bits")
		.def("epsilon", &DtypeInfoUint32::epsilon).staticmethod("epsilon")
		.def("isInteger", &DtypeInfoUint32::isInteger).staticmethod("isInteger")
		.def("min", &DtypeInfoUint32::min).staticmethod("min")
		.def("max", &DtypeInfoUint32::max).staticmethod("max");

	// Shape.hpp
	bp::class_<Shape>
		("Shape", bp::init<>())
		.def(bp::init<uint32>())
		.def(bp::init<uint32, uint32>())
		.def(bp::init<Shape>())
		.def("testListContructor", &ShapeInterface::testListContructor).staticmethod("testListContructor")
		.def_readwrite("rows", &Shape::rows)
		.def_readwrite("cols", &Shape::cols)
		.def("size", &Shape::size)
		.def("print", &Shape::print);

	// Slice.hpp
	bp::class_<Slice>
		("Slice", bp::init<>())
		.def(bp::init<int32>())
		.def(bp::init<int32, int32>())
		.def(bp::init<int32, int32, int32>())
		.def(bp::init<Slice>())
		.def("testListContructor", &SliceInterface::testListContructor).staticmethod("testListContructor")
		.def_readwrite("start", &Slice::start)
		.def_readwrite("stop", &Slice::stop)
		.def_readwrite("step", &Slice::step)
		.def("numElements", &Slice::numElements)
		.def("print", &Slice::print);

	// Timer.hpp
	typedef Timer<std::chrono::microseconds> MicroTimer;
	bp::class_<MicroTimer>
		("Timer", bp::init<>())
		.def(bp::init<std::string>())
		.def("tic", &MicroTimer::tic)
		.def("toc", &MicroTimer::toc);

	// Types.hpp
	bp::enum_<Axis::Type>("Axis")
		.value("NONE", Axis::NONE)
		.value("ROW", Axis::ROW)
		.value("COL", Axis::COL);

	bp::enum_<Endian::Type>("Endian")
		.value("NATIVE", Endian::NATIVE)
		.value("BIG", Endian::BIG)
		.value("LITTLE", Endian::LITTLE);

	// NdArray.hpp
	typedef NdArray<double> NdArrayDouble;
	bp::class_<NdArrayDouble>
		("NdArray", bp::init<>())
		.def(bp::init<uint32>())
		.def(bp::init<uint32, uint32>())
		.def(bp::init<Shape>())
		.def(bp::init<NdArrayDouble>())
		.def("test1DListContructor", &NdArrayInterface::test1DListContructor<double>).staticmethod("test1DListContructor")
		.def("test2DListContructor", &NdArrayInterface::test2DListContructor<double>).staticmethod("test2DListContructor")
		.def("getNumpyArray", &NdArrayInterface::getNumpyArray<double>)
		.def("setArray", &NdArrayInterface::setArray<double>)
		.def("all", &NdArrayInterface::all<double>)
		.def("any", &NdArrayInterface::any<double>)
		.def("argmax", &NdArrayInterface::argmax<double>)
		.def("argmin", &NdArrayInterface::argmin<double>)
		.def("argsort", &NdArrayInterface::argsort<double>)
		.def("clip", &NdArrayInterface::clip<double>)
		.def("copy", &NdArrayInterface::copy<double>)
		.def("cumprod", &NdArrayInterface::cumprod<double, double>)
		.def("cumsum", &NdArrayInterface::cumsum<double, double>)
		.def("diagonal", &NdArrayInterface::diagonal<double>)
		.def("dot", &NdArrayInterface::dot<double, double>)
		.def("dump", &NdArrayDouble::dump)
		.def("fill", &NdArrayInterface::fill<double>)
		.def("flatten", &NdArrayInterface::flatten<double>)
		.def("get", &NdArrayInterface::getValueFlat<double>)
		.def("get", &NdArrayInterface::getValueRowCol<double>)
		.def("get", &NdArrayInterface::getSlice1D<double>)
		.def("get", &NdArrayInterface::getSlice2D<double>)
		.def("testGetSlice1DList", &NdArrayInterface::testGetSlice1DList<double>).staticmethod("testGetSlice1DList")
		.def("testGetSlice2DList", &NdArrayInterface::testGetSlice2DList<double>).staticmethod("testGetSlice2DList")
		.def("item", &NdArrayDouble::item)
		.def("max", &NdArrayInterface::max<double>)
		.def("min", &NdArrayInterface::min<double>)
		.def("mean", &NdArrayInterface::mean<double>)
		.def("median", &NdArrayInterface::median<double>)
		.def("nbytes", &NdArrayDouble::nbytes)
		.def("nonzero", &NdArrayInterface::nonzero<double>)
		.def("norm", &NdArrayInterface::norm<double, double>)
		.def("ones", &NdArrayInterface::ones<double>)
		.def("partition", &NdArrayInterface::partition<double>)
		.def("print", &NdArrayDouble::print)
		.def("prod", &NdArrayInterface::prod<double, double>)
		.def("ptp", &NdArrayInterface::ptp<double>)
		.def("put", &NdArrayInterface::putFlat<double>)
		.def("put", &NdArrayInterface::putRowCol<double>)
		.def("put", &NdArrayInterface::putSlice1DValue<double>)
		.def("put", &NdArrayInterface::putSlice1DValues<double>)
		.def("put", &NdArrayInterface::putSlice2DValue<double>)
		.def("put", &NdArrayInterface::putSlice2DValues<double>)
		.def("testPutSlice1DValueList", &NdArrayInterface::testPutSlice1DValueList<double>).staticmethod("testPutSlice1DValueList")
		.def("testPutSlice1DValuesList", &NdArrayInterface::testPutSlice1DValuesList<double>).staticmethod("testPutSlice1DValuesList")
		.def("testPutSlice2DValueList", &NdArrayInterface::testPutSlice2DValueList<double>).staticmethod("testPutSlice2DValueList")
		.def("testPutSlice2DValuesList", &NdArrayInterface::testPutSlice2DValuesList<double>).staticmethod("testPutSlice2DValuesList")
		.def("repeat", &NdArrayInterface::repeat<double>)
		.def("reshape", &NdArrayInterface::reshape<double>)
		.def("reshapeList", &NdArrayInterface::reshapeList<double>)
		.def("resizeFast", &NdArrayInterface::resizeFast<double>)
		.def("resizeFastList", &NdArrayInterface::resizeFastList<double>)
		.def("resizeSlow", &NdArrayInterface::resizeSlow<double>)
		.def("resizeSlowList", &NdArrayInterface::resizeSlowList<double>)
		.def("round", &NdArrayInterface::round<double>)
		.def("shape", &NdArrayDouble::shape)
		.def("size", &NdArrayDouble::size)
		.def("sort", &NdArrayInterface::sort<double>)
		.def("std", &NdArrayInterface::std<double>)
		.def("sum", &NdArrayInterface::sum<double, double>)
		.def("swapaxes", &NdArrayInterface::swapaxes<double>)
		.def("tofile", &NdArrayDouble::tofile)
		.def("toStlVector", &NdArrayDouble::toStlVector)
		.def("trace", &NdArrayDouble::trace<double>)
		.def("transpose", &NdArrayInterface::transpose<double>)
		.def("var", &NdArrayInterface::var<double>)
		.def("zeros", &NdArrayDouble::zeros)
		.def("operatorPlusScalar", &NdArrayInterface::operatorPlusScalar<double>)
		.def("operatorPlusArray", &NdArrayInterface::operatorPlusArray<double>)
		.def("operatorMinusScalar", &NdArrayInterface::operatorMinusScalar<double>)
		.def("operatorMinusArray", &NdArrayInterface::operatorMinusArray<double>)
		.def("operatorMultiplyScalar", &NdArrayInterface::operatorMultiplyScalar<double>)
		.def("operatorMultiplyArray", &NdArrayInterface::operatorMultiplyArray<double>)
		.def("operatorDivideScalar", &NdArrayInterface::operatorDivideScalar<double>)
		.def("operatorDivideArray", &NdArrayInterface::operatorDivideArray<double>)
		.def("operatorEquality", &NdArrayInterface::operatorEqualityScalar<double>)
		.def("operatorEquality", &NdArrayInterface::operatorEqualityArray<double>)
		.def("operatorLess", &NdArrayInterface::operatorLessScalar<double>)
		.def("operatorLess", &NdArrayInterface::operatorLessArray<double>)
		.def("operatorGreater", &NdArrayInterface::operatorGreaterScalar<double>)
		.def("operatorGreater", &NdArrayInterface::operatorGreaterArray<double>)
		.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualScalar<double>)
		.def("operatorLessEqual", &NdArrayInterface::operatorLessEqualArray<double>)
		.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualScalar<double>)
		.def("operatorGreaterEqual", &NdArrayInterface::operatorGreaterEqualArray<double>)
		.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityScalar<double>)
		.def("operatorNotEquality", &NdArrayInterface::operatorNotEqualityArray<double>)
		.def("operatorPrePlusPlus", &NdArrayInterface::operatorPrePlusPlus<double>)
		.def("operatorPostPlusPlus", &NdArrayInterface::operatorPostPlusPlus<double>)
		.def("operatorPreMinusMinus", &NdArrayInterface::operatorPreMinusMinus<double>)
		.def("operatorPostMinusMinus", &NdArrayInterface::operatorPostMinusMinus<double>);
	
	typedef NdArray<uint32> NdArrayInt;
	bp::class_<NdArrayInt>
		("NdArrayInt", bp::init<>())
		.def(bp::init<uint32>())
		.def(bp::init<uint32, uint32>())
		.def(bp::init<Shape>())
		.def("shape", &NdArrayInt::shape)
		.def("size", &NdArrayInt::size)
		.def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint32>)
		.def("endianess", &NdArrayInt::endianess)
		.def("setArray", &NdArrayInterface::setArray<uint32>)
		.def("byteswap", &NdArrayInt::byteswap)
		.def("newbyteorder", &NdArrayInterface::newbyteorder<uint32>)
		.def("operatorModulusScalar", &NdArrayInterface::operatorModulusScalar<uint32>)
		.def("operatorModulusArray", &NdArrayInterface::operatorModulusArray<uint32>)
		.def("operatorBitwiseOrScalar", &NdArrayInterface::operatorBitwiseOrScalar<uint32>)
		.def("operatorBitwiseOrArray", &NdArrayInterface::operatorBitwiseOrArray<uint32>)
		.def("operatorBitwiseAndScalar", &NdArrayInterface::operatorBitwiseAndScalar<uint32>)
		.def("operatorBitwiseAndArray", &NdArrayInterface::operatorBitwiseAndArray<uint32>)
		.def("operatorBitwiseXorScalar", &NdArrayInterface::operatorBitwiseXorScalar<uint32>)
		.def("operatorBitwiseXorArray", &NdArrayInterface::operatorBitwiseXorArray<uint32>)
		.def("operatorBitwiseNot", &NdArrayInterface::operatorBitwiseNot<uint32>)
		.def("operatorBitshiftLeft", &NdArrayInterface::operatorBitshiftLeft<uint32>)
		.def("operatorBitshiftRight", &NdArrayInterface::operatorBitshiftRight<uint32>);

	typedef NdArray<int32> NdArrayInt32;
	bp::class_<NdArrayInt32>
		("NdArrayInt32", bp::init<>())
		.def(bp::init<uint32>())
		.def(bp::init<uint32, uint32>())
		.def(bp::init<Shape>())
		.def("shape", &NdArrayInt32::shape)
		.def("size", &NdArrayInt32::size)
		.def("getNumpyArray", &NdArrayInterface::getNumpyArray<int32>)
		.def("endianess", &NdArrayInt32::endianess)
		.def("setArray", &NdArrayInterface::setArray<int32>);

	typedef NdArray<uint64> NdArrayInt64;
	bp::class_<NdArrayInt64>
		("NdArrayInt64", bp::init<>())
		.def(bp::init<uint32>())
		.def(bp::init<uint32, uint32>())
		.def(bp::init<Shape>())
		.def("shape", &NdArrayInt64::shape)
		.def("size", &NdArrayInt64::size)
		.def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint64>)
		.def("endianess", &NdArrayInt64::endianess)
		.def("setArray", &NdArrayInterface::setArray<uint64>);

	typedef NdArray<uint8> NdArrayInt8;
	bp::class_<NdArrayInt8>
		("NdArrayInt8", bp::init<>())
		.def(bp::init<uint32>())
		.def(bp::init<uint32, uint32>())
		.def(bp::init<Shape>())
		.def("shape", &NdArrayInt8::shape)
		.def("size", &NdArrayInt8::size)
		.def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint8>)
		.def("endianess", &NdArrayInt8::endianess)
		.def("setArray", NdArrayInterface::setArray<uint8>);

	typedef NdArray<bool> NdArrayBool;
	bp::class_<NdArrayBool>
		("NdArrayBool", bp::init<>())
		.def(bp::init<uint32>())
		.def(bp::init<uint32, uint32>())
		.def(bp::init<Shape>())
		.def("shape", &NdArrayBool::shape)
		.def("size", &NdArrayBool::size)
		.def("getNumpyArray", &NdArrayInterface::getNumpyArray<bool>)
		.def("endianess", &NdArrayBool::endianess)
		.def("setArray", NdArrayInterface::setArray<bool>);

	// Methods.hpp
	boost::python::def("abs", &MethodsInterface::absScalar<double>);
	boost::python::def("abs", &MethodsInterface::absArray<double>);
	boost::python::def("add", &MethodsInterface::addArrays<double, double>);
	boost::python::def("alen", &NumC::alen<double>);
	boost::python::def("all", &MethodsInterface::allArray<double>);
	boost::python::def("allclose", &NumC::allclose<double>);
	boost::python::def("amin", &MethodsInterface::aminArray<double>);
	boost::python::def("amax", &MethodsInterface::amaxArray<double>);
	boost::python::def("any", &MethodsInterface::anyArray<double>);
	boost::python::def("arange", &MethodsInterface::arangeArray<double>);
	boost::python::def("argmax", &MethodsInterface::argmaxArray<double>);
	boost::python::def("argmin", &MethodsInterface::argminArray<double>);
	boost::python::def("argsort", &MethodsInterface::argsortArray<double>);
	boost::python::def("arccos", &MethodsInterface::arccosScalar<double>);
	boost::python::def("arccos", &MethodsInterface::arccosArray<double>);
	boost::python::def("arccosh", &MethodsInterface::arccoshScalar<double>);
	boost::python::def("arccosh", &MethodsInterface::arccoshArray<double>);
	boost::python::def("arcsin", &MethodsInterface::arcsinScalar<double>);
	boost::python::def("arcsin", &MethodsInterface::arcsinArray<double>);
	boost::python::def("arcsinh", &MethodsInterface::arcsinhScalar<double>);
	boost::python::def("arcsinh", &MethodsInterface::arcsinhArray<double>);
	boost::python::def("arctan", &MethodsInterface::arctanScalar<double>);
	boost::python::def("arctan", &MethodsInterface::arctanArray<double>);
	boost::python::def("arctan2", &MethodsInterface::arctan2Scalar<double>);
	boost::python::def("arctan2", &MethodsInterface::arctan2Array<double>);
	boost::python::def("arctanh", &MethodsInterface::arctanhScalar<double>);
	boost::python::def("arctanh", &MethodsInterface::arctanhArray<double>);
	boost::python::def("around", &MethodsInterface::aroundScalar<double>);
	boost::python::def("around", &MethodsInterface::aroundArray<double>);
	boost::python::def("array_equal", &NumC::array_equal<double>);
	boost::python::def("array_equiv", &NumC::array_equiv<double>);
	boost::python::def("average", &MethodsInterface::average<double>);
	boost::python::def("average", &MethodsInterface::averageWeighted<double>);
	boost::python::def("bincount", &MethodsInterface::bincount<uint32>);
	boost::python::def("bincount", &MethodsInterface::bincountWeighted<uint32>);
	boost::python::def("bitwise_and", &MethodsInterface::bitwise_and<uint64>);
	boost::python::def("bitwise_not", &MethodsInterface::bitwise_not<uint64>);
	boost::python::def("bitwise_or", &MethodsInterface::bitwise_or<uint64>);
	boost::python::def("bitwise_xor", &MethodsInterface::bitwise_xor<uint64>);
	boost::python::def("byteswap", &MethodsInterface::byteswap<uint64>);
	boost::python::def("cbrt", &MethodsInterface::cbrtScalar<double>);
	boost::python::def("cbrt", &MethodsInterface::cbrtArray<double>);
	boost::python::def("ceil", &MethodsInterface::ceilScalar<double>);
	boost::python::def("ceil", &MethodsInterface::ceilArray<double>);
	boost::python::def("clip", &MethodsInterface::clipScalar<double>);
	boost::python::def("clip", &MethodsInterface::clipArray<double>);
	boost::python::def("copy", &MethodsInterface::copy<double>);
	boost::python::def("copysign", &MethodsInterface::copySign<double>);
	boost::python::def("copyto", &MethodsInterface::copyto<double>);
	boost::python::def("cos", &MethodsInterface::cosScalar<double>);
	boost::python::def("cos", &MethodsInterface::cosArray<double>);
	boost::python::def("cosh", &MethodsInterface::coshScalar<double>);
	boost::python::def("cosh", &MethodsInterface::coshArray<double>);
	boost::python::def("count_nonzero", &MethodsInterface::count_nonzero<double>);
	boost::python::def("cube", &MethodsInterface::cubeArray<double, double>);
	boost::python::def("cumprod", &MethodsInterface::cumprodArray<double, double>);
	boost::python::def("cumsum", &MethodsInterface::cumsumArray<double, double>);
	boost::python::def("deg2rad", &MethodsInterface::deg2radScalar<double>);
	boost::python::def("deg2rad", &MethodsInterface::deg2radArray<double>);
	boost::python::def("diagflat", &MethodsInterface::diagflat<double>);
	boost::python::def("diagonal", &MethodsInterface::diagonal<double>);
	boost::python::def("diff", &MethodsInterface::diff<double>);
	boost::python::def("divide", &MethodsInterface::divide<double, double>);
	boost::python::def("dot", &MethodsInterface::dot<double, double>);
	boost::python::def("dump", &NumC::dump<double>);
	boost::python::def("empty", &MethodsInterface::emptyRowCol<double>);
	boost::python::def("empty", &MethodsInterface::emptyShape<double>);
	boost::python::def("testEmptyList", &MethodsInterface::testEmptyList<double>);
	boost::python::def("empty_like", &NumC::empty_like<double, double>);
	boost::python::def("endianess", &NumC::endianess<double>);
	boost::python::def("equal", &MethodsInterface::equal<double>);
	boost::python::def("exp", &MethodsInterface::expScalar<double>);
	boost::python::def("exp", &MethodsInterface::expArray<double>);
	boost::python::def("exp2", &MethodsInterface::exp2Scalar<double>);
	boost::python::def("exp2", &MethodsInterface::exp2Array<double>);
	boost::python::def("expm1", &MethodsInterface::expm1Scalar<double>);
	boost::python::def("expm1", &MethodsInterface::expm1Array<double>);
	boost::python::def("eye", &MethodsInterface::eye1D<double>);
	boost::python::def("eye", &MethodsInterface::eye2D<double>);
	boost::python::def("eye", &MethodsInterface::eyeShape<double>);
	boost::python::def("fix", &MethodsInterface::fixScalar<double>);
	boost::python::def("fix", &MethodsInterface::fixArray<double>);
	boost::python::def("flatten", &NumC::flatten<double>);
	boost::python::def("flatnonzero", &NumC::flatnonzero<double>);
	boost::python::def("flip", &NumC::flip<double>);
	boost::python::def("fliplr", &NumC::fliplr<double>);
	boost::python::def("flipud", &NumC::flipud<double>);
	boost::python::def("floor", &MethodsInterface::floorScalar<double>);
	boost::python::def("floor", &MethodsInterface::floorArray<double>);
	boost::python::def("floor_divide", &MethodsInterface::floor_divideScalar<double>);
	boost::python::def("floor_divide", &MethodsInterface::floor_divideArray<double>);
	boost::python::def("fmax", &MethodsInterface::fmaxScalar<double>);
	boost::python::def("fmax", &MethodsInterface::fmaxArray<double>);
	boost::python::def("fmin", &MethodsInterface::fminScalar<double>);
	boost::python::def("fmin", &MethodsInterface::fminArray<double>);
	boost::python::def("fmod", &MethodsInterface::fmodScalar<uint32>);
	boost::python::def("fmod", &MethodsInterface::fmodArray<uint32>);
	boost::python::def("full", &MethodsInterface::fullRowCol<double>);
	boost::python::def("full", &MethodsInterface::fullShape<double>);
	boost::python::def("testFullList", &MethodsInterface::testFullList<double>);
	boost::python::def("full_like", &NumC::full_like<double, double>);
	boost::python::def("full", &MethodsInterface::fullShape<double>);
	boost::python::def("greater", &NumC::greater<double>);
	boost::python::def("greater_equal", &NumC::greater_equal<double>);
	boost::python::def("hypot", &MethodsInterface::hypotScalar<double, double>);
	boost::python::def("hypot", &MethodsInterface::hypotArray<double, double>);
	boost::python::def("identity", &NumC::identity<double>);
	boost::python::def("intersect1d", &NumC::intersect1d<uint32>);
	boost::python::def("invert", &NumC::invert<uint32>);
	boost::python::def("isnan", &MethodsInterface::isnanScalar<double>);
	boost::python::def("isnan", &MethodsInterface::isnanArray<double>);
	boost::python::def("ldexp", &MethodsInterface::ldexpScalar<double>);
	boost::python::def("ldexp", &MethodsInterface::ldexpArray<double>);
	boost::python::def("left_shift", &NumC::left_shift<uint32>);
	boost::python::def("less", &NumC::less<double>);
	boost::python::def("less_equal", &NumC::less_equal<double>);
	boost::python::def("log", &MethodsInterface::logScalar<double>);
	boost::python::def("log", &MethodsInterface::logArray<double>);
	boost::python::def("log10", &MethodsInterface::log10Scalar<double>);
	boost::python::def("log10", &MethodsInterface::log10Array<double>);
	boost::python::def("log1p", &MethodsInterface::log1pScalar<double>);
	boost::python::def("log1p", &MethodsInterface::log1pArray<double>);
	boost::python::def("log2", &MethodsInterface::log2Scalar<double>);
	boost::python::def("log2", &MethodsInterface::log2Array<double>);
	boost::python::def("logical_and", &NumC::logical_and<double>);
	boost::python::def("logical_not", &NumC::logical_not<double>);
	boost::python::def("logical_or", &NumC::logical_or<double>);
	boost::python::def("logical_xor", &NumC::logical_xor<double>);
	boost::python::def("matmul", &NumC::matmul<double, double>);
	boost::python::def("max", &NumC::max<double>);
	boost::python::def("maximum", &NumC::maximum<double>);
	boost::python::def("mean", &NumC::mean<double>);
	boost::python::def("median", &NumC::median<double>);
	boost::python::def("min", &NumC::min<double>);
	boost::python::def("minimum", &NumC::minimum<double>);
	boost::python::def("mod", &NumC::mod<uint32>);
	boost::python::def("multiply", &NumC::multiply<double>);
	boost::python::def("nbytes", &NumC::nbytes<double>);
	boost::python::def("newbyteorder", &MethodsInterface::newbyteorderScalar<uint32>);
	boost::python::def("newbyteorder", &MethodsInterface::newbyteorderArray<uint32>);

	boost::python::def("negative", &NumC::negative<double, double>);

	boost::python::def("power", &MethodsInterface::powerArrayScalar<double, double>);
	boost::python::def("power", &MethodsInterface::powerArrayArray<double, double>);

	boost::python::def("right_shift", &NumC::right_shift<uint32>);
	boost::python::def("setdiff1d", &NumC::setdiff1d<uint32>);
	boost::python::def("sqr", &MethodsInterface::sqrArray<double, double>);

	boost::python::def("union1d", &NumC::union1d<uint32>);
	boost::python::def("unique", &NumC::unique<double>);


	// Utils.hpp
	boost::python::def("num2str", &NumC::num2str<double>);
	boost::python::def("num2str", &NumC::num2str<float>);
	boost::python::def("num2str", &NumC::num2str<int8>);
	boost::python::def("num2str", &NumC::num2str<int16>);
	boost::python::def("num2str", &NumC::num2str<int32>);
	boost::python::def("num2str", &NumC::num2str<int64>);
	boost::python::def("num2str", &NumC::num2str<uint8>);
	boost::python::def("num2str", &NumC::num2str<uint16>);
	boost::python::def("num2str", &NumC::num2str<uint32>);
	boost::python::def("num2str", &NumC::num2str<uint64>);

	boost::python::def("sqr", &NumC::sqr<double>);
	boost::python::def("sqr", &NumC::sqr<float>);
	boost::python::def("sqr", &NumC::sqr<int8>);
	boost::python::def("sqr", &NumC::sqr<int16>);
	boost::python::def("sqr", &NumC::sqr<int32>);
	boost::python::def("sqr", &NumC::sqr<int64>);
	boost::python::def("sqr", &NumC::sqr<uint8>);
	boost::python::def("sqr", &NumC::sqr<uint16>);
	boost::python::def("sqr", &NumC::sqr<uint32>);
	boost::python::def("sqr", &NumC::sqr<uint64>);

	boost::python::def("cube", &NumC::cube<double>);
	boost::python::def("cube", &NumC::cube<float>);
	boost::python::def("cube", &NumC::cube<int8>);
	boost::python::def("cube", &NumC::cube<int16>);
	boost::python::def("cube", &NumC::cube<int32>);
	boost::python::def("cube", &NumC::cube<int64>);
	boost::python::def("cube", &NumC::cube<uint8>);
	boost::python::def("cube", &NumC::cube<uint16>);
	boost::python::def("cube", &NumC::cube<uint32>);
	boost::python::def("cube", &NumC::cube<uint64>);

	boost::python::def("power", &NumC::power<double>);
	boost::python::def("power", &NumC::power<float>);
	boost::python::def("power", &NumC::power<int8>);
	boost::python::def("power", &NumC::power<int16>);
	boost::python::def("power", &NumC::power<int32>);
	boost::python::def("power", &NumC::power<int64>);
	boost::python::def("power", &NumC::power<uint8>);
	boost::python::def("power", &NumC::power<uint16>);
	boost::python::def("power", &NumC::power<uint32>);
	boost::python::def("power", &NumC::power<uint64>);

	// Random.hpp
	boost::python::def("bernoulli", &NumC::Random::bernoulli<double>);
	boost::python::def("beta", &NumC::Random::beta<double>);
	boost::python::def("binomial", &NumC::Random::binomial<int32>);
	boost::python::def("chiSquare", &NumC::Random::chiSquare<double>);
	boost::python::def("choice", &NumC::Random::choice<double>);
	boost::python::def("cauchy", &NumC::Random::cauchy<double>);
	boost::python::def("discrete", &NumC::Random::discrete<int32>);
	boost::python::def("exponential", &NumC::Random::exponential<double>);
	boost::python::def("extremeValue", &NumC::Random::extremeValue<double>);
	boost::python::def("f", &NumC::Random::f<double>);
	boost::python::def("gamma", &NumC::Random::gamma<double>);
	boost::python::def("geometric", &NumC::Random::geometric<int32>);
	boost::python::def("laplace", &NumC::Random::laplace<double>);
	boost::python::def("lognormal", &NumC::Random::lognormal<double>);
	boost::python::def("negativeBinomial", &NumC::Random::negativeBinomial<int32>);
	boost::python::def("nonCentralChiSquared", &NumC::Random::nonCentralChiSquared<double>);
	boost::python::def("normal", &NumC::Random::normal<double>);
	boost::python::def("permutation", &RandomInterface::permutationScalar<double>);
	boost::python::def("permutation", &RandomInterface::permutationArray<double>);
	boost::python::def("poisson", &NumC::Random::poisson<int32>);
	boost::python::def("rand", &NumC::Random::rand<double>);
	boost::python::def("randN", &NumC::Random::randN<double>);
	boost::python::def("randFloat", &NumC::Random::randFloat<double>);
	boost::python::def("randInt", &NumC::Random::randInt<int32>);
	boost::python::def("seed", &NumC::Random::seed);
	boost::python::def("shuffle", &NumC::Random::shuffle<double>);
	boost::python::def("studentT", &NumC::Random::studentT<double>);
	boost::python::def("standardNormal", &NumC::Random::standardNormal<double>);
	boost::python::def("triangle", &NumC::Random::triangle<double>);
	boost::python::def("uniform", &NumC::Random::uniform<double>);
	boost::python::def("uniformOnSphere", &NumC::Random::uniformOnSphere<double>);
	boost::python::def("weibull", &NumC::Random::weibull<double>);
}