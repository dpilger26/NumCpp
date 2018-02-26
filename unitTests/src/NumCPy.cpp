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
	void setArray(NdArray<dtype>& inArray, np::ndarray& inBoostArray)
	{
		BoostNdarrayHelper newNdArrayHelper(&inBoostArray);
		uint8 numDims = newNdArrayHelper.numDimensions();
		if (numDims > 2)
		{
			std::string errorString = "ERROR: Input array can only have up to 2 dimensions!";
			PyErr_SetString(PyExc_RuntimeError, errorString.c_str());
		}

		inArray = boostToNumC<dtype>(inBoostArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray all(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.all(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray any(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.any(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray argmax(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.argmax(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray argmin(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.argmin(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray argsort(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.argsort(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray clip(NdArray<dtype>& inArray, dtype inMin, dtype inMax)
	{
		return numCToBoost(inArray.clip(inMin, inMax));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray cumprod(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.cumprod<dtypeOut>(inAxis));
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray cumsum(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.cumsum<dtypeOut>(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray diagonal(NdArray<dtype>& inArray, uint32 inOffset = 0, Axis::Type inAxis = Axis::ROW)
	{
		return numCToBoost(inArray.diagonal(inOffset, inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray fill(NdArray<dtype>& inArray, dtype inFillValue)
	{
		inArray.fill(inFillValue);
		return numCToBoost(inArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray flatten(NdArray<dtype>& inArray)
	{
		return numCToBoost(inArray.flatten());
	}

	//================================================================================

	template<typename dtype>
	dtype getValueFlat(NdArray<dtype>& inArray, int32 inIndex)
	{
		return inArray.at(inIndex);
	}

	//================================================================================

	template<typename dtype>
	dtype getValueRowCol(NdArray<dtype>& inArray, int32 inRow, int32 inCol)
	{
		return inArray.at(inRow, inCol);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray getSlice1D(NdArray<dtype>& inArray, const Slice& inSlice)
	{
		return numCToBoost(inArray.at(inSlice));
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
	np::ndarray getSlice2D(NdArray<dtype>& inArray, const Slice& inRowSlice, const Slice& inColSlice)
	{
		return numCToBoost(inArray.at(inRowSlice, inColSlice));
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
	np::ndarray max(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.max(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray min(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.min(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray mean(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.mean(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray median(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.median(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray newbyteorder(NdArray<dtype>& inArray, Endian::Type inEndiness = Endian::NATIVE)
	{
		return numCToBoost(inArray.newbyteorder(inEndiness));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray nonzero(NdArray<dtype>& inArray)
	{
		return numCToBoost(inArray.nonzero());
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray norm(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost<dtypeOut>(inArray.norm<dtypeOut>(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray ones(NdArray<dtype>& inArray)
	{
		inArray.ones();
		return numCToBoost<dtype>(inArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray partition(NdArray<dtype>& inArray, uint32 inKth, Axis::Type inAxis = Axis::NONE)
	{
		inArray.partition(inKth, inAxis);
		return numCToBoost<dtype>(inArray);
	}

	//================================================================================

	template<typename dtype, typename dtypeOut>
	np::ndarray prod(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost<dtypeOut>(inArray.prod<dtypeOut>(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray ptp(NdArray<dtype>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.ptp(inAxis));
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putFlat(NdArray<dtype>& inArray, int32 inIndex, dtype inValue)
	{
		inArray.put(inIndex, inValue);
		return numCToBoost(inArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putRowCol(NdArray<dtype>& inArray, int32 inRow, int32 inCol, dtype inValue)
	{
		inArray.put(inRow, inCol, inValue);
		return numCToBoost(inArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putSlice1DValue(NdArray<dtype>& inArray, const Slice& inSlice, dtype inValue)
	{
		inArray.put(inSlice, inValue);
		return numCToBoost(inArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putSlice1DValues(NdArray<dtype>& inArray, const Slice& inSlice, np::ndarray& inArrayValues)
	{
		NdArray<dtype> inValues = boostToNumC<dtype>(inArrayValues);
		inArray.put(inSlice, inValues);
		return numCToBoost(inArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putSlice2DValue(NdArray<dtype>& inArray, const Slice& inSliceRow, const Slice& inSliceCol, dtype inValue)
	{
		inArray.put(inSliceRow, inSliceCol, inValue);
		return numCToBoost(inArray);
	}

	//================================================================================

	template<typename dtype>
	np::ndarray putSlice2DValues(NdArray<dtype>& inArray, const Slice& inSliceRow, const Slice& inSliceCol, np::ndarray& inArrayValues)
	{
		NdArray<dtype> inValues = boostToNumC<dtype>(inArrayValues);
		inArray.put(inSliceRow, inSliceCol, inValues);
		return numCToBoost(inArray);
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
}

//================================================================================

BOOST_PYTHON_MODULE(NumC)
{
	Py_Initialize();
	np::initialize(); // needs to be called first thing in the BOOST_PYTHON_MODULE for numpy

	//http://www.boost.org/doc/libs/1_60_0/libs/python/doc/html/tutorial/tutorial/exposing.html

	bp::scope().attr("e") = NumC::e;
	bp::scope().attr("pi") = NumC::pi;
	bp::scope().attr("nan") = NumC::nan;
	bp::scope().attr("version") = NumC::version;

	bp::class_<Shape>
		("Shape", bp::init<>())
		.def(bp::init<uint32, uint32>())
		.def(bp::init<Shape>())
		.def("testListContructor", &ShapeInterface::testListContructor).staticmethod("testListContructor")
		.def_readwrite("rows", &Shape::rows)
		.def_readwrite("cols", &Shape::cols)
		.def("size", &Shape::size)
		.def("print", &Shape::print);

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

	typedef Timer<std::chrono::microseconds> MicroTimer;
	bp::class_<MicroTimer>
		("Timer", bp::init<>())
		.def(bp::init<std::string>())
		.def("tic", &MicroTimer::tic)
		.def("toc", &MicroTimer::toc);

	bp::enum_<Axis::Type>("Axis")
		.value("NONE", Axis::NONE)
		.value("ROW", Axis::ROW)
		.value("COL", Axis::COL);

	bp::enum_<Endian::Type>("Endian")
		.value("NATIVE", Endian::NATIVE)
		.value("BIG", Endian::BIG)
		.value("LITTLE", Endian::LITTLE);

	typedef NdArray<double> NdArrayDouble;
	bp::class_<NdArrayDouble>
		("NdArray", bp::init<>())
		.def(bp::init<int16>())
		.def(bp::init<int16, int16>())
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
		.def("cumprod", &NdArrayInterface::cumprod<double, double>)
		.def("cumsum", &NdArrayInterface::cumsum<double, double>)
		.def("diagonal", &NdArrayInterface::diagonal<double>)
		.def("dump", &NdArray<double>::dump)
		.def("fill", &NdArrayInterface::fill<double>)
		.def("flatten", &NdArrayInterface::flatten<double>)
		.def("get", &NdArrayInterface::getValueFlat<double>)
		.def("get", &NdArrayInterface::getValueRowCol<double>)
		.def("get", &NdArrayInterface::getSlice1D<double>)
		.def("get", &NdArrayInterface::getSlice2D<double>)
		.def("testGetSlice1DList", &NdArrayInterface::testGetSlice1DList<double>).staticmethod("testGetSlice1DList")
		.def("testGetSlice2DList", &NdArrayInterface::testGetSlice2DList<double>).staticmethod("testGetSlice2DList")
		.def("item", &NdArray<double>::item)
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
		.def("shape", &NdArrayDouble::shape)
		.def("size", &NdArrayDouble::size);

	typedef NdArray<uint32> NdArrayInt;
	bp::class_<NdArrayInt>
		("NdArrayInt", bp::init<>())
		.def(bp::init<int16>())
		.def(bp::init<int16, int16>())
		.def(bp::init<Shape>())
		.def("shape", &NdArrayInt::shape)
		.def("size", &NdArrayInt::size)
		.def("getNumpyArray", &NdArrayInterface::getNumpyArray<uint32>)
		.def("setArray", &NdArrayInterface::setArray<uint32>)
		.def("newbyteorder", &NdArrayInterface::newbyteorder<uint32>);

	boost::python::def("sqr", NumC::sqr<double>);
	boost::python::def("sqr", NumC::sqr<float>);
	boost::python::def("sqr", NumC::sqr<int8>);
	boost::python::def("sqr", NumC::sqr<int16>);
	boost::python::def("sqr", NumC::sqr<int32>);
	boost::python::def("sqr", NumC::sqr<int64>);
	boost::python::def("sqr", NumC::sqr<uint8>);
	boost::python::def("sqr", NumC::sqr<uint16>);
	boost::python::def("sqr", NumC::sqr<uint32>);
	boost::python::def("sqr", NumC::sqr<uint64>);

	boost::python::def("cube", NumC::cube<double>);
	boost::python::def("cube", NumC::cube<float>);
	boost::python::def("cube", NumC::cube<int8>);
	boost::python::def("cube", NumC::cube<int16>);
	boost::python::def("cube", NumC::cube<int32>);
	boost::python::def("cube", NumC::cube<int64>);
	boost::python::def("cube", NumC::cube<uint8>);
	boost::python::def("cube", NumC::cube<uint16>);
	boost::python::def("cube", NumC::cube<uint32>);
	boost::python::def("cube", NumC::cube<uint64>);

	boost::python::def("power", NumC::power<double>);
	boost::python::def("power", NumC::power<float>);
	boost::python::def("power", NumC::power<int8>);
	boost::python::def("power", NumC::power<int16>);
	boost::python::def("power", NumC::power<int32>);
	boost::python::def("power", NumC::power<int64>);
	boost::python::def("power", NumC::power<uint8>);
	boost::python::def("power", NumC::power<uint16>);
	boost::python::def("power", NumC::power<uint32>);
	boost::python::def("power", NumC::power<uint64>);
}