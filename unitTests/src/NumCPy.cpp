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

namespace Interface
{
	int zeros(uint16 inNumRows, uint16 inNumCols)
	{
		NdArray<double> zeroArray = NumC::zeros<double>(Shape(inNumRows, inNumCols));
		return 666;
	}
}

//================================================================================

namespace NdArrayInterface
{
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

		uint32 numRows = 0;
		uint32 numCols = 0;
		if (numDims == 1)
		{
			numCols = static_cast<uint32>(newNdArrayHelper.shape()[0]);
			numRows = 1;
		}
		else
		{
			numRows = static_cast<uint32>(newNdArrayHelper.shape()[0]);
			numCols = static_cast<uint32>(newNdArrayHelper.shape()[1]);
		}

		Shape boostArrayShape(numRows, numCols);
		Shape arrayShape = inArray.shape();
		if (!(arrayShape.rows == numRows && arrayShape.cols == numCols))
		{
			inArray.resizeFast(boostArrayShape);
		}

		for (uint32 row = 0; row < numRows; ++row)
		{
			for (uint32 col = 0; col < numCols; ++col)
			{
				inArray(row, col) = newNdArrayHelper(row, col);
			}
		}
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
	np::ndarray flatten(NdArray<dtype>& inArray)
	{
		return numCToBoost(inArray.flatten());
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
		.def(bp::init<uint16, uint16>())
		.def_readwrite("rows", &Shape::rows)
		.def_readwrite("cols", &Shape::cols)
		.def("print", &Shape::print);

	bp::class_<Slice>
		("Slice", bp::init<>())
		.def(bp::init<int16>())
		.def(bp::init<int16, int16>())
		.def(bp::init<int16, int16, int16>())
		.def_readwrite("start", &Slice::start)
		.def_readwrite("stop", &Slice::stop)
		.def_readwrite("step", &Slice::step)
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
		.def("shape", &NdArrayDouble::shape)
		.def("size", &NdArrayDouble::size)
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
		.def("fill", &NdArray<double>::fill)
		.def("flatten", &NdArrayInterface::flatten<double>)
		.def("item", &NdArray<double>::item)
		.def("max", &NdArrayInterface::max<double>)
		.def("min", &NdArrayInterface::min<double>)
		.def("mean", &NdArrayInterface::mean<double>)
		.def("median", &NdArrayInterface::median<double>)
		.def("nbytes", &NdArrayDouble::nbytes)
		.def("newbyteorder", &NdArrayInterface::newbyteorder<uint32>);

	boost::python::def("zeros", Interface::zeros);

}