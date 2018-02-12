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
	template<typename T>
	np::ndarray getNumpyArray(const NdArray<T>& inArray) 
	{
		return numCToBoost(inArray);
	}

	//================================================================================

	template<typename T>
	void setArray(NdArray<T>& inArray, np::ndarray& inBoostArray)
	{
		BoostNdarrayHelper newNdArrayHelper(&inBoostArray);
		uint8 numDims = newNdArrayHelper.numDimensions();
		if (numDims > 2)
		{
			std::string errorString = "ERROR: Input array can only have up to 2 dimensions!";
			PyErr_SetString(PyExc_RuntimeError, errorString.c_str());
		}

		uint16 numRows = 0;
		uint16 numCols = 0;
		if (numDims == 1)
		{
			numCols = static_cast<uint16>(newNdArrayHelper.shape()[0]);
			numRows = 1;
		}
		else
		{
			numRows = static_cast<uint16>(newNdArrayHelper.shape()[0]);
			numCols = static_cast<uint16>(newNdArrayHelper.shape()[1]);
		}

		Shape boostArrayShape(numRows, numCols);
		Shape arrayShape = inArray.shape();
		if (!(arrayShape.rows == numRows && arrayShape.cols == numCols))
		{
			inArray.resizeFast(boostArrayShape);
		}

		for (uint16 row = 0; row < numRows; ++row)
		{
			for (uint16 col = 0; col < numCols; ++col)
			{
				inArray(row, col) = newNdArrayHelper(row, col);
			}
		}
	}

	//================================================================================

	template<typename T>
	np::ndarray all(NdArray<T>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.all(inAxis));
	}

	//================================================================================

	template<typename T>
	np::ndarray any(NdArray<T>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.any(inAxis));
	}

	//================================================================================

	template<typename T>
	np::ndarray argmax(NdArray<T>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.argmax(inAxis));
	}

	//================================================================================

	template<typename T>
	np::ndarray argmin(NdArray<T>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.argmin(inAxis));
	}

	//================================================================================

	template<typename T>
	np::ndarray argsort(NdArray<T>& inArray, Axis::Type inAxis = Axis::NONE)
	{
		return numCToBoost(inArray.argsort(inAxis));
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
		.def("argsort", &NdArrayInterface::argsort<double>);

	boost::python::def("zeros", Interface::zeros);

}