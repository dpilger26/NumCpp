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
template<typename T>
class NdArrayInterface
{
private:
	NumC::NdArray<T>	theArray_;

public:
	//================================================================================

	NdArrayInterface() :
		theArray_()
	{};

	//================================================================================

	Shape shape()
	{
		return theArray_.shape();
	}

	//================================================================================

	uint32 size()
	{
		return theArray_.size();
	}
};

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
		.def(bp::init<uint16>())
		.def(bp::init<uint16, uint16>())
		.def(bp::init<uint16, uint16, uint16>())
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

	typedef NdArrayInterface<double> NdArrayInterfaceDouble;
	bp::class_<NdArrayInterfaceDouble>
		("NdArray", bp::init<>())
		.def("shape", &NdArrayInterfaceDouble::shape)
		.def("size", &NdArrayInterfaceDouble::size);

	boost::python::def("zeros", Interface::zeros);

}