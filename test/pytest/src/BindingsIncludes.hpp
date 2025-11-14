#pragma once

#if defined(NUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE)

#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#elif defined(NUMCPP_INCLUDE_NANOBIND_PYTHON_INTERFACE)

#include "nanobind/nanobind.h"
#include "nanobind/nb_types.h"
#include "nanobind/ndarray.h"

#endif

#include <algorithm>
#include <array>
#include <complex>
#include <cstdio>
#include <deque>
#include <forward_list>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "NumCpp/Core/Internal/StdComplexOperators.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/PythonInterface/PybindInterface.hpp"

using namespace nc;

#if defined(NUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE)

using namespace nc::pybindInterface;
namespace python_interface = pybind11;
using interface_module     = python_interface::module;

#elif defined(NUMCPP_INCLUDE_NANOBIND_PYTHON_INTERFACE)

using namespace nc::nanobindInterface;
namespace python_interface = nanobind;
using interface_module     = python_interface::module_;

#endif

// NdArray.hpp
using NdArrayDouble                           = NdArray<double>;
using NdArrayDoubleIterator                   = NdArrayDouble::iterator;
using NdArrayDoubleConstIterator              = NdArrayDouble::const_iterator;
using NdArrayDoubleReverseIterator            = NdArrayDouble::reverse_iterator;
using NdArrayDoubleConstReverseIterator       = NdArrayDouble::const_reverse_iterator;
using NdArrayDoubleColumnIterator             = NdArrayDouble::column_iterator;
using NdArrayDoubleConstColumnIterator        = NdArrayDouble::const_column_iterator;
using NdArrayDoubleReverseColumnIterator      = NdArrayDouble::reverse_column_iterator;
using NdArrayDoubleConstReverseColumnIterator = NdArrayDouble::const_reverse_column_iterator;

using ComplexDouble                                  = std::complex<double>;
using NdArrayComplexDouble                           = NdArray<ComplexDouble>;
using NdArrayComplexDoubleIterator                   = NdArrayComplexDouble::iterator;
using NdArrayComplexDoubleConstIterator              = NdArrayComplexDouble::const_iterator;
using NdArrayComplexDoubleReverseIterator            = NdArrayComplexDouble::reverse_iterator;
using NdArrayComplexDoubleConstReverseIterator       = NdArrayComplexDouble::const_reverse_iterator;
using NdArrayComplexDoubleColumnIterator             = NdArrayComplexDouble::column_iterator;
using NdArrayComplexDoubleConstColumnIterator        = NdArrayComplexDouble::const_column_iterator;
using NdArrayComplexDoubleReverseColumnIterator      = NdArrayComplexDouble::reverse_column_iterator;
using NdArrayComplexDoubleConstReverseColumnIterator = NdArrayComplexDouble::const_reverse_column_iterator;
