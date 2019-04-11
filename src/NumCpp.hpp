/// @section Description
/// A C++ Implementation of the Python Numpy Library
///
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// @version 1.0
///
/// @section License
/// Copyright 2018 David Pilger
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
/// @section Testing
/// Compiled and tested with Visual Studio 2017, and g++ 7.3.0, with Boost version 1.68.
///
#pragma once

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // for fopen with Visual Studio
#endif

#include"NumCpp/Constants.hpp"
#include"NumCpp/Coordinates.hpp"
#include"NumCpp/DataCube.hpp"
#include"NumCpp/DtypeInfo.hpp"
#include"NumCpp/FFT.hpp"
#include"NumCpp/Filter.hpp"
#include"NumCpp/ImageProcessing.hpp"
#include"NumCpp/Linalg.hpp"
#include"NumCpp/Methods.hpp"
#include"NumCpp/NdArray.hpp"
#include"NumCpp/Polynomial.hpp"
#include"NumCpp/Random.hpp"
#include"NumCpp/Rotations.hpp"
#include"NumCpp/Shape.hpp"
#include"NumCpp/Slice.hpp"
#include"NumCpp/Timer.hpp"
#include"NumCpp/Types.hpp"
#include"NumCpp/Utils.hpp"

#ifdef INCLUDE_PYTHON_INTERFACE
#include"NumCpp/PythonInterface.hpp"
#endif

/// \example Example.cpp
/// Examples from the Quick Start Guide in README.md at [GitHub Repository](https://github.com/dpilger26/NumCpp)
