/// @section Description
/// A Templatized Header Only C++ Implementation of the Python Numpy Library
///
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
///
/// @section License
/// Copyright 2018-2023 David Pilger
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
/// **C++ Standards:**
/// C++14
/// C++17
/// C++20
///
/// **Compilers:**
/// Visual Studio: 2017, 2019
/// GNU: 6.5, 7.5, 8.4, 9.3, 10.1
/// Clang: 6, 7, 8, 9, 10
///
/// **Boost Versions:**
/// 1.68, 1.70, 1.72, and 1.73
///
#pragma once

#include "NumCpp/Coordinates.hpp"
#include "NumCpp/Core.hpp"
#include "NumCpp/DateTime.hpp"
#include "NumCpp/Filter.hpp"
#include "NumCpp/Functions.hpp"
#include "NumCpp/ImageProcessing.hpp"
#include "NumCpp/Integrate.hpp"
#include "NumCpp/Linalg.hpp"
#include "NumCpp/Logging.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Polynomial.hpp"
#include "NumCpp/PythonInterface.hpp"
#include "NumCpp/Random.hpp"
#include "NumCpp/Roots.hpp"
#include "NumCpp/Rotations.hpp"
#include "NumCpp/Special.hpp"
#include "NumCpp/Utils.hpp"
#include "NumCpp/Vector.hpp"

/// \example GaussNewtonNlls.cpp
/// Example for using the linalg::gaussNewtonNlls function
///
/// \example InterfaceWithEigen.cpp
/// Example for interfaceing with Eigen Matrix
///
/// \example InterfaceWithOpenCV.cpp
/// Example for interfaceing with OpenCV Mat
///
/// \example ReadMe.cpp
/// Examples from the Quick Start Guide in README.md at [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
