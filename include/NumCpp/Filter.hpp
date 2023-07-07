/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
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
/// Description
/// Image and signal filtering module
///
#pragma once

#include "NumCpp/Filter/Boundaries/Boundary.hpp"
#include "NumCpp/Filter/Filters/Filters1d/complementaryMeanFilter1d.hpp"
#include "NumCpp/Filter/Filters/Filters1d/complementaryMedianFilter1d.hpp"
#include "NumCpp/Filter/Filters/Filters1d/convolve1d.hpp"
#include "NumCpp/Filter/Filters/Filters1d/gaussianFilter1d.hpp"
#include "NumCpp/Filter/Filters/Filters1d/maximumFilter1d.hpp"
#include "NumCpp/Filter/Filters/Filters1d/meanFilter1d.hpp"
#include "NumCpp/Filter/Filters/Filters1d/medianFilter1d.hpp"
#include "NumCpp/Filter/Filters/Filters1d/minimumFilter1d.hpp"
#include "NumCpp/Filter/Filters/Filters1d/percentileFilter1d.hpp"
#include "NumCpp/Filter/Filters/Filters1d/rankFilter1d.hpp"
#include "NumCpp/Filter/Filters/Filters1d/uniformFilter1d.hpp"
#include "NumCpp/Filter/Filters/Filters2d/complementaryMeanFilter.hpp"
#include "NumCpp/Filter/Filters/Filters2d/complementaryMedianFilter.hpp"
#include "NumCpp/Filter/Filters/Filters2d/convolve.hpp"
#include "NumCpp/Filter/Filters/Filters2d/gaussianFilter.hpp"
#include "NumCpp/Filter/Filters/Filters2d/laplace.hpp"
#include "NumCpp/Filter/Filters/Filters2d/maximumFilter.hpp"
#include "NumCpp/Filter/Filters/Filters2d/meanFilter.hpp"
#include "NumCpp/Filter/Filters/Filters2d/medianFilter.hpp"
#include "NumCpp/Filter/Filters/Filters2d/minimumFilter.hpp"
#include "NumCpp/Filter/Filters/Filters2d/percentileFilter.hpp"
#include "NumCpp/Filter/Filters/Filters2d/rankFilter.hpp"
#include "NumCpp/Filter/Filters/Filters2d/uniformFilter.hpp"
