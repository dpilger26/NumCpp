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
/// Coordinates transformation functions
///
#pragma once

#include "NumCpp/Coordinates/Transforms/AzElToENU.hpp"
#include "NumCpp/Coordinates/Transforms/AzElToNED.hpp"
#include "NumCpp/Coordinates/Transforms/ECEFEulerToENURollPitchYaw.hpp"
#include "NumCpp/Coordinates/Transforms/ECEFEulerToNEDRollPitchYaw.hpp"
#include "NumCpp/Coordinates/Transforms/ECEFtoAzElGeocentric.hpp"
#include "NumCpp/Coordinates/Transforms/ECEFtoAzElGeodetic.hpp"
#include "NumCpp/Coordinates/Transforms/ECEFtoENU.hpp"
#include "NumCpp/Coordinates/Transforms/ECEFtoLLA.hpp"
#include "NumCpp/Coordinates/Transforms/ECEFtoNED.hpp"
#include "NumCpp/Coordinates/Transforms/ENURollPitchYawToECEFEuler.hpp"
#include "NumCpp/Coordinates/Transforms/ENUUnitVecsInECEF.hpp"
#include "NumCpp/Coordinates/Transforms/ENUtoAzEl.hpp"
#include "NumCpp/Coordinates/Transforms/ENUtoECEF.hpp"
#include "NumCpp/Coordinates/Transforms/ENUtoLLA.hpp"
#include "NumCpp/Coordinates/Transforms/ENUtoNED.hpp"
#include "NumCpp/Coordinates/Transforms/LLAtoAzElGeocentric.hpp"
#include "NumCpp/Coordinates/Transforms/LLAtoAzElGeodetic.hpp"
#include "NumCpp/Coordinates/Transforms/LLAtoECEF.hpp"
#include "NumCpp/Coordinates/Transforms/LLAtoENU.hpp"
#include "NumCpp/Coordinates/Transforms/LLAtoNED.hpp"
#include "NumCpp/Coordinates/Transforms/NEDRollPitchYawToECEFEuler.hpp"
#include "NumCpp/Coordinates/Transforms/NEDUnitVecsInECEF.hpp"
#include "NumCpp/Coordinates/Transforms/NEDtoAzEl.hpp"
#include "NumCpp/Coordinates/Transforms/NEDtoENU.hpp"
#include "NumCpp/Coordinates/Transforms/NEDtoLLA.hpp"
