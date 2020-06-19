/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.0.0
///
/// @section License
/// Copyright 2020 David Pilger
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
/// @section Description
/// Degree seperation between the two Coordinates
///
#pragma once

#include "NumCpp/Coordinates/Coordinate.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    namespace coordinates
    {
        //============================================================================
        ///						Returns the degree seperation between the two Coordinates
        ///
        /// @param				inCoordinate1
        /// @param              inCoordinate2
        ///
        /// @return             degrees
        ///
        inline double degreeSeperation(const Coordinate& inCoordinate1, const Coordinate& inCoordinate2) 
        {
            return inCoordinate1.degreeSeperation(inCoordinate2);
        }

        //============================================================================
        ///						Returns the degree seperation between the Coordinate
        ///                     and the input vector
        ///
        /// @param				inVector1
        /// @param              inVector2
        ///
        /// @return             degrees
        ///
        inline double degreeSeperation(const NdArray<double>& inVector1, const NdArray<double>& inVector2)
        {
            const Coordinate inCoord1(inVector1);
            return inCoord1.degreeSeperation(inVector2);
        }
    }
}
