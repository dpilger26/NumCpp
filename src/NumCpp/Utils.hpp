/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section LICENSE
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
/// @section DESCRIPTION
/// Usefull utility type functions
///
#pragma once

#include<string>

namespace NC
{
    //================================================================================
    /// Usefull utility type functions
    template<typename dtype>
    class Utils
    {
    public:
        //============================================================================
        ///						Converts the number into a string
        ///		
        /// @param      number
        ///
        /// @return     string
        ///
        static std::string num2str(dtype inNumber)
        {
            return std::to_string(inNumber);
        }

        //============================================================================
        ///						Squares in input value
        ///		
        /// @param      dtype
        ///
        /// @return     dtype
        ///
        static dtype sqr(dtype inValue)
        {
            return inValue * inValue;
        }

        //============================================================================
        ///						Cubes in input value
        ///		
        /// @param      dtype
        ///
        /// @return     dtype
        ///
        static dtype cube(dtype inValue)
        {
            return inValue * inValue * inValue;
        }

        //============================================================================
        ///						Raises the input value to a power
        ///		
        /// @param      dtype
        ///
        /// @return     dtype
        ///
        static dtype power(dtype inValue, uint8 inPower)
        {
            dtype returnVal = 1;
            for (uint8 i = 0; i < inPower; ++i)
            {
                returnVal *= inValue;
            }
            return returnVal;
        }
    };
}
