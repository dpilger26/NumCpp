/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
/// Copyright 2019 David Pilger
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
/// Standard NumCpp error
///
#pragma once

#include <iostream>
#include <string>
#include <stdexcept>

namespace nc
{
    namespace error
    {
        //============================================================================
        ///						Throws an invalid argument error
        ///
        /// @param      msg: the message to throw
        ///
        void throwInvalidArgument(const std::string msg = "")
        {
            std::cerr << msg << std::endl;
            throw std::invalid_argument(msg);
        }

        //============================================================================
        ///						Throws an runtime error
        ///
        /// @param      msg: the message to throw
        ///
        void throwRuntime(const std::string msg = "")
        {
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }
    }
}
