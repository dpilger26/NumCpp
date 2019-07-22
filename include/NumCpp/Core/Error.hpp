/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
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
/// Standard NumCpp errors
///
#pragma once

#include "NumCpp/Core/Types.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#define THROW_INVALID_ARGUMENT_ERROR(msg) nc::error::throwError<std::invalid_argument>(__FILE__, __func__, __LINE__, msg)
#define THROW_RUNTIME_ERROR(msg) nc::error::throwError<std::runtime_error>(__FILE__, __func__, __LINE__, msg)

namespace nc
{
    namespace error
    {
        //============================================================================
        ///						Makes the full error message string
        ///
        /// @param      file: the file
        /// @param      function: the function
        /// @param      line: the line of the file
        /// @param      msg: the message to throw (default "")
        ///
        template<typename ErrorType>
        void throwError(const std::string& file,
            const std::string& function,
            uint32 line,
            const std::string& msg = "")
        {
            std::string errMsg = "File: " + file + "\n\tFunction: " + function + "\n\tLine: " + std::to_string(line) + "\n\tError: " + msg;
            std::cerr << errMsg;
            throw ErrorType(errMsg);
        }
    }
}

