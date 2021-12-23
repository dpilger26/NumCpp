/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2022 David Pilger
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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/Filesystem.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Construct an array from data in a binary file.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fromfile.html
    ///
    /// @param				inFilename
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fromfile(const std::string& inFilename)
    {
        if (!filesystem::File(inFilename).exists())
        {
            THROW_INVALID_ARGUMENT_ERROR("input filename does not exist.\n\t" + inFilename);
        }

        // read in as binary file
        std::ifstream file(inFilename.c_str(), std::ios::in | std::ios::binary);
        if (!file.is_open())
        {
            THROW_INVALID_ARGUMENT_ERROR("unable to open file\n\t" + inFilename);
        }

        file.seekg(0, std::ifstream::end);
        const uint32 fileSize = static_cast<uint32>(file.tellg());
        file.seekg(0, std::ifstream::beg);

        std::vector<char> fileBuffer;
        fileBuffer.reserve(fileSize);
        file.read(fileBuffer.data(), fileSize);

        if (file.bad() || file.fail())
        {
            THROW_INVALID_ARGUMENT_ERROR("error occured while reading the file");
        }

        file.close();

        NdArray<dtype> returnArray(reinterpret_cast<dtype*>(fileBuffer.data()), fileSize / sizeof(dtype));

        return returnArray;
    }
        
    //============================================================================
    // Method Description:
    ///						Construct an array from data in a text file.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fromfile.html
    ///
    /// @param				inFilename
    /// @param				inSep: Delimiter separator between values in the file
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fromfile(const std::string& inFilename, const char inSep)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::ifstream file(inFilename.c_str());
        if (!file.is_open())
        {
            THROW_INVALID_ARGUMENT_ERROR("unable to open file\n\t" + inFilename);
        }

        std::vector<dtype> values;
        uint32 lineNumber = 0;
        while (!file.eof())
        {
            std::string line;
            std::getline(file, line, inSep);

            std::istringstream iss(line);
            try
            {
                dtype value;
                while (iss >> value)
                {
                    values.push_back(value);
                }
            }
            catch (const std::invalid_argument& ia)
            {
                std::cout << "Warning: fromfile: line " << lineNumber << '\n' << ia.what() << std::endl;
            }
            catch (...)
            {
                std::cout << "Warning: fromfile: line " << lineNumber << std::endl;
            }

            ++lineNumber;
        }
        file.close();

        return NdArray<dtype>(values);
    }
}  // namespace nc
