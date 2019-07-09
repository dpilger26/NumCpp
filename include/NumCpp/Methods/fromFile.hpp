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
/// Methods for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include "boost/filesystem.hpp"

#include <fstream>
#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Construct an array from data in a text or binary file.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fromfile.html
    ///
    /// @param				inFilename
    /// @param				inSep: Separator between items if file is a text file. Empty ("")
    ///							separator means the file should be treated as binary.
    ///							Right now the only supported seperators are " ", "\t", "\n"
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fromfile(const std::string& inFilename, const std::string& inSep = "")
    {
        boost::filesystem::path p(inFilename);
        if (!boost::filesystem::exists(inFilename))
        {
            THROW_INVALID_ARGUMENT_ERROR("input filename does not exist.\n\t" + inFilename);
        }

        if (inSep.compare("") == 0)
        {
            // read in as binary file
            std::ifstream file(inFilename.c_str(), std::ios::in | std::ios::binary);
            if (!file.is_open())
            {
                THROW_INVALID_ARGUMENT_ERROR("unable to open file.");
            }

            file.seekg(0, file.end);
            const uint32 fileSize = static_cast<uint32>(file.tellg());
            file.seekg(0, file.beg);

            char* fileBuffer = new char[fileSize];
            file.read(fileBuffer, fileSize);

            if (file.bad() || file.fail())
            {
                THROW_INVALID_ARGUMENT_ERROR("error occured while reading the file");
            }

            file.close();

            NdArray<dtype> returnArray(reinterpret_cast<dtype*>(fileBuffer), fileSize);
            delete[] fileBuffer;

            return returnArray;
        }
        else
        {
            // read in as txt file
            if (!(inSep.compare(" ") == 0 || inSep.compare("\t") == 0 || inSep.compare("\n") == 0))
            {
                THROW_INVALID_ARGUMENT_ERROR("only [' ', '\\t', '\\n'] seperators are supported");
            }

            std::vector<dtype> values;

            std::ifstream file(inFilename.c_str());
            if (file.is_open())
            {
                while (!file.eof())
                {
                    std::string line;
                    std::getline(file, line);

                    std::istringstream iss(line);
                    try
                    {
                        values.push_back(static_cast<dtype>(std::stod(iss.str())));
                    }
                    catch (const std::invalid_argument& ia)
                    {
                        std::cout << "Warning: fromfile: " << ia.what() << std::endl;
                        ///throw;
                    }
                }
                file.close();
            }
            else
            {
                THROW_INVALID_ARGUMENT_ERROR("unable to open file.");
            }

            return NdArray<dtype>(values);
        }
    }
}
