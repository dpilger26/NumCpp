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
/// Provides simple filesystem functions
///
#pragma once

#include <fstream>
#include <string>

namespace nc
{
    namespace filesystem
    {
        //================================================================================
        /// Provides simple filesystem functions
        class File
        {
        public:
            //============================================================================
            // Method Description:
            /// Constructor
            ///
            /// @param filename: the full filename
            ///
            explicit File(const std::string& filename) :
                fullFilename_(filename)
            {
                const size_t dot = filename.find_last_of('.');

                filename_ = filename.substr(0, dot);

                if (dot != std::string::npos)
                {
                    extension_ = filename.substr(dot + 1, std::string::npos);
                }

                std::ifstream f(filename.c_str());
                exists_ = f.good();
            }

            //============================================================================
            // Method Description:
            /// Tests whether or not the file exists
            ///
            /// @return bool
            ///
            bool exists() const noexcept
            {
                return exists_;
            }

            //============================================================================
            // Method Description:
            /// Returns the file extension without the dot
            ///
            /// @return std::string
            ///
            const std::string& ext() const noexcept
            {
                return extension_;
            }

            //============================================================================
            // Method Description:
            /// Returns the input full filename
            ///
            /// @return std::string
            ///
            std::string fullName() const
            {
                return filename_ + "." + extension_;
            }

            //============================================================================
            // Method Description:
            /// Returns true if the file has an extension
            ///
            /// @return bool
            ///
            bool hasExt() const
            {
                return !extension_.empty();
            }

            //============================================================================
            // Method Description:
            /// Returns the filename
            ///
            /// @return std::string
            ///
            const std::string& name() const noexcept
            {
                return filename_;
            }

            //============================================================================
            // Method Description:
            /// Sets the extension to the input extension.  Do not input the dot.
            /// E.g. input "txt", not ".txt"
            ///
            /// @return std::string
            ///
            std::string withExt(const std::string& ext)
            {
                extension_ = ext;
                return fullName();
            }

        private:
            //================================Attributes==================================
            std::string fullFilename_{ "" };
            std::string filename_{ "" };
            std::string extension_{ "" };
            bool        exists_{ false };
        };
    } // namespace filesystem
} // namespace nc
