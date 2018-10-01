/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
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
/// @section Description
/// A Shape Class for NdArrays
///
#pragma once

#include"NumCpp/Types.hpp"
#include"NumCpp/Utils.hpp"

#include<iostream>
#include<stdexcept>
#include<string>

namespace NC
{
    //================================================================================
    ///						A Shape Class for NdArrays
    class Shape
    {
    public:
        //====================================Attributes==============================
        uint32	rows{0};
        uint32	cols{0};

        //============================================================================
        ///						Constructor
        ///
        Shape() = default;

        //============================================================================
        ///						Constructor
        ///		
        /// @param      inSquareSize
        ///
        explicit Shape(uint32 inSquareSize) :
            rows(inSquareSize),
            cols(inSquareSize)
        {};

        //============================================================================
        ///						Constructor
        ///		
        /// @param      inRows
        /// @param      inCols
        ///
        Shape(uint32 inRows, uint32 inCols) :
            rows(inRows),
            cols(inCols)
        {};

        //============================================================================
        ///						Equality operator
        ///		
        /// @param      inOtherShape
        ///
        /// @return     bool
        ///
        bool operator==(const Shape& inOtherShape) const
        {
            return rows == inOtherShape.rows && cols == inOtherShape.cols;
        }

        //============================================================================
        ///						Not equality operator
        ///		
        /// @param      inOtherShape
        ///
        /// @return     bool
        ///
        bool operator!=(const Shape& inOtherShape) const
        {
            return !(*this == inOtherShape);
        }

        //============================================================================
        ///						Returns the size of the shape
        ///
        /// @return     size
        ///
        uint32 size() const
        {
            return rows * cols;
        }

        //============================================================================
        ///						Returns whether the shape is null (constructed with the 
        ///						default constructor).
        ///
        /// @return     bool
        ///
        bool isnull()
        {
            return rows == 0 && cols == 0;
        }

        //============================================================================
        ///						Returns the shape as a string representation
        ///
        /// @return     std::string
        ///
        std::string str() const
        {
            std::string out = "[" + Utils<uint32>::num2str(rows) + ", " + Utils<uint32>::num2str(cols) + "]\n";
            return out;
        }

        //============================================================================
        ///						Prints the shape to the console
        ///
        void print() const
        {
            std::cout << *this;
        }

        //============================================================================
        ///						IO operator for the Shape class
        ///		
        /// @param      inOStream
        /// @param      inShape
        ///
        /// @return     std::ostream
        ///
        friend std::ostream& operator<<(std::ostream& inOStream, const Shape& inShape)
        {
            inOStream << inShape.str();
            return inOStream;
        }
    };
}
