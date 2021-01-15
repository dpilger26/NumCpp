/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2021 David Pilger
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
/// A Class for slicing into NdArrays
///

#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Utils/num2str.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace nc
{
    //================================================================================
    ///						A Class for slicing into NdArrays
    class Slice
    {
    public:
        //====================================Attributes==============================
        int32	start{ 0 };
        int32	stop{ 1 };
        int32	step{ 1 };

        //============================================================================
        ///						Constructor
        ///
        constexpr Slice() = default;

        //============================================================================
        ///						Constructor
        ///
        /// @param      inStop (index not included)
        ///
        constexpr explicit Slice(int32 inStop) noexcept :
            stop(inStop)
        {}

        //============================================================================
        ///						Constructor
        ///
        /// @param          inStart
        /// @param			inStop (index not included)
        ///
        constexpr Slice(int32 inStart, int32 inStop) noexcept :
            start(inStart),
            stop(inStop)
        {}

        //============================================================================
        ///						Constructor
        ///
        /// @param      inStart
        /// @param      inStop (not included)
        /// @param      inStep
        ///
        constexpr Slice(int32 inStart, int32 inStop, int32 inStep) noexcept :
            start(inStart),
            stop(inStop),
            step(inStep)
        {}

        //============================================================================
        ///						Equality operator
        ///
        /// @param      inOtherSlice
        ///
        /// @return     bool
        ///
        bool operator==(const Slice& inOtherSlice) const noexcept
        {
            return start == inOtherSlice.start && stop == inOtherSlice.stop && step == inOtherSlice.step;
        }

        //============================================================================
        ///						Not equality operator
        ///
        /// @param      inOtherSlice
        ///
        /// @return     bool
        ///
        bool operator!=(const Slice& inOtherSlice) const noexcept
        {
            return !(*this == inOtherSlice);
        }

        //============================================================================
        ///						Prints the shape to the console
        ///
        /// @return     std::string
        ///
        std::string str() const 
        {
            std::string out = "[" + utils::num2str(start) + ":" + utils::num2str(stop) + ":" + utils::num2str(step) + "]\n";
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
        ///						Make the slice all positive and does some error checking
        ///
        /// @param      inArraySize
        ///
        void makePositiveAndValidate(uint32 inArraySize)
        {
            /// convert the start value
            if (start < 0)
            {
                start += inArraySize;
            }
            if (start > static_cast<int32>(inArraySize - 1))
            {
                THROW_INVALID_ARGUMENT_ERROR("Invalid start value for array of size " + utils::num2str(inArraySize) + ".");
            }

            /// convert the stop value
            if (stop < 0)
            {
                stop += inArraySize;
            }
            if (stop > static_cast<int32>(inArraySize))
            {
                THROW_INVALID_ARGUMENT_ERROR("Invalid stop value for array of size " + utils::num2str(inArraySize) + ".");
            }

            /// do some error checking
            if (start < stop)
            {
                if (step < 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("Invalid slice values.");
                }
            }

            if (stop < start)
            {
                if (step > 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("Invalid slice values.");
                }

                /// otherwise flip things around for my own sanity
                std::swap(start, stop);
                step *= -1;
            }
        }

        //============================================================================
        ///						Returns the number of elements that the slice contains.
        ///						be aware that this method will also make the slice all 
        ///						positive! 
        ///
        /// @param      inArraySize
        ///
        uint32 numElements(uint32 inArraySize) 
        { 
            makePositiveAndValidate(inArraySize); 

            uint32 num = 0; 
            for (int32 i = start; i < stop; i += step) 
            { 
                ++num; 
            } 
            return num; 
        } 

        //============================================================================
        ///						IO operator for the Slice class
        ///
        /// @param      inOStream
        /// @param      inSlice
        ///
        /// @return     std::ostream
        ///
        friend std::ostream& operator<<(std::ostream& inOStream, const Slice& inSlice) 
        {
            inOStream << inSlice.str();
            return inOStream;
        }
    };
} // namespace nc
