// Copyright 2018 David Pilger
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files(the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, 
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
// permit persons to whom the Software is furnished to do so, subject to the following 
// conditions :
//
// The above copyright notice and this permission notice shall be included in all copies 
// or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
// DEALINGS IN THE SOFTWARE.

#pragma once

#include"NdArray.hpp"
#include"Types.hpp"

#include<boost/filesystem.hpp>

#include<deque>
#include<limits>
#include<stdexcept>

namespace NumC
{
    //================================================================================
    // Class Description:
    //						convience container for holding a uniform array of NdArrays
    //
    template<typename dtype>
    class DataCube
    {
    public:
        //================================Typedefs==================================
        typedef typename std::deque<NdArray<dtype> >::iterator       iterator;
        typedef typename std::deque<NdArray<dtype> >::const_iterator const_iterator;

    private:
        //================================Attributes==================================
        std::deque<NdArray<dtype> >  cube_;
        Shape                       elementShape_;

    public:
        //============================================================================
        // Method Description: 
        //						Default Constructor
        //		
        // Inputs:
        //				None
        // Outputs:
        //				None
        //
        DataCube() :
            elementShape_(0, 0)
        {};

        //============================================================================
        // Method Description: 
        //						Constructor, preallocates to the input size
        //		
        // Inputs:
        //				size
        // Outputs:
        //				None
        //
        DataCube(uint32 inSize) :
            cube_(inSize),
            elementShape_(0, 0)
        {};

        //============================================================================
        // Method Description: 
        //						access method, with bounds checking
        //		
        // Inputs:
        //				index
        // Outputs:
        //				NdArray
        //
        NdArray<dtype>& at(uint32 inIndex)
        {
            return cube_.at(inIndex);
        }

        //============================================================================
        // Method Description: 
        //						const access method, with bounds checking
        //		
        // Inputs:
        //				index
        // Outputs:
        //				NdArray
        //
        const NdArray<dtype>& at(uint32 inIndex) const
        {
            return cube_.at(inIndex);
        }

        //============================================================================
        // Method Description: 
        //						returns a reference to the last element of the array
        //		
        // Inputs:
        //				None
        // Outputs:
        //				NdArray&
        //
        NdArray<dtype>& back()
        {
            return cube_.back();
        }

        //============================================================================
        // Method Description: 
        //						returns an iterator to the beginning of the container
        //		
        // Inputs:
        //				None
        // Outputs:
        //				iterator
        //
        iterator begin()
        {
            return cube_.begin();
        }

        //============================================================================
        // Method Description: 
        //						returns a const_iterator to the beginning of the container
        //		
        // Inputs:
        //				None
        // Outputs:
        //				const_iterator
        //
        const_iterator cbegin() const
        {
            return cube_.cbegin();
        }

        //============================================================================
        // Method Description: 
        //						outputs the DataCube as a .bin file
        //		
        // Inputs:
        //				None
        // Outputs:
        //				None
        //
        void dump(const std::string& inFilename) const
        {
            boost::filesystem::path p(inFilename);
            if (!boost::filesystem::exists(p.parent_path()))
            {
                std::string errStr = "ERROR: DataCube::dump: Input path does not exist:\n\t" + p.parent_path().string();
                throw std::runtime_error(errStr);
            }

            std::string ext = "";
            if (!p.has_extension())
            {
                ext += ".bin";
            }

            std::ofstream ofile((inFilename + ext).c_str(), std::ios::binary);
            for (const_iterator it = cbegin(); it < cend(); ++it)
            {
                ofile.write(reinterpret_cast<const char*>(it->cbegin()), it->size() * sizeof(dtype));
            }

            ofile.close();
        }

        //============================================================================
        // Method Description: 
        //						tests whether or not the container is empty
        //		
        // Inputs:
        //				None
        // Outputs:
        //				bool
        //
        bool isempty()
        {
            return cube_.empty();
        }

        //============================================================================
        // Method Description: 
        //						returns an iterator to 1 past the end of the container
        //		
        // Inputs:
        //				None
        // Outputs:
        //				iterator
        //
        iterator end()
        {
            return cube_.end();
        }

        //============================================================================
        // Method Description: 
        //						returns a const_iterator to 1 past the end of the container
        //		
        // Inputs:
        //				None
        // Outputs:
        //				const_iterator
        //
        const_iterator cend() const
        {
            return cube_.cend();
        }

        //============================================================================
        // Method Description: 
        //						returns a reference to the first element of the array
        //		
        // Inputs:
        //				None
        // Outputs:
        //				NdArray&
        //
        NdArray<dtype>& front()
        {
            return cube_.front();
        }

        //============================================================================
        // Method Description: 
        //						returns the number shape of the element arrays
        //		
        // Inputs:
        //				None
        // Outputs:
        //				Shape
        //
        const Shape& shape() const
        {
            return elementShape_;
        }

        //============================================================================
        // Method Description: 
        //						returns the size of the container array
        //		
        // Inputs:
        //				None
        // Outputs:
        //				uint16 size
        //
        uint32 size() const
        {
            return static_cast<uint32>(cube_.size());
        }

        //============================================================================
        // Method Description: 
        //						Removes the last element in the container
        //		
        // Inputs:
        //				None
        // Outputs:
        //				None
        //
        void pop_back()
        {
            cube_.pop_back();
        }

        //============================================================================
        // Method Description: 
        //						Removes the first element in the container
        //		
        // Inputs:
        //				None
        // Outputs:
        //				None
        //
        void pop_front()
        {
            cube_.pop_front();
        }

        //============================================================================
        // Method Description: 
        //						Adds a new element at the end of the container
        //		
        // Inputs:
        //				NdArray
        // Outputs:
        //				None
        //
        void push_back(const NdArray<dtype>& inArray)
        {
            Shape inputShape = inArray.shape();

            if (elementShape_.rows == 0 && elementShape_.cols == 0)
            {
                // initialize to the first input array size
                elementShape_.rows = inputShape.rows;
                elementShape_.cols = inputShape.cols;
            }

            if (inputShape != elementShape_)
            {
                throw std::invalid_argument("ERROR: NumC::DataCube::push_back: element arrays must all be the same shape.");
            }

            cube_.push_back(inArray);
        }

        //============================================================================
        // Method Description: 
        //						Adds a new element at the beginning of the container
        //		
        // Inputs:
        //				NdArray
        // Outputs:
        //				None
        //
        void push_front(const NdArray<dtype>& inArray)
        {
            Shape inputShape = inArray.shape();

            if (elementShape_.rows == 0 && elementShape_.cols == 0)
            {
                // initialize to the first input array size
                elementShape_.rows = inputShape.rows;
                elementShape_.cols = inputShape.cols;
            }

            if (inputShape != elementShape_)
            {
                throw std::invalid_argument("ERROR: NumC::DataCube::push_front: element arrays must all be the same shape.");
            }

            cube_.push_front(inArray);
        }

        //============================================================================
        // Method Description: 
        //						access operator, no bounds checking
        //		
        // Inputs:
        //				index
        // Outputs:
        //				NdArray
        //
        NdArray<dtype>& operator[](uint32 inIndex)
        {
            return cube_[inIndex];
        }

        //============================================================================
        // Method Description: 
        //						const access operator, no bounds checking
        //		
        // Inputs:
        //				index
        // Outputs:
        //				NdArray
        //
        const NdArray<dtype>& operator[](uint32 inIndex) const
        {
            return cube_[inIndex];
        }
    };
}
