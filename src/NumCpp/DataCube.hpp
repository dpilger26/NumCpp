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
/// Convience container for holding a uniform array of NdArrays
///
#pragma once

#include"NumCpp/NdArray.hpp"
#include"NumCpp/Types.hpp"

#include"boost/filesystem.hpp"

#include<deque>
#include<limits>
#include<stdexcept>

namespace NC
{
    //================================================================================
    ///						Convenience container for holding a uniform array of NdArrays
    template<typename dtype>
    class DataCube
    {
    public:
        //================================Typedefs==================================
        typedef typename std::deque<NdArray<dtype> >::iterator       iterator;
        typedef typename std::deque<NdArray<dtype> >::const_iterator const_iterator;

    private:
        //================================Attributes==================================
        std::deque<NdArray<dtype> >     cube_;
        Shape                           elementShape_;

    public:
        //============================================================================
        ///						Default Constructor
        ///		
        /// @param      None
        ///
        /// @return     None
        ///
        DataCube() :
            elementShape_(0, 0)
        {};

        //============================================================================
        ///						Constructor, preallocates to the input size
        ///		
        /// @param      size
        ///
        /// @return     None
        ///
        DataCube(uint32 inSize) :
            cube_(inSize),
            elementShape_(0, 0)
        {};

        //============================================================================
        ///						Access method, with bounds checking
        ///		
        /// @param      index
        ///
        /// @return     NdArray
        ///
        NdArray<dtype>& at(uint32 inIndex)
        {
            return cube_.at(inIndex);
        }

        //============================================================================
        ///						Const access method, with bounds checking
        ///		
        /// @param      index
        ///
        /// @return     NdArray
        ///
        const NdArray<dtype>& at(uint32 inIndex) const
        {
            return cube_.at(inIndex);
        }

        //============================================================================
        ///						Returns a reference to the last element of the array
        ///		
        /// @param      None
        ///
        /// @return     NdArray&
        ///
        NdArray<dtype>& back()
        {
            return cube_.back();
        }

        //============================================================================
        ///						Returns an iterator to the beginning of the container
        ///		
        /// @param      None
        ///
        /// @return     iterator
        ///
        iterator begin()
        {
            return cube_.begin();
        }

        //============================================================================
        ///						Returns a const_iterator to the beginning of the container
        ///		
        /// @param      None
        ///
        /// @return     const_iterator
        ///
        const_iterator cbegin() const
        {
            return cube_.cbegin();
        }

        //============================================================================
        ///						Outputs the DataCube as a .bin file
        ///		
        /// @param      None
        ///
        /// @return     None
        ///
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
        ///						Tests whether or not the container is empty
        ///		
        /// @param      None
        ///
        /// @return     bool
        ///
        bool isempty()
        {
            return cube_.empty();
        }

        //============================================================================
        ///						Returns an iterator to 1 past the end of the container
        ///		
        /// @param      None
        ///
        /// @return     iterator
        ///
        iterator end()
        {
            return cube_.end();
        }

        //============================================================================
        ///						Returns a const_iterator to 1 past the end of the container
        ///		
        /// @param      None
        ///
        /// @return     const_iterator
        ///
        const_iterator cend() const
        {
            return cube_.cend();
        }

        //============================================================================ 
        ///						returns a reference to the first element of the array
        ///		
        /// @param      None
        ///
        /// @return     NdArray&
        ///
        NdArray<dtype>& front()
        {
            return cube_.front();
        }

        //============================================================================
        ///						returns the number shape of the element arrays
        ///		
        /// @param      None
        ///
        /// @return     Shape
        ///
        const Shape& shape() const
        {
            return elementShape_;
        }

        //============================================================================
        ///						Returns the size of the container array
        ///		
        /// @param      None
        ///
        /// @return     size
        ///
        uint32 size() const
        {
            return static_cast<uint32>(cube_.size());
        }

        //============================================================================
        ///						Removes the last element in the container
        ///		
        /// @param      None
        ///
        /// @return     None
        ///
        void pop_back()
        {
            cube_.pop_back();
        }

        //============================================================================
        ///						Removes the first element in the container
        ///		
        /// @param      None
        ///
        /// @return     None
        ///
        void pop_front()
        {
            cube_.pop_front();
        }

        //============================================================================
        ///						Adds a new element at the end of the container
        ///		
        /// @param      NdArray
        ///
        /// @return     None
        ///
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
                throw std::invalid_argument("ERROR: NC::DataCube::push_back: element arrays must all be the same shape.");
            }

            cube_.push_back(inArray);
        }

        //============================================================================
        ///						Adds a new element at the beginning of the container
        ///		
        /// @param      NdArray
        ///
        /// @return     None
        ///
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
                throw std::invalid_argument("ERROR: NC::DataCube::push_front: element arrays must all be the same shape.");
            }

            cube_.push_front(inArray);
        }

        //============================================================================
        ///						Access operator, no bounds checking
        ///		
        /// @param      index
        ///
        /// @return     NdArray
        ///
        NdArray<dtype>& operator[](uint32 inIndex)
        {
            return cube_[inIndex];
        }

        //============================================================================
        ///						Const access operator, no bounds checking
        ///		
        /// @param      index
        ///
        /// @return     NdArray
        ///
        const NdArray<dtype>& operator[](uint32 inIndex) const
        {
            return cube_[inIndex];
        }
    };
}
