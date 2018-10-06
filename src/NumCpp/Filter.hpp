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
/// Image and signal filtering
///
#pragma once

#include<NumCpp/NdArray.hpp>
#include<NumCpp/Methods.hpp>
#include<NumCpp/Types.hpp>
#include<NumCpp/Utils.hpp>

#include<cmath>
#include<iostream>
#include<string>
#include<utility>

namespace NC
{
    //================================================================================
    ///						Image and signal filtering
    namespace Filter
    {
        //================================================================================
        // Enum Description:
        ///						Boundary condition to apply to the image filter
        enum class Boundary { REFLECT = 0, CONSTANT, NEAREST, MIRROR, WRAP };

        //============================================================================
        // Method Description: 
        ///						samples a gaussian of mean zero and input STD sigma
        ///		
        /// @param				inX
        /// @param              inY
        /// @param              inSigma
        ///              
        /// @return             dtype
        ///
        template<typename dtype>
        dtype gaussian(dtype inX, dtype inY, dtype inSigma)
        {
            double exponent = -(Utils::sqr(inX) + Utils::sqr(inY)) / (2 * Utils::sqr(inSigma));
            return static_cast<dtype>(std::exp(exponent));
        }

        //============================================================================
        // Method Description: 
        ///						extends the corner values
        ///		
        /// @param				inArray
        /// @param              inBorderWidth
        ///
        template<typename dtype>
        void fillCorners(NdArray<dtype>& inArray, uint32 inBorderWidth)
        {
            Shape inShape = inArray.shape();
            int32 numRows = static_cast<int32>(inShape.rows);
            int32 numCols = static_cast<int32>(inShape.cols);

            // top left
            inArray.put(Slice(0, inBorderWidth), Slice(0, inBorderWidth), inArray(inBorderWidth, inBorderWidth));

            // top right
            inArray.put(Slice(0, inBorderWidth), Slice(numCols - inBorderWidth, numCols), inArray(inBorderWidth, numCols - inBorderWidth - 1));

            // bottom left
            inArray.put(Slice(numRows - inBorderWidth, numRows), Slice(0, inBorderWidth), inArray(numRows - inBorderWidth - 1, inBorderWidth));

            // bottom right
            inArray.put(Slice(numRows - inBorderWidth, numRows), Slice(numCols - inBorderWidth, numCols), inArray(numRows - inBorderWidth - 1, numCols - inBorderWidth - 1));
        }

        //============================================================================
        // Method Description: 
        ///						extends the corner values
        ///		
        /// @param				inArray
        /// @param              inBorderWidth
        /// @param				inFillValue
        ///
        template<typename dtype>
        void fillCorners(NdArray<dtype>& inArray, uint32 inBorderWidth, dtype inFillValue)
        {
            Shape inShape = inArray.shape();
            int32 numRows = static_cast<int32>(inShape.rows);
            int32 numCols = static_cast<int32>(inShape.cols);

            // top left
            inArray.put(Slice(0, inBorderWidth), Slice(0, inBorderWidth), inFillValue);

            // top right
            inArray.put(Slice(0, inBorderWidth), Slice(numCols - inBorderWidth, numCols), inFillValue);

            // bottom left
            inArray.put(Slice(numRows - inBorderWidth, numRows), Slice(0, inBorderWidth), inFillValue);

            // bottom right
            inArray.put(Slice(numRows - inBorderWidth, numRows), Slice(numCols - inBorderWidth, numCols), inFillValue);
        }

        //============================================================================
        // Method Description: 
        ///						Reflects the boundaries
        ///		
        /// @param				inImage
        /// @param              inBoundarySize
        ///
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> reflectBoundary(const NdArray<dtype>& inImage, uint32 inBoundarySize)
        {
            Shape inShape = inImage.shape();
            Shape outShape(inShape);
            outShape.rows += inBoundarySize * 2;
            outShape.cols += inBoundarySize * 2;

            NdArray<dtype> outArray(outShape);
            outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows), Slice(inBoundarySize, inBoundarySize + inShape.cols), inImage);

            for (uint32 row = 0; row < inBoundarySize; ++row)
            {
                // bottom
                outArray.put(row,
                    Slice(inBoundarySize, inBoundarySize + inShape.cols),
                    inImage(inBoundarySize - row - 1, Slice(0, inShape.cols)));

                // top
                outArray.put(row + inBoundarySize + inShape.rows,
                    Slice(inBoundarySize, inBoundarySize + inShape.cols),
                    inImage(inShape.rows - row - 1, Slice(0, inShape.cols)));
            }

            for (uint32 col = 0; col < inBoundarySize; ++col)
            {
                // left
                outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                    col,
                    inImage(Slice(0, inShape.rows), inBoundarySize - col - 1));

                // right
                outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                    col + inBoundarySize + inShape.cols,
                    inImage(Slice(0, inShape.rows), inShape.cols - col - 1));
            }

            // now fill in the corners
            NdArray<dtype> lowerLeft = flipud(outArray(Slice(inBoundarySize, 2 * inBoundarySize), Slice(0, inBoundarySize)));
            NdArray<dtype> lowerRight = flipud(outArray(Slice(inBoundarySize, 2 * inBoundarySize), Slice(outShape.cols - inBoundarySize, outShape.cols)));

            uint32 upperRowStart = outShape.rows - 2 * inBoundarySize;
            NdArray<dtype> upperLeft = flipud(outArray(Slice(upperRowStart, upperRowStart + inBoundarySize), Slice(0, inBoundarySize)));
            NdArray<dtype> upperRight = flipud(outArray(Slice(upperRowStart, upperRowStart + inBoundarySize), Slice(outShape.cols - inBoundarySize, outShape.cols)));

            outArray.put(Slice(0, inBoundarySize), Slice(0, inBoundarySize), lowerLeft);
            outArray.put(Slice(0, inBoundarySize), Slice(outShape.cols - inBoundarySize, outShape.cols), lowerRight);
            outArray.put(Slice(outShape.rows - inBoundarySize, outShape.rows), Slice(0, inBoundarySize), upperLeft);
            outArray.put(Slice(outShape.rows - inBoundarySize, outShape.rows), Slice(outShape.cols - inBoundarySize, outShape.cols), upperRight);

            return std::move(outArray);
        }

        //============================================================================
        // Method Description: 
        ///						Reflects the boundaries
        ///	
        /// @param			inImage
        /// @param          inBoundarySize
        ///
        /// @return         NdArray
        ///
        template<typename dtype>
        NdArray<dtype> reflectBoundary1d(const NdArray<dtype>& inImage, uint32 inBoundarySize)
        {
            uint32 outSize = inImage.size() + inBoundarySize * 2;

            NdArray<dtype> outArray(1, outSize);
            outArray.put(Slice(inBoundarySize, inBoundarySize + inImage.size()), inImage);

            // left
            outArray.put(Slice(0, inBoundarySize), fliplr(inImage[Slice(0, inBoundarySize)]));

            // right
            outArray.put(Slice(inImage.size() + inBoundarySize, outSize), fliplr(inImage[Slice(-static_cast<int32>(inBoundarySize), inImage.size())]));

            return std::move(outArray);
        }

        //============================================================================
        // Method Description: 
        ///						Constant boundary
        ///		
        /// @param				inImage
        /// @param              inBoundarySize
        /// @param              inConstantValue
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> constantBoundary(const NdArray<dtype>& inImage, uint32 inBoundarySize, dtype inConstantValue)
        {
            Shape inShape = inImage.shape();
            Shape outShape(inShape);
            outShape.rows += inBoundarySize * 2;
            outShape.cols += inBoundarySize * 2;

            NdArray<dtype> outArray(outShape);
            outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows), Slice(inBoundarySize, inBoundarySize + inShape.cols), inImage);
            fillCorners(outArray, inBoundarySize, inConstantValue);

            outArray.put(Slice(0, inBoundarySize), Slice(inBoundarySize, inBoundarySize + inShape.cols), inConstantValue); /// bottom
            outArray.put(Slice(outShape.rows - inBoundarySize, outShape.rows), Slice(inBoundarySize, inBoundarySize + inShape.cols), inConstantValue); /// top
            outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows), Slice(0, inBoundarySize), inConstantValue); /// left
            outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows), Slice(outShape.cols - inBoundarySize, outShape.cols), inConstantValue); /// right

            return std::move(outArray);
        }

        //============================================================================
        // Method Description: 
        ///						Constant boundary1d
        ///		
        /// @param				inImage
        /// @param              inBoundarySize
        /// @param              inConstantValue
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> constantBoundary1d(const NdArray<dtype>& inImage, uint32 inBoundarySize, dtype inConstantValue)
        {
            uint32 outSize = inImage.size() + inBoundarySize * 2;

            NdArray<dtype> outArray(1, outSize);
            outArray.put(Slice(inBoundarySize, inBoundarySize + inImage.size()), inImage);

            // left
            outArray.put(Slice(0, inBoundarySize), inConstantValue);

            // right
            outArray.put(Slice(inImage.size() + inBoundarySize, outSize), inConstantValue);

            return std::move(outArray);
        }

        //============================================================================
        // Method Description: 
        ///						Nearest boundary
        ///		
        /// @param				inImage
        /// @param              inBoundarySize
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> nearestBoundary(const NdArray<dtype>& inImage, uint32 inBoundarySize)
        {
            Shape inShape = inImage.shape();
            Shape outShape(inShape);
            outShape.rows += inBoundarySize * 2;
            outShape.cols += inBoundarySize * 2;

            NdArray<dtype> outArray(outShape);
            outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows), Slice(inBoundarySize, inBoundarySize + inShape.cols), inImage);
            fillCorners(outArray, inBoundarySize);

            for (uint32 row = 0; row < inBoundarySize; ++row)
            {
                // bottom
                outArray.put(row,
                    Slice(inBoundarySize, inBoundarySize + inShape.cols),
                    inImage(0, Slice(0, inShape.cols)));

                // top
                outArray.put(row + inBoundarySize + inShape.rows,
                    Slice(inBoundarySize, inBoundarySize + inShape.cols),
                    inImage(inShape.rows - 1, Slice(0, inShape.cols)));
            }

            for (uint32 col = 0; col < inBoundarySize; ++col)
            {
                // left
                outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                    col,
                    inImage(Slice(0, inShape.rows), 0));

                // right
                outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                    col + inBoundarySize + inShape.cols,
                    inImage(Slice(0, inShape.rows), inShape.cols - 1));
            }

            return std::move(outArray);
        }

        //============================================================================
        // Method Description: 
        ///						Nearest boundary1d
        ///		
        /// @param				inImage
        /// @param              inBoundarySize
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> nearestBoundary1d(const NdArray<dtype>& inImage, uint32 inBoundarySize)
        {
            uint32 outSize = inImage.size() + inBoundarySize * 2;

            NdArray<dtype> outArray(1, outSize);
            outArray.put(Slice(inBoundarySize, inBoundarySize + inImage.size()), inImage);

            // left
            outArray.put(Slice(0, inBoundarySize), inImage[0]);

            // right
            outArray.put(Slice(inImage.size() + inBoundarySize, outSize), inImage[-1]);

            return std::move(outArray);
        }

        //============================================================================
        // Method Description: 
        ///						Mirror boundary
        ///		
        /// @param				inImage
        /// @param              inBoundarySize
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> mirrorBoundary(const NdArray<dtype>& inImage, uint32 inBoundarySize)
        {
            Shape inShape = inImage.shape();
            Shape outShape(inShape);
            outShape.rows += inBoundarySize * 2;
            outShape.cols += inBoundarySize * 2;

            NdArray<dtype> outArray(outShape);
            outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows), Slice(inBoundarySize, inBoundarySize + inShape.cols), inImage);

            for (uint32 row = 0; row < inBoundarySize; ++row)
            {
                // bottom
                outArray.put(row,
                    Slice(inBoundarySize, inBoundarySize + inShape.cols),
                    inImage(inBoundarySize - row, Slice(0, inShape.cols)));

                // top
                outArray.put(row + inBoundarySize + inShape.rows,
                    Slice(inBoundarySize, inBoundarySize + inShape.cols),
                    inImage(inShape.rows - row - 2, Slice(0, inShape.cols)));
            }

            for (uint32 col = 0; col < inBoundarySize; ++col)
            {
                // left
                outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                    col,
                    inImage(Slice(0, inShape.rows), inBoundarySize - col));

                // right
                outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                    col + inBoundarySize + inShape.cols,
                    inImage(Slice(0, inShape.rows), inShape.cols - col - 2));
            }

            // now fill in the corners
            NdArray<dtype> lowerLeft = flipud(outArray(Slice(inBoundarySize + 1, 2 * inBoundarySize + 1), Slice(0, inBoundarySize)));
            NdArray<dtype> lowerRight = flipud(outArray(Slice(inBoundarySize + 1, 2 * inBoundarySize + 1), Slice(outShape.cols - inBoundarySize, outShape.cols)));

            uint32 upperRowStart = outShape.rows - 2 * inBoundarySize - 1;
            NdArray<dtype> upperLeft = flipud(outArray(Slice(upperRowStart, upperRowStart + inBoundarySize), Slice(0, inBoundarySize)));
            NdArray<dtype> upperRight = flipud(outArray(Slice(upperRowStart, upperRowStart + inBoundarySize), Slice(outShape.cols - inBoundarySize, outShape.cols)));

            outArray.put(Slice(0, inBoundarySize), Slice(0, inBoundarySize), lowerLeft);
            outArray.put(Slice(0, inBoundarySize), Slice(outShape.cols - inBoundarySize, outShape.cols), lowerRight);
            outArray.put(Slice(outShape.rows - inBoundarySize, outShape.rows), Slice(0, inBoundarySize), upperLeft);
            outArray.put(Slice(outShape.rows - inBoundarySize, outShape.rows), Slice(outShape.cols - inBoundarySize, outShape.cols), upperRight);

            return std::move(outArray);
        }

        //============================================================================
        // Method Description: 
        ///						Mirror boundary1d
        ///		
        /// @param				inImage
        /// @param              inBoundarySize
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> mirrorBoundary1d(const NdArray<dtype>& inImage, uint32 inBoundarySize)
        {
            uint32 outSize = inImage.size() + inBoundarySize * 2;

            NdArray<dtype> outArray(1, outSize);
            outArray.put(Slice(inBoundarySize, inBoundarySize + inImage.size()), inImage);

            // left
            outArray.put(Slice(0, inBoundarySize), fliplr(inImage[Slice(1, inBoundarySize + 1)]));

            // right
            outArray.put(Slice(inImage.size() + inBoundarySize, outSize), fliplr(inImage[Slice(-static_cast<int32>(inBoundarySize) - 1, -1)]));

            return std::move(outArray);
        }

        //============================================================================
        // Method Description: 
        ///						Wrap boundary
        ///		
        /// @param				inImage
        /// @param              inBoundarySize
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> wrapBoundary(const NdArray<dtype>& inImage, uint32 inBoundarySize)
        {
            Shape inShape = inImage.shape();
            Shape outShape(inShape);
            outShape.rows += inBoundarySize * 2;
            outShape.cols += inBoundarySize * 2;

            NdArray<dtype> outArray(outShape);
            outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows), Slice(inBoundarySize, inBoundarySize + inShape.cols), inImage);

            // bottom
            outArray.put(Slice(0, inBoundarySize),
                Slice(inBoundarySize, inBoundarySize + inShape.cols),
                inImage(Slice(inShape.rows - inBoundarySize, inShape.rows), Slice(0, inShape.cols)));

            // top
            outArray.put(Slice(inShape.rows + inBoundarySize, outShape.rows),
                Slice(inBoundarySize, inBoundarySize + inShape.cols),
                inImage(Slice(0, inBoundarySize), Slice(0, inShape.cols)));

            // left
            outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                Slice(0, inBoundarySize),
                inImage(Slice(0, inShape.rows), Slice(inShape.cols - inBoundarySize, inShape.cols)));

            // right
            outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                Slice(inShape.cols + inBoundarySize, outShape.cols),
                inImage(Slice(0, inShape.rows), Slice(0, inBoundarySize)));

            // now fill in the corners
            NdArray<dtype> lowerLeft = outArray(Slice(inBoundarySize, 2 * inBoundarySize), Slice(0, inBoundarySize));
            NdArray<dtype> lowerRight = outArray(Slice(inBoundarySize, 2 * inBoundarySize), Slice(outShape.cols - inBoundarySize, outShape.cols));

            uint32 upperRowStart = outShape.rows - 2 * inBoundarySize;
            NdArray<dtype> upperLeft = outArray(Slice(upperRowStart, upperRowStart + inBoundarySize), Slice(0, inBoundarySize));
            NdArray<dtype> upperRight = outArray(Slice(upperRowStart, upperRowStart + inBoundarySize), Slice(outShape.cols - inBoundarySize, outShape.cols));

            outArray.put(Slice(0, inBoundarySize), Slice(0, inBoundarySize), upperLeft);
            outArray.put(Slice(0, inBoundarySize), Slice(outShape.cols - inBoundarySize, outShape.cols), upperRight);
            outArray.put(Slice(outShape.rows - inBoundarySize, outShape.rows), Slice(0, inBoundarySize), lowerLeft);
            outArray.put(Slice(outShape.rows - inBoundarySize, outShape.rows), Slice(outShape.cols - inBoundarySize, outShape.cols), lowerRight);

            return std::move(outArray);
        }

        //============================================================================
        // Method Description: 
        ///						Wrap boundary1d
        ///		
        /// @param				inImage
        /// @param              inBoundarySize
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> wrapBoundary1d(const NdArray<dtype>& inImage, uint32 inBoundarySize)
        {
            uint32 outSize = inImage.size() + inBoundarySize * 2;

            NdArray<dtype> outArray(1, outSize);
            outArray.put(Slice(inBoundarySize, inBoundarySize + inImage.size()), inImage);

            // left
            outArray.put(Slice(0, inBoundarySize), inImage[Slice(inImage.size() - inBoundarySize, inImage.size())]);

            // right
            outArray.put(Slice(inImage.size() + inBoundarySize, outSize), inImage[Slice(0, inBoundarySize)]);

            return std::move(outArray);
        }

        //============================================================================
        // Method Description: 
        ///						Wrap boundary
        ///		
        /// @param				inImage
        /// @param              inBoundaryType
        /// @param              inKernalSize
        /// @param              inConstantValue (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> addBoundary(const NdArray<dtype>& inImage, Boundary inBoundaryType, uint32 inKernalSize, dtype inConstantValue = 0)
        {
            if (inKernalSize % 2 == 0)
            {
                std::string errStr = "ERROR: ImageProcessing::Filter::addBoundary: input kernal size must be an odd value.";
                std::cout << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            uint32 boundarySize = inKernalSize / 2; // integer division

            switch (inBoundaryType)
            {
                case Boundary::REFLECT:
                {
                    return std::move(reflectBoundary(inImage, boundarySize));
                }
                case Boundary::CONSTANT:
                {
                    return std::move(constantBoundary(inImage, boundarySize, inConstantValue));
                }
                case Boundary::NEAREST:
                {
                    return std::move(nearestBoundary(inImage, boundarySize));
                }
                case Boundary::MIRROR:
                {
                    return std::move(mirrorBoundary(inImage, boundarySize));
                }
                case Boundary::WRAP:
                {
                    return std::move(wrapBoundary(inImage, boundarySize));
                }
                default:
                {
                    // This can't actually happen but just adding to get rid of compiler warning
                    throw std::invalid_argument("ERROR!");
                }
            }
        }

        //============================================================================
        // Method Description: 
        ///						Wrap boundary
        ///		
        /// @param				inImage
        /// @param              inBoundaryType
        /// @param              inKernalSize
        /// @param              inConstantValue (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> addBoundary1d(const NdArray<dtype>& inImage, Boundary inBoundaryType, uint32 inKernalSize, dtype inConstantValue = 0)
        {
            if (inKernalSize % 2 == 0)
            {
                std::string errStr = "ERROR: ImageProcessing::Filter::addBoundary1d: input kernal size must be an odd value.";
                std::cout << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            uint32 boundarySize = inKernalSize / 2; // integer division

            switch (inBoundaryType)
            {
                case Boundary::REFLECT:
                {
                    return std::move(reflectBoundary1d(inImage, boundarySize));
                }
                case Boundary::CONSTANT:
                {
                    return std::move(constantBoundary1d(inImage, boundarySize, inConstantValue));
                }
                case Boundary::NEAREST:
                {
                    return std::move(nearestBoundary1d(inImage, boundarySize));
                }
                case Boundary::MIRROR:
                {
                    return std::move(mirrorBoundary1d(inImage, boundarySize));
                }
                case Boundary::WRAP:
                {
                    return std::move(wrapBoundary1d(inImage, boundarySize));
                }
                default:
                {
                    // This can't actually happen but just adding to get rid of compiler warning
                    throw std::invalid_argument("ERROR!");
                }
            }
        }

        //============================================================================
        // Method Description: 
        ///						trims the boundary off to make the image back to the original size
        ///		
        /// @param				inImageWithBoundary
        /// @param              inSize
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> trimBoundary(const NdArray<dtype>& inImageWithBoundary, uint32 inSize)
        {
            Shape inShape = inImageWithBoundary.shape();
            uint32 boundarySize = inSize / 2; /// integer division

            inShape.rows -= boundarySize * 2;
            inShape.cols -= boundarySize * 2;

            return std::move(inImageWithBoundary(Slice(boundarySize, boundarySize + inShape.rows), Slice(boundarySize, boundarySize + inShape.cols)));
        }

        //============================================================================
        // Method Description: 
        ///						trims the boundary off to make the image back to the original size
        ///		
        /// @param				inImageWithBoundary
        /// @param              inSize
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> trimBoundary1d(const NdArray<dtype>& inImageWithBoundary, uint32 inSize)
        {
            uint32 boundarySize = inSize / 2; // integer division
            uint32 imageSize = inImageWithBoundary.size() - boundarySize * 2;

            return std::move(inImageWithBoundary[Slice(boundarySize, boundarySize + imageSize)]);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a multidimensional complemenatry median filter.
        ///		
        /// @param				inImageArray
        /// @param				inSize: square size of the kernel to apply
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> complementaryMedianFilter(const NdArray<dtype>& inImageArray, uint32 inSize,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> inImageArrayCopy(inImageArray);
            inImageArrayCopy -= medianFilter(inImageArray, inSize, inBoundaryType, inConstantValue);

            return std::move(inImageArrayCopy);
        }

        //============================================================================
        // Method Description: 
        ///						Calculate a one-dimensional complemenatry median filter.
        ///		
        /// @param				inImageArray
        /// @param				inSize: square size of the kernel to apply
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> complementaryMedianFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> inImageArrayCopy(inImageArray);
            inImageArrayCopy -= medianFilter1d(inImageArray, inSize, inBoundaryType, inConstantValue);

            return std::move(inImageArrayCopy);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a multidimensional kernel convolution.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html#scipy.ndimage.convolve
        ///		
        /// @param				inImageArray
        /// @param				inSize: square size of the kernel to apply
        /// @param              inWeights
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> convolve(const NdArray<dtype>& inImageArray, uint32 inSize,
            const NdArray<dtype>& inWeights, Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            if (inWeights.size() != Utils::sqr(inSize))
            {
                std::string errStr = "ERROR: NC::Filters::convolve: input weights do no match input kernal size.";
                std::cout << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            NdArray<dtype> arrayWithBoundary = addBoundary(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(inImageArray.shape());

            NdArray<dtype> weightsFlat = rot90(inWeights, 2).flatten();
            Shape inShape = inImageArray.shape();
            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPointRow = boundarySize + inShape.rows;
            uint32 endPointCol = boundarySize + inShape.cols;

            for (uint32 row = boundarySize; row < endPointRow; ++row)
            {
                for (uint32 col = boundarySize; col < endPointCol; ++col)
                {
                    NdArray<dtype> window = arrayWithBoundary(Slice(row - boundarySize, row + boundarySize + 1),
                        Slice(col - boundarySize, col + boundarySize + 1)).flatten();

                    output(row - boundarySize, col - boundarySize) = dot<dtype, dtype>(window, weightsFlat).item();
                }
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a one-dimensional kernel convolution.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve1d.html#scipy.ndimage.convolve1d
        ///		
        /// @param				inImageArray
        /// @param              inWeights
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> convolve1d(const NdArray<dtype>& inImageArray, const NdArray<dtype>& inWeights,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            uint32 boundarySize = inWeights.size() / 2; // integer division
            NdArray<dtype> arrayWithBoundary = addBoundary1d(inImageArray, inBoundaryType, inWeights.size(), inConstantValue);
            NdArray<dtype> output(1, inImageArray.size());

            NdArray<dtype> weightsFlat = fliplr(inWeights.flatten());

            uint32 endPointRow = boundarySize + inImageArray.size();

            for (uint32 i = boundarySize; i < endPointRow; ++i)
            {
                NdArray<dtype> window = arrayWithBoundary[Slice(i - boundarySize, i + boundarySize + 1)].flatten();

                output[i - boundarySize] = dot<dtype, dtype>(window, weightsFlat).item();
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a multidimensional gaussian filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter
        ///		
        /// @param				inImageArray
        /// @param				inSigma: Standard deviation for Gaussian kernel
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> gaussianFilter(const NdArray<dtype>& inImageArray, double inSigma,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            if (inSigma <= 0)
            {
                std::string errStr = "ERROR: NC::Filters::gaussianFilter: input sigma value must be greater than zero.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            // calculate the kernel size based off of the input sigma value
            const uint32 MIN_KERNEL_SIZE = 5;
            uint32 kernelSize = std::max(static_cast<uint32>(std::ceil(inSigma * 2.0 * 4.0)), MIN_KERNEL_SIZE); // 4 standard deviations
            if (kernelSize % 2 == 0)
            {
                ++kernelSize; // make sure the kernel is an odd size
            }

            double kernalHalfSize = static_cast<double>(kernelSize / 2); // integer division

            // calculate the gaussian kernel
            NdArray<double> kernel(kernelSize);
            for (double row = 0; row < kernelSize; ++row)
            {
                for (double col = 0; col < kernelSize; ++col)
                {
                    kernel(static_cast<uint32>(row), static_cast<uint32>(col)) = gaussian(row - kernalHalfSize, col - kernalHalfSize, inSigma);
                }
            }

            // normalize the kernel
            kernel /= kernel.sum<double>().item();

            // perform the convolution
            NdArray<dtype> output = convolve(inImageArray.astype<double>(),
                kernelSize,
                kernel,
                inBoundaryType,
                inConstantValue).astype<dtype>();

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculate a one-dimensional gaussian filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter1d.html#scipy.ndimage.generic_filter1d
        ///		
        /// @param				inImageArray
        /// @param				inSigma: Standard deviation for Gaussian kernel
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> gaussianFilter1d(const NdArray<dtype>& inImageArray, double inSigma,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            if (inSigma <= 0)
            {
                std::string errStr = "ERROR: NC::Filters::gaussianFilter: input sigma value must be greater than zero.";
                std::cerr << errStr << std::endl;
                throw std::invalid_argument(errStr);
            }

            // calculate the kernel size based off of the input sigma value
            const uint32 MIN_KERNEL_SIZE = 5;
            uint32 kernelSize = std::max(static_cast<uint32>(std::ceil(inSigma * 2.0 * 4.0)), MIN_KERNEL_SIZE); // 4 standard deviations
            if (kernelSize % 2 == 0)
            {
                ++kernelSize; // make sure the kernel is an odd size
            }

            double kernalHalfSize = static_cast<double>(kernelSize / 2); // integer division

            // calculate the gaussian kernel
            NdArray<double> kernel(1, kernelSize);
            for (double i = 0; i < kernelSize; ++i)
            {
                kernel[static_cast<int32>(i)] = gaussian(i - kernalHalfSize, 0.0, inSigma);
            }

            // normalize the kernel
            kernel /= kernel.sum<double>().item();

            // perform the convolution
            NdArray<dtype> output = convolve1d(inImageArray.astype<double>(),
                kernel,
                inBoundaryType,
                inConstantValue).astype<dtype>();

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a multidimensional maximum filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter.html#scipy.ndimage.maximum_filter
        ///		
        /// @param				inImageArray
        /// @param				inSize: square size of the kernel to apply
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> maximumFilter(const NdArray<dtype>& inImageArray, uint32 inSize,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> arrayWithBoundary = addBoundary(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(inImageArray.shape());

            Shape inShape = inImageArray.shape();
            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPointRow = boundarySize + inShape.rows;
            uint32 endPointCol = boundarySize + inShape.cols;

            for (uint32 row = boundarySize; row < endPointRow; ++row)
            {
                for (uint32 col = boundarySize; col < endPointCol; ++col)
                {
                    NdArray<dtype> window = arrayWithBoundary(Slice(row - boundarySize, row + boundarySize + 1),
                        Slice(col - boundarySize, col + boundarySize + 1));

                    output(row - boundarySize, col - boundarySize) = window.max().item();
                }
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a one-dimensional maximum filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter1d.html#scipy.ndimage.maximum_filter1d
        ///		
        /// @param				inImageArray
        /// @param				inSize: linear size of the kernel to apply
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> maximumFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> arrayWithBoundary = addBoundary1d(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(1, inImageArray.size());

            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPoint = boundarySize + inImageArray.size();

            for (uint32 i = boundarySize; i < endPoint; ++i)
            {
                NdArray<dtype> window = arrayWithBoundary[Slice(i - boundarySize, i + boundarySize + 1)];

                output[i - boundarySize] = window.max().item();
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a multidimensional median filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html#scipy.ndimage.median_filter
        ///		
        /// @param				inImageArray
        /// @param				inSize: square size of the kernel to apply
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> medianFilter(const NdArray<dtype>& inImageArray, uint32 inSize,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> arrayWithBoundary = addBoundary(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(inImageArray.shape());

            Shape inShape = inImageArray.shape();
            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPointRow = boundarySize + inShape.rows;
            uint32 endPointCol = boundarySize + inShape.cols;

            for (uint32 row = boundarySize; row < endPointRow; ++row)
            {
                for (uint32 col = boundarySize; col < endPointCol; ++col)
                {
                    NdArray<dtype> window = arrayWithBoundary(Slice(row - boundarySize, row + boundarySize + 1),
                        Slice(col - boundarySize, col + boundarySize + 1));

                    output(row - boundarySize, col - boundarySize) = window.median().item();
                }
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a one-dimensional median filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html#scipy.ndimage.median_filter
        ///		
        /// @param				inImageArray
        /// @param				inSize: linear size of the kernel to apply
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> medianFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> arrayWithBoundary = addBoundary1d(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(1, inImageArray.size());

            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPoint = boundarySize + inImageArray.size();

            for (uint32 i = boundarySize; i < endPoint; ++i)
            {
                NdArray<dtype> window = arrayWithBoundary[Slice(i - boundarySize, i + boundarySize + 1)];

                output[i - boundarySize] = window.median().item();
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a multidimensional minimum filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_filter.html#scipy.ndimage.minimum_filter
        ///		
        /// @param				inImageArray
        /// @param				inSize: square size of the kernel to apply
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> minimumFilter(const NdArray<dtype>& inImageArray, uint32 inSize,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> arrayWithBoundary = addBoundary(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(inImageArray.shape());

            Shape inShape = inImageArray.shape();
            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPointRow = boundarySize + inShape.rows;
            uint32 endPointCol = boundarySize + inShape.cols;

            for (uint32 row = boundarySize; row < endPointRow; ++row)
            {
                for (uint32 col = boundarySize; col < endPointCol; ++col)
                {
                    NdArray<dtype> window = arrayWithBoundary(Slice(row - boundarySize, row + boundarySize + 1),
                        Slice(col - boundarySize, col + boundarySize + 1));

                    output(row - boundarySize, col - boundarySize) = window.min().item();
                }
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a one-dimensional minumum filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_filter1d.html#scipy.ndimage.minimum_filter1d
        ///		
        /// @param				inImageArray
        /// @param				inSize: linear size of the kernel to apply
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> minumumFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> arrayWithBoundary = addBoundary1d(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(1, inImageArray.size());

            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPoint = boundarySize + inImageArray.size();

            for (uint32 i = boundarySize; i < endPoint; ++i)
            {
                NdArray<dtype> window = arrayWithBoundary[Slice(i - boundarySize, i + boundarySize + 1)];

                output[i - boundarySize] = window.min().item();
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a multidimensional percentile filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.percentile_filter.html#scipy.ndimage.percentile_filter
        ///
        /// @param				inImageArray
        /// @param				inSize: square size of the kernel to apply
        /// @param              inPercentile: percentile [0, 100]
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> percentileFilter(const NdArray<dtype>& inImageArray, uint32 inSize, uint8 inPercentile,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> arrayWithBoundary = addBoundary(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(inImageArray.shape());

            Shape inShape = inImageArray.shape();
            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPointRow = boundarySize + inShape.rows;
            uint32 endPointCol = boundarySize + inShape.cols;

            for (uint32 row = boundarySize; row < endPointRow; ++row)
            {
                for (uint32 col = boundarySize; col < endPointCol; ++col)
                {
                    NdArray<dtype> window = arrayWithBoundary(Slice(row - boundarySize, row + boundarySize + 1),
                        Slice(col - boundarySize, col + boundarySize + 1));

                    output(row - boundarySize, col - boundarySize) = percentile<dtype, dtype>(window, inPercentile, Axis::NONE, "nearest").item();
                }
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a one-dimensional percentile filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.percentile_filter.html#scipy.ndimage.percentile_filter
        ///		
        /// @param				inImageArray
        /// @param				inSize: linear size of the kernel to apply
        /// @param              inPercentile: percentile [0, 100]
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> percentileFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, uint8 inPercentile,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> arrayWithBoundary = addBoundary1d(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(1, inImageArray.size());

            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPoint = boundarySize + inImageArray.size();

            for (uint32 i = boundarySize; i < endPoint; ++i)
            {
                NdArray<dtype> window = arrayWithBoundary[Slice(i - boundarySize, i + boundarySize + 1)];

                output[i - boundarySize] = percentile<dtype, dtype>(window, inPercentile).item();
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a multidimensional rank filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rank_filter.html#scipy.ndimage.rank_filter
        ///		
        /// @param				inImageArray
        /// @param				inSize: square size of the kernel to apply
        /// @param              inRank: ([0, inSize^2 - 1])
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> rankFilter(const NdArray<dtype>& inImageArray, uint32 inSize, uint32 inRank,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            if (inRank < 0 || inRank >= Utils::sqr(inSize))
            {
                std::invalid_argument("ERROR: NC::Filters::rankFilter: rank not within filter footprint size.");
            }

            NdArray<dtype> arrayWithBoundary = addBoundary(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(inImageArray.shape());

            Shape inShape = inImageArray.shape();
            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPointRow = boundarySize + inShape.rows;
            uint32 endPointCol = boundarySize + inShape.cols;

            for (uint32 row = boundarySize; row < endPointRow; ++row)
            {
                for (uint32 col = boundarySize; col < endPointCol; ++col)
                {
                    NdArray<dtype> window = arrayWithBoundary(Slice(row - boundarySize, row + boundarySize + 1),
                        Slice(col - boundarySize, col + boundarySize + 1));

                    output(row - boundarySize, col - boundarySize) = sort(window)[inRank];
                }
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a one-dimensional rank filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rank_filter.html#scipy.ndimage.rank_filter
        ///		
        /// @param				inImageArray
        /// @param				inSize: linear size of the kernel to apply
        /// @param              inRank: ([0, inSize^2 - 1])
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> rankFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize, uint8 inRank,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> arrayWithBoundary = addBoundary1d(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(1, inImageArray.size());

            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPoint = boundarySize + inImageArray.size();

            for (uint32 i = boundarySize; i < endPoint; ++i)
            {
                NdArray<dtype> window = arrayWithBoundary[Slice(i - boundarySize, i + boundarySize + 1)];

                output[i - boundarySize] = sort(window)[inRank];
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a multidimensional uniform filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html#scipy.ndimage.uniform_filter
        ///		
        /// @param				inImageArray
        /// @param				inSize: square size of the kernel to apply
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> uniformFilter(const NdArray<dtype>& inImageArray, uint32 inSize,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> arrayWithBoundary = addBoundary(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(inImageArray.shape());

            Shape inShape = inImageArray.shape();
            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPointRow = boundarySize + inShape.rows;
            uint32 endPointCol = boundarySize + inShape.cols;

            for (uint32 row = boundarySize; row < endPointRow; ++row)
            {
                for (uint32 col = boundarySize; col < endPointCol; ++col)
                {
                    NdArray<dtype> window = arrayWithBoundary(Slice(row - boundarySize, row + boundarySize + 1),
                        Slice(col - boundarySize, col + boundarySize + 1));

                    output(row - boundarySize, col - boundarySize) = window.mean().item();
                }
            }

            return std::move(output);
        }

        //============================================================================
        // Method Description: 
        ///						Calculates a one-dimensional uniform filter.
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html#scipy.ndimage.uniform_filter1d
        ///		
        /// @param				inImageArray
        /// @param				inSize: linear size of the kernel to apply
        /// @param              inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
        /// @param				inConstantValue: contant value if boundary = 'constant' (default 0)
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<dtype> uniformFilter1d(const NdArray<dtype>& inImageArray, uint32 inSize,
            Boundary inBoundaryType = Boundary::REFLECT, dtype inConstantValue = 0)
        {
            NdArray<dtype> arrayWithBoundary = addBoundary1d(inImageArray, inBoundaryType, inSize, inConstantValue);
            NdArray<dtype> output(1, inImageArray.size());

            uint32 boundarySize = inSize / 2; // integer division
            uint32 endPoint = boundarySize + inImageArray.size();

            for (uint32 i = boundarySize; i < endPoint; ++i)
            {
                NdArray<dtype> window = arrayWithBoundary[Slice(i - boundarySize, i + boundarySize + 1)];

                output[i - boundarySize] = window.mean().item();
            }

            return std::move(output);
        }
    }
}
