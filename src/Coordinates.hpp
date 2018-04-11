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

#include"DtypeInfo.hpp"
#include"NdArray.hpp"
#include"Methods.hpp"
#include"Types.hpp"
#include"Utils.hpp"

#include<iostream>
#include<stdexcept>
#include<string>
#include<utility>

namespace NumC
{
    //================================Coordinates Namespace=============================
    namespace Coordinates
    {
        //================================================================================
        // Class Description:
        //						holds a right ascension object
        //
        template<typename dtype>
        class RA
        {
        private:
            //====================================Attributes==============================
            uint8      hours_;
            uint8      minutes_;
            dtype      seconds_;
            dtype      degrees_;

        public:
            //============================================================================
            // Method Description: 
            //						Default Constructor, not super usefull on its own
            //		
            // Inputs:
            //				None
            // Outputs:
            //				None
            //
            RA() :
                hours_(0),
                minutes_(0),
                seconds_(0.0),
                degrees_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::RA: constructor can only be called with floating point types.");
            }

            //============================================================================
            // Method Description: 
            //						Constructor
            //		
            // Inputs:
            //				fractional degrees
            // Outputs:
            //				None
            //
            RA(dtype inDegrees) :
                hours_(0),
                minutes_(0),
                seconds_(0.0),
                degrees_(inDegrees)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::RA: constructor can only be called with floating point types.");


            }

            //============================================================================
            // Method Description: 
            //						Constructor
            //		
            // Inputs:
            //				hours,
            //              minutes,
            //              seconds
            // Outputs:
            //				None
            //
            RA(uint8 inHours, uint8 inMinutes, dtype inSeconds) :
                hours_(inHours),
                minutes_(inMinutes),
                seconds_(inSeconds),
                degrees_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::RA: constructor can only be called with floating point types.");


            }

            //============================================================================
            // Method Description: 
            //						get the degrees value
            //		
            // Inputs:
            //				None
            // Outputs:
            //				uint8 minutes
            //
            dtype degrees() const
            {
                return degrees_;
            }

            //============================================================================
            // Method Description: 
            //						get the hour value
            //		
            // Inputs:
            //				None
            // Outputs:
            //				uint8 hours
            //
            uint8 hours() const
            {
                return hours_;
            }

            //============================================================================
            // Method Description: 
            //						get the minute value
            //		
            // Inputs:
            //				None
            // Outputs:
            //				uint8 minutes
            //
            uint8 minutes() const
            {
                return minutes_;
            }

            //============================================================================
            // Method Description: 
            //						get the seconds value
            //		
            // Inputs:
            //				None
            // Outputs:
            //				fractional seconds
            //
            dtype seconds() const
            {
                return seconds_;
            }

            //============================================================================
            // Method Description: 
            //						equality operator
            //		
            // Inputs:
            //				None
            // Outputs:
            //				bool
            //
            bool operator==(const RA<dtype>& inRhs) const
            {
                return degrees_ == inRhs.degrees_;
            }

            //============================================================================
            // Method Description: 
            //						not equality operator
            //		
            // Inputs:
            //				None
            // Outputs:
            //				bool
            //
            bool operator!=(const RA<dtype>& inRhs) const
            {
                return !(*this == inRhs);
            }

            //============================================================================
            // Method Description: 
            //						prints the ra
            //		
            // Inputs:
            //				None
            // Outputs:
            //				None
            //
            friend std::ostream& operator<<(std::ostream& inStream, const RA<dtype>& inRa)
            {
                std::string out = "RA = " + Utils::num2str(inRa.hours_) + " hours, " + Utils::num2str(inRa.minutes_) + " minutes, ";
                out += Utils::num2str(inRa.seconds_) + " seconds\n\tdegrees = " + Utils::num2str(inRa.degrees_) + "\n";
                inStream << out;
                return out;
            }
        };

        //================================================================================
        // Class Description:
        //						holds a declination object
        //
        template<typename dtype>
        class Dec
        {
        private:
            //====================================Attributes==============================
            uint16      degreesWhole_;
            uint8       minutes_;
            dtype       seconds_;
            dtype       degrees_;

        public:
            //============================================================================
            // Method Description: 
            //						Default Constructor, not super usefull on its own
            //		
            // Inputs:
            //				None
            // Outputs:
            //				None
            //
            Dec() :
                degreesWhole_(0),
                minutes_(0),
                seconds_(0.0),
                degrees_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
            }

            //============================================================================
            // Method Description: 
            //						Constructor
            //		
            // Inputs:
            //				fractional degrees
            // Outputs:
            //				None
            //
            Dec(dtype inDegrees) :
                degreesWhole_(0),
                minutes_(0),
                seconds_(0.0),
                degrees_(inDegrees)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");


            }

            //============================================================================
            // Method Description: 
            //						Constructor
            //		
            // Inputs:
            //				hours,
            //              minutes,
            //              seconds
            // Outputs:
            //				None
            //
            Dec(uint8 inHours, uint8 inMinutes, dtype inSeconds) :
                degreesWhole_(inHours),
                minutes_(inMinutes),
                seconds_(inSeconds),
                degrees_(0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");


            }

            //============================================================================
            // Method Description: 
            //						get the degrees value
            //		
            // Inputs:
            //				None
            // Outputs:
            //				uint8 minutes
            //
            dtype degrees() const
            {
                return degrees_;
            }

            //============================================================================
            // Method Description: 
            //						get the whole degrees value
            //		
            // Inputs:
            //				None
            // Outputs:
            //				uint8 minutes
            //
            uint16 degreesWhole() const
            {
                return degreesWhole_;
            }

            //============================================================================
            // Method Description: 
            //						get the minute value
            //		
            // Inputs:
            //				None
            // Outputs:
            //				uint8 minutes
            //
            uint8 minutes() const
            {
                return minutes_;
            }

            //============================================================================
            // Method Description: 
            //						get the seconds value
            //		
            // Inputs:
            //				None
            // Outputs:
            //				fractional seconds
            //
            dtype seconds() const
            {
                return seconds_;
            }

            //============================================================================
            // Method Description: 
            //						equality operator
            //		
            // Inputs:
            //				None
            // Outputs:
            //				bool
            //
            bool operator==(const Dec<dtype>& inRhs) const
            {
                return degrees_ == inRhs.degrees_;
            }

            //============================================================================
            // Method Description: 
            //						not equality operator
            //		
            // Inputs:
            //				None
            // Outputs:
            //				bool
            //
            bool operator!=(const Dec<dtype>& inRhs) const
            {
                return !(*this == inRhs);
            }

            //============================================================================
            // Method Description: 
            //						prints the dec
            //		
            // Inputs:
            //				None
            // Outputs:
            //				None
            //
            friend std::ostream& operator<<(std::ostream& inStream, const Dec<dtype>& inDec)
            {
                std::string out = "Dec = " + Utils::num2str(inDec.degreesWhole_) + " degrees, " + Utils::num2str(inDec.minutes_) + " minutes, ";
                out += Utils::num2str(inDec.seconds_) + " seconds\n\tdegrees = " + Utils::num2str(inDec.degrees_) + "\n";
                inStream << out;
                return out;
            }
        };

        //================================================================================
        // Class Description:
        //						holds a full coordinate object
        //
        template<typename dtype>
        class Coordinate
        {
        private:
            //====================================Attributes==============================
            RA<dtype>       ra_;
            Dec<dtype>      dec_;
            dtype           x_;
            dtype           y_;
            dtype           z_;

            //============================================================================
            // Method Description: 
            //						converts polar coordinates to cartesian coordinates
            //		
            // Inputs:
            //				None
            // Outputs:
            //				None
            //
            void cartesianToPolar()
            {

            }

            //============================================================================
            // Method Description: 
            //						converts polar coordinates to cartesian coordinates
            //		
            // Inputs:
            //				None
            // Outputs:
            //				None
            //
            void polarToCartesian()
            {

            }

        public:
            //============================================================================
            // Method Description: 
            //						Default Constructor, not super usefull on its own
            //		
            // Inputs:
            //				None
            // Outputs:
            //				None
            //
            Coordinate() :
                ra_(),
                dec_(),
                x_(1.0),
                y_(0.0),
                z_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
            }

            //============================================================================
            // Method Description: 
            //						Constructor
            //		
            // Inputs:
            //				RA fractional degrees
            //              Declination fractional degrees
            // Outputs:
            //				None
            //
            Coordinate(dtype inRaDegrees, dtype inDecDegrees) :
                ra_(inRaDegrees),
                dec_(inDecDegrees),
                x_(0.0),
                y_(0.0),
                z_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
                polarToCartesian();
            }

            //============================================================================
            // Method Description: 
            //						Constructor
            //		
            // Inputs:
            //				RA hours
            //              RA minutes
            //              RA seconds
            //              Declination degrees whole
            //              Declination minutes
            //              Declination seconds
            // Outputs:
            //				None
            //
            Coordinate(uint8 inRaHours, uint8 inRaMinutes, dtype inRaSeconds, uint16 inDecDegreesWhole, uint8 inDecMinutes, dtype inDecSeconds) :
                ra_(inRaHours, inRaMinutes, inRaSeconds),
                dec_(inDecDegreesWhole, inDecMinutes, inDecSeconds),
                x_(0.0),
                y_(0.0),
                z_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
                polarToCartesian();
            }

            //============================================================================
            // Method Description: 
            //						Constructor
            //		
            // Inputs:
            //				RA 
            //              Dec
            // Outputs:
            //				None
            //
            template<typename dtypeIn>
            Coordinate(const RA<dtypeIn>& inRA, const Dec<dtypeIn>& inDec) :
                ra_(inRA),
                dec_(inDec),
                x_(0.0),
                y_(0.0),
                z_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
                polarToCartesian();
            }

            //============================================================================
            // Method Description: 
            //						Constructor
            //		
            // Inputs:
            //				x 
            //              y
            //              z
            // Outputs:
            //				None
            //
            Coordinate(dtype inX, dtype inY, dtype inZ) :
                ra_(),
                dec_(),
                x_(inX),
                y_(inY),
                z_(inZ)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");
                cartesianToPolar();
            }

            //============================================================================
            // Method Description: 
            //						Constructor
            //		
            // Inputs:
            //				x 
            //              y
            //              z
            // Outputs:
            //				None
            //
            Coordinate(const NdArray<dtype> inCartesianVector) :
                ra_(),
                dec_(),
                x_(1.0),
                y_(0.0),
                z_(0.0)
            {
                static_assert(!DtypeInfo<dtype>::isInteger(), "ERROR: NumC::Coordinates::Dec: constructor can only be called with floating point types.");

                if (inCartesianVector.size() != 3)
                {
                    throw std::invalid_argument("ERROR: NumC::Coordinates::Dec: constructor NdArray input must be of length 3.");
                }

                x_ = inCartesianVector[0];
                y_ = inCartesianVector[1];
                z_ = inCartesianVector[2];

                cartesianToPolar();
            }

            //============================================================================
            // Method Description: 
            //						returns the Dec object
            //		
            // Inputs:
            //				None
            // Outputs:
            //				Dec
            //
            const Dec<dtype>& dec() const
            {
                return dec_;
            }

            //============================================================================
            // Method Description: 
            //						returns the RA object
            //		
            // Inputs:
            //				None
            // Outputs:
            //				RA
            //
            const RA<dtype>& ra() const
            {
                return ra_;
            }

            //============================================================================
            // Method Description: 
            //						returns the cartesian x value
            //		
            // Inputs:
            //				None
            // Outputs:
            //				x
            //
            dtype x() const
            {
                return x_;
            }

            //============================================================================
            // Method Description: 
            //						returns the cartesian y value
            //		
            // Inputs:
            //				None
            // Outputs:
            //				y
            //
            dtype y() const
            {
                return y_;
            }

            //============================================================================
            // Method Description: 
            //						returns the cartesian z value
            //		
            // Inputs:
            //				None
            // Outputs:
            //				z
            //
            dtype z() const
            {
                return z_;
            }

            //============================================================================
            // Method Description: 
            //						returns the cartesian xyz triplet as an NdArray
            //		
            // Inputs:
            //				None
            // Outputs:
            //				NdArray
            //
            NdArray xyz() const
            {
                NdArray<dtype> out = {x_, y_, z_};
                return std::move(out);
            }

            //============================================================================
            // Method Description: 
            //						returns the degree seperation between the two Coordinates
            //		
            // Inputs:
            //				Coordinate
            // Outputs:
            //				degrees
            //
            template<typename dtypeIn>
            dtype degreeSeperation(const Coordinate<dtypeIn>& inOtherCoordinate) const
            {
                return rad2deg(radianSeperation(inOtherCoordinate));
            }

            //============================================================================
            // Method Description: 
            //						returns the degree seperation between the Coordinate
            //                      and the input vector
            //		
            // Inputs:
            //				NdArray
            // Outputs:
            //				degrees
            //
            template<typename dtypeIn>
            dtype degreeSeperation(const NdArray<dtypeIn>& inVector) const
            {
                return rad2deg(radianSeperation(inVector));
            }

            //============================================================================
            // Method Description: 
            //						returns the radian seperation between the two Coordinates
            //		
            // Inputs:
            //				Coordinate
            // Outputs:
            //				radian
            //
            template<typename dtypeIn>
            dtype radianSeperation(const Coordinate<dtypeIn>& inOtherCoordinate) const
            {

            }

            //============================================================================
            // Method Description: 
            //						returns the radian seperation between the Coordinate
            //                      and the input vector
            //		
            // Inputs:
            //				NdArray
            // Outputs:
            //				radian
            //
            template<typename dtypeIn>
            dtype radianSeperation(const NdArray<dtypeIn>& inVector) const
            {

            }

            //============================================================================
            // Method Description: 
            //						equality operator
            //		
            // Inputs:
            //				None
            // Outputs:
            //				bool
            //
            bool operator==(const Coordinate<dtype>& inRhs) const
            {
                return ra_ == inRhs.ra_ && dec_ == inRhs.dec_;
            }

            //============================================================================
            // Method Description: 
            //						not equality operator
            //		
            // Inputs:
            //				None
            // Outputs:
            //				bool
            //
            bool operator!=(const Coordinate<dtype>& inRhs) const
            {
                return !(*this == inRhs);
            }

            //============================================================================
            // Method Description: 
            //						prints the dec
            //		
            // Inputs:
            //				None
            // Outputs:
            //				None
            //
            friend std::ostream& operator<<(std::ostream& inStream, const Coordinate<dtype>& inCoord)
            {
                inStream << ra_;
                inStream << dec_;
                inStream << "Cartesian = " << xyz();
                return out;
            }
        };

        //============================================================================
        // Method Description: 
        //						returns the degree seperation between the two Coordinates
        //		
        // Inputs:
        //				Coordinate1
        //              Coordinate2
        // Outputs:
        //				degrees
        //
        template<typename dtype>
        dtype degreeSeperation(const Coordinate<dtype>& inCoordinate1, const Coordinate<dtype>& inCoordinate2) const
        {
            return inCoordinate1.degreeSeperation(inCoordinate2);
        }

        //============================================================================
        // Method Description: 
        //						returns the degree seperation between the Coordinate
        //                      and the input vector
        //		
        // Inputs:
        //				NdArray1
        //              NdArray2
        // Outputs:
        //				degrees
        //
        template<typename dtype>
        dtype degreeSeperation(const NdArray<dtype>& inVector1, const NdArray<dtype>& inVector2) const
        {
            Coordinate<dtype> inCoord1(inVector1);
            return inCoord1.degreeSeperation(inVector1);
        }

        //============================================================================
        // Method Description: 
        //						returns the radian seperation between the two Coordinates
        //		
        // Inputs:
        //				Coordinate1
        //              Coordinate2
        // Outputs:
        //				radian
        //
        template<typename dtype>
        dtype radianSeperation(const Coordinate<dtype>& inCoordinate1, const Coordinate<dtype>& inCoordinate2) const
        {
            return inCoordinate1.radianSeperation(inCoordinate2);
        }

        //============================================================================
        // Method Description: 
        //						returns the radian seperation between the Coordinate
        //                      and the input vector
        //		
        // Inputs:
        //				NdArray1
        //              NdArray2
        // Outputs:
        //				radian
        //
        template<typename dtype>
        dtype radianSeperation(const NdArray<dtype>& inVector1, const NdArray<dtype>& inVector2) const
        {
            Coordinate<dtype> inCoord1(inVector1);
            return inCoord1.radianSeperation(inVector1);
        }
    }
}
