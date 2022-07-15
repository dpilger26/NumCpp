#include "NumCpp/Coordinates.hpp"

#include "BindingsIncludes.hpp"


//================================================================================

namespace RaInterface
{
    void print(const coordinates::RA& inRa)
    {
        std::cout << inRa;
    }
} // namespace RaInterface

namespace DecInterface
{
    void print(const coordinates::Dec& self)
    {
        std::cout << self;
    }
} // namespace DecInterface

namespace CoordinateInterface
{
    void print(const coordinates::Coordinate& self)
    {
        std::cout << self;
    }

    //================================================================================

    double degreeSeperationCoordinate(const coordinates::Coordinate& self,
                                      const coordinates::Coordinate& inOtherCoordinate)
    {
        return self.degreeSeperation(inOtherCoordinate);
    }

    //================================================================================

    double degreeSeperationVector(const coordinates::Coordinate& self, const NdArray<double>& inVec)
    {
        return self.degreeSeperation(inVec);
    }

    //================================================================================

    double radianSeperationCoordinate(const coordinates::Coordinate& self,
                                      const coordinates::Coordinate& inOtherCoordinate)
    {
        return self.radianSeperation(inOtherCoordinate);
    }

    //================================================================================

    double radianSeperationVector(const coordinates::Coordinate& self, const NdArray<double>& inVec)
    {
        return self.radianSeperation(inVec);
    }
} // namespace CoordinateInterface

//================================================================================

void initCoordinates(pb11::module& m)
{
    // Coordinates.hpp
    pb11::class_<coordinates::RA>(m, "Ra")
        .def(pb11::init<>())
        .def(pb11::init<double>())
        .def(pb11::init<uint8, uint8, double>())
        .def(pb11::init<coordinates::RA>())
        .def("degrees", &coordinates::RA::degrees)
        .def("radians", &coordinates::RA::radians)
        .def("hours", &coordinates::RA::hours)
        .def("minutes", &coordinates::RA::minutes)
        .def("seconds", &coordinates::RA::seconds)
        .def("__str__", &coordinates::RA::str)
        .def("print", &coordinates::RA::print)
        .def("__eq__", &coordinates::RA::operator==)
        .def("__ne__", &coordinates::RA::operator!=)
        .def("print", &RaInterface::print);

    pb11::enum_<coordinates::Sign>(m, "Sign")
        .value("POSITIVE", coordinates::Sign::POSITIVE)
        .value("NEGATIVE", coordinates::Sign::NEGATIVE);

    pb11::class_<coordinates::Dec>(m, "Dec")
        .def(pb11::init<>())
        .def(pb11::init<double>())
        .def(pb11::init<coordinates::Sign, uint8, uint8, double>())
        .def(pb11::init<coordinates::Dec>())
        .def("sign", &coordinates::Dec::sign)
        .def("degrees", &coordinates::Dec::degrees)
        .def("radians", &coordinates::Dec::radians)
        .def("degreesWhole", &coordinates::Dec::degreesWhole)
        .def("minutes", &coordinates::Dec::minutes)
        .def("seconds", &coordinates::Dec::seconds)
        .def("__str__", &coordinates::Dec::str)
        .def("print", &coordinates::Dec::print)
        .def("__eq__", &coordinates::Dec::operator==)
        .def("__ne__", &coordinates::Dec::operator!=)
        .def("print", &DecInterface::print);

    pb11::class_<coordinates::Coordinate>(m, "Coordinate")
        .def(pb11::init<>())
        .def(pb11::init<double, double>())
        .def(pb11::init<uint8, uint8, double, coordinates::Sign, uint8, uint8, double>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<coordinates::RA, coordinates::Dec>())
        .def(pb11::init<NdArrayDouble>())
        .def(pb11::init<coordinates::Coordinate>())
        .def("dec", &coordinates::Coordinate::dec, pb11::return_value_policy::reference)
        .def("ra", &coordinates::Coordinate::ra, pb11::return_value_policy::reference)
        .def("x", &coordinates::Coordinate::x)
        .def("y", &coordinates::Coordinate::y)
        .def("z", &coordinates::Coordinate::z)
        .def("xyz", &coordinates::Coordinate::xyz)
        .def("degreeSeperation", &CoordinateInterface::degreeSeperationCoordinate)
        .def("degreeSeperation", &CoordinateInterface::degreeSeperationVector)
        .def("radianSeperation", &CoordinateInterface::radianSeperationCoordinate)
        .def("radianSeperation", &CoordinateInterface::radianSeperationVector)
        .def("__str__", &coordinates::Coordinate::str)
        .def("print", &coordinates::Coordinate::print)
        .def("__eq__", &coordinates::Coordinate::operator==)
        .def("__ne__", &coordinates::Coordinate::operator!=)
        .def("print", &coordinates::Coordinate::print);
}
