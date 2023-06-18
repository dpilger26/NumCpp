#include "NumCpp/Coordinates.hpp"

#include "BindingsIncludes.hpp"

//================================================================================

void initCoordinates(pb11::module& m)
{
    // Coordinates.hpp
    m.attr("EARTH_POLAR_RADIUS")      = coordinates::reference_frames::constants::EARTH_POLAR_RADIUS;
    m.attr("EARTH_EQUATORIAL_RADIUS") = coordinates::reference_frames::constants::EARTH_EQUATORIAL_RADIUS;

    pb11::class_<coordinates::Cartesian>(m, "Cartesian")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<const Vec2&>())
        .def(pb11::init<const Vec3&>())
        .def(pb11::init<const NdArray<double>&>())
        .def_readwrite("x", &coordinates::Cartesian::x)
        .def_readwrite("y", &coordinates::Cartesian::y)
        .def_readwrite("z", &coordinates::Cartesian::z)
        .def_static("xHat", &coordinates::Cartesian::xHat)
        .def_static("yHat", &coordinates::Cartesian::yHat)
        .def_static("zHat", &coordinates::Cartesian::zHat)
        .def("__eq__", &coordinates::Cartesian::operator==)
        .def("__ne__", &coordinates::Cartesian::operator!=)
        .def("__add__",
             [](const coordinates::Cartesian& self, const coordinates::Cartesian& other) { return self + other; })
        .def("__sub__",
             [](const coordinates::Cartesian& self, const coordinates::Cartesian& other) { return self - other; })
        .def("__mul__",
             [](const coordinates::Cartesian& self, const coordinates::Cartesian& other) { return self * other; })
        .def("__mul__", [](const coordinates::Cartesian& self, double scalar) { return self * scalar; })
        .def("__truediv__", [](const coordinates::Cartesian& self, double scalar) { return self / scalar; })
        .def("print", [](const coordinates::Cartesian& self) { std::cout << self; });

    m.def("cross", &coordinates::cross);
    m.def("norm", &coordinates::norm);
    m.def("normalize", &coordinates::normalize);
    m.def("angle", &coordinates::angle);

    pb11::class_<coordinates::Euler>(m, "Euler")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def_readwrite("psi", &coordinates::Euler::psi)
        .def_readwrite("theta", &coordinates::Euler::theta)
        .def_readwrite("phi", &coordinates::Euler::phi)
        .def("__eq__", [](const coordinates::Euler& self, const coordinates::Euler& other) { return self == other; })
        .def("__ne__", [](const coordinates::Euler& self, const coordinates::Euler& other) { return self != other; })
        .def("print", [](const coordinates::Euler& self) { std::cout << self; });

    pb11::class_<coordinates::Orientation>(m, "Orientation")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def_readwrite("roll", &coordinates::Orientation::roll)
        .def_readwrite("pitch", &coordinates::Orientation::pitch)
        .def_readwrite("yaw", &coordinates::Orientation::yaw)
        .def("__eq__",
             [](const coordinates::Orientation& self, const coordinates::Orientation& other) { return self == other; })
        .def("__ne__",
             [](const coordinates::Orientation& self, const coordinates::Orientation& other) { return self != other; })
        .def("print", [](const coordinates::Orientation& self) { std::cout << self; });

    pb11::class_<coordinates::reference_frames::AzEl>(m, "AzEl")
        .def(pb11::init<>())
        .def(pb11::init<double, double>())
        .def_readwrite("az", &coordinates::reference_frames::AzEl::az)
        .def_readwrite("el", &coordinates::reference_frames::AzEl::el)
        .def("__eq__",
             [](const coordinates::reference_frames::AzEl& self, const coordinates::reference_frames::AzEl& other)
             { return self == other; })
        .def("__ne__",
             [](const coordinates::reference_frames::AzEl& self, const coordinates::reference_frames::AzEl& other)
             { return self != other; })
        .def("print", [](const coordinates::reference_frames::AzEl& self) { std::cout << self; });

    pb11::class_<coordinates::reference_frames::ECEF, coordinates::Cartesian>(m, "ECEF")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<const Vec2&>())
        .def(pb11::init<const Vec3&>())
        .def(pb11::init<const NdArray<double>&>())
        .def(pb11::init<const coordinates::Cartesian&>());

    pb11::class_<coordinates::reference_frames::ENU, coordinates::Cartesian>(m, "ENU")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<const Vec2&>())
        .def(pb11::init<const Vec3&>())
        .def(pb11::init<const NdArray<double>&>())
        .def(pb11::init<const coordinates::Cartesian&>())
        .def_property("east", &coordinates::reference_frames::ENU::east, &coordinates::reference_frames::ENU::setEast)
        .def_property("north",
                      &coordinates::reference_frames::ENU::north,
                      &coordinates::reference_frames::ENU::setNorth)
        .def_property("up", &coordinates::reference_frames::ENU::up, &coordinates::reference_frames::ENU::setUp)
        .def("print", [](const coordinates::reference_frames::ENU& self) { std::cout << self; });

    pb11::class_<coordinates::reference_frames::NED, coordinates::Cartesian>(m, "NED")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<const Vec2&>())
        .def(pb11::init<const Vec3&>())
        .def(pb11::init<const NdArray<double>&>())
        .def(pb11::init<const coordinates::Cartesian&>())
        .def_property("north",
                      &coordinates::reference_frames::NED::north,
                      &coordinates::reference_frames::NED::setNorth)
        .def_property("east", &coordinates::reference_frames::NED::east, &coordinates::reference_frames::NED::setEast)
        .def_property("down", &coordinates::reference_frames::NED::down, &coordinates::reference_frames::NED::setDown)
        .def("print", [](const coordinates::reference_frames::NED& self) { std::cout << self; });

    pb11::class_<coordinates::reference_frames::LLA>(m, "LLA")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def_readwrite("latitude", &coordinates::reference_frames::LLA::latitude)
        .def_readwrite("longitude", &coordinates::reference_frames::LLA::longitude)
        .def_readwrite("altitude", &coordinates::reference_frames::LLA::altitude)
        .def("__eq__",
             [](const coordinates::reference_frames::LLA& self, const coordinates::reference_frames::LLA& other)
             { return self == other; })
        .def("__ne__",
             [](const coordinates::reference_frames::LLA& self, const coordinates::reference_frames::LLA& other)
             { return self != other; })
        .def("print", [](const coordinates::reference_frames::LLA& self) { std::cout << self; });

    pb11::class_<coordinates::reference_frames::RA>(m, "Ra")
        .def(pb11::init<>())
        .def(pb11::init<double>())
        .def(pb11::init<uint8, uint8, double>())
        .def(pb11::init<coordinates::reference_frames::RA>())
        .def("degrees", &coordinates::reference_frames::RA::degrees)
        .def("radians", &coordinates::reference_frames::RA::radians)
        .def("hours", &coordinates::reference_frames::RA::hours)
        .def("minutes", &coordinates::reference_frames::RA::minutes)
        .def("seconds", &coordinates::reference_frames::RA::seconds)
        .def("__str__", &coordinates::reference_frames::RA::str)
        .def("print", &coordinates::reference_frames::RA::print)
        .def("__eq__", &coordinates::reference_frames::RA::operator==)
        .def("__ne__", &coordinates::reference_frames::RA::operator!=)
        .def("print", [](const coordinates::reference_frames::RA& self) { std::cout << self; });

    pb11::enum_<coordinates::reference_frames::Dec::Sign>(m, "Sign")
        .value("POSITIVE", coordinates::reference_frames::Dec::Sign::POSITIVE)
        .value("NEGATIVE", coordinates::reference_frames::Dec::Sign::NEGATIVE);

    pb11::class_<coordinates::reference_frames::Dec>(m, "Dec")
        .def(pb11::init<>())
        .def(pb11::init<double>())
        .def(pb11::init<coordinates::reference_frames::Dec::Sign, uint8, uint8, double>())
        .def(pb11::init<coordinates::reference_frames::Dec>())
        .def("sign", &coordinates::reference_frames::Dec::sign)
        .def("degrees", &coordinates::reference_frames::Dec::degrees)
        .def("radians", &coordinates::reference_frames::Dec::radians)
        .def("degreesWhole", &coordinates::reference_frames::Dec::degreesWhole)
        .def("minutes", &coordinates::reference_frames::Dec::minutes)
        .def("seconds", &coordinates::reference_frames::Dec::seconds)
        .def("__str__", &coordinates::reference_frames::Dec::str)
        .def("print", &coordinates::reference_frames::Dec::print)
        .def("__eq__", &coordinates::reference_frames::Dec::operator==)
        .def("__ne__", &coordinates::reference_frames::Dec::operator!=)
        .def("print", [](const coordinates::reference_frames::Dec& self) { std::cout << self; });

    pb11::class_<coordinates::reference_frames::Celestial>(m, "Celestial")
        .def(pb11::init<>())
        .def(pb11::init<double, double>())
        .def(pb11::init<uint8, uint8, double, coordinates::reference_frames::Dec::Sign, uint8, uint8, double>())
        .def(pb11::init<coordinates::reference_frames::RA, coordinates::reference_frames::Dec>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<NdArrayDouble>())
        .def(pb11::init<coordinates::Cartesian>())
        .def(pb11::init<Vec3>())
        .def(pb11::init<coordinates::reference_frames::Celestial>())
        .def("dec", &coordinates::reference_frames::Celestial::dec, pb11::return_value_policy::reference)
        .def("ra", &coordinates::reference_frames::Celestial::ra, pb11::return_value_policy::reference)
        .def("x", &coordinates::reference_frames::Celestial::x)
        .def("y", &coordinates::reference_frames::Celestial::y)
        .def("z", &coordinates::reference_frames::Celestial::z)
        .def("xyz", &coordinates::reference_frames::Celestial::xyz)
        .def("degreeSeperation",
             [](const coordinates::reference_frames::Celestial& self,
                const coordinates::reference_frames::Celestial& inOtherCoordinate)
             { return self.degreeSeperation(inOtherCoordinate); })
        .def("degreeSeperation",
             [](const coordinates::reference_frames::Celestial& self, const NdArray<double>& inVec)
             { return self.degreeSeperation(inVec); })
        .def("radianSeperation",
             [](const coordinates::reference_frames::Celestial& self,
                const coordinates::reference_frames::Celestial& inOtherCoordinate)
             { return self.radianSeperation(inOtherCoordinate); })
        .def("radianSeperation",
             [](const coordinates::reference_frames::Celestial& self, const NdArray<double>& inVec)
             { return self.radianSeperation(inVec); })
        .def("__str__", &coordinates::reference_frames::Celestial::str)
        .def("print", &coordinates::reference_frames::Celestial::print)
        .def("__eq__", &coordinates::reference_frames::Celestial::operator==)
        .def("__ne__", &coordinates::reference_frames::Celestial::operator!=)
        .def("print", &coordinates::reference_frames::Celestial::print);

    m.def("AzElGeocentricToENU", &coordinates::transforms::AzElGeocentricToENU);
    m.def("AzElGeocentricToNED", &coordinates::transforms::AzElGeocentricToNED);
    m.def("ECEFEulerToENURollPitchYaw", &coordinates::transforms::ECEFEulerToENURollPitchYaw);
    m.def("ECEFEulerToNEDRollPitchYaw", &coordinates::transforms::ECEFEulerToNEDRollPitchYaw);
    //     m.def("ECEFtoAzElGeocentric", &coordinates::transforms::ECEFtoAzElGeocentric); //
    //     m.def("ECEFtoAzElGeodetic", &coordinates::transforms::ECEFtoAzElGeodetic); //
    //     m.def("ECEFtoENU", &coordinates::transforms::ECEFtoENU); //
    m.def("ECEFtoLLA", &coordinates::transforms::ECEFtoLLA);
    //     m.def("ECEFtoNED", &coordinates::transforms::ECEFtoNED); //
    m.def("ENURollPitchYawToECEFEuler", &coordinates::transforms::ENURollPitchYawToECEFEuler);
    m.def("ENUtoAzEl", &coordinates::transforms::ENUtoAzEl);
    //     m.def("ENUtoECEF", &coordinates::transforms::ENUtoECEF); //
    //     m.def("ENUtoLLA", &coordinates::transforms::ENUtoLLA); //
    m.def("ENUtoNED", &coordinates::transforms::ENUtoNED);
    //     m.def("ENUUnitVecsInECEF", &coordinates::transforms::ENUUnitVecsInECEF); //
    //     m.def("LLAtoAzElGeocentric", &coordinates::transforms::LLAtoAzElGeocentric); //
    //     m.def("LLAtoAzElGeodetic", &coordinates::transforms::LLAtoAzElGeodetic); //
    m.def("LLAtoECEF", &coordinates::transforms::LLAtoECEF);
    //     m.def("LLAtoENU", &coordinates::transforms::LLAtoENU); //
    //     m.def("LLAtoNED", &coordinates::transforms::LLAtoNED); //
    m.def("NEDRollPitchYawToECEFEuler", &coordinates::transforms::NEDRollPitchYawToECEFEuler);
    m.def("NEDtoAzEl", &coordinates::transforms::NEDtoAzEl);
    //     m.def("NEDtoECEF", &coordinates::transforms::NEDtoECEF); //
    m.def("NEDtoENU", &coordinates::transforms::NEDtoENU);
    //     m.def("NEDtoLLA", &coordinates::transforms::NEDtoLLA); //
    m.def("NEDUnitVecsInECEF", &coordinates::transforms::NEDUnitVecsInECEF);
}
