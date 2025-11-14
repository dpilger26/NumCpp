#include "NumCpp/Core.hpp"

#include "BindingsIncludes.hpp"

//================================================================================

namespace ShapeInterface
{
    bool testListContructor()
    {
        const Shape test = { 357, 666 };
        return test.rows == 357 && test.cols == 666;
    }
} // namespace ShapeInterface

//================================================================================

void initCore(python_interface::module &m)
{ // DtypeInfo.hpp
    using DtypeInfoUint32 = DtypeInfo<uint32>;
    python_interface::class_<DtypeInfoUint32>(m, "DtypeIntoUint32")
        .def(python_interface::init<>())
        .def_static("bits", &DtypeInfoUint32::bits)
        .def_static("epsilon", &DtypeInfoUint32::epsilon)
        .def_static("isInteger", &DtypeInfoUint32::isInteger)
        .def_static("isSigned", &DtypeInfoUint32::isSigned)
        .def_static("min", &DtypeInfoUint32::min)
        .def_static("max", &DtypeInfoUint32::max);

    using DtypeInfoComplexDouble = DtypeInfo<std::complex<double>>;
    python_interface::class_<DtypeInfoComplexDouble>(m, "DtypeInfoComplexDouble")
        .def(python_interface::init<>())
        .def_static("bits", &DtypeInfoComplexDouble::bits)
        .def_static("epsilon", &DtypeInfoComplexDouble::epsilon)
        .def_static("isInteger", &DtypeInfoComplexDouble::isInteger)
        .def_static("isSigned", &DtypeInfoComplexDouble::isSigned)
        .def_static("min", &DtypeInfoComplexDouble::min)
        .def_static("max", &DtypeInfoComplexDouble::max);

    // Shape.hpp
    python_interface::class_<Shape>(m, "Shape")
        .def(python_interface::init<>())
        .def(python_interface::init<uint32>())
        .def(python_interface::init<uint32, uint32>())
        .def(python_interface::init<Shape>())
        .def_static("testListContructor", &ShapeInterface::testListContructor)
        .def_readwrite("rows", &Shape::rows)
        .def_readwrite("cols", &Shape::cols)
        .def("size", &Shape::size)
        .def("print", &Shape::print)
        .def("__str__", &Shape::str)
        .def("__eq__", &Shape::operator==)
        .def("__neq__", &Shape::operator!=);

    // Slice.hpp
    python_interface::class_<Slice>(m, "Slice")
        .def(python_interface::init<>())
        .def(python_interface::init<int32>())
        .def(python_interface::init<int32, int32>())
        .def(python_interface::init<int32, int32, int32>())
        .def(python_interface::init<Slice>())
        .def_readwrite("start", &Slice::start)
        .def_readwrite("stop", &Slice::stop)
        .def_readwrite("step", &Slice::step)
        .def("numElements", &Slice::numElements)
        .def("toIndices", &Slice::toIndices)
        .def("print", &Slice::print)
        .def("__str__", &Slice::str)
        .def("__eq__", &Slice::operator==)
        .def("__neq__", &Slice::operator!=);

    python_interface::class_<std::chrono::microseconds>(m, "ChronoMicroseconds")
        .def("count", &std::chrono::microseconds::count);

    // Timer.hpp
    using MicroTimer = Timer<std::chrono::microseconds>;
    python_interface::class_<MicroTimer>(m, "Timer")
        .def(python_interface::init<>())
        .def(python_interface::init<std::string>())
        .def("sleep", &MicroTimer::sleep)
        .def("tic", &MicroTimer::tic)
        .def("toc", &MicroTimer::toc);

    // Enums.hpp
    python_interface::enum_<Axis>(m, "Axis").value("NONE", Axis::NONE).value("ROW", Axis::ROW).value("COL", Axis::COL);

    python_interface::enum_<Endian>(m, "Endian")
        .value("NATIVE", Endian::NATIVE)
        .value("BIG", Endian::BIG)
        .value("LITTLE", Endian::LITTLE);

    python_interface::enum_<PointerPolicy>(m, "PointerPolicy")
        .value("COPY", PointerPolicy::COPY)
        .value("SHELL", PointerPolicy::SHELL);

    python_interface::enum_<Bias>(m, "Bias").value("YES", Bias::YES).value("NO", Bias::NO);

    python_interface::enum_<EndPoint>(m, "EndPoint").value("YES", EndPoint::YES).value("NO", EndPoint::NO);

    python_interface::enum_<Increasing>(m, "Increasing").value("YES", Increasing::YES).value("NO", Increasing::NO);

    python_interface::enum_<IsRoots>(m, "IsRoots").value("YES", IsRoots::YES).value("NO", IsRoots::NO);

    python_interface::enum_<Replace>(m, "Replace").value("YES", Replace::YES).value("NO", Replace::NO);

    python_interface::enum_<PrintElapsedTime>(m, "PrintElapsedTime")
        .value("YES", PrintElapsedTime::YES)
        .value("NO", PrintElapsedTime::NO);

    python_interface::enum_<PrintResults>(m, "PrintResults")
        .value("YES", PrintResults::YES)
        .value("NO", PrintResults::NO);

    python_interface::enum_<Side>(m, "Side").value("LEFT", Side::LEFT).value("RIGHT", Side::RIGHT);

    python_interface::enum_<InterpolationMethod>(m, "InterpolationMethod")
        .value("LINEAR", InterpolationMethod::LINEAR)
        .value("LOWER", InterpolationMethod::LOWER)
        .value("HIGHER", InterpolationMethod::HIGHER)
        .value("NEAREST", InterpolationMethod::NEAREST)
        .value("MIDPOINT", InterpolationMethod::MIDPOINT);
}