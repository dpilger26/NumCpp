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

void initCore(pb11::module &m)
{ // DtypeInfo.hpp
    using DtypeInfoUint32 = DtypeInfo<uint32>;
    pb11::class_<DtypeInfoUint32>(m, "DtypeIntoUint32")
        .def(pb11::init<>())
        .def_static("bits", &DtypeInfoUint32::bits)
        .def_static("epsilon", &DtypeInfoUint32::epsilon)
        .def_static("isInteger", &DtypeInfoUint32::isInteger)
        .def_static("isSigned", &DtypeInfoUint32::isSigned)
        .def_static("min", &DtypeInfoUint32::min)
        .def_static("max", &DtypeInfoUint32::max);

    using DtypeInfoComplexDouble = DtypeInfo<std::complex<double>>;
    pb11::class_<DtypeInfoComplexDouble>(m, "DtypeInfoComplexDouble")
        .def(pb11::init<>())
        .def_static("bits", &DtypeInfoComplexDouble::bits)
        .def_static("epsilon", &DtypeInfoComplexDouble::epsilon)
        .def_static("isInteger", &DtypeInfoComplexDouble::isInteger)
        .def_static("isSigned", &DtypeInfoComplexDouble::isSigned)
        .def_static("min", &DtypeInfoComplexDouble::min)
        .def_static("max", &DtypeInfoComplexDouble::max);

    // Shape.hpp
    pb11::class_<Shape>(m, "Shape")
        .def(pb11::init<>())
        .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>())
        .def_static("testListContructor", &ShapeInterface::testListContructor)
        .def_readwrite("rows", &Shape::rows)
        .def_readwrite("cols", &Shape::cols)
        .def("size", &Shape::size)
        .def("print", &Shape::print)
        .def("__str__", &Shape::str)
        .def("__eq__", &Shape::operator==)
        .def("__neq__", &Shape::operator!=);

    // Slice.hpp
    pb11::class_<Slice>(m, "Slice")
        .def(pb11::init<>())
        .def(pb11::init<int32>())
        .def(pb11::init<int32, int32>())
        .def(pb11::init<int32, int32, int32>())
        .def(pb11::init<Slice>())
        .def_readwrite("start", &Slice::start)
        .def_readwrite("stop", &Slice::stop)
        .def_readwrite("step", &Slice::step)
        .def("numElements", &Slice::numElements)
        .def("toIndices", &Slice::toIndices)
        .def("print", &Slice::print)
        .def("__str__", &Slice::str)
        .def("__eq__", &Slice::operator==)
        .def("__neq__", &Slice::operator!=);

    pb11::class_<std::chrono::microseconds>(m, "ChronoMicroseconds").def("count", &std::chrono::microseconds::count);

    // Timer.hpp
    using MicroTimer = Timer<std::chrono::microseconds>;
    pb11::class_<MicroTimer>(m, "Timer")
        .def(pb11::init<>())
        .def(pb11::init<std::string>())
        .def("sleep", &MicroTimer::sleep)
        .def("tic", &MicroTimer::tic)
        .def("toc", &MicroTimer::toc);

    // Enums.hpp
    pb11::enum_<Axis>(m, "Axis").value("NONE", Axis::NONE).value("ROW", Axis::ROW).value("COL", Axis::COL);

    pb11::enum_<Endian>(m, "Endian")
        .value("NATIVE", Endian::NATIVE)
        .value("BIG", Endian::BIG)
        .value("LITTLE", Endian::LITTLE);

    pb11::enum_<PointerPolicy>(m, "PointerPolicy")
        .value("COPY", PointerPolicy::COPY)
        .value("SHELL", PointerPolicy::SHELL);

    pb11::enum_<Bias>(m, "Bias").value("YES", Bias::YES).value("NO", Bias::NO);

    pb11::enum_<EndPoint>(m, "EndPoint").value("YES", EndPoint::YES).value("NO", EndPoint::NO);

    pb11::enum_<Increasing>(m, "Increasing").value("YES", Increasing::YES).value("NO", Increasing::NO);

    pb11::enum_<IsRoots>(m, "IsRoots").value("YES", IsRoots::YES).value("NO", IsRoots::NO);

    pb11::enum_<Replace>(m, "Replace").value("YES", Replace::YES).value("NO", Replace::NO);

    pb11::enum_<PrintElapsedTime>(m, "PrintElapsedTime")
        .value("YES", PrintElapsedTime::YES)
        .value("NO", PrintElapsedTime::NO);

    pb11::enum_<PrintResults>(m, "PrintResults").value("YES", PrintResults::YES).value("NO", PrintResults::NO);

    pb11::enum_<Side>(m, "Side").value("LEFT", Side::LEFT).value("RIGHT", Side::RIGHT);

    pb11::enum_<InterpolationMethod>(m, "InterpolationMethod")
        .value("LINEAR", InterpolationMethod::LINEAR)
        .value("LOWER", InterpolationMethod::LOWER)
        .value("HIGHER", InterpolationMethod::HIGHER)
        .value("NEAREST", InterpolationMethod::NEAREST)
        .value("MIDPOINT", InterpolationMethod::MIDPOINT);
}