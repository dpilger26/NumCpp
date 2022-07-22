#include "NumCpp/Vector.hpp"

#include "BindingsIncludes.hpp"


//================================================================================

namespace Vec2Interface
{
    pbArray<double> toNdArray(const Vec2& self)
    {
        return nc2pybind(self.toNdArray());
    }

    //================================================================================

    Vec2& plusEqualScaler(Vec2& self, double scaler)
    {
        return self += scaler;
    }

    //================================================================================

    Vec2& plusEqualVec2(Vec2& self, const Vec2& rhs)
    {
        return self += rhs;
    }

    //================================================================================

    Vec2& minusEqualScaler(Vec2& self, double scaler)
    {
        return self -= scaler;
    }

    //================================================================================

    Vec2& minusEqualVec2(Vec2& self, const Vec2& rhs)
    {
        return self -= rhs;
    }

    //================================================================================

    Vec2 addVec2(const Vec2& vec1, const Vec2& vec2)
    {
        return vec1 + vec2;
    }

    //================================================================================

    Vec2 addVec2Scaler(const Vec2& vec1, double scaler)
    {
        return vec1 + scaler;
    }

    //================================================================================

    Vec2 addScalerVec2(const Vec2& vec1, double scaler)
    {
        return scaler + vec1;
    }

    //================================================================================

    Vec2 minusVec2(const Vec2& vec1, const Vec2& vec2)
    {
        return vec1 - vec2;
    }

    //================================================================================

    Vec2 minusVec2Scaler(const Vec2& vec1, double scaler)
    {
        return vec1 - scaler;
    }

    //================================================================================

    Vec2 minusScalerVec2(const Vec2& vec1, double scaler)
    {
        return scaler - vec1;
    }

    //================================================================================

    Vec2 multVec2Scaler(const Vec2& vec1, double scaler)
    {
        return vec1 * scaler;
    }

    //================================================================================

    Vec2 multScalerVec2(const Vec2& vec1, double scaler)
    {
        return scaler * vec1;
    }

    //================================================================================

    Vec2 divVec2Scaler(const Vec2& vec1, double scaler)
    {
        return vec1 / scaler;
    }

    //================================================================================

    void print(const Vec2& vec)
    {
        std::cout << vec;
    }
} // namespace Vec2Interface

namespace Vec3Interface
{
    pbArray<double> toNdArray(const Vec3& self)
    {
        return nc2pybind(self.toNdArray());
    }

    //================================================================================

    Vec3& plusEqualScaler(Vec3& self, double scaler)
    {
        return self += scaler;
    }

    //================================================================================

    Vec3& plusEqualVec3(Vec3& self, const Vec3& rhs)
    {
        return self += rhs;
    }

    //================================================================================

    Vec3& minusEqualScaler(Vec3& self, double scaler)
    {
        return self -= scaler;
    }

    //================================================================================

    Vec3& minusEqualVec3(Vec3& self, const Vec3& rhs)
    {
        return self -= rhs;
    }

    //================================================================================

    Vec3 addVec3(const Vec3& vec1, const Vec3& vec2)
    {
        return vec1 + vec2;
    }

    //================================================================================

    Vec3 addVec3Scaler(const Vec3& vec1, double scaler)
    {
        return vec1 + scaler;
    }

    //================================================================================

    Vec3 addScalerVec3(const Vec3& vec1, double scaler)
    {
        return vec1 + scaler;
    }

    //================================================================================

    Vec3 minusVec3(const Vec3& vec1, const Vec3& vec2)
    {
        return vec1 - vec2;
    }

    //================================================================================

    Vec3 minusVec3Scaler(const Vec3& vec1, double scaler)
    {
        return vec1 - scaler;
    }

    //================================================================================

    Vec3 minusScalerVec3(const Vec3& vec1, double scaler)
    {
        return scaler - vec1;
    }

    //================================================================================

    Vec3 multVec3Scaler(const Vec3& vec1, double scaler)
    {
        return vec1 * scaler;
    }

    //================================================================================

    Vec3 multScalerVec3(const Vec3& vec1, double scaler)
    {
        return vec1 * scaler;
    }

    //================================================================================

    Vec3 divVec3Scaler(const Vec3& vec1, double scaler)
    {
        return vec1 / scaler;
    }

    //================================================================================

    void print(const Vec3& vec)
    {
        std::cout << vec;
    }
} // namespace Vec3Interface

//================================================================================

void initVector(pb11::module& m)
{
    // Vec2.hpp
    pb11::class_<Vec2>(m, "Vec2")
        .def(pb11::init<>())
        .def(pb11::init<double, double>())
        .def(pb11::init<NdArray<double>>())
        .def_readwrite("x", &Vec2::x)
        .def_readwrite("y", &Vec2::y)
        .def("angle", &Vec2::angle)
        .def("clampMagnitude", &Vec2::clampMagnitude)
        .def("distance", &Vec2::distance)
        .def("dot", &Vec2::dot)
        .def_static("down", &Vec2::down)
        .def_static("left", &Vec2::left)
        .def("lerp", &Vec2::lerp)
        .def("norm", &Vec2::norm)
        .def("normalize", &Vec2::normalize)
        .def("project", &Vec2::project)
        .def_static("right", &Vec2::right)
        .def("__str__", &Vec2::toString)
        .def("toNdArray", &Vec2Interface::toNdArray)
        .def_static("up", &Vec2::up)
        .def("__eq__", &Vec2::operator==)
        .def("__ne__", &Vec2::operator!=)
        .def("__iadd__", &Vec2Interface::plusEqualScaler, pb11::return_value_policy::reference)
        .def("__iadd__", &Vec2Interface::plusEqualVec2, pb11::return_value_policy::reference)
        .def("__isub__", &Vec2Interface::minusEqualVec2, pb11::return_value_policy::reference)
        .def("__isub__", &Vec2Interface::minusEqualScaler, pb11::return_value_policy::reference)
        .def("__imul__", &Vec2::operator*=, pb11::return_value_policy::reference)
        .def("__itruediv__", &Vec2::operator/=, pb11::return_value_policy::reference);

    m.def("Vec2_addVec2", &Vec2Interface::addVec2);
    m.def("Vec2_addVec2Scaler", &Vec2Interface::addVec2Scaler);
    m.def("Vec2_addScalerVec2", &Vec2Interface::addScalerVec2);
    m.def("Vec2_minusVec2", &Vec2Interface::minusVec2);
    m.def("Vec2_minusVec2Scaler", &Vec2Interface::minusVec2Scaler);
    m.def("Vec2_minusScalerVec2", &Vec2Interface::minusScalerVec2);
    m.def("Vec2_multVec2Scaler", &Vec2Interface::multVec2Scaler);
    m.def("Vec2_multScalerVec2", &Vec2Interface::multScalerVec2);
    m.def("Vec2_divVec2Scaler", &Vec2Interface::divVec2Scaler);
    m.def("Vec2_print", &Vec2Interface::print);

    // Vec3.hpp
    pb11::class_<Vec3>(m, "Vec3")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<NdArray<double>>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("angle", &Vec3::angle)
        .def_static("back", &Vec3::back)
        .def("clampMagnitude", &Vec3::clampMagnitude)
        .def("cross", &Vec3::cross)
        .def("distance", &Vec3::distance)
        .def("dot", &Vec3::dot)
        .def_static("down", &Vec3::down)
        .def_static("forward", &Vec3::forward)
        .def_static("left", &Vec3::left)
        .def("lerp", &Vec3::lerp)
        .def("norm", &Vec3::norm)
        .def("normalize", &Vec3::normalize)
        .def("project", &Vec3::project)
        .def_static("right", &Vec3::right)
        .def("__str__", &Vec3::toString)
        .def("toNdArray", &Vec3Interface::toNdArray)
        .def_static("up", &Vec3::up)
        .def("__eq__", &Vec3::operator==)
        .def("__ne__", &Vec3::operator!=)
        .def("__iadd__", &Vec3Interface::plusEqualScaler, pb11::return_value_policy::reference)
        .def("__iadd__", &Vec3Interface::plusEqualVec3, pb11::return_value_policy::reference)
        .def("__isub__", &Vec3Interface::minusEqualScaler, pb11::return_value_policy::reference)
        .def("__isub__", &Vec3Interface::minusEqualVec3, pb11::return_value_policy::reference)
        .def("__imul__", &Vec3::operator*=, pb11::return_value_policy::reference)
        .def("__itruediv__", &Vec3::operator/=, pb11::return_value_policy::reference);

    m.def("Vec3_addVec3", &Vec3Interface::addVec3);
    m.def("Vec3_addVec3Scaler", &Vec3Interface::addVec3Scaler);
    m.def("Vec3_addScalerVec3", &Vec3Interface::addScalerVec3);
    m.def("Vec3_minusVec3", &Vec3Interface::minusVec3);
    m.def("Vec3_minusVec3Scaler", &Vec3Interface::minusVec3Scaler);
    m.def("Vec3_minusScalerVec3", &Vec3Interface::minusScalerVec3);
    m.def("Vec3_multVec3Scaler", &Vec3Interface::multVec3Scaler);
    m.def("Vec3_multScalerVec3", &Vec3Interface::multScalerVec3);
    m.def("Vec3_divVec3Scaler", &Vec3Interface::divVec3Scaler);
    m.def("Vec3_print", &Vec3Interface::print);
}