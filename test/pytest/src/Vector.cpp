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

    Vec2& plusEqualScalar(Vec2& self, double scalar)
    {
        return self += scalar;
    }

    //================================================================================

    Vec2& plusEqualVec2(Vec2& self, const Vec2& rhs)
    {
        return self += rhs;
    }

    //================================================================================

    Vec2& minusEqualScalar(Vec2& self, double scalar)
    {
        return self -= scalar;
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

    Vec2 addVec2Scalar(const Vec2& vec1, double scalar)
    {
        return vec1 + scalar;
    }

    //================================================================================

    Vec2 addScalarVec2(const Vec2& vec1, double scalar)
    {
        return scalar + vec1;
    }

    //================================================================================

    Vec2 minusVec2(const Vec2& vec1, const Vec2& vec2)
    {
        return vec1 - vec2;
    }

    //================================================================================

    Vec2 minusVec2Scalar(const Vec2& vec1, double scalar)
    {
        return vec1 - scalar;
    }

    //================================================================================

    Vec2 minusScalarVec2(const Vec2& vec1, double scalar)
    {
        return scalar - vec1;
    }

    //================================================================================

    Vec2 multVec2Scalar(const Vec2& vec1, double scalar)
    {
        return vec1 * scalar;
    }

    //================================================================================

    Vec2 multScalarVec2(const Vec2& vec1, double scalar)
    {
        return scalar * vec1;
    }

    //================================================================================

    Vec2 divVec2Scalar(const Vec2& vec1, double scalar)
    {
        return vec1 / scalar;
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

    Vec3& plusEqualScalar(Vec3& self, double scalar)
    {
        return self += scalar;
    }

    //================================================================================

    Vec3& plusEqualVec3(Vec3& self, const Vec3& rhs)
    {
        return self += rhs;
    }

    //================================================================================

    Vec3& minusEqualScalar(Vec3& self, double scalar)
    {
        return self -= scalar;
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

    Vec3 addVec3Scalar(const Vec3& vec1, double scalar)
    {
        return vec1 + scalar;
    }

    //================================================================================

    Vec3 addScalarVec3(const Vec3& vec1, double scalar)
    {
        return vec1 + scalar;
    }

    //================================================================================

    Vec3 minusVec3(const Vec3& vec1, const Vec3& vec2)
    {
        return vec1 - vec2;
    }

    //================================================================================

    Vec3 minusVec3Scalar(const Vec3& vec1, double scalar)
    {
        return vec1 - scalar;
    }

    //================================================================================

    Vec3 minusScalarVec3(const Vec3& vec1, double scalar)
    {
        return scalar - vec1;
    }

    //================================================================================

    Vec3 multVec3Scalar(const Vec3& vec1, double scalar)
    {
        return vec1 * scalar;
    }

    //================================================================================

    Vec3 multScalarVec3(const Vec3& vec1, double scalar)
    {
        return vec1 * scalar;
    }

    //================================================================================

    Vec3 divVec3Scalar(const Vec3& vec1, double scalar)
    {
        return vec1 / scalar;
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
        .def("__iadd__", &Vec2Interface::plusEqualScalar, pb11::return_value_policy::reference)
        .def("__iadd__", &Vec2Interface::plusEqualVec2, pb11::return_value_policy::reference)
        .def("__isub__", &Vec2Interface::minusEqualVec2, pb11::return_value_policy::reference)
        .def("__isub__", &Vec2Interface::minusEqualScalar, pb11::return_value_policy::reference)
        .def("__imul__", &Vec2::operator*=, pb11::return_value_policy::reference)
        .def("__itruediv__", &Vec2::operator/=, pb11::return_value_policy::reference);

    m.def("Vec2_addVec2", &Vec2Interface::addVec2);
    m.def("Vec2_addVec2Scalar", &Vec2Interface::addVec2Scalar);
    m.def("Vec2_addScalarVec2", &Vec2Interface::addScalarVec2);
    m.def("Vec2_minusVec2", &Vec2Interface::minusVec2);
    m.def("Vec2_minusVec2Scalar", &Vec2Interface::minusVec2Scalar);
    m.def("Vec2_minusScalarVec2", &Vec2Interface::minusScalarVec2);
    m.def("Vec2_multVec2Scalar", &Vec2Interface::multVec2Scalar);
    m.def("Vec2_multScalarVec2", &Vec2Interface::multScalarVec2);
    m.def("Vec2_divVec2Scalar", &Vec2Interface::divVec2Scalar);
    m.def("Vec2_print", &Vec2Interface::print);

    // Vec3.hpp
    pb11::class_<Vec3>(m, "Vec3")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<NdArray<double>>())
        .def(pb11::init<Vec2>())
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
        .def("__iadd__", &Vec3Interface::plusEqualScalar, pb11::return_value_policy::reference)
        .def("__iadd__", &Vec3Interface::plusEqualVec3, pb11::return_value_policy::reference)
        .def("__isub__", &Vec3Interface::minusEqualScalar, pb11::return_value_policy::reference)
        .def("__isub__", &Vec3Interface::minusEqualVec3, pb11::return_value_policy::reference)
        .def("__imul__", &Vec3::operator*=, pb11::return_value_policy::reference)
        .def("__itruediv__", &Vec3::operator/=, pb11::return_value_policy::reference);

    m.def("Vec3_addVec3", &Vec3Interface::addVec3);
    m.def("Vec3_addVec3Scalar", &Vec3Interface::addVec3Scalar);
    m.def("Vec3_addScalarVec3", &Vec3Interface::addScalarVec3);
    m.def("Vec3_minusVec3", &Vec3Interface::minusVec3);
    m.def("Vec3_minusVec3Scalar", &Vec3Interface::minusVec3Scalar);
    m.def("Vec3_minusScalarVec3", &Vec3Interface::minusScalarVec3);
    m.def("Vec3_multVec3Scalar", &Vec3Interface::multVec3Scalar);
    m.def("Vec3_multScalarVec3", &Vec3Interface::multScalarVec3);
    m.def("Vec3_divVec3Scalar", &Vec3Interface::divVec3Scalar);
    m.def("Vec3_print", &Vec3Interface::print);
}