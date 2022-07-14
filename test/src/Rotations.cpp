#include "BindingsIncludes.hpp"

//================================================================================

namespace RotationsInterface
{
    pbArray<double> angleAxisRotationNdArray(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2pybind(rotations::Quaternion(inAxis, inAngle).toNdArray());
    }

    //================================================================================

    pbArray<double> angleAxisRotationVec3(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2pybind(rotations::Quaternion(Vec3(inAxis), inAngle).toNdArray());
    }

    //================================================================================

    pbArray<double>
        angularVelocity(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2, double inTime)
    {
        return nc2pybind(inQuat1.angularVelocity(inQuat2, inTime));
    }

    //================================================================================

    pbArray<double> nlerp(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2, double inPercent)
    {
        return nc2pybind(inQuat1.nlerp(inQuat2, inPercent).toNdArray());
    }

    //================================================================================

    pbArray<double> rotateNdArray(const rotations::Quaternion& inQuat, const NdArray<double>& inVec)
    {
        return nc2pybind(inQuat.rotate(inVec));
    }

    //================================================================================

    pbArray<double> rotateVec3(const rotations::Quaternion& inQuat, const NdArray<double>& inVec)
    {
        return nc2pybind(inQuat.rotate(Vec3(inVec)).toNdArray());
    }

    //================================================================================

    pbArray<double> slerp(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2, double inPercent)
    {
        return nc2pybind(inQuat1.slerp(inQuat2, inPercent).toNdArray());
    }

    //================================================================================

    pbArray<double> toDCM(const rotations::Quaternion& inQuat)
    {
        return nc2pybind(inQuat.toDCM());
    }

    pbArray<double> subtract(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2)
    {
        return nc2pybind((inQuat1 - inQuat2).toNdArray());
    }

    //================================================================================

    pbArray<double> negative(const rotations::Quaternion& inQuat)
    {
        return nc2pybind((-inQuat).toNdArray());
    }

    //================================================================================

    pbArray<double> multiplyScaler(const rotations::Quaternion& inQuat, double inScaler)
    {
        const rotations::Quaternion returnQuat = inQuat * inScaler;
        return nc2pybind(returnQuat.toNdArray());
    }

    //================================================================================

    pbArray<double> multiplyArray(const rotations::Quaternion& inQuat, const NdArray<double>& inArray)
    {
        NdArray<double> returnArray = inQuat * inArray;
        return nc2pybind(returnArray);
    }

    //================================================================================

    pbArray<double> multiplyQuaternion(const rotations::Quaternion& inQuat1, const rotations::Quaternion& inQuat2)
    {
        const rotations::Quaternion returnQuat = inQuat1 * inQuat2;
        return nc2pybind(returnQuat.toNdArray());
    }

    //================================================================================

    pbArray<double> eulerAnglesValues(double roll, double pitch, double yaw)
    {
        return nc2pybind(rotations::DCM::eulerAngles(roll, pitch, yaw));
    }

    //================================================================================

    pbArray<double> eulerAnglesArray(const NdArray<double>& angles)
    {
        return nc2pybind(rotations::DCM::eulerAngles(angles));
    }

    //================================================================================

    pbArray<double> angleAxisRotationDcmNdArray(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2pybind(rotations::DCM::eulerAxisAngle(inAxis, inAngle));
    }

    //================================================================================

    pbArray<double> angleAxisRotationDcmVec3(const NdArray<double>& inAxis, double inAngle)
    {
        return nc2pybind(rotations::DCM::eulerAxisAngle(Vec3(inAxis), inAngle));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric rodriguesRotation(pbArray<T>& inK, double inTheta, pbArray<T>& inV)
    {
        auto k = pybind2nc(inK);
        auto v = pybind2nc(inV);

        return nc2pybind(rotations::rodriguesRotation(k, inTheta, v));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric wahbasProblem(pbArray<T>& inWk, pbArray<T>& inVk)
    {
        auto wk = pybind2nc(inWk);
        auto vk = pybind2nc(inVk);
        return nc2pybind(rotations::wahbasProblem(wk, vk));
    }

    //================================================================================

    template<typename T>
    pbArrayGeneric wahbasProblemWeighted(pbArray<T>& inWk, pbArray<T>& inVk, pbArray<T>& inAk)
    {
        auto wk = pybind2nc(inWk);
        auto vk = pybind2nc(inVk);
        auto ak = pybind2nc(inAk);
        return nc2pybind(rotations::wahbasProblem(wk, vk, ak));
    }
} // namespace RotationsInterface

//================================================================================

void initRotations(pb11::module& m)
{
    // Rotations.hpp
    pb11::class_<rotations::Quaternion>(m, "Quaternion")
        .def(pb11::init<>())
        .def(pb11::init<double, double, double>())
        .def(pb11::init<double, double, double, double>())
        .def(pb11::init<Vec3, double>())
        .def(pb11::init<NdArray<double>, double>())
        .def(pb11::init<NdArray<double>>())
        .def_static("angleAxisRotationNdArray", &RotationsInterface::angleAxisRotationNdArray)
        .def_static("angleAxisRotationVec3", &RotationsInterface::angleAxisRotationVec3)
        .def("angularVelocity", &RotationsInterface::angularVelocity)
        .def("conjugate", &rotations::Quaternion::conjugate)
        .def("i", &rotations::Quaternion::i)
        .def_static("identity", &rotations::Quaternion::identity)
        .def("inverse", &rotations::Quaternion::inverse)
        .def("j", &rotations::Quaternion::j)
        .def("k", &rotations::Quaternion::k)
        .def("nlerp", &RotationsInterface::nlerp)
        .def("pitch", &rotations::Quaternion::pitch)
        .def("print", &rotations::Quaternion::print)
        .def("roll", &rotations::Quaternion::roll)
        .def("rotateNdArray", &RotationsInterface::rotateNdArray)
        .def("rotateVec3", &RotationsInterface::rotateVec3)
        .def("s", &rotations::Quaternion::s)
        .def("slerp", &RotationsInterface::slerp)
        .def("toDCM", &RotationsInterface::toDCM)
        .def("toNdArray", &rotations::Quaternion::toNdArray)
        .def_static("xRotation", &rotations::Quaternion::xRotation)
        .def("yaw", &rotations::Quaternion::yaw)
        .def_static("yRotation", &rotations::Quaternion::yRotation)
        .def_static("zRotation", &rotations::Quaternion::zRotation)
        .def("__eq__", &rotations::Quaternion::operator==)
        .def("__neq__", &rotations::Quaternion::operator!=)
        .def("__add__", &rotations::Quaternion::operator+)
        .def("__sub__", &RotationsInterface::subtract)
        .def("__neg__", &RotationsInterface::negative)
        .def("__mul__", &RotationsInterface::multiplyScaler)
        .def("__mul__", &RotationsInterface::multiplyQuaternion)
        .def("__mul__", &RotationsInterface::multiplyArray)
        .def("__truediv__", &rotations::Quaternion::operator/)
        .def("__str__", &rotations::Quaternion::str);

    pb11::class_<rotations::DCM>(m, "DCM")
        .def(pb11::init<>())
        .def_static("eulerAnglesValues", &RotationsInterface::eulerAnglesValues)
        .def_static("eulerAnglesArray", &RotationsInterface::eulerAnglesArray)
        .def_static("angleAxisRotationNdArray", &RotationsInterface::angleAxisRotationDcmNdArray)
        .def_static("angleAxisRotationVec3", &RotationsInterface::angleAxisRotationDcmVec3)
        .def_static("isValid", &rotations::DCM::isValid)
        .def_static("roll", &rotations::DCM::roll)
        .def_static("pitch", &rotations::DCM::pitch)
        .def_static("yaw", &rotations::DCM::yaw)
        .def_static("xRotation", &rotations::DCM::xRotation)
        .def_static("yRotation", &rotations::DCM::yRotation)
        .def_static("zRotation", &rotations::DCM::zRotation);

    m.def("rodriguesRotation", &RotationsInterface::rodriguesRotation<double>);
    m.def("wahbasProblem", &RotationsInterface::wahbasProblem<double>);
    m.def("wahbasProblemWeighted", &RotationsInterface::wahbasProblemWeighted<double>);
}