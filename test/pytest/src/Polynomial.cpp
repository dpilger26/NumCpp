#include "NumCpp/Polynomial.hpp"

#include "BindingsIncludes.hpp"

//================================================================================

namespace PolynomialInterface
{
#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype chebyshev_t_Scalar(uint32 n, dtype inValue)
    {
        return polynomial::chebyshev_t(n, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric chebyshev_t_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::chebyshev_t(n, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype chebyshev_u_Scalar(uint32 n, dtype inValue)
    {
        return polynomial::chebyshev_u(n, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric chebyshev_u_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::chebyshev_u(n, inArray));
    }
#endif

    //================================================================================

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    dtype hermite_Scalar(uint32 n, dtype inValue)
    {
        return polynomial::hermite(n, inValue);
    }
#endif

    //================================================================================

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    pbArrayGeneric hermite_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::hermite(n, inArray));
    }
#endif

    //================================================================================

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    dtype laguerre_Scalar1(uint32 n, dtype inValue)
    {
        return polynomial::laguerre(n, inValue);
    }
#endif

    //================================================================================

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    dtype laguerre_Scalar2(uint32 n, uint32 m, dtype inValue)
    {
        return polynomial::laguerre(n, m, inValue);
    }
#endif

    //================================================================================

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    pbArrayGeneric laguerre_Array1(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::laguerre(n, inArray));
    }
#endif

    //================================================================================

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    pbArrayGeneric laguerre_Array2(uint32 n, uint32 m, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::laguerre(n, m, inArray));
    }
#endif

    //================================================================================

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    dtype legendre_p_Scalar1(int32 n, dtype inValue)
    {
        return polynomial::legendre_p(n, inValue);
    }
#endif

    //================================================================================

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    dtype legendre_p_Scalar2(int32 n, int32 m, dtype inValue)
    {
        return polynomial::legendre_p(n, m, inValue);
    }
#endif

    //================================================================================

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    pbArrayGeneric legendre_p_Array1(int32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::legendre_p(n, inArray));
    }
#endif

    //================================================================================

#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    template<typename dtype>
    pbArrayGeneric legendre_p_Array2(int32 n, int32 m, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::legendre_p(n, m, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype legendre_q_Scalar(int32 n, dtype inValue)
    {
        return polynomial::legendre_q(n, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric legendre_q_Array(int32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(polynomial::legendre_q(n, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    std::vector<double> spherical_harmonic(uint32 n, int32 m, dtype theta, dtype phi)
    {
        auto                value    = polynomial::spherical_harmonic(n, m, theta, phi);
        std::vector<double> valueVec = { value.real(), value.imag() };
        return valueVec;
    }
#endif
} // namespace PolynomialInterface

//================================================================================

void initPolynomial(pb11::module& m)
{
    // Polynomial.hpp
    using Poly1d = polynomial::Poly1d<double>;

    Poly1d (*fit)(const NdArrayDouble&, const NdArrayDouble&, uint8)                               = &Poly1d::fit;
    Poly1d (*fitWeighted)(const NdArrayDouble&, const NdArrayDouble&, const NdArrayDouble&, uint8) = &Poly1d::fit;

    pb11::class_<Poly1d>(m, "Poly1d")
        .def(pb11::init<>())
        .def(pb11::init<NdArray<double>, IsRoots>())
        .def("area", &Poly1d::area)
        .def("coefficients", &Poly1d::coefficients)
        .def("deriv", &Poly1d::deriv)
        .def("eval", [](const Poly1d& self, double xValue) { return self.eval(xValue); })
        .def("eval", [](const Poly1d& self, const NdArray<double>& xValues) { return self.eval(xValues); })
        .def_static("fit", fit)
        .def_static("fitWeighted", fitWeighted)
        .def("integ", &Poly1d::integ)
        .def("order", &Poly1d::order)
        .def("print", &Poly1d::print)
        .def("__str__", &Poly1d::str)
        .def("__repr__", &Poly1d::str)
        .def("__getitem__", [](const Poly1d& self, double xValue) { return self(xValue); })
        .def("__getitem__", [](const Poly1d& self, const NdArray<double>& xValues) { return self(xValues); })
        .def("__add__", &Poly1d::operator+)
        .def("__iadd__", &Poly1d::operator+=, pb11::return_value_policy::reference)
        .def("__sub__", &Poly1d::operator-)
        .def("__isub__", &Poly1d::operator-=, pb11::return_value_policy::reference)
        .def("__mul__", &Poly1d::operator*)
        .def("__imul__", &Poly1d::operator*=, pb11::return_value_policy::reference)
        .def("__pow__", &Poly1d::operator^)
        .def("__ipow__", &Poly1d::operator^=, pb11::return_value_policy::reference);

#ifndef NUMCPP_NO_USE_BOOST
    m.def("chebyshev_t_Scalar", &PolynomialInterface::chebyshev_t_Scalar<double>);
    m.def("chebyshev_t_Array", &PolynomialInterface::chebyshev_t_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("chebyshev_u_Scalar", &PolynomialInterface::chebyshev_u_Scalar<double>);
    m.def("chebyshev_u_Array", &PolynomialInterface::chebyshev_u_Array<double>);
#endif
#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    m.def("hermite_Scalar", &PolynomialInterface::hermite_Scalar<double>);
    m.def("hermite_Array", &PolynomialInterface::hermite_Array<double>);
#endif
#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    m.def("laguerre_Scalar1", &PolynomialInterface::laguerre_Scalar1<double>);
    m.def("laguerre_Array1", &PolynomialInterface::laguerre_Array1<double>);
    m.def("laguerre_Scalar2", &PolynomialInterface::laguerre_Scalar2<double>);
    m.def("laguerre_Array2", &PolynomialInterface::laguerre_Array2<double>);
#endif
#if defined(__cpp_lib_math_special_functions) || !defined(NUMCPP_NO_USE_BOOST)
    m.def("legendre_p_Scalar1", &PolynomialInterface::legendre_p_Scalar1<double>);
    m.def("legendre_p_Array1", &PolynomialInterface::legendre_p_Array1<double>);
    m.def("legendre_p_Scalar2", &PolynomialInterface::legendre_p_Scalar2<double>);
    m.def("legendre_p_Array2", &PolynomialInterface::legendre_p_Array2<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("legendre_q_Scalar", &PolynomialInterface::legendre_q_Scalar<double>);
    m.def("legendre_q_Array", &PolynomialInterface::legendre_q_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("spherical_harmonic", &PolynomialInterface::spherical_harmonic<double>);
    m.def("spherical_harmonic_r", &polynomial::spherical_harmonic_r<double, double>);
    m.def("spherical_harmonic_i", &polynomial::spherical_harmonic_i<double, double>);
#endif
}