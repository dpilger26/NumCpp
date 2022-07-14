#include "BindingsIncludes.hpp"

//================================================================================

namespace SpecialInterface
{
#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype airy_ai_Scaler(dtype inValue)
    {
        return special::airy_ai(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric airy_ai_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::airy_ai(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype airy_ai_prime_Scaler(dtype inValue)
    {
        return special::airy_ai_prime(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric airy_ai_prime_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::airy_ai_prime(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype airy_bi_Scaler(dtype inValue)
    {
        return special::airy_bi(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric airy_bi_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::airy_bi(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype airy_bi_prime_Scaler(dtype inValue)
    {
        return special::airy_bi_prime(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric airy_bi_prime_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::airy_bi_prime(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    double bernoulli_Scaler(uint32 n)
    {
        return special::bernoilli(n);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    pbArray<double> bernoulli_Array(const NdArray<uint32>& inArray)
    {
        return nc2pybind(special::bernoilli(inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype bessel_in_Scaler(dtype inV, dtype inValue)
    {
        return special::bessel_in(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric bessel_in_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_in(inV, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype bessel_in_prime_Scaler(dtype inV, dtype inValue)
    {
        return special::bessel_in_prime(inV, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric bessel_in_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_in_prime(inV, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype bessel_jn_Scaler(dtype inV, dtype inValue)
    {
        return special::bessel_jn(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric bessel_jn_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_jn(inV, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype bessel_jn_prime_Scaler(dtype inV, dtype inValue)
    {
        return special::bessel_jn_prime(inV, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric bessel_jn_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_jn_prime(inV, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype bessel_kn_Scaler(dtype inV, dtype inValue)
    {
        return special::bessel_kn(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric bessel_kn_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_kn(inV, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype bessel_kn_prime_Scaler(dtype inV, dtype inValue)
    {
        return special::bessel_kn_prime(inV, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric bessel_kn_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_kn_prime(inV, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype bessel_yn_Scaler(dtype inV, dtype inValue)
    {
        return special::bessel_yn(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric bessel_yn_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_yn(inV, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype bessel_yn_prime_Scaler(dtype inV, dtype inValue)
    {
        return special::bessel_yn_prime(inV, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric bessel_yn_prime_Array(dtype inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::bessel_yn_prime(inV, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype beta_Scaler(dtype a, dtype b)
    {
        return special::beta(a, b);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric beta_Array(const NdArray<dtype>& a, const NdArray<dtype>& b)
    {
        return nc2pybind(special::beta(a, b));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype comp_ellint_1_Scaler(dtype k)
    {
        return special::comp_ellint_1(k);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric comp_ellint_1_Array(const NdArray<dtype>& k)
    {
        return nc2pybind(special::comp_ellint_1(k));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype comp_ellint_2_Scaler(dtype k)
    {
        return special::comp_ellint_2(k);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric comp_ellint_2_Array(const NdArray<dtype>& k)
    {
        return nc2pybind(special::comp_ellint_2(k));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype comp_ellint_3_Scaler(dtype k, dtype v)
    {
        return special::comp_ellint_3(k, v);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype1, typename dtype2>
    pbArrayGeneric comp_ellint_3_Array(const NdArray<dtype1>& k, const NdArray<dtype2>& v)
    {
        return nc2pybind(special::comp_ellint_3(k, v));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    std::complex<dtype> cyclic_hankel_1_Scaler(dtype v, dtype x)
    {
        return special::cyclic_hankel_1(v, x);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric cyclic_hankel_1_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2pybind(special::cyclic_hankel_1(v, x));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    std::complex<dtype> cyclic_hankel_2_Scaler(dtype v, dtype x)
    {
        return special::cyclic_hankel_2(v, x);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric cyclic_hankel_2_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2pybind(special::cyclic_hankel_2(v, x));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype ellint_1_Scaler(dtype k, dtype p)
    {
        return special::ellint_1(k, p);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype1, typename dtype2>
    pbArrayGeneric ellint_1_Array(const NdArray<dtype1>& k, const NdArray<dtype2>& p)
    {
        return nc2pybind(special::ellint_1(k, p));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype ellint_2_Scaler(dtype k, dtype p)
    {
        return special::ellint_2(k, p);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype1, typename dtype2>
    pbArrayGeneric ellint_2_Array(const NdArray<dtype1>& k, const NdArray<dtype2>& p)
    {
        return nc2pybind(special::ellint_2(k, p));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype ellint_3_Scaler(dtype k, dtype v, dtype p)
    {
        return special::ellint_3(k, v, p);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype1, typename dtype2, typename dtype3>
    pbArrayGeneric ellint_3_Array(const NdArray<dtype1>& k, const NdArray<dtype2>& v, const NdArray<dtype3>& p)
    {
        return nc2pybind(special::ellint_3(k, v, p));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype expint_Scaler(dtype k)
    {
        return special::expint(k);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric expint_Array(const NdArray<dtype>& k)
    {
        return nc2pybind(special::expint(k));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype digamma_Scaler(dtype inValue)
    {
        return special::digamma(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric digamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::digamma(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype erf_Scaler(dtype inValue)
    {
        return special::erf(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric erf_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::erf(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype erf_inv_Scaler(dtype inValue)
    {
        return special::erf_inv(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric erf_inv_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::erf_inv(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype erfc_Scaler(dtype inValue)
    {
        return special::erfc(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric erfc_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::erfc(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype erfc_inv_Scaler(dtype inValue)
    {
        return special::erfc_inv(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric erfc_inv_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::erfc_inv(inArray));
    }
#endif

    //================================================================================

    double factorial_Scaler(uint32 inValue)
    {
        return special::factorial(inValue);
    }

    //================================================================================

    pbArray<double> factorial_Array(const NdArray<uint32>& inArray)
    {
        return nc2pybind(special::factorial(inArray));
    }

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype gamma_Scaler(dtype inValue)
    {
        return special::gamma(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric gamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::gamma(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype gamma1pm1_Scaler(dtype inValue)
    {
        return special::gamma1pm1(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric gamma1pm1_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::gamma1pm1(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype log_gamma_Scaler(dtype inValue)
    {
        return special::log_gamma(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric log_gamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::log_gamma(inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype polygamma_Scaler(uint32 n, dtype inValue)
    {
        return special::polygamma(n, inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric polygamma_Array(uint32 n, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::polygamma(n, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    double prime_Scaler(uint32 inValue)
    {
        return special::prime(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    pbArray<uint32> prime_Array(const NdArray<uint32>& inArray)
    {
        return nc2pybind(special::prime(inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype riemann_zeta_Scaler(dtype inValue)
    {
        return special::riemann_zeta(inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric riemann_zeta_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::riemann_zeta(inArray));
    }
#endif

    //================================================================================

    template<typename dtype>
    pbArrayGeneric softmax(const NdArray<dtype>& inArray, Axis inAxis)
    {
        return nc2pybind(special::softmax(inArray, inAxis));
    }

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype spherical_bessel_jn_Scaler(uint32 inV, dtype inValue)
    {
        return special::spherical_bessel_jn(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric spherical_bessel_jn_Array(uint32 inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::spherical_bessel_jn(inV, inArray));
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    dtype spherical_bessel_yn_Scaler(uint32 inV, dtype inValue)
    {
        return special::spherical_bessel_yn(inV, inValue);
    }
#endif

    //================================================================================

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    template<typename dtype>
    pbArrayGeneric spherical_bessel_yn_Array(uint32 inV, const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::spherical_bessel_yn(inV, inArray));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    std::complex<dtype> spherical_hankel_1_Scaler(dtype v, dtype x)
    {
        return special::spherical_hankel_1(v, x);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric spherical_hankel_1_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2pybind(special::spherical_hankel_1(v, x));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    std::complex<dtype> spherical_hankel_2_Scaler(dtype v, dtype x)
    {
        return special::spherical_hankel_2(v, x);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric spherical_hankel_2_Array(dtype v, const NdArray<dtype>& x)
    {
        return nc2pybind(special::spherical_hankel_2(v, x));
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    dtype trigamma_Scaler(dtype inValue)
    {
        return special::trigamma(inValue);
    }
#endif

    //================================================================================

#ifndef NUMCPP_NO_USE_BOOST
    template<typename dtype>
    pbArrayGeneric trigamma_Array(const NdArray<dtype>& inArray)
    {
        return nc2pybind(special::trigamma(inArray));
    }
#endif
} // namespace SpecialInterface

//================================================================================

void initSpecial(pb11::module& m)
{
// Special.hpp
#ifndef NUMCPP_NO_USE_BOOST
    m.def("airy_ai_Scaler", &SpecialInterface::airy_ai_Scaler<double>);
    m.def("airy_ai_Array", &SpecialInterface::airy_ai_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("airy_ai_prime_Scaler", &SpecialInterface::airy_ai_prime_Scaler<double>);
    m.def("airy_ai_prime_Array", &SpecialInterface::airy_ai_prime_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("airy_bi_Scaler", &SpecialInterface::airy_bi_Scaler<double>);
    m.def("airy_bi_Array", &SpecialInterface::airy_bi_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("airy_bi_prime_Scaler", &SpecialInterface::airy_bi_prime_Scaler<double>);
    m.def("airy_bi_prime_Array", &SpecialInterface::airy_bi_prime_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("bernoulli_Scaler", &SpecialInterface::bernoulli_Scaler);
    m.def("bernoulli_Array", &SpecialInterface::bernoulli_Array);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("bessel_in_Scaler", &SpecialInterface::bessel_in_Scaler<double>);
    m.def("bessel_in_Array", &SpecialInterface::bessel_in_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("bessel_in_prime_Scaler", &SpecialInterface::bessel_in_prime_Scaler<double>);
    m.def("bessel_in_prime_Array", &SpecialInterface::bessel_in_prime_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("bessel_jn_Scaler", &SpecialInterface::bessel_jn_Scaler<double>);
    m.def("bessel_jn_Array", &SpecialInterface::bessel_jn_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("bessel_jn_prime_Scaler", &SpecialInterface::bessel_jn_prime_Scaler<double>);
    m.def("bessel_jn_prime_Array", &SpecialInterface::bessel_jn_prime_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("bessel_kn_Scaler", &SpecialInterface::bessel_kn_Scaler<double>);
    m.def("bessel_kn_Array", &SpecialInterface::bessel_kn_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("bessel_kn_prime_Scaler", &SpecialInterface::bessel_kn_prime_Scaler<double>);
    m.def("bessel_kn_prime_Array", &SpecialInterface::bessel_kn_prime_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("bessel_yn_Scaler", &SpecialInterface::bessel_yn_Scaler<double>);
    m.def("bessel_yn_Array", &SpecialInterface::bessel_yn_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("bessel_yn_prime_Scaler", &SpecialInterface::bessel_yn_prime_Scaler<double>);
    m.def("bessel_yn_prime_Array", &SpecialInterface::bessel_yn_prime_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("beta_Scaler", &SpecialInterface::beta_Scaler<double>);
    m.def("beta_Array", &SpecialInterface::beta_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("comp_ellint_1_Scaler", &SpecialInterface::comp_ellint_1_Scaler<double>);
    m.def("comp_ellint_1_Array", &SpecialInterface::comp_ellint_1_Array<double>);
#endif

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("comp_ellint_2_Scaler", &SpecialInterface::comp_ellint_2_Scaler<double>);
    m.def("comp_ellint_2_Array", &SpecialInterface::comp_ellint_2_Array<double>);
#endif

#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("comp_ellint_3_Scaler", &SpecialInterface::comp_ellint_3_Scaler<double>);
    m.def("comp_ellint_3_Array", &SpecialInterface::comp_ellint_3_Array<double, double>);
#endif
    m.def("cnr", &special::cnr);
#ifndef NUMCPP_NO_USE_BOOST
    m.def("cyclic_hankel_1_Scaler", &SpecialInterface::cyclic_hankel_1_Scaler<double>);
    m.def("cyclic_hankel_1_Array", &SpecialInterface::cyclic_hankel_1_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("cyclic_hankel_2_Scaler", &SpecialInterface::cyclic_hankel_2_Scaler<double>);
    m.def("cyclic_hankel_2_Array", &SpecialInterface::cyclic_hankel_2_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("ellint_1_Scaler", &SpecialInterface::ellint_1_Scaler<double>);
    m.def("ellint_1_Array", &SpecialInterface::ellint_1_Array<double, double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("ellint_2_Scaler", &SpecialInterface::ellint_2_Scaler<double>);
    m.def("ellint_2_Array", &SpecialInterface::ellint_2_Array<double, double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("ellint_3_Scaler", &SpecialInterface::ellint_3_Scaler<double>);
    m.def("ellint_3_Array", &SpecialInterface::ellint_3_Array<double, double, double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("expint_Scaler", &SpecialInterface::expint_Scaler<double>);
    m.def("expint_Array", &SpecialInterface::expint_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("digamma_Scaler", &SpecialInterface::digamma_Scaler<double>);
    m.def("digamma_Array", &SpecialInterface::digamma_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("erf_Scaler", &SpecialInterface::erf_Scaler<double>);
    m.def("erf_Array", &SpecialInterface::erf_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("erf_inv_Scaler", &SpecialInterface::erf_inv_Scaler<double>);
    m.def("erf_inv_Array", &SpecialInterface::erf_inv_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("erfc_Scaler", &SpecialInterface::erfc_Scaler<double>);
    m.def("erfc_Array", &SpecialInterface::erfc_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("erfc_inv_Scaler", &SpecialInterface::erfc_inv_Scaler<double>);
    m.def("erfc_inv_Array", &SpecialInterface::erfc_inv_Array<double>);
#endif
    m.def("factorial_Scaler", &SpecialInterface::factorial_Scaler);
    m.def("factorial_Array", &SpecialInterface::factorial_Array);
#ifndef NUMCPP_NO_USE_BOOST
    m.def("gamma_Scaler", &SpecialInterface::gamma_Scaler<double>);
    m.def("gamma_Array", &SpecialInterface::gamma_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("gamma1pm1_Scaler", &SpecialInterface::gamma1pm1_Scaler<double>);
    m.def("gamma1pm1_Array", &SpecialInterface::gamma1pm1_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("log_gamma_Scaler", &SpecialInterface::log_gamma_Scaler<double>);
    m.def("log_gamma_Array", &SpecialInterface::log_gamma_Array<double>);
#endif
    m.def("pnr", &special::pnr);
#ifndef NUMCPP_NO_USE_BOOST
    m.def("polygamma_Scaler", &SpecialInterface::polygamma_Scaler<double>);
    m.def("polygamma_Array", &SpecialInterface::polygamma_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("prime_Scaler", &SpecialInterface::prime_Scaler);
    m.def("prime_Array", &SpecialInterface::prime_Array);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("riemann_zeta_Scaler", &SpecialInterface::riemann_zeta_Scaler<double>);
    m.def("riemann_zeta_Array", &SpecialInterface::riemann_zeta_Array<double>);
#endif
    m.def("softmax", &SpecialInterface::softmax<double>);
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("spherical_bessel_jn_Scaler", &SpecialInterface::spherical_bessel_jn_Scaler<double>);
    m.def("spherical_bessel_jn_Array", &SpecialInterface::spherical_bessel_jn_Array<double>);
#endif
#if !defined(NUMCPP_NO_USE_BOOST) || defined(__cpp_lib_math_special_functions)
    m.def("spherical_bessel_yn_Scaler", &SpecialInterface::spherical_bessel_yn_Scaler<double>);
    m.def("spherical_bessel_yn_Array", &SpecialInterface::spherical_bessel_yn_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("spherical_hankel_1_Scaler", &SpecialInterface::spherical_hankel_1_Scaler<double>);
    m.def("spherical_hankel_1_Array", &SpecialInterface::spherical_hankel_1_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("spherical_hankel_2_Scaler", &SpecialInterface::spherical_hankel_2_Scaler<double>);
    m.def("spherical_hankel_2_Array", &SpecialInterface::spherical_hankel_2_Array<double>);
#endif
#ifndef NUMCPP_NO_USE_BOOST
    m.def("trigamma_Scaler", &SpecialInterface::trigamma_Scaler<double>);
    m.def("trigamma_Array", &SpecialInterface::trigamma_Array<double>);
#endif
}