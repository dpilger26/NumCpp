#include "NumCpp/Linalg.hpp"

#include "BindingsIncludes.hpp"

//================================================================================

namespace LinalgInterface
{
    template<typename dtype>
    std::pair<pbArrayGeneric, pbArrayGeneric> eig(const NdArray<dtype>& inArray)
    {
        const auto& [eigenValues, eigenVectors] = linalg::eig(inArray);
        return std::make_pair(nc2pybind(eigenValues), nc2pybind(eigenVectors));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric eigvals(const NdArray<dtype>& inArray)
    {
        return nc2pybind(linalg::eigvals(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric hatArray(const NdArray<dtype>& inArray)
    {
        return nc2pybind(linalg::hat(inArray));
    }

    //================================================================================

    template<typename dtype>
    pbArrayGeneric multi_dot(const NdArray<dtype>& inArray1,
                             const NdArray<dtype>& inArray2,
                             const NdArray<dtype>& inArray3,
                             const NdArray<dtype>& inArray4)
    {
        return nc2pybind(linalg::multi_dot({ inArray1, inArray2, inArray3, inArray4 }));
    }

    //================================================================================

    template<typename dtype>
    pb11::tuple pivotLU_decomposition(const NdArray<dtype>& inArray)
    {
        auto  lup = linalg::pivotLU_decomposition(inArray);
        auto& l   = std::get<0>(lup);
        auto& u   = std::get<1>(lup);
        auto& p   = std::get<2>(lup);
        return pb11::make_tuple(nc2pybind(l), nc2pybind(u), nc2pybind(p));
    }

    //================================================================================

    template<typename dtype>
    pbArray<double> solve(const NdArray<dtype>& inA, const NdArray<dtype>& inB)
    {
        return nc2pybind(linalg::solve(inA, inB));
    }
} // namespace LinalgInterface

//================================================================================

void initLinalg(pb11::module& m)
{
    // Linalg.hpp
    m.def("cholesky", &linalg::cholesky<double>);
    m.def("det", &linalg::det<double>);
    m.def("det", &linalg::det<int64>);
    m.def("eig", &LinalgInterface::eig<double>);
    m.def("eigvals", &LinalgInterface::eigvals<double>);
    m.def("hat", &LinalgInterface::hatArray<double>);
    m.def("inv", &linalg::inv<double>);
    m.def("lstsq", &linalg::lstsq<double>);
    m.def("lu_decomposition", &linalg::lu_decomposition<double>);
    m.def("matrix_power", &linalg::matrix_power<double>);
    m.def("multi_dot", &LinalgInterface::multi_dot<double>);
    m.def("multi_dot", &LinalgInterface::multi_dot<ComplexDouble>);
    m.def("pinv", &linalg::pinv<double>);
    m.def("pivotLU_decomposition", &LinalgInterface::pivotLU_decomposition<double>);
    m.def("solve", &LinalgInterface::solve<double>);
    m.def("svd", &linalg::svd<double>);
}
