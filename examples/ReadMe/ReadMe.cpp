#include"NumCpp.hpp"

#include"boost/filesystem.hpp"

#include<iostream>

int main()
{
    // Containers
    nc::NdArray<int> a0 = { {1, 2}, {3, 4} };
    nc::NdArray<int> a1 = { {1, 2}, {3, 4}, {5, 6} };
    a1.reshape(2, 3);
    auto a2 = a1.astype<double>();

    // Initializers
    auto a3 = nc::linspace<int>(1, 10, 5);
    auto a4 = nc::arange<int>(3, 7);
    auto a5 = nc::eye<int>(4);
    auto a6 = nc::zeros<int>(3, 4);
    auto a7 = nc::NdArray<int>(3, 4) = 0;
    auto a8 = nc::ones<int>(3, 4);
    auto a9 = nc::NdArray<int>(3, 4) = 1;
    auto a10 = nc::nans(3, 4);
    auto a11 = nc::NdArray<double>(3, 4) = nc::constants::nan;
    auto a12 = nc::empty<int>(3, 4);
    auto a13 = nc::NdArray<int>(3, 4);

    // Slicing/Broadcasting
    auto a14 = nc::random::randInt<int>({ 10, 10 }, 0, 100);
    auto value = a14(2, 3);
    auto slice = a14({ 2, 5 }, { 2, 5 });
    auto rowSlice = a14(a14.rSlice(), 7);
    auto values = a14[a14 > 50];
    a14.putMask(a14 > 50, 666);

    // random
    nc::random::seed(666);
    auto a15 = nc::random::randN<double>({3, 4});
    auto a16 = nc::random::randInt<int>({3, 4}, 0, 10);
    auto a17 = nc::random::rand<double>({3, 4});
    auto a18 = nc::random::choice(a17, 3);

    // Concatenation
    auto a = nc::random::randInt<int>({3, 4}, 0, 10);
    auto b = nc::random::randInt<int>({3, 4}, 0, 10);
    auto c = nc::random::randInt<int>({3, 4}, 0, 10);

    auto a19 = nc::stack({ a, b, c }, nc::Axis::ROW);
    auto a20 = nc::vstack({ a, b, c });
    auto a21 = nc::hstack({ a, b, c });
    auto a22 = nc::append(a, b, nc::Axis::COL);

    // Diagonal, Traingular, and Flip
    auto d = nc::random::randInt<int>({5, 5}, 0, 10);
    auto a23 = nc::diagonal(d);
    auto a24 = nc::triu(a);
    auto a25 = nc::tril(a);
    auto a26 = nc::flip(d, nc::Axis::ROW);
    auto a27 = nc::flipud(d);
    auto a28 = nc::fliplr(d);

    // iteration
    for (auto it = a.begin(); it < a.end(); ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    for (auto& arrayValue : a)
    {
        std::cout << arrayValue << " ";
    }
    std::cout << std::endl;

    // Logical
    auto a29 = nc::where(a > 5, a, b);
    auto a30 = nc::any(a);
    auto a31 = nc::all(a);
    auto a32 = nc::logical_and(a, b);
    auto a33 = nc::logical_or(a, b);
    auto a34 = nc::isclose(a15, a17);
    auto a35 = nc::allclose(a, b);

    // Comparisons
    auto a36 = nc::equal(a, b);
    auto a37 = a == b;
    auto a38 = nc::not_equal(a, b);
    auto a39 = a != b;

#ifdef __cpp_structured_bindings
    auto [rows, cols] = nc::nonzero(a);
#else
    auto rowsCols = nc::nonzero(a);
    auto& rows = rowsCols.first;
    auto& cols = rowsCols.second;
#endif

    // Minimum, Maximum, Sorting
    auto value1 = nc::min(a);
    auto value2 = nc::max(a);
    auto value3 = nc::argmin(a);
    auto value4 = nc::argmax(a);
    auto a41 = nc::sort(a, nc::Axis::ROW);
    auto a42 = nc::argsort(a, nc::Axis::COL);
    auto a43 = nc::unique(a);
    auto a44 = nc::setdiff1d(a, b);
    auto a45 = nc::diff(a);

    // Reducers
    auto value5 = nc::sum<int>(a);
    auto a46 = nc::sum<int>(a, nc::Axis::ROW);
    auto value6 = nc::prod<int>(a);
    auto a47 = nc::prod<int>(a, nc::Axis::ROW);
    auto value7 = nc::mean(a);
    auto a48 = nc::mean(a, nc::Axis::ROW);
    auto value8 = nc::count_nonzero(a);
    auto a49 = nc::count_nonzero(a, nc::Axis::ROW);

    // I/O
    a.print();
    std::cout << a << std::endl;

    auto tempDir = boost::filesystem::temp_directory_path();
    auto tempTxt = (tempDir / "temp.txt").string();
    a.tofile(tempTxt, '\n');
    auto a50 = nc::fromfile<int>(tempTxt, '\n');

    auto tempBin = (tempDir / "temp.bin").string();
    nc::dump(a, tempBin);
    auto a51 = nc::load<int>(tempBin);

    // Mathematical Functions

    // Basic Functions
    auto a52 = nc::abs(a);
    auto a53 = nc::sign(a);
    auto a54 = nc::remainder(a, b);
    auto a55 = nc::clip(a, 3, 8);
    auto xp = nc::linspace<double>(0.0, 2.0 * nc::constants::pi, 100);
    auto fp = nc::sin(xp);
    auto x = nc::linspace<double>(0.0, 2.0 * nc::constants::pi, 1000);
    auto f = nc::interp(x, xp, fp);

    // Exponential Functions
    auto a56 = nc::exp(a);
    auto a57 = nc::expm1(a);
    auto a58 = nc::log(a);
    auto a59 = nc::log1p(a);

    // Power Functions
    auto a60 = nc::power<int>(a, 4);
    auto a61 = nc::sqrt(a);
    auto a62 = nc::square(a);
    auto a63 = nc::cbrt(a);

    // Trigonometric Functions
    auto a64 = nc::sin(a);
    auto a65 = nc::cos(a);
    auto a66 = nc::tan(a);

    // Hyperbolic Functions
    auto a67 = nc::sinh(a);
    auto a68 = nc::cosh(a);
    auto a69 = nc::tanh(a);

    // Classification Functions
    auto a70 = nc::isnan(a.astype<double>());
    //nc::isinf(a);

    // Linear Algebra
    auto a71 = nc::norm<int>(a);
    auto a72 = nc::dot<int>(a, b.transpose());

    auto a73 = nc::random::randInt<int>({3, 3}, 0, 10);
    auto a74 = nc::random::randInt<int>({4, 3}, 0, 10);
    auto a75 = nc::random::randInt<int>({1, 4}, 0, 10);
    auto value9 = nc::linalg::det(a73);
    auto a76 = nc::linalg::inv(a73);
    auto a77 = nc::linalg::lstsq(a74, a75);
    auto a78 = nc::linalg::matrix_power<int>(a73, 3);
    auto a79 = nc::linalg::multi_dot<int>({ a, b.transpose(), c });

    nc::NdArray<double> u;
    nc::NdArray<double> s;
    nc::NdArray<double> vt;
    nc::linalg::svd(a.astype<double>(), u, s, vt);

    return 0;
}
