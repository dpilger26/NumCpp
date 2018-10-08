#include"NumCpp.hpp"

#include<iostream>

int main()
{
    // Containers
    NC::NdArray<int> a0 = { {1, 2}, {3, 4} };
    NC::NdArray<int> a1 = { {1, 2}, {3, 4}, {5, 6} };
    a1.reshape(2, 3);
    auto a2 = a1.astype<double>();

    // Initializers
    auto a3 = NC::linspace<int>(1, 10, 5);
    auto a4 = NC::arange<int>(3, 7);
    auto a5 = NC::eye<int>(4);
    auto a6 = NC::zeros<int>(3, 4);
    auto a7 = NC::NdArray<int>(3, 4) = 0;
    auto a8 = NC::ones<int>(3, 4);
    auto a9 = NC::NdArray<int>(3, 4) = 1;
    auto a10 = NC::nans<double>(3, 4);
    auto a11 = NC::NdArray<double>(3, 4) = NC::Constants::nan;
    auto a12 = NC::empty<int>(3, 4);
    auto a13 = NC::NdArray<int>(3, 4);

    // Slicing/Broadcasting
    auto a14 = NC::Random<int>::randInt(NC::Shape(10, 10), 0, 100);
    auto value = a14(2, 3);
    auto slice = a14(NC::Slice(2, 5), NC::Slice(5, 8));
    auto rowSlice = a14(a14.rSlice(), 7);
    auto values = a14[a14 > 50];
    a14.putMask(a14 > 50, 666);

    // Random
    NC::Random<>::seed(666);
    auto a15 = NC::Random<double>::randN(NC::Shape(3, 4));
    auto a16 = NC::Random<int>::randInt(NC::Shape(3, 4), 0, 10);
    auto a17 = NC::Random<double>::rand(NC::Shape(3, 4));
    auto a18 = NC::Random<double>::choice(a17, 3);

    // Concatenation
    auto a = NC::Random<int>::randInt(NC::Shape(3, 4), 0, 10);
    auto b = NC::Random<int>::randInt(NC::Shape(3, 4), 0, 10);
    auto c = NC::Random<int>::randInt(NC::Shape(3, 4), 0, 10);

    auto a19 = NC::stack({ a, b, c }, NC::Axis::ROW);
    auto a20 = NC::vstack({ a, b, c });
    auto a21 = NC::hstack({ a, b, c });
    auto a22 = NC::append(a, b, NC::Axis::COL);

    // Diagonal, Traingular, and Flip
    auto d = NC::Random<int>::randInt(NC::Shape(5, 5), 0, 10);
    auto a23 = NC::diagonal(d);
    auto a24 = NC::triu(a);
    auto a25 = NC::tril(a);
    auto a26 = NC::flip(d, NC::Axis::ROW);
    auto a27 = NC::flipud(d);
    auto a28 = NC::fliplr(d);

    // iteration
    for (auto it = a.begin(); it < a.end(); ++it)
    {
        std::cout << *it << " ";
    }

    for (auto& arrayValue : a)
    {
        std::cout << arrayValue << " ";
    }


    // Logical
    auto a29 = NC::where(a > 5, a, b);
    auto a30 = NC::any(a);
    auto a31 = NC::all(a);
    auto a32 = NC::logical_and(a, b);
    auto a33 = NC::logical_or(a, b);
    auto a34 = NC::isclose(a, b);
    auto a35 = NC::allclose(a, b);

    // Comparisons
    auto a36 = NC::equal(a, b);
    auto a37 = a == b;
    auto a38 = NC::not_equal(a, b);
    auto a39 = a != b;
    auto a40 = NC::nonzero(a);

    // Minimum, Maximum, Sorting
    auto value1 = NC::min(a);
    auto value2 = NC::max(a);
    auto value3 = NC::argmin(a);
    auto value4 = NC::argmax(a);
    auto a41 = NC::sort(a, NC::Axis::ROW);
    auto a42 = NC::argsort(a, NC::Axis::COL);
    auto a43 = NC::unique(a);
    auto a44 = NC::setdiff1d(a, b);
    auto a45 = NC::diff(a);

    // Reducers
    auto value5 = NC::sum<int>(a);
    auto a46 = NC::sum<int>(a, NC::Axis::ROW);
    auto value6 = NC::prod<int>(a);
    auto a47 = NC::prod<int>(a, NC::Axis::ROW);
    auto value7 = NC::mean(a);
    auto a48 = NC::mean(a, NC::Axis::ROW);
    auto value8 = NC::count_nonzero(a);
    auto a49 = NC::count_nonzero(a, NC::Axis::ROW);

    // I/O
    a.print();
    std::cout << a << std::endl;
    a.tofile("C:/Temp/temp.txt", "\n");
    auto a50 = NC::fromfile<int>("C:/Temp/temp.txt", "\n");
    NC::dump(a, "C:/Temp/temp.bin");
    auto a51 = NC::load<int>("C:/Temp/temp.bin");

    // Mathematical Functions

    // Basic Functions
    auto a52 = NC::abs(a);
    auto a53 = NC::sign(a);
    auto a54 = NC::remainder<double>(a, b);
    auto a55 = NC::clip(a, 3, 8);
    auto xp = NC::linspace<double>(0.0, 2.0 * NC::Constants::pi, 100);
    auto fp = NC::sin(xp);
    auto x = NC::linspace<double>(0.0, 2.0 * NC::Constants::pi, 1000);
    auto f = NC::interp(x, xp, fp);

    // Exponential Functions
    auto a56 = NC::exp(a);
    auto a57 = NC::expm1(a);
    auto a58 = NC::log(a);
    auto a59 = NC::log1p(a);

    // Power Functions
    auto a60 = NC::power<int>(a, 4);
    auto a61 = NC::sqrt(a);
    auto a62 = NC::square(a);
    auto a63 = NC::cbrt(a);

    // Trigonometric Functions
    auto a64 = NC::sin(a);
    auto a65 = NC::cos(a);
    auto a66 = NC::tan(a);

    // Hyperbolic Functions
    auto a67 = NC::sinh(a);
    auto a68 = NC::cosh(a);
    auto a69 = NC::tanh(a);

    // Classification Functions
    auto a70 = NC::isnan(a.astype<double>());
    //NC::isinf(a);

    // Linear Algebra
    auto a71 = NC::norm<int>(a);
    auto a72 = NC::dot<int>(a, b.transpose());
    auto value9 = NC::Linalg::det(a);
    auto a73 = NC::Linalg::inv(a);
    auto a74 = NC::Linalg::lstsq(a, b);
    auto a75 = NC::Linalg::matrix_power<int>(a, 3);
    auto a77 = NC::Linalg::multi_dot<int>({ a, b.transpose(), c });

    NC::NdArray<double> u;
    NC::NdArray<double> s;
    NC::NdArray<double> vt;
    NC::Linalg::svd(a.astype<double>(), u, s, vt);

    return 0;
}