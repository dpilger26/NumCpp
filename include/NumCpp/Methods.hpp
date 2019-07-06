/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
/// Copyright 2019 David Pilger
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy of this
/// software and associated documentation files(the "Software"), to deal in the Software
/// without restriction, including without limitation the rights to use, copy, modify,
/// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
/// permit persons to whom the Software is furnished to do so, subject to the following
/// conditions :
///
/// The above copyright notice and this permission notice shall be included in all copies
/// or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
/// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
/// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
/// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
/// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
/// DEALINGS IN THE SOFTWARE.
///
/// @section Description
/// Simple Vector classes
///
#pragma once

#include"NumCpp/Methods/abs.hpp"
#include"NumCpp/Methods/add.hpp"
#include"NumCpp/Methods/alen.hpp"
#include"NumCpp/Methods/all.hpp"
#include"NumCpp/Methods/allclose.hpp"
#include"NumCpp/Methods/amax.hpp"
#include"NumCpp/Methods/amin.hpp"
#include"NumCpp/Methods/any.hpp"
#include"NumCpp/Methods/append.hpp"
#include"NumCpp/Methods/applyFunction.hpp"
#include"NumCpp/Methods/applyPoly1d.hpp"
#include"NumCpp/Methods/arange.hpp"
#include"NumCpp/Methods/arccos.hpp"
#include"NumCpp/Methods/arccosh.hpp"
#include"NumCpp/Methods/arcsin.hpp"
#include"NumCpp/Methods/arcsinh.hpp"
#include"NumCpp/Methods/arctan.hpp"
#include"NumCpp/Methods/arctan2.hpp"
#include"NumCpp/Methods/arctanh.hpp"
#include"NumCpp/Methods/argmax.hpp"
#include"NumCpp/Methods/argmin.hpp"
#include"NumCpp/Methods/argsort.hpp"
#include"NumCpp/Methods/argwhere.hpp"
#include"NumCpp/Methods/around.hpp"
#include"NumCpp/Methods/array_equal.hpp"
#include"NumCpp/Methods/array_equiv.hpp"
#include"NumCpp/Methods/asarray.hpp"
#include"NumCpp/Methods/astype.hpp"
#include"NumCpp/Methods/average.hpp"

#include"NumCpp/Methods/binaryRepr.hpp"
#include"NumCpp/Methods/bincount.hpp"
#include"NumCpp/Methods/bitwise_and.hpp"
#include"NumCpp/Methods/bitwise_not.hpp"
#include"NumCpp/Methods/bitwise_or.hpp"
#include"NumCpp/Methods/bitwise_xor.hpp"
#include"NumCpp/Methods/byteswap.hpp"

#include"NumCpp/Methods/cbrt.hpp"
#include"NumCpp/Methods/ceil.hpp"
#include"NumCpp/Methods/clip.hpp"
#include"NumCpp/Methods/column_stack.hpp"
#include"NumCpp/Methods/concatenate.hpp"
#include"NumCpp/Methods/contains.hpp"
#include"NumCpp/Methods/copy.hpp"
#include"NumCpp/Methods/copySign.hpp"
#include"NumCpp/Methods/cos.hpp"
#include"NumCpp/Methods/cosh.hpp"
#include"NumCpp/Methods/count_nonzero.hpp"
#include"NumCpp/Methods/cross.hpp"
#include"NumCpp/Methods/cube.hpp"
#include"NumCpp/Methods/cumprod.hpp"
#include"NumCpp/Methods/cumsum.hpp"

#include"NumCpp/Methods/deg2rad.hpp"
#include"NumCpp/Methods/degrees.hpp"
#include"NumCpp/Methods/deleteIndices.hpp"
#include"NumCpp/Methods/diagFlat.hpp"
#include"NumCpp/Methods/diagnol.hpp"
#include"NumCpp/Methods/diff.hpp"
#include"NumCpp/Methods/divide.hpp"
#include"NumCpp/Methods/dot.hpp"
#include"NumCpp/Methods/dump.hpp"

#include"NumCpp/Methods/empty.hpp"
#include"NumCpp/Methods/empty_like.hpp"
#include"NumCpp/Methods/endianess.hpp"
#include"NumCpp/Methods/equal.hpp"
#include"NumCpp/Methods/erf.hpp"
#include"NumCpp/Methods/erfc.hpp"
#include"NumCpp/Methods/exp.hpp"
#include"NumCpp/Methods/exp2.hpp"
#include"NumCpp/Methods/expm1.hpp"
#include"NumCpp/Methods/eye.hpp"

#include"NumCpp/Methods/fillDiagnol.hpp"
#include"NumCpp/Methods/find.hpp"
#include"NumCpp/Methods/fix.hpp"
#include"NumCpp/Methods/flatnonzero.hpp"
#include"NumCpp/Methods/flatten.hpp"
#include"NumCpp/Methods/flip.hpp"
#include"NumCpp/Methods/fliplr.hpp"
#include"NumCpp/Methods/flipud.hpp"
#include"NumCpp/Methods/floor.hpp"
#include"NumCpp/Methods/floor_divide.hpp"
#include"NumCpp/Methods/fmax.hpp"
#include"NumCpp/Methods/fmin.hpp"
#include"NumCpp/Methods/fmod.hpp"
#include"NumCpp/Methods/fromBuffer.hpp"
#include"NumCpp/Methods/fromFile.hpp"
#include"NumCpp/Methods/fromIter.hpp"
#include"NumCpp/Methods/full.hpp"
#include"NumCpp/Methods/full_like.hpp"

#include"NumCpp/Methods/gcd.hpp"
#include"NumCpp/Methods/gradient.hpp"
#include"NumCpp/Methods/greater.hpp"
#include"NumCpp/Methods/greater_equal.hpp"

#include"NumCpp/Methods/histogram.hpp"
#include"NumCpp/Methods/hstack.hpp"
#include"NumCpp/Methods/hypot.hpp"

#include"NumCpp/Methods/identity.hpp"
#include"NumCpp/Methods/interp.hpp"
#include"NumCpp/Methods/intersect1d.hpp"
#include"NumCpp/Methods/invert.hpp"
#include"NumCpp/Methods/isclose.hpp"
#include"NumCpp/Methods/isinf.hpp"
#include"NumCpp/Methods/isnan.hpp"

#include"NumCpp/Methods/lcm.hpp"
#include"NumCpp/Methods/ldexp.hpp"
#include"NumCpp/Methods/left_shift.hpp"
#include"NumCpp/Methods/less.hpp"
#include"NumCpp/Methods/less_equal.hpp"
#include"NumCpp/Methods/linspace.hpp"
#include"NumCpp/Methods/load.hpp"
#include"NumCpp/Methods/log.hpp"
#include"NumCpp/Methods/log1p.hpp"
#include"NumCpp/Methods/log2.hpp"
#include"NumCpp/Methods/log10.hpp"
#include"NumCpp/Methods/logical_and.hpp"
#include"NumCpp/Methods/logical_not.hpp"
#include"NumCpp/Methods/logical_or.hpp"
#include"NumCpp/Methods/logical_xor.hpp"

#include"NumCpp/Methods/matmul.hpp"

#include"NumCpp/Methods/right_shift.hpp"
