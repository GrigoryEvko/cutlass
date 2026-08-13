/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/


#include "cutlass_unit_test.h"

#include <cute/tensor_impl.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cute;

namespace {

// Evaluate a layout over its full domain and hold cosize to its postcondition.
// The postcondition is co_min(l) <= l(c) and l(c) < co_min(l) + cosize(l).
template <class Layout>
void
test_cosize_bounds(Layout const& layout, int expected_cosize)
{
  EXPECT_EQ(int(cosize(layout)), expected_cosize);
  EXPECT_EQ(int(size(coshape(layout))), int(cosize(layout)));

  int lo = 0;
  if constexpr (!is_composed_layout<Layout>::value) {
    lo = int(co_min(layout));
  }
  for (int c = 0; c < size(layout); ++c) {
    int idx = int(layout(c));
    EXPECT_GE(idx, lo);
    EXPECT_LT(idx, lo + int(cosize(layout)));
  }
}

} // namespace

// A Swizzle takes part in the codomain of a ComposedLayout, thus cosize(layout_b)
// alone does not bound it. Sw<1,2,3> o (2):(32) emits the index 36, and the
// codomain of the layout_b alone is only 33 elements.
TEST(CuTe_core, CosizeSwizzledComposedLayout)
{
  test_cosize_bounds(composition(Swizzle<1,2,3>{}, Layout<Shape<_2>,Stride<_32>>{}), 64);
  test_cosize_bounds(composition(Swizzle<1,0,-5>{}, Layout<Shape<_2>,Stride<_1>>{}), 34);
  test_cosize_bounds(composition(Swizzle<1,4,1>{},
                                 Layout<Shape<_1,_2>,Stride<_0,_32>>{}), 64);
}

// The offset of a ComposedLayout takes part in the codomain too.
TEST(CuTe_core, CosizeSwizzledComposedLayoutOffset)
{
  test_cosize_bounds(composition(Swizzle<1,4,-6>{}, Int<8>{},
                                 Layout<Shape<_1,_2>,Stride<_0,_8>>{}), 1056);
}

// The bound must stay exact for each canonical shared memory layout, because a
// looser bound would grow every shared memory allocation.
TEST(CuTe_core, CosizeSwizzledComposedLayoutIsExact)
{
  EXPECT_EQ(int(cosize(composition(Swizzle<3,3,3>{},
                                   Layout<Shape<_8,_64>,Stride<_64,_1>>{}))), 512);
  EXPECT_EQ(int(cosize(composition(Swizzle<2,3,3>{},
                                   Layout<Shape<_8,_32>,Stride<_32,_1>>{}))), 256);
  EXPECT_EQ(int(cosize(composition(Swizzle<1,3,3>{},
                                   Layout<Shape<_8,_16>,Stride<_16,_1>>{}))), 128);
  EXPECT_EQ(int(cosize(composition(Swizzle<2,0,-2>{},
                                   Layout<Shape<_4,_4>,Stride<_4,_1>>{}))),  16);
  EXPECT_EQ(int(cosize(composition(Swizzle<1,4,3>{},
                                   Layout<Shape<_16,_8>,Stride<_1,_16>>{}))), 128);
}

// coshape gives the SPAN of the codomain. Where a stride is negative the codomain
// does not start at zero, and co_min gives that start.
TEST(CuTe_core, CoMinNegativeStride)
{
  EXPECT_EQ(int(co_min(Layout<Shape<_3>,Stride<Int<-3>>>{})), -6);
  EXPECT_EQ(int(cosize(Layout<Shape<_3>,Stride<Int<-3>>>{})),  7);
  test_cosize_bounds(Layout<Shape<_3>,Stride<Int<-3>>>{}, 7);

  EXPECT_EQ(int(co_min(Layout<Shape<_2>,Stride<Int<-8>>>{})), -8);
  EXPECT_EQ(int(cosize(Layout<Shape<_2>,Stride<Int<-8>>>{})),  9);
  test_cosize_bounds(Layout<Shape<_2>,Stride<Int<-8>>>{}, 9);

  // A layout with no negative stride has a codomain that starts at zero.
  EXPECT_EQ(int(co_min(Layout<Shape<_4,_8>,Stride<_1,_4>>{})), 0);
  test_cosize_bounds(Layout<Shape<_4,_8>,Stride<_1,_4>>{}, 32);
}

// co_min keeps the basis of a ScaledBasis stride, as abs does.
TEST(CuTe_core, CoMinScaledBasis)
{
  auto l = make_layout(make_shape (Int<2>{}, Int<2>{}),
                       make_stride(E<0>{}, E<1>{} * Int<-4>{}));
  EXPECT_TRUE(bool(co_min(l) == make_arithmetic_tuple(Int<0>{}, Int<-4>{})));
  EXPECT_TRUE(bool(coshape(l) == make_arithmetic_tuple(Int<2>{}, Int<5>{})));
}
