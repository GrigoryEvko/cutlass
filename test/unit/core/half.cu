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
/*! \file
    \brief Statically sized array of elements that accommodates all CUTLASS-supported numeric types
           and is safe to use in a union.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/device_memory.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(half_t, host_conversion) {
  for (int i = -1024; i < 1024; ++i) {
    float f = static_cast<float>(i);

    cutlass::half_t x = static_cast<cutlass::half_t>(i);
    cutlass::half_t y = static_cast<cutlass::half_t>(f);

    EXPECT_TRUE(static_cast<int>(x) == i);
    EXPECT_TRUE(static_cast<float>(y) == f);
  }

  // Try out default-ctor (zero initialization of primitive proxy type)
  EXPECT_TRUE(cutlass::half_t() == 0.0_hf);

  // Try out user-defined literals
  EXPECT_TRUE(cutlass::half_t(7) == 7_hf);
  EXPECT_TRUE(7 == static_cast<int>(7_hf));
}

TEST(half_t, host_arithmetic) {

  for (int i = -100; i < 100; ++i) {
    for (int j = -100; j < 100; ++j) {

      cutlass::half_t x = static_cast<cutlass::half_t>(i);
      cutlass::half_t y = static_cast<cutlass::half_t>(j);

      EXPECT_TRUE(static_cast<int>(x + y) == (i + j));
    }
  }

  for (int i = -6; i < 6; ++i) {
    for (int j = -6; j < 6; ++j) {

      cutlass::half_t x = static_cast<cutlass::half_t>(i);
      cutlass::half_t y = static_cast<cutlass::half_t>(j);

      EXPECT_TRUE(static_cast<int>(x * y) == (i * j));
    }
  }
}

TEST(half_t, host_round_toward_zero) {

  // Round toward zero must not increase a magnitude. Thus an overflow gives
  // 0x7bff, which is the largest finite half_t. IEEE 754-2019 clause 7.4 gives
  // this result, and __float2half_rz gives it too. Each value below comes from
  // __float2half_rz on an sm_120 device.
  struct {
    uint32_t f32_bits;
    uint16_t expected;
  } tests[] = {
    {0x477fffff, 0x7bff},  // 65535.996, the largest float that half_t holds
    {0x47800000, 0x7bff},  // 65536, the smallest float that overflows half_t
    {0xc7800000, 0xfbff},  // -65536
    {0x7f7fffff, 0x7bff},  // the largest finite float
    {0xff7fffff, 0xfbff},  // the smallest finite float
    {0x7f800000, 0x7c00},  // +infinity in, +infinity out
    {0xff800000, 0xfc00},  // -infinity in, -infinity out
    {0x7fc00000, 0x7fff},  // NaN gives the canonical NaN
    {0xffc00000, 0x7fff},  // a negative NaN gives the canonical NaN
    {0x3fc00000, 0x3e00},  // 1.5, which half_t holds exactly
    {0x00000000, 0x0000}   // the end of the list, and zero gives zero
  };

  int const count = int(sizeof(tests) / sizeof(tests[0]));

  cutlass::NumericConverter<cutlass::half_t, float,
                            cutlass::FloatRoundStyle::round_toward_zero> convert;

  for (int i = 0; i < count; ++i) {

    // A read of the table through a float reference breaks the aliasing rules
    // of the language, thus this test copies the bits.
    float f32;
    std::memcpy(&f32, &tests[i].f32_bits, sizeof(f32));

    cutlass::half_t f16 = convert(f32);

    EXPECT_TRUE(tests[i].expected == f16.raw())
      << "Error - convert(f32: 0x" << std::hex << tests[i].f32_bits
      << ") -> 0x" << std::hex << tests[i].expected
      << "\ngot: 0x" << std::hex << f16.raw();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
