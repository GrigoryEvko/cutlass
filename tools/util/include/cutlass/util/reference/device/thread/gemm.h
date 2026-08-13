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
    \brief Reference implementation for GEMM in host-side code.
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_view.h"
#include "cutlass/gemm/gemm.h"

namespace cutlass {
namespace reference {
namespace device {
namespace thread {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Thread-level blocked general matrix product.
//
// Note, this is a reference implementation. Performance is not expected to approach peak.
//
template <
  typename TensorRefA,
  typename TensorRefB,
  typename TensorRefC,
  typename ScalarType,
  typename AccumulatorType,
  typename OutputTile,
  typename InnerProductOp = multiply_add<AccumulatorType>,
  typename ConvertOp = NumericConverter<typename TensorRefC::Element, ScalarType>
>
struct Gemm {

  using ElementA = typename TensorRefA::Element;
  using ElementB = typename TensorRefB::Element;
  using ElementC = typename TensorRefC::Element;

  //
  // Data members
  //

  // The thread covers OutputTile::kRow rows and OutputTile::kColumn columns of
  // the output. kernel::Gemm gives each thread the same extent, thus the two
  // files must use the same convention. The first index is always the row.

  /// Tile for A operand. The index is the row offset in the output tile.
  ElementA A_tile[OutputTile::kRow];

  /// Tile for B operand. The index is the column offset in the output tile.
  ElementB B_tile[OutputTile::kColumn];

  /// Tile for Accumulator. The first index is the row offset.
  AccumulatorType accum[OutputTile::kRow][OutputTile::kColumn];

  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Gemm(AccumulatorType initial_accum = AccumulatorType(0)) {

    // Clear fetch registers
    for (int i = 0; i < OutputTile::kRow; ++i) {
      A_tile[i] = ElementA(0);
    }

    for (int j = 0; j < OutputTile::kColumn; ++j) {
      B_tile[j] = ElementB(0);
    }

    // Clear accumulators
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < OutputTile::kRow; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < OutputTile::kColumn; ++j) {
        accum[i][j] = initial_accum;
      }
    }
  }

  /// Computes a matrix product
  CUTLASS_HOST_DEVICE
  Gemm & multiply_add(
    gemm::GemmCoord problem_size,
    TensorRefA tensor_a,
    TensorRefB tensor_b,
    MatrixCoord output_coord = MatrixCoord()) {

    InnerProductOp inner_product_op;

    // Loop over the GEMM K dimension
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < problem_size.k(); ++k) {

      // Fetch a slice of the A matrix
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < OutputTile::kRow; ++i) {
        if (output_coord.row() + i < problem_size.m()) {
          A_tile[i] = tensor_a.at(make_Coord(output_coord.row() + i, k));
        }
      }

      // Fetch a slice of the B matrix
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < OutputTile::kColumn; ++j) {
        if (output_coord.column() + j < problem_size.n()) {
          B_tile[j] = tensor_b.at(make_Coord(k, output_coord.column() + j));
        }
      }

      // Compute an accumulated matrix product
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < OutputTile::kRow; ++i) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < OutputTile::kColumn; ++j) {
          accum[i][j] = inner_product_op(A_tile[i], B_tile[j], accum[i][j]);
        }
      }
    }

    return *this;
  }

  /// Performs linear scaling of matrix product and updates output tensor
  CUTLASS_HOST_DEVICE
  Gemm & epilogue(
    gemm::GemmCoord problem_size,
    ScalarType alpha,
    ScalarType beta,
    TensorRefC tensor_c,
    TensorRefC tensor_d,
    MatrixCoord output_coord = MatrixCoord()) {

    ConvertOp convert_op;
    
    // Update the output tensor
    for (int i = 0; i < OutputTile::kRow; ++i) {
      for (int j = 0; j < OutputTile::kColumn; ++j) {
        MatrixCoord coord = output_coord + MatrixCoord(i, j);
        if (coord.row() < problem_size.m() && coord.column() < problem_size.n()) {

          tensor_d.at(coord) = convert_op(
            alpha * ScalarType(accum[i][j]) +
            beta * ScalarType(tensor_c.at(coord))
          );
        }
      }
    }

    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace device
} // namespace reference
} // namespace cutlass
