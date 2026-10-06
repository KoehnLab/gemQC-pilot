#pragma once
/// \file
/// Values of the basis functions at points in space (e.g. for cube files).
/// Normalization and function ordering as in the integrals (libint).

#include "gemqc/basis.h"
#include "gemqc/integrals.h"

namespace gemqc {

/// points: (npoints, 3) in bohr; result (npoints, nbf)
Matrix basis_values(const BasisSet& basis, const Matrix& points);

}  // namespace gemqc
