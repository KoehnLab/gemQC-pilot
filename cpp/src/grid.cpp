// Basis function values on points: contracted Cartesian Gaussians with libint's
// normalization (all components normalized like x^l), spherical functions by
// libint's solid-harmonic coefficients (m = -l..l).

#include "gemqc/grid.h"

#include <libint2/solidharmonics.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace gemqc {

namespace {

constexpr double kExpCut = 46.0;  // exp(-46) ~ 1e-20: primitive negligible

// Cartesian components x^i y^j z^k in libint (CCA) order
std::vector<std::array<int, 3>> cartesian_powers(int l) {
  std::vector<std::array<int, 3>> p;
  for (int i = l; i >= 0; --i)
    for (int j = l - i; j >= 0; --j) p.push_back({i, j, l - i - j});
  return p;
}

}  // namespace

Matrix basis_values(const BasisSet& basis, const Matrix& points) {
  init_libint();
  const Eigen::Index np = points.rows();
  Matrix v = Matrix::Zero(np, basis.nbf());
  const auto& shells = basis.shells();
  const auto& offsets = basis.shell_offsets();
  // per shell: Cartesian powers and minimal exponent (for screening)
  std::vector<std::vector<std::array<int, 3>>> powers(shells.size());
  std::vector<double> amin(shells.size());
  for (size_t s = 0; s < shells.size(); ++s) {
    powers[s] = cartesian_powers(shells[s].contr[0].l);
    amin[s] = *std::min_element(shells[s].alpha.begin(), shells[s].alpha.end());
  }

#pragma omp parallel
  {
    std::vector<double> cart, xp(1), yp(1), zp(1);
#pragma omp for schedule(dynamic, 64)
    for (Eigen::Index p = 0; p < np; ++p) {
      for (size_t s = 0; s < shells.size(); ++s) {
        const auto& sh = shells[s];
        const double x = points(p, 0) - sh.O[0], y = points(p, 1) - sh.O[1],
                     z = points(p, 2) - sh.O[2];
        const double r2 = x * x + y * y + z * z;
        if (amin[s] * r2 > kExpCut) continue;
        double radial = 0.0;  // contracted radial part
        for (size_t k = 0; k < sh.alpha.size(); ++k) {
          const double e = sh.alpha[k] * r2;
          if (e < kExpCut) radial += sh.contr[0].coeff[k] * std::exp(-e);
        }
        const int l = sh.contr[0].l;
        xp.resize(l + 1), yp.resize(l + 1), zp.resize(l + 1);
        xp[0] = yp[0] = zp[0] = 1.0;
        for (int i = 1; i <= l; ++i)
          xp[i] = xp[i - 1] * x, yp[i] = yp[i - 1] * y, zp[i] = zp[i - 1] * z;
        const auto& pw = powers[s];
        cart.resize(pw.size());
        for (size_t c = 0; c < pw.size(); ++c)
          cart[c] = radial * xp[pw[c][0]] * yp[pw[c][1]] * zp[pw[c][2]];
        double* out = &v(p, offsets[s]);
        if (!sh.contr[0].pure || l == 0) {
          for (size_t c = 0; c < cart.size(); ++c) out[c] = cart[c];
          continue;
        }
        const auto& coef = libint2::solidharmonics::SolidHarmonicsCoefficients<double>::instance(l);
        for (int m = 0; m < 2 * l + 1; ++m) {
          const auto* idx = coef.row_idx(m);
          const auto* val = coef.row_values(m);
          double sum = 0.0;
          for (unsigned k = 0; k < coef.nnz(m); ++k) sum += val[k] * cart[idx[k]];
          out[m] = sum;
        }
      }
    }
  }
  return v;
}

}  // namespace gemqc
