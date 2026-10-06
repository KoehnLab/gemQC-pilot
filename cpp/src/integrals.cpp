#include "gemqc/integrals.h"

#include <libint2.hpp>

#include <algorithm>
#include <mutex>
#include <vector>

namespace gemqc {

void init_libint() {
  static std::once_flag flag;
  std::call_once(flag, [] { libint2::initialize(); });
}

namespace {

// All shell-pair blocks <b1|op|b2> for the nop operators of the engine;
// if b1 and b2 are the same object, only s2 <= s1 is computed (symmetric)
std::vector<Matrix> compute_1body(const BasisSet& b1, const BasisSet& b2,
                                  libint2::Engine& engine, size_t nop) {
  std::vector<Matrix> mats(nop, Matrix::Zero(b1.nbf(), b2.nbf()));
  const bool same = &b1 == &b2;
  const auto& sh1 = b1.shells();
  const auto& sh2 = b2.shells();
  const auto& off1 = b1.shell_offsets();
  const auto& off2 = b2.shell_offsets();
  const auto& buf = engine.results();

  for (size_t s1 = 0; s1 < sh1.size(); ++s1) {
    const auto n1 = sh1[s1].size();
    for (size_t s2 = 0; s2 < (same ? s1 + 1 : sh2.size()); ++s2) {
      const auto n2 = sh2[s2].size();
      engine.compute(sh1[s1], sh2[s2]);
      for (size_t op = 0; op < nop; ++op) {
        if (buf[op] == nullptr) continue;  // screened out
        Eigen::Map<const Matrix> blk(buf[op], n1, n2);
        mats[op].block(off1[s1], off2[s2], n1, n2) = blk;
        if (same && s1 != s2)
          mats[op].block(off2[s2], off1[s1], n2, n1) = blk.transpose();
      }
    }
  }
  return mats;
}

// engine sized for shells of both bases
libint2::Engine make_engine(const BasisSet& b1, const BasisSet& b2,
                            libint2::Operator op) {
  init_libint();
  return libint2::Engine(op, std::max(b1.max_nprim(), b2.max_nprim()),
                         std::max(b1.max_l(), b2.max_l()), 0);
}

}  // namespace

Matrix overlap(const BasisSet& b1, const BasisSet& b2) {
  auto engine = make_engine(b1, b2, libint2::Operator::overlap);
  return compute_1body(b1, b2, engine, 1)[0];
}

Matrix kinetic(const BasisSet& b1, const BasisSet& b2) {
  auto engine = make_engine(b1, b2, libint2::Operator::kinetic);
  return compute_1body(b1, b2, engine, 1)[0];
}

Matrix nuclear_attraction(const BasisSet& b1, const BasisSet& b2,
                          const Molecule& mol) {
  auto engine = make_engine(b1, b2, libint2::Operator::nuclear);
  std::vector<std::pair<double, std::array<double, 3>>> charges;
  for (const auto& a : mol.atoms()) charges.push_back({double(a.z), a.r});
  engine.set_params(charges);
  return compute_1body(b1, b2, engine, 1)[0];
}

std::array<Matrix, 3> dipole(const BasisSet& b1, const BasisSet& b2,
                             const std::array<double, 3>& origin) {
  auto engine = make_engine(b1, b2, libint2::Operator::emultipole1);
  engine.set_params(origin);
  auto m = compute_1body(b1, b2, engine, 4);  // [S, x, y, z]
  return {m[1], m[2], m[3]};
}

std::array<Matrix, 6> second_moments(const BasisSet& b, const std::array<double, 3>& origin) {
  auto engine = make_engine(b, b, libint2::Operator::emultipole2);
  engine.set_params(origin);
  auto m = compute_1body(b, b, engine, 10);  // [S, x, y, z, xx, xy, xz, yy, yz, zz]
  return {m[4], m[5], m[6], m[7], m[8], m[9]};
}

// single-basis versions (square, symmetric)
Matrix overlap(const BasisSet& b) { return overlap(b, b); }
Matrix kinetic(const BasisSet& b) { return kinetic(b, b); }
Matrix nuclear_attraction(const BasisSet& b, const Molecule& mol) {
  return nuclear_attraction(b, b, mol);
}
std::array<Matrix, 3> dipole(const BasisSet& b,
                             const std::array<double, 3>& origin) {
  return dipole(b, b, origin);
}

}  // namespace gemqc
