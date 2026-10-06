#include "gemqc/hessian.h"

#include <stdexcept>

#include "gemqc/resources.h"

namespace gemqc {

namespace {
// irrep of a product from the characters (abelian groups: 1D irreps)
int irrep_product(const std::vector<std::vector<double>>& chars, int a, int b) {
  if (chars.empty()) return 0;
  for (size_t k = 0; k < chars.size(); ++k) {
    bool match = true;
    for (size_t g = 0; g < chars[k].size() && match; ++g)
      match = chars[k][g] == chars[a][g] * chars[b][g];
    if (match) return int(k);
  }
  throw std::runtime_error("irrep product not found in character table");
}
}  // namespace

Pairs rotation_pairs(const std::vector<bool>& occ, const std::vector<int>& irreps,
                     const std::vector<std::vector<double>>& characters, int irrep) {
  Pairs p;
  for (size_t a = 0; a < occ.size(); ++a)
    for (size_t i = 0; i < occ.size(); ++i)
      if (!occ[a] && occ[i] && irrep_product(characters, irreps[a], irreps[i]) == irrep)
        p.push_back({Eigen::Index(a), Eigen::Index(i)});
  return p;
}

Matrix pairs_to_matrix(const Pairs& pairs, const Eigen::VectorXd& v, Eigen::Index nmo) {
  Matrix K = Matrix::Zero(nmo, nmo);
  for (size_t q = 0; q < pairs.size(); ++q) K(pairs[q].first, pairs[q].second) = v(q);
  return K;
}

Eigen::VectorXd matrix_to_pairs(const Pairs& pairs, const Matrix& M) {
  Eigen::VectorXd v(pairs.size());
  for (size_t q = 0; q < pairs.size(); ++q) v(q) = M(pairs[q].first, pairs[q].second);
  return v;
}

namespace {
// J and K for densities, in batches that fit into memory ((anti)symmetry detected)
JK batched_jk(JKBuilder& fb, const std::vector<Matrix>& dens, const std::vector<int>& irreps) {
  JK out;
  size_t batch = std::max<size_t>(1, dens.size());
  while (batch > 1 && !resources::fits(fb.memory_estimate(batch, true, 1))) batch /= 2;
  for (size_t b0 = 0; b0 < dens.size(); b0 += batch) {
    const size_t nb = std::min(batch, dens.size() - b0);
    const std::vector<Matrix> d(dens.begin() + b0, dens.begin() + b0 + nb);
    const std::vector<int> ir(irreps.begin() + b0, irreps.begin() + b0 + nb);
    auto jk = fb.compute(d, false, true, ir);
    for (size_t k = 0; k < nb; ++k) out.J.push_back(jk.J[k]), out.K.push_back(jk.K[k]);
  }
  return out;
}
}  // namespace

std::vector<Eigen::VectorXd> hessian_products(JKBuilder& fb, const Matrix& C, const Matrix& Fmo,
                                              const std::vector<HessianRequest>& requests) {
  const Eigen::Index nmo = C.cols();
  // pseudo-densities: real rotations X + X^T, imaginary ones X - X^T
  std::vector<Matrix> dens;
  std::vector<int> irreps;
  for (const auto& r : requests) {
    const Matrix X = C * pairs_to_matrix(*r.pairs, r.v, nmo) * C.transpose();
    dens.push_back(r.type == HessianType::complex ? Matrix(X - X.transpose())
                                                  : Matrix(X + X.transpose()));
    irreps.push_back(r.irrep);
  }
  const JK jk = batched_jk(fb, dens, irreps);
  const auto& J = jk.J;
  const auto& K = jk.K;
  std::vector<Eigen::VectorXd> sigma;
  for (size_t k = 0; k < requests.size(); ++k) {
    const auto& r = requests[k];
    const Matrix G = r.type == HessianType::singlet ? Matrix(2.0 * J[k] - K[k]) : Matrix(-K[k]);
    const Matrix Kv = pairs_to_matrix(*r.pairs, r.v, nmo);
    sigma.push_back(matrix_to_pairs(*r.pairs, Fmo * Kv - Kv * Fmo + C.transpose() * G * C));
  }
  return sigma;
}

std::vector<Eigen::VectorXd> uhf_hessian_products(JKBuilder& fb, const Matrix* C,
                                                  const Matrix* Fmo,
                                                  const std::vector<UhfHessianRequest>& requests) {
  const Eigen::Index nmo = C[0].cols();
  // two pseudo-densities per request (alpha, beta)
  std::vector<Matrix> dens;
  std::vector<int> irreps;
  for (const auto& r : requests) {
    const Eigen::Index na = r.pairs[0]->size();
    for (int s = 0; s < 2; ++s) {
      const Eigen::VectorXd vs = s == 0 ? r.v.head(na) : r.v.tail(r.v.size() - na);
      const Matrix X = C[s] * pairs_to_matrix(*r.pairs[s], vs, nmo) * C[s].transpose();
      dens.push_back(r.complex ? Matrix(X - X.transpose()) : Matrix(X + X.transpose()));
      irreps.push_back(r.irrep);
    }
  }
  const JK jk = batched_jk(fb, dens, irreps);
  std::vector<Eigen::VectorXd> sigma;
  for (size_t k = 0; k < requests.size(); ++k) {
    const auto& r = requests[k];
    const Eigen::Index na = r.pairs[0]->size();
    Eigen::VectorXd out(r.v.size());
    for (int s = 0; s < 2; ++s) {
      const Eigen::VectorXd vs = s == 0 ? r.v.head(na) : r.v.tail(r.v.size() - na);
      const Matrix G = r.complex ? Matrix(-jk.K[2 * k + s])
                                 : Matrix(jk.J[2 * k] + jk.J[2 * k + 1] - jk.K[2 * k + s]);
      const Matrix Kv = pairs_to_matrix(*r.pairs[s], vs, nmo);
      const Eigen::VectorXd o =
          matrix_to_pairs(*r.pairs[s], Fmo[s] * Kv - Kv * Fmo[s] + C[s].transpose() * G * C[s]);
      if (s == 0) out.head(na) = o;
      else out.tail(out.size() - na) = o;
    }
    sigma.push_back(out);
  }
  return sigma;
}

Eigen::VectorXd energy_hessian_product(JKBuilder& fb, const std::vector<Matrix>& C,
                                       const std::vector<Matrix>& Fmo,
                                       const std::vector<Pairs>& pairs, const Eigen::VectorXd& v) {
  if (C.size() == 1)
    return 4.0 * hessian_products(fb, C[0], Fmo[0], {{HessianType::singlet, 0, &pairs[0], v}})[0];
  return 2.0 * uhf_hessian_products(fb, C.data(), Fmo.data(),
                                    {{false, 0, {&pairs[0], &pairs[1]}, v}})[0];
}

}  // namespace gemqc
