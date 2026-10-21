# Theory notes

Atomic units throughout; $h$ is the one-electron Hamiltonian, $J[D]$ and $K[D]$ are the
Coulomb and exchange matrices of an AO density $D$ (direct builds from screened
4-index integrals).

## Open-shell SCF in general form

ROHF uses the general open-shell energy (Roothaan, McWeeny) with shells $k$
(closed, open) of occupation fractions $f_k$ and coupling coefficients
$a_{kl}$, $b_{kl}$, $D_k$ the density of the orbitals of shell $k$:

$$
E = \sum_k 2 f_k \operatorname{tr}(D_k h)
  + \sum_{kl} f_k f_l \operatorname{tr}\!\left(D_k (2a_{kl} J[D_l] - b_{kl} K[D_l])\right)
  + E_\mathrm{nuc} .
$$

The shell Fock matrices are $F_k = f_k\,(h + \sum_l f_l (2a_{kl}J_l - b_{kl}K_l))$ and
the orbital gradient between shells $k$ and $l$ is $4(F_k - F_l)_{pq}$. DIIS works on
the Guest-Saunders effective Fock matrix (diagonal blocks the total Fock matrix,
off-diagonal blocks $(F_k - F_l)/(f_k - f_l)$), QN on the rotations between all
shells. Canonical orbital convention: closed shells from
$(F_\alpha + F_\beta)/2$, open shells and virtuals from $F_\alpha$ (i.e. $F_k/f_k$).

High-spin ROHF: closed ($f = 1$) and open shells ($f = 1/2$, $a = 1$, $b = 2$ among
open shells).

## Density fitting

With the Coulomb metric $V_{PQ} = (P|Q)$ of the fitting functions,
$(\mu\nu|\lambda\sigma) \approx \sum_{PQ} (\mu\nu|P)\,[V^{-1}]_{PQ}\,(Q|\lambda\sigma)$ and
$V^{-1} = M M^T$ from the eigenvectors of $V$ (eigenvalues below $10^{-10}$ of the
largest dropped). Then

$$
J[D] = \sum_P (\mu\nu|P)\, c_P,\quad c = V^{-1}\gamma,\quad \gamma_P = \sum_{\mu\nu}(P|\mu\nu) D_{\mu\nu};
\qquad
K[D] = \sum_Q \sum_k s_k\, X^Q_{\cdot k} (Y^Q_{\cdot k})^T,
$$

with $D = \sum_k s_k a_k b_k^T$ (eigen-decomposition, or SVD for non-symmetric
densities) and $X^Q_{\mu k} = \sum_P M_{PQ} (P|\mu\nu) a_{\nu k}$ ($Y$ with $b$). The cost
of K is proportional to the rank of $D$ (e.g. the number of occupied orbitals).

## Properties

Moments refer to the centre of mass; the traceless quadrupole is
$\Theta_{ab} = \tfrac12 \sum_i q_i (3 r_{ia} r_{ib} - r_i^2 \delta_{ab})$.

## MP2-F12 with fixed amplitudes

Orbitals: $i, j, k, l, m, n$ active occupied, $o$ all occupied (frozen core
included), $a, b$ virtual, $p, q, r, s$ orbital basis, $x, y$ CABS, $p'$ all of them.
The geminal is $F = f_{12} = -e^{-\gamma r_{12}}/\gamma$ (exact, or STG-nG:
$e^{-\gamma r} \approx \sum_i c_i e^{-a_i r^2}$), $F^{kl}_{p'q'} = \langle kl|F|p'q'\rangle$.
Ansatz 3: $Q_{12} = (1 - o_1)(1 - o_2) - v_1 v_2$; its complement $\hat P = 1 - Q_{12}$
comprises the pairs $(rs)$, $(ox)$, $(xo)$, and the resolution of the identity runs over
$p'$.

**Intermediates** ($kl, mn$ active pairs):

$$
V^{kl}_{ij} = \langle kl|F r_{12}^{-1}|ij\rangle - \sum_{\hat P} F^{kl}_{pq}\, g^{pq}_{ij},\qquad
X^{kl}_{mn} = \langle kl|F^2|mn\rangle - \sum_{\hat P} F^{kl}_{pq} F^{pq}_{mn},
$$

$$
B^{kl}_{mn} = \tfrac12 \hat S\big(A - Z - F^{kl}_{ab} C^{ab}_{mn}\big),\qquad
\hat S M^{kl}_{mn} = M^{kl}_{mn} + M^{mn}_{kl},
$$

$$
A^{kl}_{mn} = U^{kl}_{mn} + [\bar{F^2}]^{kl}_{mn} - \sum_{\hat P} \bar F^{pq}_{kl} F^{pq}_{mn}
- \tilde Y^{kl}_{mn},
$$

- $U^{kl}_{mn} = \langle kl|(\nabla_1 F)^2|mn\rangle = \tfrac12\langle kl|[F,[\hat T_1 + \hat T_2, F]]|mn\rangle$
  (libint `delcgtg2`, or $e^{-2\gamma r_{12}}$ for the exact Slater function);
- bars: transformation with the Hartree operator $h + J = f + K$ over all $p'$,
  $[\bar{F^2}]^{kl}_{mn} = \sum_{p'} (h+J)_{kp'} \langle p'l|F^2|mn\rangle + (h+J)_{lp'}
  \langle kp'|F^2|mn\rangle$, $\bar F^{pq}_{kl} = \sum_{p'} (h+J)_{pp'} F^{kl}_{p'q} +
  (h+J)_{qp'} F^{kl}_{pp'}$;
- $\tilde Y^{kl}_{mn} = \sum_{Q} \tilde F^{kl}_{\alpha\beta} F^{\alpha\beta}_{mn}$ over the
  $Q_{12}$ pairs $\alpha\beta = (xa), (ax), (xy)$, with $\tilde F = K F + F K$ (exchange
  terms of the double RI expansion);
- $Z^{kl}_{mn} = \sum_o (F^{kl}_{\check o x} F^{ox}_{mn} + F^{kl}_{x\check o} F^{xo}_{mn} +
  F^{kl}_{\check o a} F^{oa}_{mn} + F^{kl}_{a\check o} F^{ao}_{mn})$, $F^{kl}_{\check o x} =
  \sum_y F^{kl}_{yx} f_{yo}$ etc. (occupied–CABS Fock elements);
- $C^{ab}_{mn} = \sum_x f_{ax} F^{xb}_{mn} + F^{ax}_{mn} f_{bx}$.

Approximations (Molpro names): **3C** all terms;
**3\*C** without the $F\,C$ term in $B$ and without the coupling below; **NOZ** omits
$Z$ (generalized Brillouin condition in $B$); **HY1** omits $\tilde Y$.

**Energy** (closed shell; fixed amplitudes $T^{ij}_{ij} = 3/8$, $T^{ij}_{ji} = 1/8$, i.e.
1/2 for singlet and 1/4 for triplet pairs; $\tilde T = 2T - T^{\mathrm{swap}}$):

$$
E_{\mathrm{F12}} = \sum_{ij}\Big[2\sum_{kl} \tilde T^{ij}_{kl} V^{kl}_{ij}
 + \sum_{kl,mn} \tilde T^{ij}_{kl}\big(B^{kl}_{mn} - (\varepsilon_i + \varepsilon_j) X^{kl}_{mn}\big) T^{ij}_{mn}\Big].
$$

Singlet and triplet parts follow from the symmetric ($\tilde T = T$) and antisymmetric
($\tilde T = 3T$) amplitude components. In 3C the conventional amplitudes couple to the
geminals: $T^{ab}_{ij} = (K + Cc)^{ab}_{ij} / (\varepsilon_i + \varepsilon_j - \varepsilon_a -
\varepsilon_b)$, $(Cc)^{ab}_{ij} = \sum_{kl} C^{ab}_{kl} T^{ij}_{kl}$, and the MP2 part
becomes $\sum \tilde T^{ab}_{ij} (K + Cc)^{ab}_{ij}$; its change is part of the 3C
correction. The CABS singles correction is added to the reference energy.

**Density fitting.** With the Coulomb fit coefficients $d^P_{kp'} = \sum_Q [J^{-1}]_{PQ}
(Q|kp')$, the geminal integrals are fitted robustly,
$(kp'|g|lq') = d_{kp'}^T (P|g|lq') + (P|g|kp')^T d_{lq'} - d_{kp'}^T (P|g|Q) d_{lq'}$
(Coulomb: $d^T (P|lq')$). With exact integrals, $\langle ij|g|p'q'\rangle = C^T K^{ij} C$ from
the exchange-type matrices $K^{ij}_{\mu\lambda} = (\mu i|g|\lambda j)$ of the pair densities
over the RI basis (one integral pass per operator). Only the pairs $kl, mn \in \{ij, ji\}$ are needed; the pair
matrices $F^{ij}_{p'q'}$ are assembled per pair from the 3-index quantities. With
STG-6G, the JK fitting basis as CABS basis and DF-JK fitting basis, DF-MP2-F12
energies of water agree with the Molpro implementation to $10^{-11}$ Eh for 3C, 3\*C, NOZ and HY1.

### Extended geminals (F12XG pilot)

The correlation factor depends on the basis functions of the geminal orbitals: with
basis-function groups $G$ and one exponent $\tilde\gamma_G$ each,
$f_{GH} = -e^{-\gamma_{GH} r_{12}}/\gamma_{GH}$, $\gamma_{GH} = \frac12(\tilde\gamma_G +
\tilde\gamma_H)$, and the group-masked orbitals $\phi_i^G = \sum_{\mu \in G} \chi_\mu
C_{\mu i}$,

$$
g_{ij}(1,2) = \sum_{\mu\nu} f_{\mu\nu}(r_{12})\,\chi_\mu(1)\chi_\nu(2)\,C_{\mu i}C_{\nu j}
 = \sum_{GH} f_{GH}(r_{12})\,\phi_i^G(1)\,\phi_j^H(2).
$$

All formulas above hold with $F|ij\rangle$ replaced by $|g_{ij}\rangle$; every integral is a
sum over group pairs of standard integrals with one operator and masked orbitals:

- $F^{ij}_{p'q'} = \sum_{GH} \langle p'q'|f_{GH}|i^G j^H\rangle$, the $V$ term
  $\sum_{GH} \langle kl|f_{GH} r_{12}^{-1}|i^G j^H\rangle$;
- $\langle g_{kl}|g_{mn}\rangle = \sum_{G'H'GH} \langle k^{G'} l^{H'}|f_{G'H'} f_{GH}|m^G
  n^H\rangle$ (a contracted Gaussian geminal again, or $e^{-(\gamma_a+\gamma_b) r_{12}}$
  exactly);
- $U$: $\langle k^{G'} l^{H'}|(\nabla_1 f_{G'H'})\cdot(\nabla_1 f_{GH})|m^G n^H\rangle$
  (libint `delcgtga_delcgtgb`; exactly $e^{-(\gamma_a+\gamma_b) r_{12}}$);
- $[\bar{F^2}]$: $h + J$ acts on the masked orbital, $\sum_{G'H'GH} \langle (h+J)k^{G'}\,
  l^{H'}|f_{G'H'} f_{GH}|m^G n^H\rangle + (\text{electron } 2)$, with $(h+J)\phi_k^{G'}$ in
  the RI basis. The operator of the transformed index is fixed by its untransformed
  group $G'$ (the five-index structure of the integrals).

Approximation C does not use $f\phi_i = \varepsilon_i\phi_i$ for the geminal orbitals, so
the masked (non-canonical) orbitals need no other change there. Since $f'_{GH}(0) = 1$ for
every pair, the fixed amplitudes still satisfy the cusp conditions.

Kinetic term with different bra and ket geminals $f_a = f_{G'H'}$, $f_b = f_{GH}$: the
rewriting $f T f = \frac12[f,[T,f]] + \frac12(f^2 T + T f^2)$ gives for $f_a \ne f_b$ only
the symmetric part $\frac12(f_a T f_b + f_b T f_a)$; the remainder $\frac12(f_a T f_b -
f_b T f_a)$ does not cancel in the sum over the group pairs when an orbital has components
in both groups. For Slater functions, $f_a\nabla f_b - f_b\nabla f_a = (1/\gamma_b -
1/\gamma_a)\,\nabla\big(\gamma_a\gamma_b f_a f_b/(\gamma_a+\gamma_b)\big)$, hence

$$
f_a T f_b = \tfrac12[f_a,[T,f_b]] + \frac{\gamma_b\, f_a f_b T + \gamma_a\, T f_a f_b}
{\gamma_a + \gamma_b},
$$

with $T$ acting to the right or on the bra. The local parts of $h + J$ cancel in the
antisymmetric remainder, so in $[\bar{F^2}]$ the term with $h + J$ on the bra orbitals
(geminal $f_a$) carries the weight $2\gamma_b/(\gamma_a+\gamma_b)$ before the
symmetrization $\hat S$ (1 for equal geminals; nominal exponents for fitted geminals,
for which the relation holds to the accuracy of the fit).

With density fitting, each group-pair term has a fixed operator, so the robust formula
applies term by term with the fit coefficients of the masked densities,
$d_{k^G p'} = \sum_Q [J^{-1}]_{PQ} (Q|k^G p')$:
$(k^G p'|f_{GH}|l^H q') \approx d_{k^G p'}^T \tilde G^{GH}_{l^H q'} + (\tilde G^{GH}_{k^G p'})^T
d_{l^H q'}$, $\tilde G^{GH}_{k^G} = (P|f_{GH}|k^G p') - \frac12 (P|f_{GH}|Q)\, d_{k^G}$. The
group of the other electron only selects the operator. The coefficients are linear in
the orbitals, so those of doubly masked and $(h+J)$-transformed densities are
contractions of $d_{k^G}$; for equal exponents the sum over the groups is standard
DF-MP2-F12 exactly.

## Orbital Hessian and Newton

For real rotations $\kappa_{ai}$ the RHF energy Hessian is $4(A+B)$ with gradient
$4F_{ai}$; the UHF Hessian is $2(A+B)$ with gradient $2F$. Products with $A \pm B$ are
computed directly from Fock builds of transition densities (batched). The Newton solver takes augmented-Hessian steps, i.e. the
lowest eigenvector of
$\begin{pmatrix} 0 & g^T \\ g & H \end{pmatrix}$, in a trust region.

