"""Integral transformation to the correlated orbital spaces (shared by correlation methods).

`MOIntegrals` provides integral classes named by index types over the active occupied
(o) and active virtual (v) orbitals of `correlation.orbital_spaces`, per spin case:
'aa' (closed shells: the only case), 'bb' and 'ab' (open shells; the first index pair
alpha, the second beta). Implemented class: 'ovov' (ia|jb), as needed for MP2.

- Exact integrals: AO exchange matrices K[c_i c_j^T] of occupied orbital pairs
  (JKBuilder.pair_jk, one pass over the shell quartets per batch of pairs, sized by the
  memory left), (ia|jb) = C_a^T K C_b; stored as (o, v, o, v).
- Density fitting: B^Q_ia = sum_P M_PQ (P|ia) per spin (DFBuilder.mo_three_index) are
  stored as (o, v, nfit); (ia|jb) blocks are assembled as B_i B^T on request.

Storage: in memory if the stored arrays fit into MEMORY_FRACTION of the memory limit
(mode 'auto'), else in an HDF5 file in the scratch directory, deleted by close(). The
consumers read blocks of the first occupied index (`blocks`), so the same code serves
both storages.
"""

import time

import numpy as np

from . import _core, scratch

MEMORY_FRACTION = 0.5    # share of the memory limit for the stored arrays
BLOCK_BYTES = 256 << 20  # size of a block of rows i returned by blocks()
DOUBLE = 8


class MOIntegrals:
    """Integrals over the correlated orbital spaces.

    Args:
        spaces: correlation.OrbitalSpaces.
        builder: J/K builder (exact: FockBuilder; density fitting: DFBuilder in the RI
            fitting basis).
        df: Use the 3-index quantities of `builder` (density fitting).
        mode: 'auto', 'memory' or 'disk'.
        block_bytes: Size of the blocks returned by blocks().

    Attributes:
        cases: Spin cases ('aa', or 'aa', 'bb', 'ab').
        on_disk: The arrays are in a scratch file.
        path: Scratch file, or None.
        size: Bytes of the stored arrays.
        time: Seconds for the transformation.
    """

    def __init__(self, spaces, builder, df=False, mode="auto", block_bytes=BLOCK_BYTES):
        t0 = time.time()
        self.spaces, self.df = spaces, df
        self.closed = spaces.alpha is spaces.beta
        self.cases = ("aa",) if self.closed else ("aa", "bb", "ab")
        self.block_bytes = block_bytes
        spin = {"a": spaces.alpha, "b": spaces.beta}
        self._orb = {s: (sp.C[:, sp.occ], sp.C[:, sp.virt]) for s, sp in spin.items()}
        self.spins = ("a",) if self.closed else ("a", "b")
        if df:
            self.size = sum(self._orb[s][0].shape[1] * self._orb[s][1].shape[1]
                            for s in self.spins) * builder.nfit * DOUBLE
        else:
            self.size = sum(int(np.prod(self._shape(c))) for c in self.cases) * DOUBLE
        limit = _core.memory()
        if mode == "auto":
            mode = "memory" if not limit or self.size <= MEMORY_FRACTION * limit else "disk"
        self.on_disk = mode == "disk"
        self.path, self._h5, self._mem = None, None, {}
        if self.on_disk:
            import h5py
            self.path = scratch.new_file("mo_integrals")
            self._h5 = h5py.File(self.path, "w")
        if df:
            for s in self.spins:
                o, v = self._orb[s]
                B = np.asarray(builder.mo_three_index(o, v))  # (nfit, o*v)
                self._create("B" + s, (o.shape[1], v.shape[1], B.shape[0]))
                self._target("B" + s)[...] = B.T.reshape(o.shape[1], v.shape[1], B.shape[0])
        else:
            self._exact(builder)
        self.time = time.time() - t0

    # --- storage ---

    def _shape(self, case):
        """(o1, v1, o2, v2) of the ovov class of a spin case."""
        (o1, v1), (o2, v2) = self._orb[case[0]], self._orb[case[1]]
        return o1.shape[1], v1.shape[1], o2.shape[1], v2.shape[1]

    def _create(self, name, shape):
        if self.on_disk:
            self._h5.create_dataset(name, shape=shape, dtype="f8")
        else:
            self._mem[name] = np.zeros(shape)

    def _target(self, name):
        """The stored array (HDF5 dataset or ndarray; both take slice assignments)."""
        return self._h5[name] if self.on_disk else self._mem[name]

    def _read(self, name, rows):
        return np.asarray(self._target(name)[rows])

    def _rows_per_block(self, row_bytes):
        return max(1, int(self.block_bytes // max(row_bytes, 1)))

    def _exact(self, builder):
        """(ia|jb) of all spin cases from AO exchange matrices of occupied pairs."""
        nbf = self._orb["a"][0].shape[0]
        limit = _core.memory()
        for case in self.cases:
            o1, v1, o2, v2 = self._shape(case)
            self._create("ovov_" + case, (o1, v1, o2, v2))
            if o1 == 0 or o2 == 0 or v1 == 0 or v2 == 0:
                continue
            (Co1, Cv1), (Co2, Cv2) = self._orb[case[0]], self._orb[case[1]]
            same = case[0] == case[1]
            Cocc = Co1 if same else np.hstack([Co1, Co2])
            pairs = ([(i, j) for i in range(o1) for j in range(i, o1)] if same
                     else [(i, o1 + j) for i in range(o1) for j in range(o2)])
            # J and K of each pair (2 nbf^2), plus the transformed block per pair
            per_pair = (2 * nbf * nbf + v1 * v2) * DOUBLE
            free = max(limit - self.size, 0) if limit and not self.on_disk else limit
            nbatch = max(1, int(0.5 * free // per_pair)) if free else len(pairs)
            X = self._target("ovov_" + case)
            for k in range(0, len(pairs), nbatch):
                batch = pairs[k:k + nbatch]
                _, K = builder.pair_jk(Cocc, batch)
                for (i, j), Kij in zip(batch, K):
                    V = Cv1.T @ np.asarray(Kij) @ Cv2  # (i a|j b), (v1, v2)
                    if same:
                        X[i, :, j, :] = V
                        if i != j:  # (j a|i b) = (i b|j a)
                            X[j, :, i, :] = V.T
                    else:
                        X[i, :, j - o1, :] = V

    # --- access ---

    def blocks(self, case="aa"):
        """Blocks of (ia|jb) of a spin case: (i0, i1, X) with X (i1 - i0, v1, o2, v2)."""
        o1, v1, o2, v2 = self._shape(case)
        if self.df:
            Bi = "B" + case[0]
            Bj = self._read("B" + case[1], slice(None)).reshape(o2 * v2, -1)
            nfit = Bj.shape[1]
            step = self._rows_per_block((v1 * o2 * v2 + v1 * nfit) * DOUBLE)
            for i0 in range(0, o1, step):
                i1 = min(o1, i0 + step)
                B = self._read(Bi, slice(i0, i1)).reshape((i1 - i0) * v1, nfit)
                yield i0, i1, (B @ Bj.T).reshape(i1 - i0, v1, o2, v2)
            return
        step = self._rows_per_block(v1 * o2 * v2 * DOUBLE)
        for i0 in range(0, o1, step):
            i1 = min(o1, i0 + step)
            yield i0, i1, np.asarray(self._read("ovov_" + case, slice(i0, i1)))

    def describe(self):
        """One line on the integrals and their storage."""
        kind = "density-fitted 3-index B^Q_ia" if self.df else "(ia|jb)"
        where = f"on disk ({self.path})" if self.on_disk else "in memory"
        return (f"{kind}, {_core.format_bytes(self.size)} {where}, transformation "
                f"{self.time:.2f} s")

    def close(self):
        """Release the arrays; delete the scratch file."""
        self._mem = {}
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self.path.unlink(missing_ok=True)


class ThreeIndexStore:
    """Named 3-index arrays (n, naux, m), indexed by an orbital in the first dimension, in
    memory or in an HDF5 scratch file (same rule as MOIntegrals: `auto` keeps them in
    memory if `size` fits into MEMORY_FRACTION of the memory limit).

    Args:
        size: Total bytes of the arrays to be stored.
        mode: 'auto', 'memory' or 'disk'.
        label: Name stem of the scratch file.
    """

    def __init__(self, size, mode="auto", label="three_index"):
        limit = _core.memory()
        if mode == "auto":
            mode = "memory" if not limit or size <= MEMORY_FRACTION * limit else "disk"
        self.size, self.on_disk, self.path = size, mode == "disk", None
        self._h5, self._mem = None, {}
        if self.on_disk:
            import h5py
            self.path = scratch.new_file(label)
            self._h5 = h5py.File(self.path, "w")

    def create(self, name, shape):
        """An empty array (n, a, b) to be filled by set()."""
        if self.on_disk:
            self._h5.create_dataset(name.replace("/", "|"), shape=shape, dtype="f8",
                                    chunks=(1,) + tuple(shape[1:]))
        else:
            self._mem[name] = np.zeros(shape)

    def set(self, name, k, value):
        """Set the slice k of an array made by create()."""
        if self.on_disk:
            self._h5[name.replace("/", "|")][k] = value
        else:
            self._mem[name][k] = value

    def put(self, name, array):
        """Store an array (n, naux, m)."""
        if self.on_disk:  # '/' would create HDF5 groups
            self._h5.create_dataset(name.replace("/", "|"), data=array,
                                    chunks=(1,) + array.shape[1:])
        else:
            self._mem[name] = np.ascontiguousarray(array)

    def get(self, name, k):
        """The slice k (naux, m) of an array."""
        if self.on_disk:
            return np.asarray(self._h5[name.replace("/", "|")][k])
        return self._mem[name][k]

    def describe(self):
        where = f"on disk ({self.path})" if self.on_disk else "in memory"
        return f"{_core.format_bytes(self.size)} {where}"

    def close(self):
        """Release the arrays; delete the scratch file."""
        self._mem = {}
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self.path.unlink(missing_ok=True)
