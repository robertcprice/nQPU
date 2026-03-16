"""Electron integral computation for Gaussian basis sets.

Implements analytical evaluation of one-electron and two-electron
integrals over contracted s-type Gaussian basis functions, sufficient
for STO-3G calculations on small molecules.

Integral types
--------------
- **Overlap** (S): ``<a|b>``
- **Kinetic** (T): ``<a|-1/2 nabla^2|b>``
- **Nuclear attraction** (V): ``<a|-Z/|r-R||b>``
- **Electron repulsion** (ERI): ``(ab|cd) = <ab|1/r12|cd>``

The Boys function ``F_n(x)`` is evaluated via a series expansion for
small x and an asymptotic formula for large x, avoiding any dependency
on scipy.special.

References
----------
- Szabo & Ostlund, *Modern Quantum Chemistry* (1996), Appendix A.
- Obara & Saika, J. Chem. Phys. 84, 3963 (1986).
- McMurchie & Davidson, J. Comp. Phys. 26, 218 (1978).
"""

from __future__ import annotations

import math

import numpy as np

from .molecular import BasisFunction, BasisSet, Molecule, PrimitiveGaussian

# ============================================================
# Boys function
# ============================================================


def boys_function(n: int, x: float) -> float:
    """Evaluate the Boys function F_n(x).

    .. math::

        F_n(x) = \\int_0^1 t^{2n} \\exp(-x t^2) \\, dt

    Uses a Taylor series expansion for small x (x < 25) and an
    asymptotic formula for large x.

    Parameters
    ----------
    n : int
        Order of the Boys function.
    x : float
        Argument (must be >= 0).

    Returns
    -------
    float
        Value of F_n(x).
    """
    if x < 1e-12:
        # F_n(0) = 1 / (2n + 1)
        return 1.0 / (2.0 * n + 1.0)

    if x > 25.0:
        # Asymptotic: F_n(x) ~ (2n-1)!! / (2^(n+1)) * sqrt(pi/x^(2n+1))
        return (
            _double_factorial(2 * n - 1)
            / (2.0 ** (n + 1))
            * math.sqrt(math.pi / x ** (2 * n + 1))
        )

    # Taylor series: F_n(x) = sum_{k=0}^{inf} (-1)^k * x^k / (k! * (2n + 2k + 1))
    # This converges well for moderate x.
    max_terms = 80
    total = 0.0
    term = 1.0 / (2.0 * n + 1.0)  # k=0 term
    total = term
    for k in range(1, max_terms):
        term *= -x / k
        contribution = term / (2.0 * n + 2.0 * k + 1.0)
        total += contribution
        if abs(contribution) < 1e-15 * abs(total):
            break

    return total


def _double_factorial(n: int) -> float:
    """Compute n!! (double factorial). Returns 1.0 for n <= 0."""
    if n <= 0:
        return 1.0
    result = 1.0
    k = n
    while k > 0:
        result *= k
        k -= 2
    return result


# ============================================================
# Primitive Gaussian integrals (s-type only)
# ============================================================


def _gaussian_product_center(
    alpha: float, center_a: tuple[float, float, float],
    beta: float, center_b: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Gaussian product theorem: center of product Gaussian."""
    gamma = alpha + beta
    px = (alpha * center_a[0] + beta * center_b[0]) / gamma
    py = (alpha * center_a[1] + beta * center_b[1]) / gamma
    pz = (alpha * center_a[2] + beta * center_b[2]) / gamma
    return (px, py, pz)


def _dist_squared(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    """Squared Euclidean distance between two 3D points."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _primitive_overlap(pa: PrimitiveGaussian, pb: PrimitiveGaussian) -> float:
    """Overlap integral between two primitive s-type Gaussians.

    .. math::

        S_{ab} = N_a N_b (\\pi/(\\alpha+\\beta))^{3/2}
                 \\exp(-\\alpha\\beta/(\\alpha+\\beta) |R_A-R_B|^2)
    """
    alpha = pa.exponent
    beta = pb.exponent
    gamma = alpha + beta
    rab2 = _dist_squared(pa.center, pb.center)

    na = pa.normalization
    nb = pb.normalization

    return na * nb * (math.pi / gamma) ** 1.5 * math.exp(-alpha * beta / gamma * rab2)


def _primitive_kinetic(pa: PrimitiveGaussian, pb: PrimitiveGaussian) -> float:
    """Kinetic energy integral between two primitive s-type Gaussians.

    .. math::

        T_{ab} = \\alpha\\beta/(\\alpha+\\beta)
                 (3 - 2\\alpha\\beta/(\\alpha+\\beta)|R_A-R_B|^2)
                 S_{ab}
    """
    alpha = pa.exponent
    beta = pb.exponent
    gamma = alpha + beta
    rab2 = _dist_squared(pa.center, pb.center)

    s = _primitive_overlap(pa, pb)
    t = alpha * beta / gamma * (3.0 - 2.0 * alpha * beta / gamma * rab2) * s
    return t


def _primitive_nuclear(
    pa: PrimitiveGaussian,
    pb: PrimitiveGaussian,
    nucleus_center: tuple[float, float, float],
    nuclear_charge: float,
) -> float:
    """Nuclear attraction integral for two primitive s-type Gaussians.

    .. math::

        V_{ab}^C = -Z_C \\frac{2\\pi}{\\gamma} N_a N_b
                   \\exp(-\\alpha\\beta/\\gamma |R_A-R_B|^2)
                   F_0(\\gamma |R_P - R_C|^2)

    where P is the Gaussian product center.
    """
    alpha = pa.exponent
    beta = pb.exponent
    gamma = alpha + beta
    rab2 = _dist_squared(pa.center, pb.center)

    p = _gaussian_product_center(alpha, pa.center, beta, pb.center)
    rpc2 = _dist_squared(p, nucleus_center)

    na = pa.normalization
    nb = pb.normalization

    prefactor = -nuclear_charge * 2.0 * math.pi / gamma
    exponential = math.exp(-alpha * beta / gamma * rab2)

    return prefactor * na * nb * exponential * boys_function(0, gamma * rpc2)


def _primitive_eri(
    pa: PrimitiveGaussian,
    pb: PrimitiveGaussian,
    pc: PrimitiveGaussian,
    pd: PrimitiveGaussian,
) -> float:
    """Two-electron repulsion integral (ab|cd) for four s-type primitives.

    .. math::

        (ab|cd) = \\frac{2\\pi^{5/2}}{\\gamma_{ab}\\gamma_{cd}
                  \\sqrt{\\gamma_{ab}+\\gamma_{cd}}}
                  N_a N_b N_c N_d
                  \\exp(-\\alpha\\beta/\\gamma_{ab}|R_A-R_B|^2)
                  \\exp(-\\gamma\\delta/\\gamma_{cd}|R_C-R_D|^2)
                  F_0(\\rho|R_P-R_Q|^2)

    where P, Q are Gaussian product centers and rho is the reduced exponent.
    """
    alpha = pa.exponent
    beta = pb.exponent
    gamma_ab = alpha + beta

    gamma_val = pc.exponent
    delta = pd.exponent
    gamma_cd = gamma_val + delta

    rab2 = _dist_squared(pa.center, pb.center)
    rcd2 = _dist_squared(pc.center, pd.center)

    p = _gaussian_product_center(alpha, pa.center, beta, pb.center)
    q = _gaussian_product_center(gamma_val, pc.center, delta, pd.center)
    rpq2 = _dist_squared(p, q)

    rho = gamma_ab * gamma_cd / (gamma_ab + gamma_cd)

    na = pa.normalization
    nb = pb.normalization
    nc = pc.normalization
    nd = pd.normalization

    prefactor = (
        2.0
        * math.pi ** 2.5
        / (gamma_ab * gamma_cd * math.sqrt(gamma_ab + gamma_cd))
    )
    exp_ab = math.exp(-alpha * beta / gamma_ab * rab2)
    exp_cd = math.exp(-gamma_val * delta / gamma_cd * rcd2)

    return prefactor * na * nb * nc * nd * exp_ab * exp_cd * boys_function(0, rho * rpq2)


# ============================================================
# Contracted basis function integrals
# ============================================================


def overlap_integral(bf_a: BasisFunction, bf_b: BasisFunction) -> float:
    """Compute the overlap integral <a|b> between two contracted basis functions.

    Parameters
    ----------
    bf_a, bf_b : BasisFunction
        Contracted Gaussian basis functions.

    Returns
    -------
    float
        Overlap integral S_ab.
    """
    total = 0.0
    for pa in bf_a.primitives:
        for pb in bf_b.primitives:
            total += pa.coefficient * pb.coefficient * _primitive_overlap(pa, pb)
    return total


def kinetic_integral(bf_a: BasisFunction, bf_b: BasisFunction) -> float:
    """Compute the kinetic energy integral <a|-1/2 nabla^2|b>.

    Parameters
    ----------
    bf_a, bf_b : BasisFunction
        Contracted Gaussian basis functions.

    Returns
    -------
    float
        Kinetic energy integral T_ab.
    """
    total = 0.0
    for pa in bf_a.primitives:
        for pb in bf_b.primitives:
            total += pa.coefficient * pb.coefficient * _primitive_kinetic(pa, pb)
    return total


def nuclear_attraction_integral(
    bf_a: BasisFunction,
    bf_b: BasisFunction,
    nuclei: list[tuple[tuple[float, float, float], float]],
) -> float:
    """Compute the nuclear attraction integral.

    Sums over all nuclei: V_ab = sum_C <a|-Z_C/|r-R_C||b>.

    Parameters
    ----------
    bf_a, bf_b : BasisFunction
        Contracted Gaussian basis functions.
    nuclei : list of (center, charge)
        Nuclear centers (in Bohr) and charges.

    Returns
    -------
    float
        Nuclear attraction integral V_ab.
    """
    total = 0.0
    for center_c, charge_c in nuclei:
        for pa in bf_a.primitives:
            for pb in bf_b.primitives:
                total += (
                    pa.coefficient
                    * pb.coefficient
                    * _primitive_nuclear(pa, pb, center_c, charge_c)
                )
    return total


def electron_repulsion_integral(
    bf_a: BasisFunction,
    bf_b: BasisFunction,
    bf_c: BasisFunction,
    bf_d: BasisFunction,
) -> float:
    """Compute the two-electron repulsion integral (ab|cd).

    Parameters
    ----------
    bf_a, bf_b, bf_c, bf_d : BasisFunction
        Four contracted Gaussian basis functions.

    Returns
    -------
    float
        Two-electron integral (ab|cd) in chemists' notation.
    """
    total = 0.0
    for pa in bf_a.primitives:
        for pb in bf_b.primitives:
            for pc_prim in bf_c.primitives:
                for pd_prim in bf_d.primitives:
                    total += (
                        pa.coefficient
                        * pb.coefficient
                        * pc_prim.coefficient
                        * pd_prim.coefficient
                        * _primitive_eri(pa, pb, pc_prim, pd_prim)
                    )
    return total


# ============================================================
# Full integral matrices
# ============================================================


def compute_one_electron_integrals(
    molecule: Molecule,
    basis: BasisSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute all one-electron integrals for a molecule in a given basis.

    Returns the overlap matrix S, kinetic energy matrix T, and the
    core Hamiltonian H_core = T + V (where V is the nuclear attraction).

    Parameters
    ----------
    molecule : Molecule
        The molecular system.
    basis : BasisSet
        The basis set applied to this molecule.

    Returns
    -------
    S : np.ndarray
        Overlap matrix, shape ``(n, n)``.
    T : np.ndarray
        Kinetic energy matrix, shape ``(n, n)``.
    H_core : np.ndarray
        Core Hamiltonian ``T + V``, shape ``(n, n)``.
    """
    n = basis.num_functions
    S = np.zeros((n, n), dtype=np.float64)
    T = np.zeros((n, n), dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)

    # Build nuclear list: (center_in_bohr, charge)
    nuclei = [
        (atom.position_bohr, atom.nuclear_charge)
        for atom in molecule.atoms
    ]

    for i in range(n):
        for j in range(i, n):
            s_ij = overlap_integral(basis.functions[i], basis.functions[j])
            t_ij = kinetic_integral(basis.functions[i], basis.functions[j])
            v_ij = nuclear_attraction_integral(
                basis.functions[i], basis.functions[j], nuclei
            )

            S[i, j] = s_ij
            S[j, i] = s_ij
            T[i, j] = t_ij
            T[j, i] = t_ij
            V[i, j] = v_ij
            V[j, i] = v_ij

    H_core = T + V
    return S, T, H_core


def compute_two_electron_integrals(
    molecule: Molecule,
    basis: BasisSet,
) -> np.ndarray:
    """Compute the full two-electron integral tensor.

    .. math::

        g_{pqrs} = (pq|rs) = \\int\\int \\phi_p(1)\\phi_q(1)
                   \\frac{1}{r_{12}} \\phi_r(2)\\phi_s(2) \\, d1 \\, d2

    Uses 8-fold symmetry to reduce computation.

    Parameters
    ----------
    molecule : Molecule
        The molecular system (unused but kept for API consistency).
    basis : BasisSet
        The basis set applied to this molecule.

    Returns
    -------
    np.ndarray
        Two-electron integral tensor, shape ``(n, n, n, n)``.
    """
    n = basis.num_functions
    eri = np.zeros((n, n, n, n), dtype=np.float64)

    for p in range(n):
        for q in range(p + 1):
            for r in range(n):
                for s in range(r + 1):
                    if (p * (p + 1)) // 2 + q < (r * (r + 1)) // 2 + s:
                        continue
                    val = electron_repulsion_integral(
                        basis.functions[p],
                        basis.functions[q],
                        basis.functions[r],
                        basis.functions[s],
                    )
                    # 8-fold symmetry: (pq|rs) = (qp|rs) = (pq|sr) = (qp|sr)
                    #                           = (rs|pq) = (sr|pq) = (rs|qp) = (sr|qp)
                    eri[p, q, r, s] = val
                    eri[q, p, r, s] = val
                    eri[p, q, s, r] = val
                    eri[q, p, s, r] = val
                    eri[r, s, p, q] = val
                    eri[s, r, p, q] = val
                    eri[r, s, q, p] = val
                    eri[s, r, q, p] = val

    return eri
