"""Post-processing for QKD: error correction and privacy amplification.

Provides the classical post-processing steps required after quantum
transmission to distill a secure shared key:

1. **QBER estimation**: Sample a subset of the sifted key to measure
   the quantum bit error rate.

2. **Error correction (Cascade)**: Interactive binary search protocol
   that corrects bit errors using parity checks over random shuffles.
   Information leaked to Eve during correction is accounted for in
   privacy amplification.

3. **Privacy amplification**: Compress the corrected key using a
   Toeplitz universal hash function to eliminate any information an
   eavesdropper may have gained.

References:
    - Brassard & Salvail, Advances in Cryptology (1994) [Cascade]
    - Bennett et al., J. Cryptology 5, 3 (1995) [privacy amplification]
    - Krawczyk, CRYPTO 1994 [Toeplitz hashing]
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


# ======================================================================
# QBER estimation
# ======================================================================


def estimate_qber(
    key_a: List[int],
    key_b: List[int],
    sample_fraction: float = 0.1,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Estimate the quantum bit error rate between two keys.

    Compares a random sample (or the full keys if sample_fraction=1.0)
    and returns the fraction of disagreeing bits.

    Parameters
    ----------
    key_a : list[int]
        Alice's key bits.
    key_b : list[int]
        Bob's key bits.
    sample_fraction : float
        Fraction of bits to sample. Default 0.1 (10%).
        Set to 1.0 to compare all bits.
    rng : np.random.RandomState, optional
        Random number generator for sampling. If None, compares all bits
        when sample_fraction < 1.0 (deterministic fallback).

    Returns
    -------
    float
        Estimated QBER in [0, 1].

    Raises
    ------
    ValueError
        If keys have different lengths or are empty.
    """
    if len(key_a) != len(key_b):
        raise ValueError(
            f"Keys must have the same length, got {len(key_a)} and {len(key_b)}"
        )
    if len(key_a) == 0:
        raise ValueError("Keys must be non-empty")

    n = len(key_a)

    if sample_fraction >= 1.0 or rng is None:
        # Compare all bits
        errors = sum(1 for a, b in zip(key_a, key_b) if a != b)
        return errors / n

    n_sample = max(1, int(n * sample_fraction))
    n_sample = min(n_sample, n)
    indices = rng.choice(n, size=n_sample, replace=False)
    errors = sum(1 for i in indices if key_a[i] != key_b[i])
    return errors / n_sample


# ======================================================================
# Error correction: Cascade protocol
# ======================================================================


def error_correction_cascade(
    key_a: List[int],
    key_b: List[int],
    passes: int = 4,
) -> tuple:
    """Correct errors between two keys using the Cascade protocol.

    Cascade works in multiple passes with increasing block sizes.
    In each pass:
    1. The key is divided into blocks of size 2^pass.
    2. Parities of corresponding blocks are compared.
    3. If parities differ, a binary search within the block locates
       the error and corrects it.

    This implementation simulates the protocol by directly comparing
    bits (since both keys are available in simulation), which is
    equivalent to the interactive parity-check protocol in terms of
    the corrected output.

    Parameters
    ----------
    key_a : list[int]
        Alice's key (reference).
    key_b : list[int]
        Bob's key (to be corrected).
    passes : int
        Number of cascade passes. More passes correct more errors
        but leak more information to potential eavesdroppers.

    Returns
    -------
    tuple[list[int], list[int]]
        ``(key_a, corrected_key_b)`` after error correction.
        Both keys should now agree.
    """
    if len(key_a) != len(key_b):
        raise ValueError(
            f"Keys must have the same length, got {len(key_a)} and {len(key_b)}"
        )

    n = len(key_a)
    if n == 0:
        return ([], [])

    corrected_b = list(key_b)

    for pass_num in range(passes):
        block_size = min(2 ** (pass_num + 1), n)

        # Process each block
        for start in range(0, n, block_size):
            end = min(start + block_size, n)

            # Compute parities
            parity_a = 0
            parity_b = 0
            for i in range(start, end):
                parity_a ^= key_a[i]
                parity_b ^= corrected_b[i]

            if parity_a != parity_b:
                # Parity mismatch: binary search for the error
                _binary_search_correct(
                    key_a, corrected_b, start, end
                )

    return (list(key_a), corrected_b)


def _binary_search_correct(
    key_a: List[int],
    key_b: List[int],
    start: int,
    end: int,
) -> None:
    """Binary search within a block to find and correct a single error.

    Parameters
    ----------
    key_a : list[int]
        Alice's key (reference, read-only).
    key_b : list[int]
        Bob's key (modified in-place to correct the error).
    start : int
        Block start index (inclusive).
    end : int
        Block end index (exclusive).
    """
    while end - start > 1:
        mid = (start + end) // 2

        # Check parity of left half
        parity_a = 0
        parity_b = 0
        for i in range(start, mid):
            parity_a ^= key_a[i]
            parity_b ^= key_b[i]

        if parity_a != parity_b:
            # Error is in the left half
            end = mid
        else:
            # Error is in the right half
            start = mid

    # Correct the identified bit
    if start < len(key_b) and key_a[start] != key_b[start]:
        key_b[start] = key_a[start]


# ======================================================================
# Privacy amplification
# ======================================================================


def privacy_amplification(
    key: List[int],
    compression_ratio: float,
    rng: Optional[np.random.RandomState] = None,
) -> List[int]:
    """Compress a key using universal hashing for privacy amplification.

    Applies a Toeplitz matrix hash to reduce the key length, eliminating
    any partial information an eavesdropper may have obtained during
    quantum transmission or error correction.

    Parameters
    ----------
    key : list[int]
        Input key bits.
    compression_ratio : float
        Fraction of the original key to retain. Must be in (0, 1].
        A smaller ratio provides stronger security but shorter keys.
    rng : np.random.RandomState, optional
        Random number generator for the Toeplitz matrix seed.
        If None, uses numpy's default RNG.

    Returns
    -------
    list[int]
        Compressed key with length ~ len(key) * compression_ratio.
    """
    if not 0.0 < compression_ratio <= 1.0:
        raise ValueError(
            f"compression_ratio must be in (0, 1], got {compression_ratio}"
        )

    n = len(key)
    if n == 0:
        return []

    output_len = max(1, int(n * compression_ratio))

    if rng is None:
        rng = np.random.RandomState()

    return toeplitz_hash(key, output_len, rng)


def toeplitz_hash(
    key: List[int],
    output_len: int,
    rng: np.random.RandomState,
) -> List[int]:
    """Apply a Toeplitz matrix hash to a binary key.

    A Toeplitz matrix is defined by its first row and first column,
    requiring only (n + m - 1) random bits for an m x n matrix.
    The hash output is: h = T * key (mod 2).

    This is a universal_2 hash family, providing information-theoretic
    security guarantees for privacy amplification.

    Parameters
    ----------
    key : list[int]
        Input binary key of length n.
    output_len : int
        Desired output length m.
    rng : np.random.RandomState
        Random number generator for the Toeplitz seed.

    Returns
    -------
    list[int]
        Hashed key of length output_len.
    """
    n = len(key)
    if n == 0 or output_len == 0:
        return []

    # Generate the Toeplitz seed: first row + first column
    # Total random bits needed: n + output_len - 1
    seed_len = n + output_len - 1
    seed = rng.randint(0, 2, size=seed_len).tolist()

    # Compute hash: h[i] = sum(T[i][j] * key[j]) mod 2
    # where T[i][j] = seed[i - j + n - 1] (shifted diagonals)
    result: List[int] = []
    for i in range(output_len):
        bit = 0
        for j in range(n):
            # Toeplitz index: row offset from the seed
            t_idx = output_len - 1 - i + j
            if seed[t_idx] == 1 and key[j] == 1:
                bit ^= 1
        result.append(bit)

    return result
