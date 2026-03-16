"""BB84 Quantum Key Distribution protocol.

Implements the full Bennett-Brassard 1984 protocol with:
  - Random bit and basis generation by Alice
  - Qubit transmission through a noisy/lossy quantum channel
  - Random basis measurement by Bob
  - Basis sifting (keeping only matching-basis bits)
  - QBER estimation from a sampled subset
  - Error correction via cascade protocol
  - Privacy amplification via Toeplitz universal hashing

The BB84 security threshold is QBER < 11%: above this the protocol
aborts because the error rate is consistent with an eavesdropper
extracting too much information.

References:
    - Bennett & Brassard, Proc. IEEE Int. Conf. Computers, Systems,
      and Signal Processing, Bangalore, India (1984)
    - Shor & Preskill, Phys. Rev. Lett. 85, 441 (2000) [security proof]
    - Scarani et al., Rev. Mod. Phys. 81, 1301 (2009) [review]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .channel import QuantumChannel
from .privacy import error_correction_cascade, estimate_qber, privacy_amplification


# BB84 QBER security threshold (Shor-Preskill bound)
BB84_QBER_THRESHOLD = 0.11


@dataclass
class QKDResult:
    """Result of a QKD protocol run.

    Attributes
    ----------
    protocol : str
        Name of the protocol ('BB84', 'E91', 'B92').
    n_bits_sent : int
        Total number of qubits Alice transmitted.
    n_bits_received : int
        Number of qubits that arrived at Bob (not lost).
    raw_key_alice : list[int]
        Alice's raw key before sifting.
    raw_key_bob : list[int]
        Bob's raw key before sifting.
    sifted_key_alice : list[int]
        Alice's key after basis sifting.
    sifted_key_bob : list[int]
        Bob's key after basis sifting.
    final_key : list[int]
        The final secure key after error correction and privacy amplification.
    qber : float
        Quantum bit error rate estimated from sampled bits.
    key_rate : float
        Effective key rate: len(final_key) / n_bits_sent.
    security_parameter : float
        Security parameter epsilon (probability of key compromise).
    secure : bool
        Whether the protocol completed successfully (QBER below threshold).
    chsh_s_value : Optional[float]
        CHSH S-value (only for E91 protocol).
    sifting_efficiency : float
        Fraction of received bits that survived sifting.
    """

    protocol: str = "BB84"
    n_bits_sent: int = 0
    n_bits_received: int = 0
    raw_key_alice: List[int] = field(default_factory=list)
    raw_key_bob: List[int] = field(default_factory=list)
    sifted_key_alice: List[int] = field(default_factory=list)
    sifted_key_bob: List[int] = field(default_factory=list)
    final_key: List[int] = field(default_factory=list)
    qber: float = 0.0
    key_rate: float = 0.0
    security_parameter: float = 1e-10
    secure: bool = True
    chsh_s_value: Optional[float] = None
    sifting_efficiency: float = 0.0


class BB84Protocol:
    """Full BB84 QKD protocol implementation.

    Runs the complete protocol pipeline: preparation, transmission,
    sifting, error estimation, error correction, and privacy amplification.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    sample_fraction : float
        Fraction of sifted bits to sacrifice for QBER estimation.
    cascade_passes : int
        Number of cascade error correction passes.
    security_parameter : float
        Target security parameter (epsilon) for privacy amplification.

    Examples
    --------
    >>> from nqpu.qkd import BB84Protocol, QuantumChannel
    >>> channel = QuantumChannel(error_rate=0.03)
    >>> protocol = BB84Protocol(seed=42)
    >>> result = protocol.generate_key(n_bits=5000, channel=channel)
    >>> assert result.secure
    >>> assert result.qber < 0.11
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        sample_fraction: float = 0.1,
        cascade_passes: int = 4,
        security_parameter: float = 1e-10,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.sample_fraction = sample_fraction
        self.cascade_passes = cascade_passes
        self.security_parameter = security_parameter

    # ------------------------------------------------------------------
    # Main protocol entry point
    # ------------------------------------------------------------------

    def generate_key(
        self,
        n_bits: int,
        channel: QuantumChannel,
    ) -> QKDResult:
        """Run the full BB84 protocol and return the result.

        Parameters
        ----------
        n_bits : int
            Number of qubits Alice prepares and sends.
        channel : QuantumChannel
            The quantum channel connecting Alice and Bob.

        Returns
        -------
        QKDResult
            Complete protocol result including keys and diagnostics.
        """
        if n_bits < 10:
            raise ValueError("n_bits must be >= 10 for meaningful key generation")

        result = QKDResult(protocol="BB84", n_bits_sent=n_bits)

        # Step 1: Alice generates random bits and bases
        alice_bits = self.rng.randint(0, 2, size=n_bits).tolist()
        alice_bases = self.rng.randint(0, 2, size=n_bits).tolist()

        # Step 2: Bob chooses random measurement bases
        bob_bases = self.rng.randint(0, 2, size=n_bits).tolist()

        # Step 3: Transmit qubits through the channel
        bob_bits: List[int] = []
        received_alice_bits: List[int] = []
        received_alice_bases: List[int] = []
        received_bob_bases: List[int] = []

        for i in range(n_bits):
            transmission = channel.transmit_qubit(
                alice_bits[i], alice_bases[i], self.rng
            )
            if transmission is None:
                # Photon lost
                continue

            received_bit, _original_basis = transmission

            # Bob measures in his chosen basis
            if bob_bases[i] == alice_bases[i]:
                # Matching basis: Bob gets the (possibly noisy) bit
                bob_measured = received_bit
            else:
                # Mismatched basis: Bob gets a random result
                bob_measured = self.rng.randint(0, 2)

            bob_bits.append(bob_measured)
            received_alice_bits.append(alice_bits[i])
            received_alice_bases.append(alice_bases[i])
            received_bob_bases.append(bob_bases[i])

        result.n_bits_received = len(bob_bits)
        result.raw_key_alice = received_alice_bits
        result.raw_key_bob = bob_bits

        if result.n_bits_received < 4:
            result.secure = False
            result.qber = 1.0
            return result

        # Step 4: Basis sifting -- keep only matching-basis bits
        sifted_alice: List[int] = []
        sifted_bob: List[int] = []

        for i in range(len(bob_bits)):
            if received_alice_bases[i] == received_bob_bases[i]:
                sifted_alice.append(received_alice_bits[i])
                sifted_bob.append(bob_bits[i])

        result.sifted_key_alice = sifted_alice
        result.sifted_key_bob = sifted_bob
        result.sifting_efficiency = (
            len(sifted_alice) / result.n_bits_received
            if result.n_bits_received > 0
            else 0.0
        )

        if len(sifted_alice) < 4:
            result.secure = False
            result.qber = 1.0
            return result

        # Step 5: QBER estimation from a sample
        n_sample = max(2, int(len(sifted_alice) * self.sample_fraction))
        n_sample = min(n_sample, len(sifted_alice) - 2)

        sample_indices = self.rng.choice(
            len(sifted_alice), size=n_sample, replace=False
        )
        sample_indices_set = set(sample_indices.tolist())

        sample_a = [sifted_alice[i] for i in sample_indices]
        sample_b = [sifted_bob[i] for i in sample_indices]

        result.qber = estimate_qber(sample_a, sample_b, sample_fraction=1.0)

        # Remove sampled bits from keys
        remaining_alice = [
            sifted_alice[i]
            for i in range(len(sifted_alice))
            if i not in sample_indices_set
        ]
        remaining_bob = [
            sifted_bob[i]
            for i in range(len(sifted_bob))
            if i not in sample_indices_set
        ]

        # Step 6: Security check
        if result.qber > BB84_QBER_THRESHOLD:
            result.secure = False
            result.final_key = []
            result.key_rate = 0.0
            return result

        if len(remaining_alice) < 2:
            result.secure = True
            result.final_key = []
            result.key_rate = 0.0
            return result

        # Step 7: Error correction (cascade protocol)
        corrected_alice, corrected_bob = error_correction_cascade(
            remaining_alice, remaining_bob, passes=self.cascade_passes
        )

        # Step 8: Privacy amplification
        # Compression ratio based on QBER: r = 1 - h(QBER) where h is
        # binary entropy.  More conservative with higher QBER.
        if result.qber > 0:
            h_qber = self._binary_entropy(result.qber)
            compression_ratio = max(0.1, 1.0 - 2.0 * h_qber)
        else:
            compression_ratio = 0.9

        result.final_key = privacy_amplification(
            corrected_alice, compression_ratio, self.rng
        )

        result.security_parameter = self.security_parameter
        result.key_rate = (
            len(result.final_key) / n_bits if n_bits > 0 else 0.0
        )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _binary_entropy(p: float) -> float:
        """Compute binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).

        Parameters
        ----------
        p : float
            Probability in [0, 1].

        Returns
        -------
        float
            Binary entropy in bits.
        """
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
