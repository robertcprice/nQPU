"""B92 Quantum Key Distribution protocol.

Implements Bennett's simplified 1992 protocol using only two non-orthogonal
quantum states. Alice sends |0> for bit 0 and |+> for bit 1. Bob measures
in a randomly chosen basis and can sometimes conclusively determine Alice's
bit (when his measurement is incompatible with the state Alice did NOT send).

The conclusive detection rate is ~25% for a perfect channel, making B92
less efficient than BB84 but requiring fewer quantum resources.

References:
    - Bennett, Phys. Rev. Lett. 68, 3121 (1992)
    - Tamaki et al., Phys. Rev. A 67, 032310 (2003) [security analysis]
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .bb84 import QKDResult
from .channel import QuantumChannel
from .privacy import error_correction_cascade, estimate_qber, privacy_amplification


class B92Protocol:
    """Full B92 QKD protocol implementation.

    Alice encodes bit 0 as |0> and bit 1 as |+> = (|0> + |1>) / sqrt(2).
    Bob measures each qubit in a randomly chosen basis:
      - Z-basis: {|0>, |1>}
      - X-basis: {|+>, |->}

    Conclusive results occur when Bob's measurement outcome is incompatible
    with one of Alice's possible states:
      - If Bob measures |1> in Z-basis: Alice must have sent |+> (bit 1),
        because |0> can never give outcome |1> in Z-basis.
      - If Bob measures |-> in X-basis: Alice must have sent |0> (bit 0),
        because |+> can never give outcome |-> in X-basis.

    All other outcomes are inconclusive and discarded.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    sample_fraction : float
        Fraction of conclusive bits to sacrifice for QBER estimation.
    cascade_passes : int
        Number of cascade error correction passes.

    Examples
    --------
    >>> from nqpu.qkd import B92Protocol, QuantumChannel
    >>> channel = QuantumChannel(error_rate=0.01)
    >>> protocol = B92Protocol(seed=42)
    >>> result = protocol.generate_key(n_bits=10000, channel=channel)
    >>> assert result.secure
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        sample_fraction: float = 0.1,
        cascade_passes: int = 4,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.sample_fraction = sample_fraction
        self.cascade_passes = cascade_passes

    # ------------------------------------------------------------------
    # Main protocol entry point
    # ------------------------------------------------------------------

    def generate_key(
        self,
        n_bits: int,
        channel: QuantumChannel,
    ) -> QKDResult:
        """Run the full B92 protocol.

        Parameters
        ----------
        n_bits : int
            Number of qubits Alice sends.
        channel : QuantumChannel
            Quantum channel connecting Alice and Bob.

        Returns
        -------
        QKDResult
            Protocol result with keys and diagnostics.
        """
        if n_bits < 10:
            raise ValueError("n_bits must be >= 10 for meaningful key generation")

        result = QKDResult(protocol="B92", n_bits_sent=n_bits)

        # Step 1: Alice generates random bits
        # bit 0 -> sends |0> (Z-basis state)
        # bit 1 -> sends |+> (X-basis state)
        alice_bits = self.rng.randint(0, 2, size=n_bits).tolist()

        # Step 2: Bob chooses random measurement bases
        # 0 = Z-basis, 1 = X-basis
        bob_bases = self.rng.randint(0, 2, size=n_bits).tolist()

        # Step 3: Transmit and measure
        conclusive_alice: List[int] = []
        conclusive_bob: List[int] = []
        n_received = 0

        for i in range(n_bits):
            alice_bit = alice_bits[i]
            # Alice's preparation basis: 0 for bit 0 (Z), 1 for bit 1 (X)
            alice_basis = alice_bit

            transmission = channel.transmit_qubit(
                alice_bit, alice_basis, self.rng
            )
            if transmission is None:
                continue  # Photon lost

            n_received += 1
            received_bit, _original_basis = transmission
            bob_basis = bob_bases[i]

            # Determine Bob's measurement outcome
            outcome, conclusive = self._bob_measure(
                received_bit, alice_bit, alice_basis, bob_basis, self.rng
            )

            if conclusive:
                # Bob announces he got a conclusive result (but not which one)
                conclusive_alice.append(alice_bit)
                conclusive_bob.append(outcome)

        result.n_bits_received = n_received
        result.sifted_key_alice = conclusive_alice
        result.sifted_key_bob = conclusive_bob
        result.sifting_efficiency = (
            len(conclusive_alice) / n_received if n_received > 0 else 0.0
        )

        if len(conclusive_alice) < 4:
            result.secure = False
            result.qber = 1.0
            return result

        # Step 4: QBER estimation
        n_sample = max(2, int(len(conclusive_alice) * self.sample_fraction))
        n_sample = min(n_sample, len(conclusive_alice) - 2)

        sample_indices = self.rng.choice(
            len(conclusive_alice), size=n_sample, replace=False
        )
        sample_indices_set = set(sample_indices.tolist())

        sample_a = [conclusive_alice[i] for i in sample_indices]
        sample_b = [conclusive_bob[i] for i in sample_indices]
        result.qber = estimate_qber(sample_a, sample_b, sample_fraction=1.0)

        remaining_alice = [
            conclusive_alice[i]
            for i in range(len(conclusive_alice))
            if i not in sample_indices_set
        ]
        remaining_bob = [
            conclusive_bob[i]
            for i in range(len(conclusive_bob))
            if i not in sample_indices_set
        ]

        # B92 is less tolerant to noise than BB84
        if result.qber > 0.11:
            result.secure = False
            result.final_key = []
            result.key_rate = 0.0
            return result

        if len(remaining_alice) < 2:
            result.final_key = []
            result.key_rate = 0.0
            return result

        # Step 5: Error correction
        corrected_alice, corrected_bob = error_correction_cascade(
            remaining_alice, remaining_bob, passes=self.cascade_passes
        )

        # Step 6: Privacy amplification
        compression = 0.5
        result.final_key = privacy_amplification(
            corrected_alice, compression, self.rng
        )

        result.key_rate = (
            len(result.final_key) / n_bits if n_bits > 0 else 0.0
        )

        return result

    # ------------------------------------------------------------------
    # Measurement logic
    # ------------------------------------------------------------------

    @staticmethod
    def _bob_measure(
        received_bit: int,
        alice_bit: int,
        alice_basis: int,
        bob_basis: int,
        rng: np.random.RandomState,
    ) -> tuple:
        """Simulate Bob's measurement and determine if it is conclusive.

        In an ideal (noiseless, no-Eve) scenario:
        - Alice sends |0> (bit 0): Bob measuring Z gets 0 (inconclusive),
          Bob measuring X gets +/- with equal probability.
          If Bob gets |-> (X-basis, outcome 1), conclusive: Alice sent |0> (bit 0).
        - Alice sends |+> (bit 1): Bob measuring X gets + (inconclusive),
          Bob measuring Z gets 0/1 with equal probability.
          If Bob gets |1> (Z-basis, outcome 1), conclusive: Alice sent |+> (bit 1).

        With channel noise, the received_bit may differ from alice_bit,
        which introduces errors in the conclusive results.

        Parameters
        ----------
        received_bit : int
            The bit value after channel transmission.
        alice_bit : int
            Alice's original bit (for determining the physical state).
        alice_basis : int
            Basis of Alice's preparation (0=Z for bit 0, 1=X for bit 1).
        bob_basis : int
            Bob's measurement basis (0=Z, 1=X).
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        tuple[int, bool]
            (inferred_bit, is_conclusive).
        """
        if alice_basis == bob_basis:
            # Same basis: always inconclusive in B92
            # If Alice sent |0> and Bob measures Z: always gets 0 (inconclusive)
            # If Alice sent |+> and Bob measures X: always gets + (inconclusive)
            # With noise, received_bit might differ, but the protocol logic
            # still treats same-basis measurements as inconclusive.
            return (0, False)

        # Different bases: possibility of conclusive outcome
        if alice_basis == 0 and bob_basis == 1:
            # Alice sent |0>, Bob measures X-basis
            # Without noise: |0> = (|+> + |->)/sqrt(2), so 50/50
            # If Bob gets |-> (outcome 1): conclusive, Alice sent |0> (bit 0)
            # Channel noise: received_bit might be flipped
            if received_bit == 0:
                # State is close to |0>, measuring in X gives 50/50
                outcome = rng.randint(0, 2)
            else:
                # State is close to |1>, measuring in X gives 50/50
                outcome = rng.randint(0, 2)

            if outcome == 1:  # Got |->
                return (0, True)
            else:
                return (0, False)

        elif alice_basis == 1 and bob_basis == 0:
            # Alice sent |+>, Bob measures Z-basis
            # Without noise: |+> = (|0> + |1>)/sqrt(2), so 50/50
            # If Bob gets |1> (outcome 1): conclusive, Alice sent |+> (bit 1)
            if received_bit == 1:
                # State is close to |+>, measuring in Z gives 50/50
                outcome = rng.randint(0, 2)
            else:
                # State close to |-> or |0> depending on noise
                outcome = rng.randint(0, 2)

            if outcome == 1:  # Got |1>
                return (1, True)
            else:
                return (0, False)

        return (0, False)
