"""Ekert-91 (E91) entanglement-based QKD protocol.

Uses entangled Bell pairs |Phi+> = (|00> + |11>) / sqrt(2) shared between
Alice and Bob. Security is guaranteed by the violation of the CHSH Bell
inequality: if the measured S-value is close to 2*sqrt(2) ~ 2.828, no
eavesdropper can have significant information about the key.

Measurement bases:
  - Alice: {0 deg, 45 deg, 90 deg}  (bases A1, A2, A3)
  - Bob:   {45 deg, 90 deg, 135 deg} (bases B1, B2, B3)

The CHSH test uses correlations between the non-matching bases
(A1-B1, A1-B3, A3-B1, A3-B3), while matching bases (A2-B1, A3-B2,
effectively A2=B1 at 45 deg, A3=B2 at 90 deg) generate the shared key.

References:
    - Ekert, Phys. Rev. Lett. 67, 661 (1991)
    - Clauser et al., Phys. Rev. Lett. 23, 880 (1969) [CHSH inequality]
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .bb84 import QKDResult
from .channel import QuantumChannel
from .privacy import error_correction_cascade, estimate_qber, privacy_amplification


# Alice's three measurement angles (radians)
ALICE_ANGLES = [0.0, math.pi / 4, math.pi / 2]  # 0, 45, 90 deg

# Bob's three measurement angles (radians)
BOB_ANGLES = [math.pi / 4, math.pi / 2, 3 * math.pi / 4]  # 45, 90, 135 deg

# Theoretical maximum CHSH S-value (Tsirelson bound)
TSIRELSON_BOUND = 2.0 * math.sqrt(2.0)

# CHSH classical bound: S <= 2 for local hidden variable theories
CHSH_CLASSICAL_BOUND = 2.0


class E91Protocol:
    """Full E91 QKD protocol implementation.

    Generates entangled Bell pairs, distributes them through a quantum
    channel, performs measurements in randomly chosen bases, tests the
    CHSH inequality for eavesdropping detection, and extracts a secret
    key from matching-basis measurements.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    sample_fraction : float
        Fraction of key bits sacrificed for QBER estimation.
    cascade_passes : int
        Number of cascade error correction passes.
    chsh_threshold : float
        Minimum S-value to consider the channel secure.
        Default is 2.0 (classical bound). Below this, abort.

    Examples
    --------
    >>> from nqpu.qkd import E91Protocol, QuantumChannel
    >>> channel = QuantumChannel(error_rate=0.01)
    >>> protocol = E91Protocol(seed=42)
    >>> result = protocol.generate_key(n_pairs=10000, channel=channel)
    >>> assert result.chsh_s_value > 2.0
    >>> assert result.secure
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        sample_fraction: float = 0.1,
        cascade_passes: int = 4,
        chsh_threshold: float = 2.0,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.sample_fraction = sample_fraction
        self.cascade_passes = cascade_passes
        self.chsh_threshold = chsh_threshold

    # ------------------------------------------------------------------
    # Main protocol entry point
    # ------------------------------------------------------------------

    def generate_key(
        self,
        n_pairs: int,
        channel: QuantumChannel,
    ) -> QKDResult:
        """Run the full E91 protocol.

        Parameters
        ----------
        n_pairs : int
            Number of entangled pairs to generate and distribute.
        channel : QuantumChannel
            Quantum channel for Bob's photon (Alice's is local).

        Returns
        -------
        QKDResult
            Protocol result with CHSH S-value and keys.
        """
        if n_pairs < 20:
            raise ValueError("n_pairs must be >= 20 for meaningful statistics")

        result = QKDResult(protocol="E91", n_bits_sent=n_pairs)

        # Step 1: Generate pairs and choose measurement bases
        alice_basis_choices = self.rng.randint(0, 3, size=n_pairs)
        bob_basis_choices = self.rng.randint(0, 3, size=n_pairs)

        # Step 2: Distribute pairs and measure
        alice_outcomes: List[int] = []
        bob_outcomes: List[int] = []
        alice_angles_used: List[float] = []
        bob_angles_used: List[float] = []
        alice_basis_list: List[int] = []
        bob_basis_list: List[int] = []

        for i in range(n_pairs):
            # Transmit entangled pair through channel
            pair = channel.transmit_entangled_pair(self.rng)
            if pair is None:
                continue  # Photon lost

            # For Bell state |Phi+> = (|00> + |11>)/sqrt(2), measurements
            # are correlated: E(a,b) = cos(a - b).
            # Alice measures first (random 0 or 1 for Bell state).
            # Bob's outcome is correlated with Alice's via:
            #   P(B=Alice) = cos^2((a-b)/2), P(B!=Alice) = sin^2((a-b)/2)
            a_angle = ALICE_ANGLES[alice_basis_choices[i]]
            b_angle = BOB_ANGLES[bob_basis_choices[i]]

            # Alice's measurement: 50/50 for Bell state
            a_outcome = self.rng.randint(0, 2)

            # Check for entanglement-breaking eavesdropper
            eve_breaks = (
                channel.eavesdropper is not None
                and channel.eavesdropper.strategy == "entanglement_breaking"
                and self.rng.random() < channel.eavesdropper.interception_rate
            )

            if eve_breaks:
                # Eve's attack destroys quantum correlations:
                # Bob's outcome is now independent of Alice's
                b_outcome = self.rng.randint(0, 2)
            else:
                # Bob's measurement: correlated with Alice's outcome
                angle_diff = a_angle - b_angle
                # Probability Bob gets same outcome as Alice
                p_same = math.cos(angle_diff / 2.0) ** 2

                if self.rng.random() < p_same:
                    b_outcome = a_outcome
                else:
                    b_outcome = 1 - a_outcome

                # Apply channel noise (bit flip on Bob's side)
                if self.rng.random() < channel.error_rate:
                    b_outcome = 1 - b_outcome

            alice_outcomes.append(a_outcome)
            bob_outcomes.append(b_outcome)
            alice_angles_used.append(a_angle)
            bob_angles_used.append(b_angle)
            alice_basis_list.append(alice_basis_choices[i])
            bob_basis_list.append(bob_basis_choices[i])

        result.n_bits_received = len(alice_outcomes)

        if result.n_bits_received < 10:
            result.secure = False
            result.qber = 1.0
            result.chsh_s_value = 0.0
            return result

        # Step 3: CHSH test using non-matching basis pairs
        # CHSH uses: E(A1,B1), E(A1,B3), E(A3,B1), E(A3,B3)
        # Alice basis 0 = A1 (0 deg), Alice basis 2 = A3 (90 deg)
        # Bob basis 0 = B1 (45 deg), Bob basis 2 = B3 (135 deg)
        chsh_correlations = self._compute_chsh_correlations(
            alice_outcomes,
            bob_outcomes,
            alice_basis_list,
            bob_basis_list,
        )

        s_value = self._compute_s_value(chsh_correlations)
        result.chsh_s_value = s_value

        # Step 4: Extract key from matching-basis pairs
        # Alice basis 1 (45 deg) == Bob basis 0 (45 deg)
        # Alice basis 2 (90 deg) == Bob basis 1 (90 deg)
        key_alice: List[int] = []
        key_bob: List[int] = []

        for i in range(len(alice_outcomes)):
            a_angle = alice_angles_used[i]
            b_angle = bob_angles_used[i]
            if abs(a_angle - b_angle) < 1e-10:
                key_alice.append(alice_outcomes[i])
                key_bob.append(bob_outcomes[i])

        result.sifted_key_alice = key_alice
        result.sifted_key_bob = key_bob
        result.sifting_efficiency = (
            len(key_alice) / result.n_bits_received
            if result.n_bits_received > 0
            else 0.0
        )

        # Step 5: Security check from CHSH
        if s_value < self.chsh_threshold:
            result.secure = False
            result.final_key = []
            result.key_rate = 0.0
            return result

        if len(key_alice) < 4:
            result.secure = True
            result.final_key = []
            result.key_rate = 0.0
            return result

        # Step 6: QBER estimation
        n_sample = max(2, int(len(key_alice) * self.sample_fraction))
        n_sample = min(n_sample, len(key_alice) - 2)

        sample_indices = self.rng.choice(
            len(key_alice), size=n_sample, replace=False
        )
        sample_indices_set = set(sample_indices.tolist())

        sample_a = [key_alice[i] for i in sample_indices]
        sample_b = [key_bob[i] for i in sample_indices]
        result.qber = estimate_qber(sample_a, sample_b, sample_fraction=1.0)

        remaining_alice = [
            key_alice[i]
            for i in range(len(key_alice))
            if i not in sample_indices_set
        ]
        remaining_bob = [
            key_bob[i]
            for i in range(len(key_bob))
            if i not in sample_indices_set
        ]

        if len(remaining_alice) < 2:
            result.final_key = []
            result.key_rate = 0.0
            return result

        # Step 7: Error correction
        corrected_alice, corrected_bob = error_correction_cascade(
            remaining_alice, remaining_bob, passes=self.cascade_passes
        )

        # Step 8: Privacy amplification
        compression = max(0.1, 0.5)
        result.final_key = privacy_amplification(
            corrected_alice, compression, self.rng
        )

        result.key_rate = (
            len(result.final_key) / n_pairs if n_pairs > 0 else 0.0
        )

        return result

    # ------------------------------------------------------------------
    # Measurement simulation
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_in_basis(
        raw_bit: int,
        angle: float,
        rng: np.random.RandomState,
    ) -> int:
        """Simulate measuring a qubit at a given angle.

        For a qubit in state |raw_bit>, measuring in a rotated basis
        at angle theta gives outcome 0 with probability cos^2(theta/2)
        if raw_bit=0, or sin^2(theta/2) if raw_bit=0 (depending on the
        relative angle).

        For entangled pairs from |Phi+>, the correlation between Alice
        (angle a) and Bob (angle b) is: E(a,b) = -cos(a - b).

        Parameters
        ----------
        raw_bit : int
            The computational basis state (0 or 1).
        angle : float
            Measurement angle in radians.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        int
            Measurement outcome: 0 or 1.
        """
        # Probability of flipping from the raw bit depends on angle
        # In the Z-basis (angle=0), no flip. In X-basis (angle=pi/2),
        # 50% flip. The flip probability is sin^2(angle/2).
        p_flip = math.sin(angle / 2.0) ** 2

        if rng.random() < p_flip:
            return 1 - raw_bit
        return raw_bit

    # ------------------------------------------------------------------
    # CHSH computation
    # ------------------------------------------------------------------

    def _compute_chsh_correlations(
        self,
        alice_outcomes: List[int],
        bob_outcomes: List[int],
        alice_bases: List[int],
        bob_bases: List[int],
    ) -> dict:
        """Compute CHSH correlation coefficients.

        Computes E(a, b) = (N_same - N_diff) / (N_same + N_diff)
        for each of the four CHSH basis combinations.

        Parameters
        ----------
        alice_outcomes : list[int]
            Alice's measurement outcomes.
        bob_outcomes : list[int]
            Bob's measurement outcomes.
        alice_bases : list[int]
            Alice's basis choices (0, 1, 2).
        bob_bases : list[int]
            Bob's basis choices (0, 1, 2).

        Returns
        -------
        dict
            Correlations for (A1,B1), (A1,B3), (A3,B1), (A3,B3).
        """
        # CHSH pairs: (alice_basis, bob_basis)
        chsh_pairs = [
            (0, 0),  # A1, B1: angles 0, 45
            (0, 2),  # A1, B3: angles 0, 135
            (2, 0),  # A3, B1: angles 90, 45
            (2, 2),  # A3, B3: angles 90, 135
        ]

        correlations = {}
        for a_basis, b_basis in chsh_pairs:
            n_same = 0
            n_diff = 0
            for i in range(len(alice_outcomes)):
                if alice_bases[i] == a_basis and bob_bases[i] == b_basis:
                    if alice_outcomes[i] == bob_outcomes[i]:
                        n_same += 1
                    else:
                        n_diff += 1

            total = n_same + n_diff
            if total > 0:
                correlations[(a_basis, b_basis)] = (n_same - n_diff) / total
            else:
                correlations[(a_basis, b_basis)] = 0.0

        return correlations

    @staticmethod
    def _compute_s_value(correlations: dict) -> float:
        """Compute the CHSH S-parameter from correlations.

        S = |E(A1,B1) - E(A1,B3)| + |E(A3,B1) + E(A3,B3)|

        For quantum mechanics: S = 2*sqrt(2) ~ 2.828 (Tsirelson bound).
        For local hidden variables: S <= 2 (CHSH bound).

        Parameters
        ----------
        correlations : dict
            Correlation values for the four CHSH basis pairs.

        Returns
        -------
        float
            CHSH S-value.
        """
        e_a1_b1 = correlations.get((0, 0), 0.0)
        e_a1_b3 = correlations.get((0, 2), 0.0)
        e_a3_b1 = correlations.get((2, 0), 0.0)
        e_a3_b3 = correlations.get((2, 2), 0.0)

        s = abs(e_a1_b1 - e_a1_b3) + abs(e_a3_b1 + e_a3_b3)
        return s
