"""Quantum channel simulation for QKD protocols.

Models the physical quantum channel between Alice and Bob, including:
  - Depolarising noise (random bit/phase flips from environmental decoherence)
  - Photon loss (absorption and scattering in optical fibre)
  - Eavesdropping (Eve's intercept-resend or entanglement-breaking attacks)

The fibre-optic loss model uses the standard telecom formula:
    loss = 1 - 10^(-alpha * L / 10)
where alpha ~ 0.2 dB/km at 1550 nm wavelength (telecom C-band).

References:
    - Gisin et al., Rev. Mod. Phys. 74, 145 (2002)
    - Scarani et al., Rev. Mod. Phys. 81, 1301 (2009)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class EavesdropperConfig:
    """Configuration for an eavesdropper (Eve) on the quantum channel.

    Parameters
    ----------
    strategy : str
        Attack strategy. Supported: 'intercept_resend', 'entanglement_breaking'.
    interception_rate : float
        Fraction of qubits Eve intercepts. Range [0, 1].
        1.0 means Eve intercepts every qubit.
    """

    strategy: str = "intercept_resend"
    interception_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.strategy not in ("intercept_resend", "entanglement_breaking"):
            raise ValueError(
                f"Unknown eavesdropper strategy: {self.strategy!r}. "
                f"Supported: 'intercept_resend', 'entanglement_breaking'."
            )
        if not 0.0 <= self.interception_rate <= 1.0:
            raise ValueError(
                f"interception_rate must be in [0, 1], got {self.interception_rate}"
            )


@dataclass
class QuantumChannel:
    """Simulated quantum channel for QKD.

    Models fibre-optic or free-space quantum communication links with
    configurable noise, loss, and eavesdropping.

    Parameters
    ----------
    error_rate : float
        Depolarising error probability per qubit. Each transmitted qubit
        has this probability of being flipped to the wrong value.
    loss_probability : float
        Probability that a photon is lost in transit. If a distance and
        attenuation are specified, this is computed automatically.
    eavesdropper : EavesdropperConfig, optional
        If set, an eavesdropper intercepts qubits according to this config.
    distance_km : float, optional
        Fibre length in kilometres. If specified, ``loss_probability`` is
        computed from the fibre attenuation model and overrides any
        manually-set value.
    attenuation_db_per_km : float
        Fibre attenuation coefficient in dB/km. Default 0.2 dB/km
        corresponds to standard single-mode fibre at 1550 nm.
    """

    error_rate: float = 0.0
    loss_probability: float = 0.0
    eavesdropper: Optional[EavesdropperConfig] = None
    distance_km: Optional[float] = None
    attenuation_db_per_km: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError(
                f"error_rate must be in [0, 1], got {self.error_rate}"
            )
        if self.distance_km is not None:
            if self.distance_km < 0:
                raise ValueError(
                    f"distance_km must be non-negative, got {self.distance_km}"
                )
            self.loss_probability = self._fiber_loss(
                self.distance_km, self.attenuation_db_per_km
            )
        if not 0.0 <= self.loss_probability <= 1.0:
            raise ValueError(
                f"loss_probability must be in [0, 1], got {self.loss_probability}"
            )

    # ------------------------------------------------------------------
    # Fibre-optic loss model
    # ------------------------------------------------------------------

    @staticmethod
    def _fiber_loss(distance_km: float, alpha_db_per_km: float) -> float:
        """Compute photon loss probability from fibre attenuation.

        Uses the standard model:
            transmittance = 10^(-alpha * L / 10)
            loss = 1 - transmittance

        Parameters
        ----------
        distance_km : float
            Fibre length in kilometres.
        alpha_db_per_km : float
            Attenuation in dB/km.

        Returns
        -------
        float
            Loss probability in [0, 1].
        """
        if distance_km <= 0.0:
            return 0.0
        total_db = alpha_db_per_km * distance_km
        transmittance = 10.0 ** (-total_db / 10.0)
        return 1.0 - transmittance

    @staticmethod
    def fiber_loss_for_distance(
        distance_km: float, alpha_db_per_km: float = 0.2
    ) -> float:
        """Public helper to compute fibre loss for a given distance.

        Parameters
        ----------
        distance_km : float
            Fibre length in kilometres.
        alpha_db_per_km : float
            Attenuation coefficient (default 0.2 dB/km).

        Returns
        -------
        float
            Loss probability.
        """
        return QuantumChannel._fiber_loss(distance_km, alpha_db_per_km)

    # ------------------------------------------------------------------
    # Qubit transmission
    # ------------------------------------------------------------------

    def transmit_qubit(
        self,
        bit: int,
        basis: int,
        rng: np.random.RandomState,
    ) -> Optional[Tuple[int, int]]:
        """Transmit a single qubit through the channel.

        Alice prepares the qubit in the given bit value and basis, and the
        channel may lose it, add noise, or let Eve intercept it.

        Parameters
        ----------
        bit : int
            The bit value Alice encodes (0 or 1).
        basis : int
            The basis Alice uses: 0 = rectilinear (Z), 1 = diagonal (X).
        rng : np.random.RandomState
            Random number generator for reproducibility.

        Returns
        -------
        tuple[int, int] or None
            ``(received_bit, original_basis)`` if the photon arrives, or
            ``None`` if the photon is lost.
        """
        # Step 1: Check for photon loss
        if rng.random() < self.loss_probability:
            return None

        received_bit = bit

        # Step 2: Eve's interception (if present)
        if self.eavesdropper is not None:
            received_bit = self._apply_eavesdropper(
                received_bit, basis, rng
            )

        # Step 3: Channel noise (depolarising)
        if self.error_rate > 0 and rng.random() < self.error_rate:
            received_bit = 1 - received_bit

        return (received_bit, basis)

    def transmit_entangled_pair(
        self,
        rng: np.random.RandomState,
    ) -> Optional[Tuple[int, int]]:
        """Transmit one half of an entangled Bell pair through the channel.

        For E91 protocol: the source creates |Phi+> = (|00> + |11>) / sqrt(2).
        Alice keeps one photon (assumed perfect local channel), Bob receives
        the other through this channel.

        Returns
        -------
        tuple[int, int] or None
            ``(alice_outcome, bob_outcome)`` in the computational basis
            before any basis rotation, or ``None`` if Bob's photon is lost.
        """
        # Check for photon loss on Bob's side
        if rng.random() < self.loss_probability:
            return None

        # Perfect Bell state: 50/50 |00> or |11>
        outcome = rng.randint(0, 2)

        # Entanglement-breaking attack by Eve
        if (
            self.eavesdropper is not None
            and self.eavesdropper.strategy == "entanglement_breaking"
        ):
            if rng.random() < self.eavesdropper.interception_rate:
                # Eve measures and re-prepares, destroying entanglement.
                # Bob's outcome becomes independently random.
                bob_outcome = rng.randint(0, 2)
                return (outcome, bob_outcome)

        # Channel noise on Bob's photon
        bob_outcome = outcome
        if self.error_rate > 0 and rng.random() < self.error_rate:
            bob_outcome = 1 - bob_outcome

        return (outcome, bob_outcome)

    # ------------------------------------------------------------------
    # Internal: eavesdropper attack
    # ------------------------------------------------------------------

    def _apply_eavesdropper(
        self,
        bit: int,
        basis: int,
        rng: np.random.RandomState,
    ) -> int:
        """Apply Eve's intercept-resend attack.

        Eve intercepts the qubit and measures it in a randomly chosen basis.
        If her basis matches Alice's, she learns the bit perfectly and
        re-sends it unchanged. If her basis is wrong, she measures a random
        outcome and re-sends that, introducing a 25% error rate on those
        qubits (50% chance of wrong basis * 50% chance of flipped bit).

        Parameters
        ----------
        bit : int
            The current bit value.
        basis : int
            Alice's preparation basis.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        int
            The bit value after Eve's interference.
        """
        eve = self.eavesdropper
        assert eve is not None

        if rng.random() >= eve.interception_rate:
            return bit

        # Eve chooses a random basis
        eve_basis = rng.randint(0, 2)

        if eve_basis == basis:
            # Eve measures in the correct basis: learns the bit exactly
            return bit
        else:
            # Wrong basis: Eve gets a random result
            return rng.randint(0, 2)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [
            f"error_rate={self.error_rate}",
            f"loss={self.loss_probability:.4f}",
        ]
        if self.distance_km is not None:
            parts.append(f"distance={self.distance_km}km")
        if self.eavesdropper is not None:
            parts.append(f"eve={self.eavesdropper.strategy}")
        return f"QuantumChannel({', '.join(parts)})"
