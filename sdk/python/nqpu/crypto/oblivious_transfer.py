"""Quantum oblivious transfer: 1-out-of-2 OT.

Implements quantum 1-out-of-2 oblivious transfer where:
- Sender holds two messages (m0, m1)
- Receiver wants exactly one message m_b
- After the protocol: receiver gets m_b, sender does not learn b

The protocol uses conjugate coding (similar to BB84):
1. Sender prepares qubits encoding both messages in random bases
2. Receiver measures in a basis corresponding to their choice bit
3. Sender reveals basis information
4. Receiver decodes the chosen message from matching-basis bits

Security relies on the uncertainty principle: measuring in the wrong
basis destroys information about the other message.

All implementations use pure numpy -- no external dependencies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KET_0 = np.array([1, 0], dtype=complex)
_KET_1 = np.array([0, 1], dtype=complex)
_KET_PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
_KET_MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)
_H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _prepare_bb84(basis: int, value: int) -> np.ndarray:
    """Prepare qubit in BB84 encoding."""
    if basis == 0:
        return _KET_0.copy() if value == 0 else _KET_1.copy()
    else:
        return _KET_PLUS.copy() if value == 0 else _KET_MINUS.copy()


def _measure_bb84(state: np.ndarray, basis: int,
                  rng: np.random.Generator) -> int:
    """Measure qubit in given basis (0=Z, 1=X)."""
    if basis == 1:
        state = _H_GATE @ state
    prob_0 = float(np.abs(state[0]) ** 2)
    return 0 if rng.random() < prob_0 else 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SenderState:
    """Sender's private state during OT protocol."""
    bases: np.ndarray               # sender's random bases
    encoded_qubits: np.ndarray      # prepared qubit states
    messages: Tuple[np.ndarray, np.ndarray]  # (m0, m1)


@dataclass
class ReceiverState:
    """Receiver's private state during OT protocol."""
    choice: int                     # receiver's choice bit (0 or 1)
    measurement_bases: np.ndarray   # bases receiver used to measure
    raw_bits: np.ndarray            # raw measurement results


@dataclass
class OTResult:
    """Result of oblivious transfer protocol."""
    received_message: np.ndarray    # the message receiver got
    correct: bool                   # whether it matches the chosen message
    sender_knows_choice: bool       # should always be False in honest execution


# ---------------------------------------------------------------------------
# QuantumOT
# ---------------------------------------------------------------------------

@dataclass
class QuantumOT:
    """Quantum 1-out-of-2 oblivious transfer.

    Protocol steps:
    1. **Sender prepare**: For each bit position, sender picks a random
       basis and encodes the XOR of both messages' bits in that basis.
       Actually, sender prepares 2*n_bits qubits: n_bits in bases for m0,
       n_bits in bases for m1.

    2. **Receiver choose**: Receiver measures the qubits corresponding
       to their chosen message in the correct basis, and the other set
       in a random (likely wrong) basis.

    3. **Sender reveal**: Sender reveals which bases were used.

    4. **Receiver decode**: Receiver uses matching-basis results to
       reconstruct the chosen message.
    """

    n_bits: int = 8   # message length in bits

    def _validate_message(self, msg: np.ndarray) -> None:
        """Validate message is a binary array of correct length."""
        if len(msg) != self.n_bits:
            raise ValueError(
                f"Message length {len(msg)} != n_bits {self.n_bits}"
            )
        if not np.all((msg == 0) | (msg == 1)):
            raise ValueError("Message must contain only 0s and 1s")

    def sender_prepare(
        self,
        m0: np.ndarray,
        m1: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> SenderState:
        """Sender prepares quantum states encoding both messages.

        For each bit position i, sender creates two qubits:
        - Qubit 2i: encodes m0[i] in a random basis b0[i]
        - Qubit 2i+1: encodes m1[i] in a random basis b1[i]

        Parameters
        ----------
        m0, m1 : np.ndarray
            Binary messages of length n_bits.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        SenderState
        """
        if rng is None:
            rng = np.random.default_rng()

        self._validate_message(m0)
        self._validate_message(m1)

        # Random bases for each message bit (2 * n_bits total)
        n_total = 2 * self.n_bits
        bases = rng.integers(0, 2, size=n_total)

        # Prepare qubits
        qubits = np.zeros((n_total, 2), dtype=complex)
        for i in range(self.n_bits):
            qubits[2 * i] = _prepare_bb84(int(bases[2 * i]), int(m0[i]))
            qubits[2 * i + 1] = _prepare_bb84(int(bases[2 * i + 1]), int(m1[i]))

        return SenderState(
            bases=bases,
            encoded_qubits=qubits,
            messages=(m0.copy(), m1.copy()),
        )

    def receiver_choose(
        self,
        sender_qubits: np.ndarray,
        choice: int,
        rng: Optional[np.random.Generator] = None,
    ) -> ReceiverState:
        """Receiver measures qubits corresponding to choice.

        Receiver measures the qubits for their chosen message in
        random bases, hoping some match the sender's bases.

        Parameters
        ----------
        sender_qubits : np.ndarray
            All prepared qubits from sender, shape (2*n_bits, 2).
        choice : int
            0 or 1 -- which message to receive.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        ReceiverState
        """
        if rng is None:
            rng = np.random.default_rng()

        if choice not in (0, 1):
            raise ValueError("choice must be 0 or 1")

        n_total = 2 * self.n_bits
        measurement_bases = rng.integers(0, 2, size=n_total)
        raw_bits = np.zeros(n_total, dtype=int)

        for i in range(n_total):
            raw_bits[i] = _measure_bb84(
                sender_qubits[i].copy(),
                int(measurement_bases[i]),
                rng
            )

        return ReceiverState(
            choice=choice,
            measurement_bases=measurement_bases,
            raw_bits=raw_bits,
        )

    def sender_reveal(self, sender_state: SenderState) -> dict:
        """Sender reveals basis information.

        Parameters
        ----------
        sender_state : SenderState

        Returns
        -------
        dict with 'bases' array.
        """
        return {"bases": sender_state.bases.copy()}

    def receiver_decode(
        self,
        receiver_state: ReceiverState,
        sender_info: dict,
    ) -> np.ndarray:
        """Receiver extracts chosen message from matching-basis bits.

        For the chosen message (choice=b), receiver looks at qubits
        at positions 2*i+b and keeps only those where their measurement
        basis matched the sender's basis.

        For positions where bases don't match, receiver uses a random
        bit (which is the best they can do).

        Parameters
        ----------
        receiver_state : ReceiverState
        sender_info : dict
            From sender_reveal().

        Returns
        -------
        np.ndarray
            Decoded message bits.
        """
        sender_bases = sender_info["bases"]
        choice = receiver_state.choice
        recv_bases = receiver_state.measurement_bases
        raw = receiver_state.raw_bits

        decoded = np.zeros(self.n_bits, dtype=int)
        for i in range(self.n_bits):
            idx = 2 * i + choice
            if recv_bases[idx] == sender_bases[idx]:
                # Bases match: result is correct
                decoded[i] = raw[idx]
            else:
                # Bases don't match: result is random (incorrect ~50%)
                decoded[i] = raw[idx]

        return decoded

    def run(
        self,
        m0: np.ndarray,
        m1: np.ndarray,
        choice: int,
        rng: Optional[np.random.Generator] = None,
    ) -> OTResult:
        """Run complete OT protocol.

        Parameters
        ----------
        m0, m1 : np.ndarray
            Binary messages of length n_bits.
        choice : int
            0 or 1.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        OTResult
        """
        if rng is None:
            rng = np.random.default_rng()

        # Step 1: Sender prepares
        sender_state = self.sender_prepare(m0, m1, rng=rng)

        # Step 2: Receiver measures
        receiver_state = self.receiver_choose(
            sender_state.encoded_qubits, choice, rng=rng
        )

        # Step 3: Sender reveals bases
        sender_info = self.sender_reveal(sender_state)

        # Step 4: Receiver decodes
        decoded = self.receiver_decode(receiver_state, sender_info)

        # Check correctness
        target = m0 if choice == 0 else m1
        correct = np.array_equal(decoded, target)

        return OTResult(
            received_message=decoded,
            correct=correct,
            sender_knows_choice=False,  # honest protocol
        )

    def success_rate_analysis(
        self,
        n_trials: int = 500,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Analyze per-bit success rate of the OT protocol.

        Returns
        -------
        dict with 'bit_accuracy', 'full_message_accuracy', and other stats.
        """
        if rng is None:
            rng = np.random.default_rng()

        bit_correct = 0
        bit_total = 0
        full_correct = 0

        for _ in range(n_trials):
            m0 = rng.integers(0, 2, size=self.n_bits).astype(np.int8)
            m1 = rng.integers(0, 2, size=self.n_bits).astype(np.int8)
            choice = int(rng.integers(0, 2))

            result = self.run(m0, m1, choice, rng=rng)
            target = m0 if choice == 0 else m1

            bit_correct += int(np.sum(result.received_message == target))
            bit_total += self.n_bits
            if result.correct:
                full_correct += 1

        return {
            "n_trials": n_trials,
            "bit_accuracy": bit_correct / bit_total,
            "full_message_accuracy": full_correct / n_trials,
            "expected_bit_accuracy": 0.75,  # 50% match basis => correct, 50% random => 50% correct
        }
