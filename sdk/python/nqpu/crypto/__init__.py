"""nQPU Quantum Cryptography -- Protocols and primitives for quantum-secure communication.

Implements five families of quantum cryptographic protocols:

1. **Primitives**: Quantum one-time pad, authentication codes, universal
   hashing for privacy amplification, and quantum fingerprinting.

2. **Quantum Money**: Wiesner's private-key scheme and Aaronson-Christiano
   public-key quantum money with no-cloning-based unforgeability.

3. **Blind Computation**: Broadbent-Fitzsimons-Kashefi (BFK) protocol for
   universal blind quantum computation with trap-based verification.

4. **Secret Sharing**: GHZ-based (n,n) sharing, general (k,n) threshold
   quantum secret sharing, and classical Shamir sharing with quantum RNG.

5. **Oblivious Transfer**: Quantum 1-out-of-2 oblivious transfer using
   conjugate coding for simultaneous sender and receiver privacy.

Example:
    from nqpu.crypto import QuantumOneTimePad, WiesnerMoney, BFKProtocol

    qotp = QuantumOneTimePad(n_qubits=2)
    key = qotp.generate_key()

    bank = WiesnerMoney(n_qubits=16)
    serial, note = bank.mint(serial=1)
    assert bank.verify(serial, note)

    bfk = BFKProtocol(n_computation_qubits=2, n_layers=3)
    result = bfk.run_blind([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
"""

# -- Primitives --------------------------------------------------------------
from .primitives import (
    QuantumOneTimePad,
    QuantumAuthCode,
    UniversalHash,
    QuantumFingerprinting,
)

# -- Quantum money ------------------------------------------------------------
from .quantum_money import (
    WiesnerMoney,
    PublicKeyMoney,
    MoneySecurityResult,
)

# -- Blind computation --------------------------------------------------------
from .blind_computation import (
    BlindQubit,
    BrickworkState,
    BFKProtocol,
    ClientState,
    BlindResult,
    BlindVerifier,
)

# -- Secret sharing -----------------------------------------------------------
from .secret_sharing import (
    GHZSecretSharing,
    ThresholdQSS,
    ClassicalQSS,
    QSSSecurityResult,
)

# -- Oblivious transfer ------------------------------------------------------
from .oblivious_transfer import (
    QuantumOT,
    SenderState,
    ReceiverState,
    OTResult,
)

__all__ = [
    # Primitives
    "QuantumOneTimePad",
    "QuantumAuthCode",
    "UniversalHash",
    "QuantumFingerprinting",
    # Quantum money
    "WiesnerMoney",
    "PublicKeyMoney",
    "MoneySecurityResult",
    # Blind computation
    "BlindQubit",
    "BrickworkState",
    "BFKProtocol",
    "ClientState",
    "BlindResult",
    "BlindVerifier",
    # Secret sharing
    "GHZSecretSharing",
    "ThresholdQSS",
    "ClassicalQSS",
    "QSSSecurityResult",
    # Oblivious transfer
    "QuantumOT",
    "SenderState",
    "ReceiverState",
    "OTResult",
]
