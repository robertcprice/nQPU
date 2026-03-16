"""nQPU Quantum Key Distribution -- Protocol simulation and analysis.

Implements the three foundational QKD protocols with full post-processing
pipelines, eavesdropper simulation, and multi-node network support.

Protocols:
  - BB84: Bennett-Brassard 1984 (prepare-and-measure, two conjugate bases)
  - E91: Ekert 1991 (entanglement-based, CHSH eavesdropping detection)
  - B92: Bennett 1992 (simplified two-state protocol)

Example:
    from nqpu.qkd import BB84Protocol, QuantumChannel, QKDResult

    channel = QuantumChannel(error_rate=0.02, loss_probability=0.1)
    protocol = BB84Protocol(seed=42)
    result = protocol.generate_key(n_bits=10000, channel=channel)
    print(f"QBER: {result.qber:.4f}, Key length: {len(result.final_key)}")
"""

from .channel import QuantumChannel, EavesdropperConfig
from .bb84 import BB84Protocol
from .e91 import E91Protocol
from .b92 import B92Protocol
from .privacy import (
    error_correction_cascade,
    privacy_amplification,
    estimate_qber,
    toeplitz_hash,
)
from .network import QKDNode, QKDNetwork, QKDResult

__all__ = [
    # Channel
    "QuantumChannel",
    "EavesdropperConfig",
    # Protocols
    "BB84Protocol",
    "E91Protocol",
    "B92Protocol",
    # Post-processing
    "error_correction_cascade",
    "privacy_amplification",
    "estimate_qber",
    "toeplitz_hash",
    # Network
    "QKDNode",
    "QKDNetwork",
    # Results
    "QKDResult",
]
