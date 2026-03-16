"""Quantum computing cost estimation.

Estimates costs for quantum computations across cloud providers,
including QEC overhead calculations for surface and colour codes.
Provides budget optimisation and break-even analysis.

Example::

    from nqpu.dashboard import estimate_cost, compare_providers
    est = estimate_cost(n_qubits=10, depth=50, shots=1024)
    print(est.report())
    print(compare_providers(n_qubits=10, depth=50))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class CloudProvider:
    """Cloud quantum computing provider."""

    name: str
    devices: list  # device names
    pricing_model: str  # "per_shot", "per_minute", "per_task"
    base_cost: float  # base cost per unit
    queue_time_minutes: float  # typical queue time


@dataclass
class QECOverhead:
    """Error correction overhead estimation."""

    physical_qubits_per_logical: int
    code_distance: int
    logical_error_rate: float
    space_overhead: float  # total physical / logical
    time_overhead: float  # syndrome cycles per logical operation

    @staticmethod
    def surface_code(
        target_error: float,
        physical_error: float,
    ) -> QECOverhead:
        """Estimate surface code overhead.

        Uses the threshold model: logical error ~ (p/p_th)^((d+1)/2)
        where p_th ~ 0.01 for the surface code.

        Parameters
        ----------
        target_error : float
            Target logical error rate per operation.
        physical_error : float
            Physical gate error rate.
        """
        p_th = 0.01  # surface code threshold
        if physical_error >= p_th:
            # Below threshold: can't do useful QEC
            return QECOverhead(
                physical_qubits_per_logical=0,
                code_distance=0,
                logical_error_rate=physical_error,
                space_overhead=1.0,
                time_overhead=1.0,
            )

        # d such that (p/p_th)^((d+1)/2) <= target_error
        ratio = physical_error / p_th
        if ratio <= 0 or target_error <= 0:
            d = 3
        else:
            # (d+1)/2 >= log(target_error) / log(ratio)
            d_continuous = 2.0 * math.log(target_error) / math.log(ratio) - 1.0
            d = max(3, int(math.ceil(d_continuous)))

        # Ensure d is odd
        if d % 2 == 0:
            d += 1

        physical_per_logical = 2 * d * d
        logical_rate = ratio ** ((d + 1) / 2)
        space_overhead = float(physical_per_logical)
        time_overhead = float(d)  # syndrome measurement rounds

        return QECOverhead(
            physical_qubits_per_logical=physical_per_logical,
            code_distance=d,
            logical_error_rate=logical_rate,
            space_overhead=space_overhead,
            time_overhead=time_overhead,
        )

    @staticmethod
    def color_code(
        target_error: float,
        physical_error: float,
    ) -> QECOverhead:
        """Estimate colour code overhead.

        The 2D colour code has threshold ~ 0.0082 and encodes 1 logical
        qubit in (3d^2 + 1)/4 physical qubits for distance d (triangular).
        We use a simplified model similar to surface code.

        Parameters
        ----------
        target_error : float
            Target logical error rate per operation.
        physical_error : float
            Physical gate error rate.
        """
        p_th = 0.0082
        if physical_error >= p_th:
            return QECOverhead(
                physical_qubits_per_logical=0,
                code_distance=0,
                logical_error_rate=physical_error,
                space_overhead=1.0,
                time_overhead=1.0,
            )

        ratio = physical_error / p_th
        if ratio <= 0 or target_error <= 0:
            d = 3
        else:
            d_continuous = 2.0 * math.log(target_error) / math.log(ratio) - 1.0
            d = max(3, int(math.ceil(d_continuous)))

        if d % 2 == 0:
            d += 1

        # Colour code: roughly (3d^2 + 1)/4 data qubits, but with ancillas
        # Use simplified 1.5 * d^2 physical per logical
        physical_per_logical = int(math.ceil(1.5 * d * d))
        logical_rate = ratio ** ((d + 1) / 2)

        return QECOverhead(
            physical_qubits_per_logical=physical_per_logical,
            code_distance=d,
            logical_error_rate=logical_rate,
            space_overhead=float(physical_per_logical),
            time_overhead=float(d),
        )


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""

    compute_cost: float
    error_correction_overhead: float
    classical_processing: float
    total: float

    def ascii_breakdown(self) -> str:
        """ASCII cost breakdown."""
        lines = [
            "COST BREAKDOWN",
            "-" * 40,
            f"  Compute:            ${self.compute_cost:.4f}",
            f"  QEC overhead:       ${self.error_correction_overhead:.4f}",
            f"  Classical proc.:    ${self.classical_processing:.4f}",
            "-" * 40,
            f"  TOTAL:              ${self.total:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class CostEstimate:
    """Complete cost estimate for a quantum computation."""

    provider: str
    device: str
    logical_qubits: int
    physical_qubits: int
    shots: int
    estimated_time_seconds: float
    qec_overhead: Optional[QECOverhead]
    breakdown: CostBreakdown
    confidence: str  # "high", "medium", "low"
    assumptions: List[str]

    def report(self) -> str:
        """ASCII cost report."""
        lines = [
            "=" * 60,
            "QUANTUM COST ESTIMATE",
            "=" * 60,
            f"  Provider:       {self.provider}",
            f"  Device:         {self.device}",
            f"  Logical qubits: {self.logical_qubits}",
            f"  Physical qubits:{self.physical_qubits}",
            f"  Shots:          {self.shots}",
            f"  Est. time:      {self.estimated_time_seconds:.2f} s",
            f"  Confidence:     {self.confidence}",
            "",
            self.breakdown.ascii_breakdown(),
        ]

        if self.qec_overhead and self.qec_overhead.code_distance > 0:
            lines.extend([
                "",
                "  QEC Details:",
                f"    Code distance:     {self.qec_overhead.code_distance}",
                f"    Phys/logical:      {self.qec_overhead.physical_qubits_per_logical}",
                f"    Logical error:     {self.qec_overhead.logical_error_rate:.2e}",
                f"    Space overhead:    {self.qec_overhead.space_overhead:.1f}x",
            ])

        if self.assumptions:
            lines.append("")
            lines.append("  Assumptions:")
            for a in self.assumptions:
                lines.append(f"    - {a}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ======================================================================
# Provider database
# ======================================================================


_DEFAULT_PROVIDERS: Dict[str, CloudProvider] = {
    "ibm_quantum": CloudProvider(
        name="IBM Quantum",
        devices=["ibm_eagle", "ibm_heron"],
        pricing_model="per_shot",
        base_cost=0.00015,
        queue_time_minutes=5.0,
    ),
    "amazon_braket": CloudProvider(
        name="Amazon Braket",
        devices=["ionq_aria", "rigetti_aspen_m3"],
        pricing_model="per_task",
        base_cost=0.30,
        queue_time_minutes=2.0,
    ),
    "azure_quantum": CloudProvider(
        name="Azure Quantum",
        devices=["ionq_aria", "quantinuum_h2"],
        pricing_model="per_shot",
        base_cost=0.003,
        queue_time_minutes=3.0,
    ),
    "google_cloud": CloudProvider(
        name="Google Cloud QC",
        devices=["google_sycamore"],
        pricing_model="per_minute",
        base_cost=5.0,
        queue_time_minutes=10.0,
    ),
    "ionq_direct": CloudProvider(
        name="IonQ Direct",
        devices=["ionq_aria", "ionq_forte"],
        pricing_model="per_shot",
        base_cost=0.003,
        queue_time_minutes=1.0,
    ),
    "quantinuum": CloudProvider(
        name="Quantinuum",
        devices=["quantinuum_h2"],
        pricing_model="per_shot",
        base_cost=0.01,
        queue_time_minutes=2.0,
    ),
}


# ======================================================================
# Cost estimator
# ======================================================================


class CostEstimator:
    """Estimate costs for quantum computations across providers.

    Builds a database of cloud quantum providers and uses realistic
    pricing models to estimate total cost including QEC overhead.
    """

    def __init__(self) -> None:
        self.providers = self._build_provider_database()

    def _build_provider_database(self) -> Dict[str, CloudProvider]:
        """Build database of cloud quantum providers."""
        return dict(_DEFAULT_PROVIDERS)

    def estimate(
        self,
        n_qubits: int,
        depth: int,
        shots: int = 1024,
        provider: Optional[str] = None,
        use_qec: bool = False,
        target_error: float = 1e-6,
    ) -> CostEstimate:
        """Estimate cost for a quantum computation.

        Parameters
        ----------
        n_qubits : int
            Number of logical qubits.
        depth : int
            Circuit depth.
        shots : int
            Number of measurement shots.
        provider : str, optional
            Provider key. Auto-selects cheapest if not specified.
        use_qec : bool
            Whether to include QEC overhead.
        target_error : float
            Target logical error rate (used when ``use_qec=True``).
        """
        if provider and provider in self.providers:
            prov = self.providers[provider]
        else:
            # Select cheapest
            prov = min(self.providers.values(), key=lambda p: p.base_cost)

        # QEC overhead
        qec: Optional[QECOverhead] = None
        physical_qubits = n_qubits
        qec_cost_multiplier = 1.0

        if use_qec:
            physical_error = 0.001  # assume typical physical error
            qec = QECOverhead.surface_code(target_error, physical_error)
            if qec.code_distance > 0:
                physical_qubits = n_qubits * qec.physical_qubits_per_logical
                qec_cost_multiplier = qec.space_overhead * qec.time_overhead
            else:
                qec_cost_multiplier = 1.0

        # Compute cost
        if prov.pricing_model == "per_shot":
            compute_cost = prov.base_cost * shots
        elif prov.pricing_model == "per_task":
            # Base cost + per-shot adder
            compute_cost = prov.base_cost + 0.00035 * shots
        elif prov.pricing_model == "per_minute":
            # Estimate execution time
            time_per_circuit_us = depth * 0.2  # rough estimate
            total_time_min = (time_per_circuit_us * shots) / (60.0 * 1e6)
            total_time_min = max(total_time_min, 1.0 / 60.0)  # minimum billing
            compute_cost = prov.base_cost * total_time_min
        else:
            compute_cost = prov.base_cost * shots

        qec_overhead_cost = compute_cost * (qec_cost_multiplier - 1.0) if use_qec else 0.0
        classical_cost = 0.001 * n_qubits * depth  # classical processing

        total = compute_cost + qec_overhead_cost + classical_cost

        # Estimate execution time
        gate_time_us = 0.2  # average gate time
        est_time = depth * gate_time_us * 1e-6 * shots

        # Confidence
        if use_qec:
            confidence = "low"
        elif n_qubits > 50:
            confidence = "medium"
        else:
            confidence = "high"

        assumptions = [
            f"Pricing model: {prov.pricing_model}",
            f"Base cost: ${prov.base_cost}",
        ]
        if use_qec:
            assumptions.append(f"Physical error rate: 0.001")
            assumptions.append(f"Target error: {target_error:.1e}")
            if qec:
                assumptions.append(f"Code distance: {qec.code_distance}")

        breakdown = CostBreakdown(
            compute_cost=compute_cost,
            error_correction_overhead=qec_overhead_cost,
            classical_processing=classical_cost,
            total=total,
        )

        device_name = prov.devices[0] if prov.devices else "unknown"

        return CostEstimate(
            provider=prov.name,
            device=device_name,
            logical_qubits=n_qubits,
            physical_qubits=physical_qubits,
            shots=shots,
            estimated_time_seconds=est_time,
            qec_overhead=qec,
            breakdown=breakdown,
            confidence=confidence,
            assumptions=assumptions,
        )

    def compare_providers(
        self,
        n_qubits: int,
        depth: int,
        shots: int = 1024,
    ) -> str:
        """ASCII comparison of all providers.

        Parameters
        ----------
        n_qubits : int
            Logical qubit count.
        depth : int
            Circuit depth.
        shots : int
            Shot count.
        """
        col_prov = 20
        col_cost = 12
        col_model = 12
        col_queue = 10
        col_device = 20

        header = (
            f"{'Provider':<{col_prov}} "
            f"{'Total ($)':>{col_cost}} "
            f"{'Model':<{col_model}} "
            f"{'Queue (m)':>{col_queue}} "
            f"{'Device':<{col_device}}"
        )
        sep = "-" * len(header)
        lines = [
            f"PROVIDER COMPARISON: {n_qubits}Q, depth={depth}, {shots} shots",
            sep,
            header,
            sep,
        ]

        estimates = []
        for key in sorted(self.providers.keys()):
            est = self.estimate(n_qubits, depth, shots, provider=key)
            estimates.append((key, est))

        estimates.sort(key=lambda x: x[1].breakdown.total)

        for key, est in estimates:
            prov = self.providers[key]
            lines.append(
                f"{prov.name:<{col_prov}} "
                f"{est.breakdown.total:>{col_cost}.4f} "
                f"{prov.pricing_model:<{col_model}} "
                f"{prov.queue_time_minutes:>{col_queue}.1f} "
                f"{est.device:<{col_device}}"
            )

        lines.append(sep)
        cheapest = estimates[0]
        lines.append(f"Cheapest: {self.providers[cheapest[0]].name} "
                      f"(${cheapest[1].breakdown.total:.4f})")
        return "\n".join(lines)

    def budget_optimizer(
        self,
        budget: float,
        n_qubits: int,
        depth: int,
        min_shots: int = 100,
    ) -> dict:
        """Find best provider and config within budget.

        Parameters
        ----------
        budget : float
            Maximum budget in USD.
        n_qubits : int
            Logical qubit count.
        depth : int
            Circuit depth.
        min_shots : int
            Minimum acceptable shots.

        Returns
        -------
        dict
            ``{provider, shots, total_cost, device}`` or empty dict if
            no provider fits.
        """
        best: Optional[dict] = None

        for key, prov in self.providers.items():
            # Binary search for max shots within budget
            lo, hi = min_shots, 1_000_000
            feasible_shots = 0

            while lo <= hi:
                mid = (lo + hi) // 2
                est = self.estimate(n_qubits, depth, mid, provider=key)
                if est.breakdown.total <= budget:
                    feasible_shots = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            if feasible_shots >= min_shots:
                est = self.estimate(n_qubits, depth, feasible_shots, provider=key)
                candidate = {
                    "provider": prov.name,
                    "shots": feasible_shots,
                    "total_cost": est.breakdown.total,
                    "device": est.device,
                }
                if best is None or feasible_shots > best["shots"]:
                    best = candidate

        return best or {}

    def break_even_analysis(
        self,
        classical_time_hours: float,
        n_qubits: int,
        depth: int,
    ) -> str:
        """Estimate when quantum becomes cost-effective vs classical.

        Parameters
        ----------
        classical_time_hours : float
            Time for classical computation.
        n_qubits : int
            Qubit count.
        depth : int
            Circuit depth.

        Returns
        -------
        str
            Analysis report.
        """
        classical_cost_per_hour = 3.0  # rough cloud GPU cost
        classical_total = classical_time_hours * classical_cost_per_hour

        # Estimate quantum cost at various shot counts
        shot_levels = [100, 1000, 10_000, 100_000]
        lines = [
            "BREAK-EVEN ANALYSIS",
            "=" * 60,
            f"  Classical time:  {classical_time_hours:.1f} hours",
            f"  Classical cost:  ${classical_total:.2f} "
            f"(at ${classical_cost_per_hour}/hr)",
            "",
            f"  {'Shots':>10} {'Quantum Cost':>15} {'Break-Even?':>15}",
            "  " + "-" * 42,
        ]

        cheapest_prov = min(self.providers.values(), key=lambda p: p.base_cost)
        prov_key = [k for k, v in self.providers.items() if v is cheapest_prov][0]

        break_even_shots = None
        for shots in shot_levels:
            est = self.estimate(n_qubits, depth, shots, provider=prov_key)
            q_cost = est.breakdown.total
            is_cheaper = q_cost < classical_total
            status = "YES" if is_cheaper else "no"
            lines.append(f"  {shots:>10} ${q_cost:>14.4f} {status:>15}")
            if is_cheaper and break_even_shots is None:
                break_even_shots = shots

        lines.append("")
        if break_even_shots is not None:
            lines.append(
                f"  Quantum is cost-effective at ~{break_even_shots} shots "
                f"using {cheapest_prov.name}"
            )
        else:
            lines.append(
                "  Quantum is NOT yet cost-effective for this workload "
                "at tested shot counts"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# ======================================================================
# Convenience functions
# ======================================================================


def estimate_cost(
    n_qubits: int,
    depth: int,
    shots: int = 1024,
) -> CostEstimate:
    """Quick cost estimate using best available provider.

    Automatically selects the cheapest provider.
    """
    estimator = CostEstimator()
    return estimator.estimate(n_qubits, depth, shots)


def compare_providers(
    n_qubits: int,
    depth: int,
    shots: int = 1024,
) -> str:
    """Quick provider comparison."""
    estimator = CostEstimator()
    return estimator.compare_providers(n_qubits, depth, shots)
