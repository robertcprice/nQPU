"""nQPU Tensor Networks -- MPS, MPO, DMRG, and TEBD algorithms.

A pure-Python (numpy-only) tensor network library for simulating
one-dimensional quantum many-body systems.  Mirrors the capabilities
of the Rust ``tensor_networks`` module in an accessible Python API.

Modules:

1. **Tensors**: Named-index tensors with contraction, SVD/QR decomposition,
   and network-level contraction planning.

2. **MPS**: Matrix Product States for efficient representation of 1-D
   quantum states with controllable bond dimension.

3. **MPO**: Matrix Product Operators for Hamiltonians (Ising, Heisenberg)
   with expectation values and state application.

4. **DMRG**: Two-site Density Matrix Renormalization Group for finding
   ground states via variational sweeping optimisation.

5. **TEBD**: Time Evolving Block Decimation for real and imaginary time
   evolution with Suzuki-Trotter decomposition.

Example -- Find the Ising ground state with DMRG:

    from nqpu.tensor_networks import IsingMPO, dmrg_ground_state

    H = IsingMPO(8, J=1.0, h=1.0)
    result = dmrg_ground_state(H, chi_max=32, n_sweeps=20)
    print(f"Ground state energy: {result.energy:.8f}")
    print(f"Converged: {result.converged}")

Example -- Imaginary time cooling with TEBD:

    from nqpu.tensor_networks import (
        ProductState, ising_nn_hamiltonian, ImaginaryTEBD,
    )

    psi = ProductState(6)
    H = ising_nn_hamiltonian(6, J=1.0, h=1.0)
    cooler = ImaginaryTEBD(psi, H, chi_max=16)
    result = cooler.run(dt=0.05, n_steps=200)
    print(f"Final energy approx: check via MPO expectation")
"""

from __future__ import annotations

# -- Core tensors ------------------------------------------------------
from .tensor import (
    Tensor,
    TensorNetwork,
    contract_pair,
)

# -- Matrix Product States ---------------------------------------------
from .mps import (
    MPS,
    ProductState,
    GHZState,
    RandomMPS,
    WState,
)

# -- Matrix Product Operators ------------------------------------------
from .mpo import (
    MPO,
    IsingMPO,
    HeisenbergMPO,
    IdentityMPO,
    XXModelMPO,
)

# -- DMRG --------------------------------------------------------------
from .dmrg import (
    DMRG,
    DMRGResult,
    dmrg_ground_state,
)

# -- TEBD --------------------------------------------------------------
from .tebd import (
    TEBD,
    TEBDResult,
    ImaginaryTEBD,
    NNHamiltonian,
    ising_nn_hamiltonian,
    heisenberg_nn_hamiltonian,
    tebd_evolve,
)

# -- PEPS -----------------------------------------------------------------
from .peps import (
    PEPSTensor, PEPS, BoundaryMPS, SimpleUpdate, SimpleUpdateResult,
    ising_2d_bonds, heisenberg_2d_bonds,
)

# -- TDVP -----------------------------------------------------------------
from .tdvp import (
    TDVPResult, TDVP1Site, TDVP2Site,
    matrix_exponential_action, krylov_expm,
)

# -- Autodiff -----------------------------------------------------------------
from .autodiff import (
    TensorNode, tensor_node, contract, trace, svd, backward,
    DifferentiableContraction, VariationalTN, OptimizationResult,
)

# -- TN Machine Learning ------------------------------------------------------
from .tn_ml import (
    MPSClassifier, TNKernel, MLResult,
)

__all__ = [
    # Core tensors
    "Tensor",
    "TensorNetwork",
    "contract_pair",
    # MPS
    "MPS",
    "ProductState",
    "GHZState",
    "RandomMPS",
    "WState",
    # MPO
    "MPO",
    "IsingMPO",
    "HeisenbergMPO",
    "IdentityMPO",
    "XXModelMPO",
    # DMRG
    "DMRG",
    "DMRGResult",
    "dmrg_ground_state",
    # TEBD
    "TEBD",
    "TEBDResult",
    "ImaginaryTEBD",
    "NNHamiltonian",
    "ising_nn_hamiltonian",
    "heisenberg_nn_hamiltonian",
    "tebd_evolve",
    # PEPS
    "PEPSTensor",
    "PEPS",
    "BoundaryMPS",
    "SimpleUpdate",
    "SimpleUpdateResult",
    "ising_2d_bonds",
    "heisenberg_2d_bonds",
    # TDVP
    "TDVPResult",
    "TDVP1Site",
    "TDVP2Site",
    "matrix_exponential_action",
    "krylov_expm",
    # Autodiff
    "TensorNode",
    "tensor_node",
    "contract",
    "trace",
    "svd",
    "backward",
    "DifferentiableContraction",
    "VariationalTN",
    "OptimizationResult",
    # TN Machine Learning
    "MPSClassifier",
    "TNKernel",
    "MLResult",
]
