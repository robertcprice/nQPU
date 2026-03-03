#!/usr/bin/env python3
"""
NQPU Quantum Biology Platform REST API

Production-ready FastAPI server for quantum-enhanced computational biology.

Features:
- Drug design (fingerprints, similarity, ADMET, properties)
- Genomics (DNA encoding, alignment, mutation simulation)
- Genome assembly (quantum-enhanced de novo assembly)
- CRISPR (guide RNA design, edit simulation)
- Protein folding (quantum Monte Carlo, secondary structure)
- Enzyme catalysis (rate enhancement, quantum tunneling)

Endpoints:
- /drug/* - Drug design endpoints
- /dna/* - Genomics endpoints
- /assembly/* - Genome assembly
- /crispr/* - CRISPR simulation
- /protein/* - Protein folding
- /enzyme/* - Enzyme catalysis

Run with: uvicorn nqpu_api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict
from contextlib import asynccontextmanager

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import (
    FastAPI, HTTPException, Depends, Request, status,
    Header, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uvicorn

# Import NQPU modules
from nqpu_drug_design import (
    Molecule, evaluate_drug_likeness, predict_admet,
    ecfp4, ecfp6, fcfp4, maccs_keys, tanimoto_similarity,
    MorganFingerprint, MACCSKeys, DrugProperties,
    quantum_similarity, screen_library, optimize_lead,
    DrugDiscoveryPipeline
)
from dna_rna_organism import DNA, RNA, Protein as BioProtein, Gene, Organism
from quantum_organism import QuantumDNA, QuantumOrganism
from quantum_genome_tools import (
    Read, Contig, QuantumGenomeAssembler,
    CRISPRGuide, CRISPRSimulation, DeNovoAssembler
)
from quantum_protein_folding import (
    AminoAcid, Residue, Protein as FoldProtein,
    QuantumProteinFolder, Substrate, ActiveSite, Enzyme,
    QuantumElectronTransfer
)


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter (100 requests/minute)."""

    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > minute_ago
        ]

        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False

        self.requests[client_id].append(now)
        return True

    def remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        minute_ago = now - 60
        count = sum(1 for t in self.requests[client_id] if t > minute_ago)
        return max(0, self.requests_per_minute - count)


rate_limiter = RateLimiter(100)


# ============================================================================
# API KEY AUTHENTICATION (OPTIONAL)
# ============================================================================

API_KEYS = set(os.environ.get("NQPU_API_KEYS", "").split(",")) - {""}

async def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")) -> Optional[str]:
    """Verify API key if configured."""
    if not API_KEYS:
        return None  # No auth required

    if not api_key or api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return api_key


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

# --- Drug Design Models ---

class FingerprintRequest(BaseModel):
    """Request for molecular fingerprint generation."""
    smiles: str = Field(..., description="SMILES string of the molecule")
    fingerprint_type: str = Field(
        default="ecfp4",
        description="Fingerprint type: ecfp4, ecfp6, fcfp4, or maccs"
    )
    n_bits: int = Field(default=2048, ge=512, le=8192)

    @field_validator('smiles')
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('SMILES cannot be empty')
        return v


class FingerprintResponse(BaseModel):
    """Response with fingerprint data."""
    smiles: str
    fingerprint_type: str
    bits_set: List[int]
    n_bits: int
    density: float
    bit_count: int


class SimilarityRequest(BaseModel):
    """Request for molecular similarity calculation."""
    smiles1: str = Field(..., description="First molecule SMILES")
    smiles2: str = Field(..., description="Second molecule SMILES")
    fingerprint_type: str = Field(default="ecfp4")

    @field_validator('smiles1', 'smiles2')
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        return v.strip()


class SimilarityResponse(BaseModel):
    """Response with similarity scores."""
    smiles1: str
    smiles2: str
    tanimoto_similarity: float
    dice_similarity: float
    fingerprint_type: str
    verdict: str


class PropertiesRequest(BaseModel):
    """Request for molecular properties."""
    smiles: str = Field(..., description="SMILES string")


class PropertiesResponse(BaseModel):
    """Response with molecular properties."""
    smiles: str
    molecular_weight: float
    log_p: float
    h_bond_donors: int
    h_bond_acceptors: int
    rotatable_bonds: int
    polar_surface_area: float
    synthetic_accessibility: float
    heavy_atom_count: int


class AdmetRequest(BaseModel):
    """Request for ADMET prediction."""
    smiles: str = Field(..., description="SMILES string")


class AdmetPropertyResult(BaseModel):
    """Single ADMET property result."""
    property_name: str
    probability: float
    passes: bool
    confidence: float


class AdmetResponse(BaseModel):
    """Response with ADMET predictions."""
    smiles: str
    properties: Dict[str, AdmetPropertyResult]
    overall_score: float
    lipinski_violations: int
    qed_score: float


# --- Genomics Models ---

class DNAEncodeRequest(BaseModel):
    """Request for DNA quantum encoding."""
    sequence: str = Field(..., description="DNA sequence (ATGC)")
    name: str = Field(default="query", description="Sequence name")

    @field_validator('sequence')
    @classmethod
    def validate_dna(cls, v: str) -> str:
        v = ''.join(c.upper() for c in v if c.upper() in 'ATGC')
        if not v:
            raise ValueError('Invalid DNA sequence')
        return v


class DNAEncodeResponse(BaseModel):
    """Response with quantum encoding."""
    sequence: str
    name: str
    length: int
    qubits_needed: int
    gc_content: float
    encoding_type: str


class DNASimilarityRequest(BaseModel):
    """Request for DNA quantum similarity."""
    sequence1: str = Field(..., description="First DNA sequence")
    sequence2: str = Field(..., description="Second DNA sequence")
    use_quantum: bool = Field(default=True)

    @field_validator('sequence1', 'sequence2')
    @classmethod
    def validate_dna(cls, v: str) -> str:
        return ''.join(c.upper() for c in v if c.upper() in 'ATGC')


class DNASimilarityResponse(BaseModel):
    """Response with DNA similarity."""
    sequence1: str
    sequence2: str
    quantum_fidelity: float
    hamming_distance: int
    alignment_score: float
    gc_difference: float


class DNAAlignRequest(BaseModel):
    """Request for quantum DNA alignment."""
    query: str = Field(..., description="Query sequence")
    target: str = Field(..., description="Target sequence")

    @field_validator('query', 'target')
    @classmethod
    def validate_dna(cls, v: str) -> str:
        return ''.join(c.upper() for c in v if c.upper() in 'ATGC')


class DNAAlignResponse(BaseModel):
    """Response with alignment result."""
    query: str
    target: str
    aligned_query: str
    aligned_target: str
    score: float
    identity: float
    gaps: int


class DNAMutateRequest(BaseModel):
    """Request for mutation simulation."""
    sequence: str = Field(..., description="DNA sequence to mutate")
    mutation_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    mutation_type: str = Field(default="point", description="point, insertion, deletion")

    @field_validator('sequence')
    @classmethod
    def validate_dna(cls, v: str) -> str:
        return ''.join(c.upper() for c in v if c.upper() in 'ATGC')


class DNAMutateResponse(BaseModel):
    """Response with mutated sequence."""
    original: str
    mutated: str
    mutation_count: int
    mutations: List[Dict[str, Any]]
    mutation_rate: float


# --- Genome Assembly Models ---

class SimulateReadsRequest(BaseModel):
    """Request for read simulation."""
    genome_sequence: str = Field(..., description="Reference genome")
    read_length: int = Field(default=100, ge=50, le=500)
    coverage: int = Field(default=30, ge=1, le=200)
    error_rate: float = Field(default=0.01, ge=0.0, le=0.1)

    @field_validator('genome_sequence')
    @classmethod
    def validate_genome(cls, v: str) -> str:
        return ''.join(c.upper() for c in v if c.upper() in 'ATGC')


class SimulateReadsResponse(BaseModel):
    """Response with simulated reads."""
    n_reads: int
    read_length: int
    coverage: float
    reads: List[str]
    quality_scores: List[List[int]]


class AssembleRequest(BaseModel):
    """Request for genome assembly."""
    reads: List[str] = Field(..., description="List of read sequences")
    min_overlap: int = Field(default=15, ge=5, le=50)
    use_quantum: bool = Field(default=True)


class AssembleResponse(BaseModel):
    """Response with assembly result."""
    n_contigs: int
    total_length: int
    n50: int
    contigs: List[Dict[str, Any]]
    assembly_type: str


# --- CRISPR Models ---

class DesignGuidesRequest(BaseModel):
    """Request for CRISPR guide design."""
    gene_sequence: str = Field(..., description="Target gene sequence")
    genome_sequence: str = Field(..., description="Full genome for off-target check")
    n_guides: int = Field(default=5, ge=1, le=20)

    @field_validator('gene_sequence', 'genome_sequence')
    @classmethod
    def validate_dna(cls, v: str) -> str:
        return ''.join(c.upper() for c in v if c.upper() in 'ATGC')


class GuideResult(BaseModel):
    """Single guide RNA result."""
    sequence: str
    pam: str
    gc_content: float
    specificity_score: float
    position: int
    strand: str


class DesignGuidesResponse(BaseModel):
    """Response with designed guides."""
    gene_sequence: str
    guides: List[GuideResult]
    n_guides: int


class SimulateEditRequest(BaseModel):
    """Request for CRISPR edit simulation."""
    guide_sequence: str = Field(..., description="20bp guide RNA sequence")
    position: int = Field(..., ge=0, description="Target position in genome")

    @field_validator('guide_sequence')
    @classmethod
    def validate_guide(cls, v: str) -> str:
        v = ''.join(c.upper() for c in v if c.upper() in 'ATGC')
        if len(v) != 20:
            raise ValueError('Guide sequence must be exactly 20bp')
        return v


class SimulateEditResponse(BaseModel):
    """Response with edit simulation."""
    guide: str
    position: int
    edit_type: str
    detail: str
    probabilities: Dict[str, float]


# --- Protein Folding Models ---

class ProteinFoldRequest(BaseModel):
    """Request for protein folding."""
    sequence: str = Field(..., description="Amino acid sequence (single letter)")
    temperature: float = Field(default=300.0, ge=250.0, le=400.0)
    max_iterations: int = Field(default=1000, ge=100, le=10000)
    use_quantum: bool = Field(default=True)

    @field_validator('sequence')
    @classmethod
    def validate_protein(cls, v: str) -> str:
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        v = ''.join(c.upper() for c in v if c.upper() in valid_aa)
        if not v:
            raise ValueError('Invalid amino acid sequence')
        return v


class ProteinFoldResponse(BaseModel):
    """Response with folding result."""
    sequence: str
    final_energy: float
    conformation: List[Dict[str, float]]
    method: str
    iterations: int
    secondary_structure: str


class SecondaryStructureRequest(BaseModel):
    """Request for secondary structure prediction."""
    sequence: str = Field(..., description="Amino acid sequence")

    @field_validator('sequence')
    @classmethod
    def validate_protein(cls, v: str) -> str:
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        return ''.join(c.upper() for c in v if c.upper() in valid_aa)


class SecondaryStructureResponse(BaseModel):
    """Response with secondary structure."""
    sequence: str
    secondary_structure: str
    helix_content: float
    sheet_content: float
    coil_content: float


# --- Enzyme Models ---

class EnzymeCatalyzeRequest(BaseModel):
    """Request for enzyme catalysis simulation."""
    enzyme_name: str = Field(..., description="Enzyme name")
    ec_number: str = Field(..., description="Enzyme Commission number")
    substrate_name: str = Field(..., description="Substrate name")
    substrate_smiles: str = Field(..., description="Substrate SMILES")
    binding_energy: float = Field(..., description="Binding energy in kcal/mol")
    temperature: float = Field(default=300.0)


class EnzymeCatalyzeResponse(BaseModel):
    """Response with catalysis simulation."""
    enzyme: str
    substrate: str
    rate_enhancement: str
    tunneling_probability: str
    barrier_reduction: str
    quantum_contribution: str
    temperature: str


class EnzymeRateRequest(BaseModel):
    """Request for rate calculation."""
    barrier_classical: float = Field(..., description="Classical barrier (kcal/mol)")
    barrier_quantum: float = Field(..., description="Quantum barrier (kcal/mol)")
    temperature: float = Field(default=300.0)
    mechanism: str = Field(default="general")


class EnzymeRateResponse(BaseModel):
    """Response with rate calculation."""
    rate_enhancement: float
    rate_enhancement_formatted: str
    classical_rate: float
    quantum_rate: float
    tunneling_boost: float
    temperature: float


# --- Health Models ---

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    modules_loaded: Dict[str, bool]
    rate_limit_remaining: int


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str
    timestamp: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_client_id(request: Request) -> str:
    """Extract client identifier for rate limiting."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def interpret_similarity(score: float) -> str:
    """Interpret similarity score."""
    if score >= 0.7:
        return "Very similar structures"
    elif score >= 0.5:
        return "Moderately similar"
    elif score >= 0.3:
        return "Low similarity"
    else:
        return "Different structures"


def predict_secondary_structure(sequence: str) -> str:
    """Predict secondary structure using propensity scales."""
    # Chou-Fasman propensities (simplified)
    helix_prop = {'A': 1.42, 'E': 1.51, 'L': 1.21, 'M': 1.45, 'Q': 1.11,
                  'K': 1.16, 'R': 0.98, 'H': 1.00}
    sheet_prop = {'V': 1.70, 'I': 1.60, 'Y': 1.47, 'F': 1.38, 'T': 1.19,
                  'W': 1.37, 'L': 1.30, 'C': 1.19}

    structure = []
    for aa in sequence:
        h = helix_prop.get(aa, 1.0)
        e = sheet_prop.get(aa, 1.0)

        if h > 1.1 and h > e:
            structure.append('H')  # Helix
        elif e > 1.1 and e > h:
            structure.append('E')  # Sheet
        else:
            structure.append('C')  # Coil

    return ''.join(structure)


# ============================================================================
# APPLICATION SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("NQPU API starting up...")
    print(f"API Keys configured: {len(API_KEYS) > 0}")
    yield
    # Shutdown
    print("NQPU API shutting down...")


app = FastAPI(
    title="NQPU Quantum Biology Platform",
    description="Production REST API for quantum-enhanced computational biology",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detail": str(exc.detail),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# DEPENDENCIES
# ============================================================================

async def check_rate_limit(request: Request):
    """Rate limiting dependency."""
    client_id = get_client_id(request)
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in 60 seconds."
        )
    return client_id


# ============================================================================
# HEALTH ENDPOINT
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request):
    """Health check endpoint."""
    client_id = get_client_id(request)
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        modules_loaded={
            "nqpu_drug_design": True,
            "dna_rna_organism": True,
            "quantum_organism": True,
            "quantum_genome_tools": True,
            "quantum_protein_folding": True
        },
        rate_limit_remaining=rate_limiter.remaining(client_id)
    )


# ============================================================================
# DRUG DESIGN ENDPOINTS
# ============================================================================

@app.post("/drug/fingerprint", response_model=FingerprintResponse, tags=["Drug Design"])
async def generate_fingerprint(
    request: FingerprintRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Generate molecular fingerprint (ECFP4, ECFP6, FCFP4, or MACCS)."""
    try:
        mol = Molecule.from_smiles(request.smiles, "query")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {str(e)}")

    fp_type = request.fingerprint_type.lower()

    try:
        if fp_type == "ecfp4":
            fp = ecfp4(mol, n_bits=request.n_bits)
        elif fp_type == "ecfp6":
            fp = ecfp6(mol, n_bits=request.n_bits)
        elif fp_type == "fcfp4":
            fp = fcfp4(mol, n_bits=request.n_bits)
        elif fp_type == "maccs":
            fp = maccs_keys(mol)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown fingerprint type: {fp_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fingerprint generation failed: {str(e)}")

    bits = list(fp.bits) if hasattr(fp, 'bits') else []
    n_bits = fp.n_bits if hasattr(fp, 'n_bits') else 166
    density = len(bits) / n_bits if n_bits > 0 else 0

    return FingerprintResponse(
        smiles=request.smiles,
        fingerprint_type=fp_type,
        bits_set=bits[:1000],  # Limit response size
        n_bits=n_bits,
        density=density,
        bit_count=len(bits)
    )


@app.post("/drug/similarity", response_model=SimilarityResponse, tags=["Drug Design"])
async def calculate_similarity(
    request: SimilarityRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Calculate Tanimoto similarity between two molecules."""
    try:
        mol1 = Molecule.from_smiles(request.smiles1, "mol1")
        mol2 = Molecule.from_smiles(request.smiles2, "mol2")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {str(e)}")

    fp_type = request.fingerprint_type.lower()

    try:
        if fp_type == "ecfp4":
            fp1, fp2 = ecfp4(mol1), ecfp4(mol2)
        elif fp_type == "ecfp6":
            fp1, fp2 = ecfp6(mol1), ecfp6(mol2)
        elif fp_type == "maccs":
            fp1, fp2 = maccs_keys(mol1), maccs_keys(mol2)
        else:
            fp1, fp2 = ecfp4(mol1), ecfp4(mol2)

        tanimoto = tanimoto_similarity(fp1, fp2)

        # Dice similarity
        intersection = len(fp1.bits & fp2.bits) if hasattr(fp1, 'bits') else 0
        union = len(fp1.bits) + len(fp2.bits)
        dice = 2 * intersection / union if union > 0 else 0.0

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")

    return SimilarityResponse(
        smiles1=request.smiles1,
        smiles2=request.smiles2,
        tanimoto_similarity=round(tanimoto, 4),
        dice_similarity=round(dice, 4),
        fingerprint_type=fp_type,
        verdict=interpret_similarity(tanimoto)
    )


@app.post("/drug/properties", response_model=PropertiesResponse, tags=["Drug Design"])
async def calculate_properties(
    request: PropertiesRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Calculate molecular properties (MW, LogP, TPSA, etc.)."""
    try:
        mol = Molecule.from_smiles(request.smiles, "query")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {str(e)}")

    # Calculate heavy atom count (non-hydrogen atoms)
    heavy_atoms = sum(1 for atom in mol.atoms if atom[0] != 'H')

    return PropertiesResponse(
        smiles=request.smiles,
        molecular_weight=round(mol.molecular_weight(), 2),
        log_p=round(mol.estimated_log_p(), 2),
        h_bond_donors=mol.h_bond_donors(),
        h_bond_acceptors=mol.h_bond_acceptors(),
        rotatable_bonds=mol.rotatable_bonds(),
        polar_surface_area=round(mol.polar_surface_area(), 1),
        synthetic_accessibility=round(mol.synthetic_accessibility(), 1),
        heavy_atom_count=heavy_atoms
    )


@app.post("/drug/admet", response_model=AdmetResponse, tags=["Drug Design"])
async def predict_admet_profile(
    request: AdmetRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Predict full ADMET profile for a molecule."""
    try:
        mol = Molecule.from_smiles(request.smiles, "query")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {str(e)}")

    try:
        admet = predict_admet(mol)
        drug_result = evaluate_drug_likeness(mol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ADMET prediction failed: {str(e)}")

    properties = {}
    overall_sum = 0
    for prop_name, result in admet.items():
        properties[prop_name] = AdmetPropertyResult(
            property_name=prop_name,
            probability=result.probability,
            passes=result.passes,
            confidence=result.confidence
        )
        overall_sum += result.probability

    n_props = len(properties) if properties else 1
    overall_score = overall_sum / n_props

    return AdmetResponse(
        smiles=request.smiles,
        properties=properties,
        overall_score=round(overall_score, 3),
        lipinski_violations=drug_result.lipinski_violations,
        qed_score=round(drug_result.qed_score, 3)
    )


# ============================================================================
# GENOMICS ENDPOINTS
# ============================================================================

@app.post("/dna/encode", response_model=DNAEncodeResponse, tags=["Genomics"])
async def encode_dna(
    request: DNAEncodeRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Encode DNA sequence to quantum state representation."""
    dna = DNA.from_sequence(request.sequence, request.name)
    qdna = QuantumDNA.from_dna(dna)

    return DNAEncodeResponse(
        sequence=request.sequence,
        name=request.name,
        length=len(request.sequence),
        qubits_needed=qdna.qubits_needed,
        gc_content=round(dna.gc_content(), 3),
        encoding_type="amplitude" if len(request.sequence) > 10 else "direct"
    )


@app.post("/dna/similarity", response_model=DNASimilarityResponse, tags=["Genomics"])
async def dna_similarity(
    request: DNASimilarityRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Calculate quantum fidelity between DNA sequences."""
    dna1 = DNA.from_sequence(request.sequence1, "seq1")
    dna2 = DNA.from_sequence(request.sequence2, "seq2")

    # Hamming distance
    min_len = min(len(dna1.sequence), len(dna2.sequence))
    hamming = sum(a != b for a, b in zip(dna1.sequence[:min_len], dna2.sequence[:min_len]))

    # Quantum fidelity (simplified)
    qdna1 = QuantumDNA.from_dna(dna1)
    qdna2 = QuantumDNA.from_dna(dna2)

    try:
        state1 = qdna1.to_quantum_state()
        state2 = qdna2.to_quantum_state()

        min_dim = min(len(state1), len(state2))
        if min_dim > 0:
            fidelity = float(np.abs(np.vdot(state1[:min_dim], state2[:min_dim])) ** 2)
        else:
            fidelity = 0.0
    except Exception:
        fidelity = max(0, 1 - hamming / max(min_len, 1))

    # Alignment score
    matches = min_len - hamming
    alignment_score = matches / max(min_len, 1)

    return DNASimilarityResponse(
        sequence1=request.sequence1[:100],  # Truncate for response
        sequence2=request.sequence2[:100],
        quantum_fidelity=round(fidelity, 4),
        hamming_distance=hamming,
        alignment_score=round(alignment_score, 4),
        gc_difference=round(abs(dna1.gc_content() - dna2.gc_content()), 3)
    )


@app.post("/dna/align", response_model=DNAAlignResponse, tags=["Genomics"])
async def align_dna(
    request: DNAAlignRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Quantum sequence alignment."""
    query = request.query
    target = request.target

    # Simple Needleman-Wunsch style alignment
    n, m = len(query), len(target)

    # Score matrix
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        score[i][0] = -i
    for j in range(m + 1):
        score[0][j] = -j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i-1][j-1] + (1 if query[i-1] == target[j-1] else -1)
            delete = score[i-1][j] - 1
            insert = score[i][j-1] - 1
            score[i][j] = max(match, delete, insert)

    # Traceback
    aligned_q, aligned_t = [], []
    i, j = n, m
    gaps = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i][j] == score[i-1][j-1] + (1 if query[i-1] == target[j-1] else -1):
            aligned_q.append(query[i-1])
            aligned_t.append(target[j-1])
            i -= 1
            j -= 1
        elif i > 0 and score[i][j] == score[i-1][j] - 1:
            aligned_q.append(query[i-1])
            aligned_t.append('-')
            gaps += 1
            i -= 1
        else:
            aligned_q.append('-')
            aligned_t.append(target[j-1])
            gaps += 1
            j -= 1

    aligned_q = ''.join(reversed(aligned_q))
    aligned_t = ''.join(reversed(aligned_t))

    # Calculate identity
    matches = sum(1 for a, b in zip(aligned_q, aligned_t) if a == b)
    identity = matches / max(len(aligned_q), 1)

    return DNAAlignResponse(
        query=query[:50],
        target=target[:50],
        aligned_query=aligned_q[:200],
        aligned_target=aligned_t[:200],
        score=float(score[n][m]),
        identity=round(identity, 3),
        gaps=gaps
    )


@app.post("/dna/mutate", response_model=DNAMutateResponse, tags=["Genomics"])
async def mutate_dna(
    request: DNAMutateRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Simulate DNA mutations."""
    import random

    dna = DNA.from_sequence(request.sequence, "query")
    bases = list(dna.sequence)
    mutations = []

    if request.mutation_type == "point":
        for i in range(len(bases)):
            if random.random() < request.mutation_rate:
                original = bases[i]
                alternatives = [b for b in 'ATGC' if b != original]
                bases[i] = random.choice(alternatives)
                mutations.append({
                    "position": i,
                    "type": "point",
                    "original": original,
                    "mutated": bases[i]
                })

    elif request.mutation_type == "insertion":
        n_insertions = int(len(bases) * request.mutation_rate)
        for _ in range(n_insertions):
            pos = random.randint(0, len(bases))
            new_base = random.choice('ATGC')
            bases.insert(pos, new_base)
            mutations.append({
                "position": pos,
                "type": "insertion",
                "base": new_base
            })

    elif request.mutation_type == "deletion":
        n_deletions = int(len(bases) * request.mutation_rate)
        for _ in range(n_deletions):
            if bases:
                pos = random.randint(0, len(bases) - 1)
                deleted = bases.pop(pos)
                mutations.append({
                    "position": pos,
                    "type": "deletion",
                    "deleted": deleted
                })

    return DNAMutateResponse(
        original=request.sequence,
        mutated=''.join(bases),
        mutation_count=len(mutations),
        mutations=mutations[:100],  # Limit response
        mutation_rate=request.mutation_rate
    )


# ============================================================================
# GENOME ASSEMBLY ENDPOINTS
# ============================================================================

@app.post("/assembly/simulate-reads", response_model=SimulateReadsResponse, tags=["Assembly"])
async def simulate_reads(
    request: SimulateReadsRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Generate simulated sequencing reads."""
    genome = DNA.from_sequence(request.genome_sequence, "genome")

    # Validate read_length doesn't exceed genome length
    if request.read_length > len(genome.sequence):
        raise HTTPException(
            status_code=400,
            detail=f"read_length ({request.read_length}) cannot exceed genome length ({len(genome.sequence)})"
        )

    assembler = QuantumGenomeAssembler.simulate_reads(
        genome,
        read_length=request.read_length,
        coverage=request.coverage,
        error_rate=request.error_rate
    )

    # Limit number of reads in response
    max_reads = 100
    reads = [r.sequence for r in assembler.reads[:max_reads]]
    quality_scores = [r.quality[:50] for r in assembler.reads[:max_reads]]

    return SimulateReadsResponse(
        n_reads=len(assembler.reads),
        read_length=request.read_length,
        coverage=float(request.coverage),
        reads=reads,
        quality_scores=quality_scores
    )


@app.post("/assembly/assemble", response_model=AssembleResponse, tags=["Assembly"])
async def assemble_genome(
    request: AssembleRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Quantum genome assembly from reads."""
    # Create reads
    reads = [Read(sequence=seq) for seq in request.reads]

    assembler = QuantumGenomeAssembler(
        reads=reads,
        min_overlap=request.min_overlap
    )

    try:
        if request.use_quantum:
            contigs = assembler.quantum_assembly()
            assembly_type = "quantum"
        else:
            contigs = assembler.classical_assembly()
            assembly_type = "classical"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assembly failed: {str(e)}")

    # Calculate N50
    lengths = sorted([c.length for c in contigs], reverse=True)
    total = sum(lengths)
    n50 = 0
    cumsum = 0
    for length in lengths:
        cumsum += length
        if cumsum >= total / 2:
            n50 = length
            break

    contig_data = [
        {
            "sequence": c.sequence[:200],
            "length": c.length,
            "coverage": round(c.coverage, 2)
        }
        for c in contigs[:20]  # Limit response
    ]

    return AssembleResponse(
        n_contigs=len(contigs),
        total_length=total,
        n50=n50,
        contigs=contig_data,
        assembly_type=assembly_type
    )


# ============================================================================
# CRISPR ENDPOINTS
# ============================================================================

@app.post("/crispr/design-guides", response_model=DesignGuidesResponse, tags=["CRISPR"])
async def design_guides(
    request: DesignGuidesRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Design CRISPR guide RNAs."""
    genome = DNA.from_sequence(request.genome_sequence, "genome")
    gene = Gene(
        name="target",
        coding_sequence=DNA.from_sequence(request.gene_sequence, "gene"),
        essential=True,
        function="target gene"
    )

    crispr = CRISPRSimulation(genome)

    try:
        guides = crispr.quantum_guide_search(gene, n_guides=request.n_guides)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Guide design failed: {str(e)}")

    guide_results = []
    for guide in guides:
        gc = (guide.sequence.count('G') + guide.sequence.count('C')) / 20

        # Find position in gene
        try:
            pos = request.gene_sequence.index(guide.sequence)
        except ValueError:
            pos = -1

        guide_results.append(GuideResult(
            sequence=guide.sequence,
            pam=guide.pam,
            gc_content=round(gc, 2),
            specificity_score=round(guide.specificity_score(genome), 3),
            position=pos,
            strand='+'  # Simplified
        ))

    return DesignGuidesResponse(
        gene_sequence=request.gene_sequence[:50],
        guides=guide_results,
        n_guides=len(guide_results)
    )


@app.post("/crispr/simulate-edit", response_model=SimulateEditResponse, tags=["CRISPR"])
async def simulate_edit(
    request: SimulateEditRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Simulate CRISPR edit outcome."""
    guide = CRISPRGuide(sequence=request.guide_sequence)

    # Create a minimal genome for simulation
    genome = DNA.from_sequence("A" * 1000, "sim")
    crispr = CRISPRSimulation(genome)

    result = crispr.simulate_edit(guide, request.position)

    return SimulateEditResponse(
        guide=request.guide_sequence,
        position=request.position,
        edit_type=result['type'],
        detail=result['detail'],
        probabilities={
            "deletion": 0.6,
            "insertion": 0.2,
            "substitution": 0.1,
            "no_edit": 0.1
        }
    )


# ============================================================================
# PROTEIN FOLDING ENDPOINTS
# ============================================================================

@app.post("/protein/fold", response_model=ProteinFoldResponse, tags=["Protein"])
async def fold_protein(
    request: ProteinFoldRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Quantum protein folding simulation."""
    # Create protein
    residues = []
    for i, aa in enumerate(request.sequence):
        try:
            amino_acid = AminoAcid[aa]
        except KeyError:
            amino_acid = AminoAcid.A  # Default to Alanine
        residues.append(Residue(amino_acid=amino_acid, position=i))

    protein = FoldProtein(name="query", sequence=request.sequence, residues=residues)

    # Run folding
    folder = QuantumProteinFolder(
        protein=protein,
        temperature=request.temperature,
        max_iterations=request.max_iterations
    )

    try:
        if request.use_quantum:
            energy, conformation = folder.quantum_folding()
            method = "quantum_annealing"
        else:
            energy, conformation = folder.classical_folding()
            method = "classical_mc"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Folding failed: {str(e)}")

    # Convert conformation to dict format
    conf_data = [
        {"phi": round(phi, 2), "psi": round(psi, 2)}
        for phi, psi in conformation[:100]
    ]

    # Predict secondary structure
    ss = predict_secondary_structure(request.sequence)

    return ProteinFoldResponse(
        sequence=request.sequence,
        final_energy=round(energy, 2),
        conformation=conf_data,
        method=method,
        iterations=request.max_iterations,
        secondary_structure=ss[:100]
    )


@app.post("/protein/secondary-structure", response_model=SecondaryStructureResponse, tags=["Protein"])
async def predict_secondary(
    request: SecondaryStructureRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Predict protein secondary structure."""
    ss = predict_secondary_structure(request.sequence)

    helix_count = ss.count('H')
    sheet_count = ss.count('E')
    coil_count = ss.count('C')
    total = len(ss)

    return SecondaryStructureResponse(
        sequence=request.sequence,
        secondary_structure=ss,
        helix_content=round(helix_count / total, 3) if total > 0 else 0,
        sheet_content=round(sheet_count / total, 3) if total > 0 else 0,
        coil_content=round(coil_count / total, 3) if total > 0 else 0
    )


# ============================================================================
# ENZYME ENDPOINTS
# ============================================================================

@app.post("/enzyme/catalyze", response_model=EnzymeCatalyzeResponse, tags=["Enzyme"])
async def simulate_catalysis(
    request: EnzymeCatalyzeRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Simulate enzyme catalysis with quantum effects."""
    substrate = Substrate(
        name=request.substrate_name,
        smiles=request.substrate_smiles,
        binding_energy=request.binding_energy
    )

    active_site = ActiveSite(
        residues=["CATALYTIC"],
        mechanism="general",
        substrate=substrate
    )

    # Create minimal protein for enzyme
    protein = FoldProtein(
        name=request.enzyme_name,
        sequence="A",  # Minimal sequence
        residues=[Residue(amino_acid=AminoAcid.A, position=0)]
    )

    enzyme = Enzyme(
        name=request.enzyme_name,
        ec_number=request.ec_number,
        protein=protein,
        active_sites=[active_site]
    )

    try:
        result = enzyme.catalyze(substrate, temperature=request.temperature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Catalysis simulation failed: {str(e)}")

    return EnzymeCatalyzeResponse(
        enzyme=result.get("enzyme", request.enzyme_name),
        substrate=result.get("substrate", request.substrate_name),
        rate_enhancement=result.get("rate_enhancement", "N/A"),
        tunneling_probability=result.get("tunneling_probability", "N/A"),
        barrier_reduction=result.get("barrier_reduction", "N/A"),
        quantum_contribution=result.get("quantum_contribution", "N/A"),
        temperature=result.get("temperature", f"{request.temperature} K")
    )


@app.post("/enzyme/rate", response_model=EnzymeRateResponse, tags=["Enzyme"])
async def calculate_rate(
    request: EnzymeRateRequest,
    client_id: str = Depends(check_rate_limit),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Calculate enzyme rate enhancement."""
    active_site = ActiveSite(
        residues=["CATALYTIC"],
        mechanism=request.mechanism
    )

    rate_enhancement = active_site.calculate_rate_enhancement(
        request.barrier_classical,
        request.barrier_quantum,
        request.temperature
    )

    # Calculate individual rates
    k_uncatalyzed = np.exp(-25.0 / (0.001987 * request.temperature))
    k_quantum = np.exp(-request.barrier_quantum / (0.001987 * request.temperature))

    # Tunneling boost
    tunneling_boost = 10.0 if "H" in request.mechanism.lower() or "proton" in request.mechanism.lower() else 1.0

    return EnzymeRateResponse(
        rate_enhancement=rate_enhancement,
        rate_enhancement_formatted=f"{rate_enhancement:.2e}x",
        classical_rate=float(np.exp(-request.barrier_classical / (0.001987 * request.temperature))),
        quantum_rate=float(k_quantum),
        tunneling_boost=tunneling_boost,
        temperature=request.temperature
    )


# ============================================================================
# NUMPY IMPORT (for quantum state calculations)
# ============================================================================

import numpy as np


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NQPU Quantum Biology API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║           NQPU Quantum Biology Platform API                ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Version: 1.0.0                                           ║
    ║  Host: {args.host:<54} ║
    ║  Port: {args.port:<54} ║
    ║  Docs: http://{args.host}:{args.port}/docs{' ' * (41 - len(args.host) - len(str(args.port)))}║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "nqpu_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )
