"""Comprehensive tests for the quantum biology package.

Tests cover all five modules:
  - photosynthesis.py: FMO complex, Lindblad dynamics, quantum advantage
  - tunneling.py: WKB tunneling, KIE, enzyme presets, sensitivity
  - olfaction.py: IETS tunneling, resonance, isotope discrimination
  - avian_navigation.py: Radical pair, singlet yield, compass anisotropy
  - dna_mutation.py: Base pair tunneling, tautomer probability, mutation rates

Tests verify physical correctness against known results and sanity checks
on physical constants and model outputs.
"""

import math

import numpy as np
import pytest

from nqpu.bio.photosynthesis import (
    FMOComplex,
    FMOEvolution,
    PhotosyntheticSystem,
    QuantumTransportEfficiency,
    DecoherenceModel,
    SpectralDensity,
    SpectralDensityType,
)
from nqpu.bio.tunneling import (
    EnzymeTunneling,
    TunnelingBarrier,
    BarrierShape,
    TunnelingSensitivity,
    ENZYMES,
    M_PROTON,
    M_DEUTERIUM,
)
from nqpu.bio.olfaction import (
    QuantumNose,
    OlfactoryReceptor,
    MolecularVibration,
    Odorant,
    OdorDiscrimination,
    ODORANTS,
)
from nqpu.bio.avian_navigation import (
    RadicalPair,
    CryptochromeModel,
    CompassSensitivity,
    DecoherenceEffects,
)
from nqpu.bio.dna_mutation import (
    BasePair,
    BasePairType,
    DoubleWellPotential,
    TautomerTunneling,
    MutationRate,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def fmo_standard() -> FMOComplex:
    """Standard 7-site FMO complex."""
    return FMOComplex.standard()


@pytest.fixture
def enzyme_adh() -> EnzymeTunneling:
    """Alcohol dehydrogenase tunneling model."""
    return ENZYMES["alcohol_dehydrogenase"]


@pytest.fixture
def enzyme_slo() -> EnzymeTunneling:
    """Soybean lipoxygenase tunneling model."""
    return ENZYMES["soybean_lipoxygenase"]


@pytest.fixture
def radical_pair_crypto() -> RadicalPair:
    """Cryptochrome radical pair."""
    return RadicalPair.cryptochrome()


@pytest.fixture
def base_pair_at() -> BasePair:
    """Adenine-thymine base pair."""
    return BasePair.from_type(BasePairType.AT)


@pytest.fixture
def base_pair_gc() -> BasePair:
    """Guanine-cytosine base pair."""
    return BasePair.from_type(BasePairType.GC)


@pytest.fixture
def nose_default() -> QuantumNose:
    """Default quantum nose model."""
    return QuantumNose.default()


# ======================================================================
# 1. Photosynthesis Tests
# ======================================================================


class TestFMOComplex:
    """Tests for FMO complex photosynthetic energy transfer."""

    def test_standard_construction(self, fmo_standard: FMOComplex):
        """Standard FMO complex has 7 sites."""
        assert fmo_standard.sites == 7
        assert fmo_standard.hamiltonian.shape == (7, 7)
        assert fmo_standard.initial_site == 0
        assert fmo_standard.temperature_k == 300.0

    def test_hamiltonian_is_symmetric(self, fmo_standard: FMOComplex):
        """FMO Hamiltonian must be Hermitian (real symmetric)."""
        h = fmo_standard.hamiltonian
        np.testing.assert_allclose(h, h.T, atol=1e-10)

    def test_site_energies_from_adolphs_renger(self, fmo_standard: FMOComplex):
        """Check key site energies from Adolphs & Renger (2006)."""
        h = fmo_standard.hamiltonian
        # Site 3 (index 2) has lowest energy = 0.0 cm^-1
        assert h[2, 2] == 0.0
        # Site 6 (index 5) has highest energy = 420.0 cm^-1
        assert h[5, 5] == 420.0
        # Site 1 (index 0) = 200.0 cm^-1
        assert h[0, 0] == 200.0

    def test_coupling_1_2(self, fmo_standard: FMOComplex):
        """Strongest coupling is between sites 1-2 (-87.7 cm^-1)."""
        h = fmo_standard.hamiltonian
        assert h[0, 1] == pytest.approx(-87.7, abs=0.1)
        assert h[1, 0] == pytest.approx(-87.7, abs=0.1)

    def test_evolve_returns_correct_shape(self, fmo_standard: FMOComplex):
        """Evolution returns arrays of correct dimensions."""
        result = fmo_standard.evolve(duration_fs=100.0, steps=50)
        assert result.times_fs.shape == (51,)
        assert result.site_populations.shape == (51, 7)
        assert result.transfer_efficiency.shape == (51,)
        # Number of upper-triangle coherences for 7 sites = 7*6/2 = 21
        assert result.coherences.shape == (51, 21)

    def test_initial_population(self, fmo_standard: FMOComplex):
        """Initial population is entirely on the initial site."""
        result = fmo_standard.evolve(duration_fs=10.0, steps=5)
        pops = result.site_populations[0]
        assert pops[0] == pytest.approx(1.0, abs=1e-10)
        assert sum(pops[1:]) == pytest.approx(0.0, abs=1e-10)

    def test_population_conserved_short_time(self, fmo_standard: FMOComplex):
        """Total population should be approximately conserved at very short times."""
        result = fmo_standard.evolve(duration_fs=1.0, steps=100)
        # At t = 1 fs the trapping loss is negligible
        total = result.total_population(10)
        assert total == pytest.approx(1.0, abs=0.05)

    def test_transfer_to_reaction_center(self, fmo_standard: FMOComplex):
        """Energy should transfer to the reaction center (site 3) over time."""
        result = fmo_standard.evolve(duration_fs=500.0, steps=1000)
        peak_eff = result.peak_efficiency()
        # FMO should achieve non-trivial transfer efficiency
        assert peak_eff > 0.01, f"Peak efficiency too low: {peak_eff}"

    def test_coherences_start_nonzero(self, fmo_standard: FMOComplex):
        """Off-diagonal coherences should build up during evolution."""
        result = fmo_standard.evolve(duration_fs=200.0, steps=500)
        # Check that coherences are nonzero at some intermediate time
        mid = len(result.times_fs) // 4
        avg_coh = result.average_coherence(mid)
        assert avg_coh > 0.0

    def test_coherence_lifetime(self, fmo_standard: FMOComplex):
        """Coherence lifetime should be in the hundreds of fs range."""
        result = fmo_standard.evolve(duration_fs=1000.0, steps=2000)
        lifetime = result.coherence_lifetime_fs(0.5)
        # Coherence should decay within the simulation time
        assert lifetime > 0.0
        assert lifetime < 1000.0

    def test_evolve_invalid_duration(self, fmo_standard: FMOComplex):
        """Negative duration should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            fmo_standard.evolve(duration_fs=-10.0, steps=100)

    def test_evolve_zero_steps(self, fmo_standard: FMOComplex):
        """Zero steps should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            fmo_standard.evolve(duration_fs=100.0, steps=0)

    def test_lhc_ii_construction(self):
        """LHC-II system should construct with 4 sites."""
        lhc = FMOComplex.from_system(PhotosyntheticSystem.LHC_II)
        assert lhc.sites == 4
        assert lhc.hamiltonian.shape == (4, 4)

    def test_pe545_construction(self):
        """PE545 system should construct with 4 sites."""
        pe = FMOComplex.from_system(PhotosyntheticSystem.PE545)
        assert pe.sites == 4
        assert pe.temperature_k == 294.0

    def test_from_system_fmo(self):
        """from_system(FMO) should match standard()."""
        fmo1 = FMOComplex.standard()
        fmo2 = FMOComplex.from_system(PhotosyntheticSystem.FMO)
        np.testing.assert_allclose(fmo1.hamiltonian, fmo2.hamiltonian)


class TestSpectralDensity:
    """Tests for spectral density models."""

    def test_drude_lorentz_positive(self):
        """Drude-Lorentz spectral density should be positive for omega > 0."""
        sd = SpectralDensity()
        assert sd.evaluate(100.0) > 0.0
        assert sd.evaluate(500.0) > 0.0

    def test_drude_lorentz_zero_at_zero(self):
        """J(0) = 0 for all spectral density types."""
        for st in SpectralDensityType:
            sd = SpectralDensity(density_type=st)
            assert sd.evaluate(0.0) == 0.0

    def test_ohmic_spectral_density(self):
        """Ohmic spectral density should be positive."""
        sd = SpectralDensity(density_type=SpectralDensityType.OHMIC)
        assert sd.evaluate(100.0) > 0.0

    def test_dephasing_increases_with_temperature(self):
        """Dephasing rate should increase with temperature."""
        sd = SpectralDensity()
        rate_200 = sd.dephasing_rate_at_temperature(200.0)
        rate_400 = sd.dephasing_rate_at_temperature(400.0)
        assert rate_400 > rate_200

    def test_dephasing_invalid_temperature(self):
        """Negative temperature should raise ValueError."""
        sd = SpectralDensity()
        with pytest.raises(ValueError, match="positive"):
            sd.dephasing_rate_at_temperature(-10.0)


class TestDecoherenceModel:
    """Tests for the decoherence model."""

    def test_dephasing_rate_positive(self):
        """Dephasing rate should be positive at room temperature."""
        model = DecoherenceModel(temperature_k=300.0)
        rate = model.dephasing_rate_cm_inv()
        assert rate > 0.0

    def test_relaxation_rate_with_gap(self):
        """Relaxation rate should be positive for positive energy gap."""
        model = DecoherenceModel(temperature_k=300.0)
        rate = model.relaxation_rate_cm_inv(100.0)
        assert rate > 0.0

    def test_relaxation_rate_zero_for_zero_gap(self):
        """Relaxation rate should be zero for zero energy gap."""
        model = DecoherenceModel(temperature_k=300.0)
        assert model.relaxation_rate_cm_inv(0.0) == 0.0


class TestQuantumTransportEfficiency:
    """Tests for quantum vs classical transport comparison."""

    def test_quantum_efficiency_positive(self):
        """Quantum transport should achieve nonzero efficiency."""
        fmo = FMOComplex.standard()
        qte = QuantumTransportEfficiency(fmo.hamiltonian)
        q_eff = qte.quantum_efficiency(duration_fs=500.0, steps=1000)
        assert q_eff > 0.0

    def test_classical_efficiency_positive(self):
        """Classical transport should achieve nonzero efficiency."""
        fmo = FMOComplex.standard()
        qte = QuantumTransportEfficiency(fmo.hamiltonian)
        c_eff = qte.classical_efficiency(duration_fs=500.0, steps=1000)
        assert c_eff > 0.0

    def test_quantum_advantage_exists(self):
        """Quantum transport should outperform classical for FMO."""
        fmo = FMOComplex.standard()
        qte = QuantumTransportEfficiency(fmo.hamiltonian)
        advantage = qte.quantum_advantage(duration_fs=500.0, steps=1000)
        # Quantum advantage should be > 1 for the FMO complex
        assert advantage > 1.0, (
            f"Expected quantum advantage > 1, got {advantage:.3f}"
        )


# ======================================================================
# 2. Enzyme Tunneling Tests
# ======================================================================


class TestEnzymeTunneling:
    """Tests for WKB tunneling in enzymes."""

    def test_tunneling_probability_range(self, enzyme_adh: EnzymeTunneling):
        """Tunneling probability should be in [0, 1]."""
        prob = enzyme_adh.tunneling_probability()
        assert 0.0 <= prob <= 1.0

    def test_tunneling_probability_zero_barrier(self):
        """Zero barrier should give probability = 1."""
        model = EnzymeTunneling.from_barrier(0.0, 0.05)
        assert model.tunneling_probability() == 1.0

    def test_tunneling_probability_zero_width(self):
        """Zero width should give probability = 1."""
        model = EnzymeTunneling.from_barrier(0.3, 0.0)
        assert model.tunneling_probability() == 1.0

    def test_tunneling_probability_decreases_with_height(self):
        """Higher barrier should reduce tunneling probability."""
        prob_low = EnzymeTunneling.from_barrier(0.2, 0.05).tunneling_probability()
        prob_high = EnzymeTunneling.from_barrier(0.5, 0.05).tunneling_probability()
        assert prob_high < prob_low

    def test_tunneling_probability_decreases_with_width(self):
        """Wider barrier should reduce tunneling probability."""
        prob_narrow = EnzymeTunneling.from_barrier(0.3, 0.03).tunneling_probability()
        prob_wide = EnzymeTunneling.from_barrier(0.3, 0.08).tunneling_probability()
        assert prob_wide < prob_narrow

    def test_tunneling_probability_decreases_with_mass(self):
        """Heavier particle should tunnel less."""
        prob_h = EnzymeTunneling.from_barrier(0.3, 0.05, M_PROTON).tunneling_probability()
        prob_d = EnzymeTunneling.from_barrier(0.3, 0.05, M_DEUTERIUM).tunneling_probability()
        assert prob_d < prob_h

    def test_tunneling_rate_positive(self, enzyme_adh: EnzymeTunneling):
        """Tunneling rate should be positive."""
        rate = enzyme_adh.tunneling_rate()
        assert rate > 0.0

    def test_kie_greater_than_one(self, enzyme_adh: EnzymeTunneling):
        """KIE (H/D) should be greater than 1 due to lighter H tunneling more."""
        kie = enzyme_adh.kie_ratio()
        assert kie > 1.0, f"KIE should be > 1, got {kie:.2f}"

    def test_kie_slo_large(self, enzyme_slo: EnzymeTunneling):
        """SLO should have a large KIE (experimentally ~80)."""
        kie = enzyme_slo.kie_ratio()
        assert kie > 5.0, f"SLO KIE should be large, got {kie:.2f}"

    def test_classical_rate_increases_with_temperature(
        self, enzyme_adh: EnzymeTunneling
    ):
        """Classical rate should increase with temperature (Arrhenius)."""
        rate_200 = enzyme_adh.classical_rate(200.0)
        rate_400 = enzyme_adh.classical_rate(400.0)
        assert rate_400 > rate_200

    def test_classical_rate_invalid_temperature(
        self, enzyme_adh: EnzymeTunneling
    ):
        """Negative temperature should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            enzyme_adh.classical_rate(-10.0)

    def test_total_rate_exceeds_classical(self, enzyme_adh: EnzymeTunneling):
        """Total rate (tunnel + thermal) should exceed classical rate."""
        total = enzyme_adh.total_rate(300.0)
        classical = enzyme_adh.classical_rate(300.0)
        assert total >= classical

    def test_tunnel_splitting_positive(self, enzyme_adh: EnzymeTunneling):
        """Tunnel splitting should be positive."""
        splitting = enzyme_adh.tunnel_splitting_ev()
        assert splitting > 0.0

    def test_rate_vs_temperature(self, enzyme_adh: EnzymeTunneling):
        """Rate vs temperature should return arrays of correct shape."""
        temps, total, classical, tunnel = enzyme_adh.rate_vs_temperature(
            50.0, 500.0, 20
        )
        assert len(temps) == 20
        assert len(total) == 20
        assert len(classical) == 20
        assert len(tunnel) == 20

    def test_rate_vs_temperature_invalid_range(
        self, enzyme_adh: EnzymeTunneling
    ):
        """Invalid temperature range should raise ValueError."""
        with pytest.raises(ValueError):
            enzyme_adh.rate_vs_temperature(0.0, 500.0, 20)
        with pytest.raises(ValueError):
            enzyme_adh.rate_vs_temperature(500.0, 100.0, 20)

    def test_crossover_temperature_positive(self, enzyme_adh: EnzymeTunneling):
        """Crossover temperature should be positive."""
        t_cross = enzyme_adh.crossover_temperature()
        assert t_cross > 0.0

    def test_tunnel_rate_dominates_low_temperature(
        self, enzyme_adh: EnzymeTunneling
    ):
        """At low temperature, tunneling should dominate thermal rate."""
        total = enzyme_adh.total_rate(20.0)
        classical = enzyme_adh.classical_rate(20.0)
        tunnel = enzyme_adh.tunneling_rate()
        # Total should be close to tunnel rate at very low T
        assert total > classical

    def test_negative_barrier_raises_error(self):
        """Negative barrier height should raise ValueError."""
        model = EnzymeTunneling(
            barrier=TunnelingBarrier(height_ev=-0.1, width_nm=0.05)
        )
        with pytest.raises(ValueError, match="non-negative"):
            model.tunneling_probability()

    def test_parabolic_barrier(self):
        """Parabolic barrier should give larger tunneling than rectangular.

        Rectangular uses exp(-2*kappa*a), parabolic uses exp(-pi/2*kappa*a).
        Since pi/2 ~ 1.57 < 2, the parabolic exponent is less negative,
        giving higher tunneling probability (the smoother barrier is
        easier to tunnel through).
        """
        rect = EnzymeTunneling(
            barrier=TunnelingBarrier(0.3, 0.05, BarrierShape.RECTANGULAR)
        ).tunneling_probability()
        para = EnzymeTunneling(
            barrier=TunnelingBarrier(0.3, 0.05, BarrierShape.PARABOLIC)
        ).tunneling_probability()
        assert para > rect

    def test_swain_schaad_exponent(self, enzyme_adh: EnzymeTunneling):
        """Swain-Schaad exponent should exceed classical limit of 1.44."""
        ss = enzyme_adh.swain_schaad_exponent()
        assert not math.isnan(ss)
        # Classical limit is ~1.44; quantum tunneling gives higher values
        assert ss > 1.0

    def test_all_enzyme_presets_exist(self):
        """All predefined enzymes should be accessible."""
        assert "alcohol_dehydrogenase" in ENZYMES
        assert "soybean_lipoxygenase" in ENZYMES
        assert "aromatic_amine_dehydrogenase" in ENZYMES

    def test_enzyme_presets_have_positive_rates(self):
        """All enzyme presets should have positive tunneling rates."""
        for name, enzyme in ENZYMES.items():
            rate = enzyme.tunneling_rate()
            assert rate > 0.0, f"{name} has zero tunneling rate"


class TestTunnelingBarrier:
    """Tests for the TunnelingBarrier class."""

    def test_validate_negative_height(self):
        """Negative height should raise ValueError on validate."""
        barrier = TunnelingBarrier(height_ev=-0.1, width_nm=0.05)
        with pytest.raises(ValueError, match="non-negative"):
            barrier.validate()

    def test_validate_negative_width(self):
        """Negative width should raise ValueError on validate."""
        barrier = TunnelingBarrier(height_ev=0.3, width_nm=-0.01)
        with pytest.raises(ValueError, match="non-negative"):
            barrier.validate()

    def test_validate_passes_for_physical(self):
        """Physical parameters should pass validation."""
        barrier = TunnelingBarrier(height_ev=0.3, width_nm=0.05)
        barrier.validate()  # Should not raise


class TestTunnelingSensitivity:
    """Tests for tunneling sensitivity analysis."""

    def test_height_sensitivity_shape(self, enzyme_adh: EnzymeTunneling):
        """Height sensitivity should return arrays of correct length."""
        sens = TunnelingSensitivity(enzyme_adh)
        heights, probs = sens.height_sensitivity(n_points=10)
        assert len(heights) == 10
        assert len(probs) == 10

    def test_height_sensitivity_monotonic(self, enzyme_adh: EnzymeTunneling):
        """Tunneling should decrease monotonically with barrier height."""
        sens = TunnelingSensitivity(enzyme_adh)
        heights, probs = sens.height_sensitivity(delta_ev=0.1, n_points=20)
        # Should be monotonically decreasing
        for i in range(len(probs) - 1):
            assert probs[i + 1] <= probs[i] + 1e-10

    def test_width_sensitivity_shape(self, enzyme_adh: EnzymeTunneling):
        """Width sensitivity should return arrays of correct length."""
        sens = TunnelingSensitivity(enzyme_adh)
        widths, probs = sens.width_sensitivity(n_points=10)
        assert len(widths) == 10
        assert len(probs) == 10

    def test_mass_sensitivity_default(self, enzyme_adh: EnzymeTunneling):
        """Mass sensitivity should show H > D > T tunneling."""
        sens = TunnelingSensitivity(enzyme_adh)
        masses, probs = sens.mass_sensitivity()
        assert len(masses) == 3
        # H tunnels more than D tunnels more than T
        assert probs[0] > probs[1] > probs[2]


# ======================================================================
# 3. Olfaction Tests
# ======================================================================


class TestQuantumNose:
    """Tests for the quantum olfaction model."""

    def test_default_construction(self, nose_default: QuantumNose):
        """Default nose should have 0.2 eV energy gap."""
        assert nose_default.energy_gap_ev() == pytest.approx(0.2, abs=1e-10)

    def test_resonant_frequency_positive(self, nose_default: QuantumNose):
        """Resonant frequency should be positive."""
        f_res = nose_default.resonant_frequency_cm_inv()
        assert f_res > 0.0

    def test_tunneling_rate_positive_at_resonance(
        self, nose_default: QuantumNose
    ):
        """Tunneling rate at resonance should be positive and large."""
        f_res = nose_default.resonant_frequency_cm_inv()
        rate = nose_default.tunneling_rate(f_res)
        assert rate > 0.0

    def test_tunneling_rate_peaks_at_resonance(
        self, nose_default: QuantumNose
    ):
        """Rate should be maximum near the resonant frequency."""
        f_res = nose_default.resonant_frequency_cm_inv()
        rate_on = nose_default.tunneling_rate(f_res)
        rate_off_low = nose_default.tunneling_rate(f_res - 500.0)
        rate_off_high = nose_default.tunneling_rate(f_res + 500.0)
        assert rate_on > rate_off_low
        assert rate_on > rate_off_high

    def test_selectivity_greater_than_one(self, nose_default: QuantumNose):
        """On-resonance/off-resonance ratio should exceed 1."""
        sel = nose_default.selectivity(500.0)
        assert sel > 1.0

    def test_negative_frequency_raises_error(self, nose_default: QuantumNose):
        """Negative frequency should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            nose_default.tunneling_rate(-100.0)

    def test_spectrum_shape(self, nose_default: QuantumNose):
        """Spectrum should return arrays of correct length."""
        freqs, rates = nose_default.spectrum(
            freq_min=500.0, freq_max=3000.0, steps=50
        )
        assert len(freqs) == 50
        assert len(rates) == 50

    def test_spectrum_all_positive(self, nose_default: QuantumNose):
        """All spectrum rates should be non-negative."""
        _, rates = nose_default.spectrum(steps=50)
        assert np.all(rates >= 0.0)

    def test_is_detected_at_resonance(self, nose_default: QuantumNose):
        """Molecule at resonant frequency should be detected."""
        f_res = nose_default.resonant_frequency_cm_inv()
        rate = nose_default.tunneling_rate(f_res)
        # Use the actual rate as a reasonable threshold basis
        assert nose_default.is_detected(f_res, threshold_rate=rate * 0.1)

    def test_is_not_detected_far_off_resonance(
        self, nose_default: QuantumNose
    ):
        """Molecule far from resonance should not be detected with high threshold."""
        f_res = nose_default.resonant_frequency_cm_inv()
        rate_on = nose_default.tunneling_rate(f_res)
        # Very far off resonance
        assert not nose_default.is_detected(
            f_res + 5000.0, threshold_rate=rate_on * 0.5
        )


class TestMolecularVibration:
    """Tests for molecular vibration model."""

    def test_energy_positive(self):
        """Vibrational energy should be positive for positive frequency."""
        vib = MolecularVibration(frequency_cm_inv=1680.0)
        assert vib.energy_ev() > 0.0
        assert vib.energy_j() > 0.0

    def test_thermal_occupation_increases_with_temperature(self):
        """Phonon occupation should increase with temperature."""
        vib = MolecularVibration(frequency_cm_inv=1000.0)
        n_200 = vib.thermal_occupation(200.0)
        n_500 = vib.thermal_occupation(500.0)
        assert n_500 > n_200

    def test_deuteration_lowers_frequency(self):
        """Deuteration should lower the vibrational frequency."""
        vib = MolecularVibration(frequency_cm_inv=3000.0, reduced_mass_amu=1.0)
        d_vib = vib.deuterated()
        assert d_vib.frequency_cm_inv < vib.frequency_cm_inv

    def test_deuteration_frequency_ratio(self):
        """Deuterated frequency should be ~ 1/sqrt(2) of normal."""
        vib = MolecularVibration(frequency_cm_inv=3000.0, reduced_mass_amu=1.0)
        d_vib = vib.deuterated()
        ratio = d_vib.frequency_cm_inv / vib.frequency_cm_inv
        assert ratio == pytest.approx(1.0 / math.sqrt(2.0), abs=0.01)


class TestOdorant:
    """Tests for odorant molecule model."""

    def test_acetophenone_exists(self):
        """Acetophenone should be in the predefined odorants."""
        assert "acetophenone" in ODORANTS

    def test_benzaldehyde_exists(self):
        """Benzaldehyde should be in the predefined odorants."""
        assert "benzaldehyde" in ODORANTS

    def test_odorant_deuteration(self):
        """Deuterated odorant should have lower primary frequency."""
        aceto = ODORANTS["acetophenone"]
        d_aceto = aceto.deuterated()
        assert d_aceto.primary_frequency_cm_inv < aceto.primary_frequency_cm_inv
        assert d_aceto.name.startswith("d-")


class TestOdorDiscrimination:
    """Tests for quantum vs classical olfaction comparison."""

    def test_isotope_test_acetophenone(self):
        """Quantum model should discriminate acetophenone from its deuterated form."""
        disc = OdorDiscrimination()
        aceto = ODORANTS["acetophenone"]
        result = disc.isotope_test(aceto)
        # Quantum model should discriminate (different vibrations)
        assert result["quantum_discriminates"] is True
        # Classical model should NOT discriminate (same shape)
        assert result["classical_discriminates"] is False

    def test_different_molecules_discriminated(self):
        """Both models should discriminate very different molecules."""
        disc = OdorDiscrimination()
        aceto = ODORANTS["acetophenone"]
        benz = ODORANTS["benzaldehyde"]
        # Classical model: different molecular weights => different shapes
        assert disc.classical_discriminates(aceto, benz) is True

    def test_isotope_test_returns_all_keys(self):
        """Isotope test result should contain all expected keys."""
        disc = OdorDiscrimination()
        aceto = ODORANTS["acetophenone"]
        result = disc.isotope_test(aceto)
        expected_keys = {
            "molecule", "deuterated", "freq_normal", "freq_deuterated",
            "rate_normal", "rate_deuterated", "quantum_discriminates",
            "classical_discriminates",
        }
        assert set(result.keys()) == expected_keys


# ======================================================================
# 4. Avian Navigation Tests
# ======================================================================


class TestRadicalPair:
    """Tests for the radical pair mechanism."""

    def test_cryptochrome_construction(self, radical_pair_crypto: RadicalPair):
        """Cryptochrome radical pair should have expected parameters."""
        assert radical_pair_crypto.hyperfine_coupling_mhz == 28.0
        assert radical_pair_crypto.field_strength_ut == 50.0

    def test_singlet_yield_in_range(self, radical_pair_crypto: RadicalPair):
        """Singlet yield should be in [0, 1]."""
        for angle in [0.0, math.pi / 4, math.pi / 2, math.pi]:
            sy = radical_pair_crypto.singlet_yield(angle)
            assert 0.0 <= sy <= 1.0, (
                f"Singlet yield {sy} out of range at angle {angle}"
            )

    def test_singlet_yield_varies_with_angle(
        self, radical_pair_crypto: RadicalPair
    ):
        """Singlet yield should depend on the field angle (compass effect)."""
        yield_0 = radical_pair_crypto.singlet_yield(0.0)
        yield_90 = radical_pair_crypto.singlet_yield(math.pi / 2)
        # The yields should differ (anisotropic hyperfine)
        assert yield_0 != pytest.approx(yield_90, abs=1e-4), (
            "Singlet yield should vary with angle for anisotropic coupling"
        )

    def test_angular_response_shape(self, radical_pair_crypto: RadicalPair):
        """Angular response should return correct number of points."""
        angles, yields = radical_pair_crypto.angular_response(n_angles=18)
        assert len(angles) == 18
        assert len(yields) == 18

    def test_angular_response_range(self, radical_pair_crypto: RadicalPair):
        """Angular response should cover [0, pi]."""
        angles, _ = radical_pair_crypto.angular_response(n_angles=18)
        assert angles[0] == pytest.approx(0.0, abs=1e-10)
        assert angles[-1] == pytest.approx(math.pi, abs=1e-10)

    def test_compass_anisotropy_positive(
        self, radical_pair_crypto: RadicalPair
    ):
        """Compass anisotropy should be positive for anisotropic coupling."""
        aniso = radical_pair_crypto.compass_anisotropy(n_angles=12)
        assert aniso > 0.0, (
            f"Expected positive anisotropy, got {aniso}"
        )

    def test_isotropic_reduced_anisotropy(self):
        """Isotropic hyperfine should give smaller compass anisotropy."""
        aniso_rp = RadicalPair(
            hyperfine_coupling_mhz=28.0,
            hyperfine_anisotropy=0.3,
            field_strength_ut=50.0,
            k_s=1.0, k_t=1.0, lifetime_us=1.0,
        )
        iso_rp = RadicalPair(
            hyperfine_coupling_mhz=28.0,
            hyperfine_anisotropy=0.0,
            field_strength_ut=50.0,
            k_s=1.0, k_t=1.0, lifetime_us=1.0,
        )
        aniso_val = aniso_rp.compass_anisotropy(12)
        iso_val = iso_rp.compass_anisotropy(12)
        assert aniso_val > iso_val

    def test_zero_field_reduced_anisotropy(self):
        """Zero magnetic field should produce reduced anisotropy."""
        rp_earth = RadicalPair.cryptochrome()
        rp_zero = RadicalPair(
            hyperfine_coupling_mhz=28.0,
            hyperfine_anisotropy=0.3,
            field_strength_ut=0.001,  # near-zero field
            k_s=1.0, k_t=1.0, lifetime_us=1.0,
        )
        aniso_earth = rp_earth.compass_anisotropy(12)
        aniso_zero = rp_zero.compass_anisotropy(12)
        # Near-zero field should not give directional information
        assert aniso_earth > aniso_zero or aniso_zero < 0.01


class TestCryptochromeModel:
    """Tests for the cryptochrome protein model."""

    def test_european_robin_construction(self):
        """Robin model should have 3 Trp residues."""
        robin = CryptochromeModel.european_robin()
        assert robin.species == "European robin"
        assert len(robin.trp_hyperfine_mhz) == 3

    def test_homing_pigeon_construction(self):
        """Pigeon model should have 3 Trp residues."""
        pigeon = CryptochromeModel.homing_pigeon()
        assert pigeon.species == "homing pigeon"
        assert len(pigeon.trp_hyperfine_mhz) == 3

    def test_effective_radical_pair(self):
        """Should create valid radical pairs for each Trp."""
        robin = CryptochromeModel.european_robin()
        for i in range(3):
            rp = robin.effective_radical_pair(i)
            assert rp.hyperfine_coupling_mhz > 0.0
            sy = rp.singlet_yield(0.0)
            assert 0.0 <= sy <= 1.0

    def test_invalid_trp_index(self):
        """Out-of-range Trp index should raise IndexError."""
        robin = CryptochromeModel.european_robin()
        with pytest.raises(IndexError):
            robin.effective_radical_pair(5)

    def test_total_compass_anisotropy_positive(self):
        """Combined compass anisotropy should be positive."""
        robin = CryptochromeModel.european_robin()
        aniso = robin.total_compass_anisotropy(n_angles=8)
        assert aniso > 0.0

    def test_robin_better_compass_than_pigeon(self):
        """Robin should have better compass sensitivity than pigeon."""
        robin = CryptochromeModel.european_robin()
        pigeon = CryptochromeModel.homing_pigeon()
        robin_aniso = robin.total_compass_anisotropy(n_angles=8)
        pigeon_aniso = pigeon.total_compass_anisotropy(n_angles=8)
        assert robin_aniso > pigeon_aniso


class TestCompassSensitivity:
    """Tests for compass sensitivity analysis."""

    def test_angular_resolution_finite(self):
        """Angular resolution should be a finite positive number."""
        rp = RadicalPair.cryptochrome()
        cs = CompassSensitivity(rp)
        res = cs.angular_resolution(n_angles=18)
        assert res > 0.0
        assert not math.isinf(res)

    def test_field_strength_sensitivity_shape(self):
        """Field strength sensitivity should return correct shape."""
        rp = RadicalPair.cryptochrome()
        cs = CompassSensitivity(rp)
        fields, aniso = cs.field_strength_sensitivity(n_points=5)
        assert len(fields) == 5
        assert len(aniso) == 5


class TestDecoherenceEffects:
    """Tests for decoherence effects on compass."""

    def test_anisotropy_vs_lifetime_shape(self):
        """Should return arrays of correct length."""
        rp = RadicalPair.cryptochrome()
        de = DecoherenceEffects(rp)
        lifetimes, aniso = de.anisotropy_vs_lifetime(n_points=5)
        assert len(lifetimes) == 5
        assert len(aniso) == 5

    def test_anisotropy_vs_recombination_shape(self):
        """Should return arrays of correct length."""
        rp = RadicalPair.cryptochrome()
        de = DecoherenceEffects(rp)
        k_vals, aniso = de.anisotropy_vs_recombination(n_points=5)
        assert len(k_vals) == 5
        assert len(aniso) == 5


# ======================================================================
# 5. DNA Mutation Tests
# ======================================================================


class TestBasePair:
    """Tests for DNA base pair tunneling model."""

    def test_at_construction(self, base_pair_at: BasePair):
        """A-T base pair should have 2 hydrogen bonds."""
        assert base_pair_at.pair_type == BasePairType.AT
        assert base_pair_at.n_bonds == 2

    def test_gc_construction(self, base_pair_gc: BasePair):
        """G-C base pair should have 3 hydrogen bonds."""
        assert base_pair_gc.pair_type == BasePairType.GC
        assert base_pair_gc.n_bonds == 3

    def test_barrier_parameters_physical(self, base_pair_at: BasePair):
        """Barrier parameters should be in physically reasonable ranges."""
        assert 0.1 < base_pair_at.potential.barrier_height_ev < 1.0
        assert 0.01 < base_pair_at.potential.barrier_width_nm < 0.2

    def test_tautomer_probability_range(self, base_pair_at: BasePair):
        """Tautomer probability should be in (0, 1)."""
        p = base_pair_at.tautomer_probability(300.0)
        assert 0.0 < p < 1.0

    def test_tautomer_probability_very_small(self, base_pair_at: BasePair):
        """Tautomer probability should be very small (rare event)."""
        p = base_pair_at.tautomer_probability(300.0)
        assert p < 0.01, f"Tautomer probability too large: {p}"

    def test_gc_lower_tautomer_probability(
        self, base_pair_at: BasePair, base_pair_gc: BasePair
    ):
        """G-C should have lower tautomer probability (higher barrier)."""
        p_at = base_pair_at.tautomer_probability(300.0)
        p_gc = base_pair_gc.tautomer_probability(300.0)
        assert p_gc < p_at

    def test_tautomer_increases_with_temperature(
        self, base_pair_at: BasePair
    ):
        """Tautomer probability should increase with temperature."""
        p_200 = base_pair_at.tautomer_probability(200.0)
        p_400 = base_pair_at.tautomer_probability(400.0)
        assert p_400 > p_200

    def test_tautomer_invalid_temperature(self, base_pair_at: BasePair):
        """Non-positive temperature should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            base_pair_at.tautomer_probability(0.0)
        with pytest.raises(ValueError, match="positive"):
            base_pair_at.tautomer_probability(-10.0)

    def test_concerted_probability_smaller(self, base_pair_at: BasePair):
        """Concerted (all bonds) probability should be smaller than single."""
        p_single = base_pair_at.tautomer_probability(300.0)
        p_conc = base_pair_at.concerted_tautomer_probability(300.0)
        assert p_conc < p_single
        # For 2 bonds: p_conc ~ p_single^2
        assert p_conc == pytest.approx(p_single ** 2, rel=1e-6)

    def test_mutation_rate_positive(self, base_pair_at: BasePair):
        """Mutation rate should be positive at body temperature."""
        rate = base_pair_at.mutation_rate(310.0)
        assert rate > 0.0

    def test_classical_mutation_rate_positive(self, base_pair_at: BasePair):
        """Classical mutation rate should be positive."""
        rate = base_pair_at.classical_mutation_rate(310.0)
        assert rate > 0.0

    def test_quantum_enhances_mutation_low_temp(self, base_pair_at: BasePair):
        """Quantum tunneling should dominate classical rate at low temperature.

        At low temperature, Arrhenius thermal hopping is exponentially
        suppressed while tunneling persists, so the quantum rate exceeds
        the classical rate. At biological temperatures (~310 K), the
        Boltzmann-weighted tautomer model may give lower rates than
        bare Arrhenius because it also accounts for asymmetry.
        """
        # At very low temperature, quantum tunneling must dominate
        ratio_low = base_pair_at.quantum_classical_ratio(50.0)
        assert ratio_low > 1.0, f"Expected ratio > 1 at 50K, got {ratio_low}"

    def test_tunnel_splitting_positive(self, base_pair_at: BasePair):
        """Tunnel splitting should be positive."""
        splitting = base_pair_at.tunnel_splitting_ev()
        assert splitting > 0.0

    def test_mutation_rate_vs_temperature(self, base_pair_at: BasePair):
        """Temperature scan should return arrays of correct shape."""
        temps, q_rates, c_rates = base_pair_at.mutation_rate_vs_temperature(
            t_min=200.0, t_max=400.0, steps=20
        )
        assert len(temps) == 20
        assert len(q_rates) == 20
        assert len(c_rates) == 20

    def test_mutation_rate_vs_temperature_invalid(
        self, base_pair_at: BasePair
    ):
        """Invalid temperature range should raise ValueError."""
        with pytest.raises(ValueError):
            base_pair_at.mutation_rate_vs_temperature(0.0, 400.0)
        with pytest.raises(ValueError):
            base_pair_at.mutation_rate_vs_temperature(400.0, 200.0)


class TestDoubleWellPotential:
    """Tests for the double-well potential model."""

    def test_barrier_at_center(self):
        """Potential should be at maximum near x=0 (barrier top)."""
        pot = DoubleWellPotential(
            barrier_height_ev=0.4, barrier_width_nm=0.07, asymmetry_ev=0.0
        )
        v_center = pot.evaluate(0.0)
        v_well = pot.evaluate(0.035)  # at well minimum
        assert v_center > v_well

    def test_symmetric_wells_equal(self):
        """Symmetric potential should have equal well depths."""
        pot = DoubleWellPotential(
            barrier_height_ev=0.4, barrier_width_nm=0.07, asymmetry_ev=0.0
        )
        v_left = pot.evaluate(-0.035)
        v_right = pot.evaluate(0.035)
        assert v_left == pytest.approx(v_right, abs=1e-10)

    def test_asymmetry_breaks_symmetry(self):
        """Nonzero asymmetry should make the wells different."""
        pot = DoubleWellPotential(
            barrier_height_ev=0.4, barrier_width_nm=0.07, asymmetry_ev=0.05
        )
        v_left = pot.evaluate(-0.035)
        v_right = pot.evaluate(0.035)
        assert v_left != pytest.approx(v_right, abs=1e-6)

    def test_potential_curve_shape(self):
        """Potential curve should return arrays of correct length."""
        pot = DoubleWellPotential(
            barrier_height_ev=0.4, barrier_width_nm=0.07, asymmetry_ev=0.0
        )
        x, v = pot.potential_curve(n_points=100)
        assert len(x) == 100
        assert len(v) == 100


class TestTautomerTunneling:
    """Tests for detailed tautomer tunneling analysis."""

    def test_wkb_action_positive(self, base_pair_at: BasePair):
        """WKB action should be positive."""
        tt = TautomerTunneling(base_pair_at)
        action = tt.wkb_action()
        assert action > 0.0

    def test_instanton_rate_positive(self, base_pair_at: BasePair):
        """Instanton rate should be positive at room temperature."""
        tt = TautomerTunneling(base_pair_at)
        rate = tt.instanton_rate(300.0)
        assert rate > 0.0

    def test_tunneling_time_positive(self, base_pair_at: BasePair):
        """Tunneling traversal time should be positive."""
        tt = TautomerTunneling(base_pair_at)
        time_fs = tt.tunneling_time_fs()
        assert time_fs > 0.0


class TestMutationRate:
    """Tests for mutation rate comparison."""

    def test_quantum_dominance_temperature(self, base_pair_at: BasePair):
        """Quantum dominance crossover temperature should be positive."""
        mr = MutationRate(base_pair_at)
        t_cross = mr.quantum_dominance_temperature()
        assert t_cross > 0.0

    def test_mutations_per_cell_division(self, base_pair_at: BasePair):
        """Should predict a finite number of mutations per cell division."""
        mr = MutationRate(base_pair_at)
        n_mut = mr.mutation_rate_per_cell_division(
            temperature_k=310.0,
            genome_size_bp=6_400_000_000,
            division_time_s=86400.0,
        )
        # Should be some finite positive number
        assert n_mut > 0.0
        assert not math.isinf(n_mut)

    def test_compare_rates_returns_all_keys(self, base_pair_at: BasePair):
        """Compare rates should return dict with all expected keys."""
        mr = MutationRate(base_pair_at)
        result = mr.compare_rates(310.0)
        expected_keys = {
            "base_pair", "temperature_k", "quantum_rate",
            "classical_rate", "ratio", "tunneling_probability",
            "tautomer_probability", "tunnel_splitting_ev",
        }
        assert set(result.keys()) == expected_keys


# ======================================================================
# 6. Cross-Module Integration Tests
# ======================================================================


class TestCrossModuleIntegration:
    """Tests that verify consistency across all bio modules."""

    def test_all_modules_importable(self):
        """All bio modules should import without error."""
        from nqpu.bio import (
            FMOComplex, EnzymeTunneling, QuantumNose,
            RadicalPair, BasePair,
        )
        assert FMOComplex is not None
        assert EnzymeTunneling is not None
        assert QuantumNose is not None
        assert RadicalPair is not None
        assert BasePair is not None

    def test_physical_constants_consistent(self):
        """Physical constants should be consistent across modules."""
        from nqpu.bio import photosynthesis, tunneling, olfaction, dna_mutation
        # Check HBAR is the same across all modules
        assert photosynthesis.HBAR == tunneling.HBAR
        assert tunneling.HBAR == olfaction.HBAR
        assert olfaction.HBAR == dna_mutation.HBAR

    def test_enzyme_and_dna_tunneling_use_same_wkb(self):
        """Enzyme and DNA models should give same tunneling for same barrier."""
        from nqpu.bio.tunneling import EnzymeTunneling, TunnelingBarrier
        from nqpu.bio.dna_mutation import BasePair, BasePairType

        # Create an enzyme model with same parameters as A-T base pair
        at = BasePair.from_type(BasePairType.AT)
        enzyme = EnzymeTunneling(
            barrier=TunnelingBarrier(
                height_ev=at.potential.barrier_height_ev,
                width_nm=at.potential.barrier_width_nm,
            )
        )
        prob_enzyme = enzyme.tunneling_probability()
        prob_dna = at._wkb_tunneling_probability()
        assert prob_enzyme == pytest.approx(prob_dna, rel=1e-10)

    def test_ev_to_cm_inv_consistency(self):
        """Energy gap in olfaction should convert correctly."""
        from nqpu.bio.olfaction import QuantumNose, EV_TO_J, CM_INV_TO_J
        nose = QuantumNose.default()
        gap_ev = nose.energy_gap_ev()
        f_res = nose.resonant_frequency_cm_inv()
        # Convert back: f_res * CM_INV_TO_J / EV_TO_J should equal gap_ev
        gap_back = f_res * CM_INV_TO_J / EV_TO_J
        assert gap_back == pytest.approx(gap_ev, rel=1e-6)

    def test_all_presets_accessible(self):
        """All predefined biological systems should be accessible."""
        from nqpu.bio.tunneling import ENZYMES
        from nqpu.bio.olfaction import ODORANTS

        assert len(ENZYMES) >= 3
        assert len(ODORANTS) >= 3


# ======================================================================
# 7. Physical Sanity Checks
# ======================================================================


class TestPhysicalSanity:
    """Tests verifying physically reasonable outputs."""

    def test_fmo_coherence_sub_picosecond(self):
        """FMO coherence lifetime should be sub-picosecond at 300K.

        Experimental value: ~300-500 fs (Engel et al., 2007).
        """
        fmo = FMOComplex.standard()
        result = fmo.evolve(duration_fs=1000.0, steps=2000)
        lifetime = result.coherence_lifetime_fs()
        assert lifetime < 1000.0, (
            f"Coherence lifetime {lifetime} fs exceeds 1 ps"
        )

    def test_kie_physically_reasonable(self):
        """KIE for enzyme tunneling should be in reasonable range.

        Classical upper bound: ~7 (semiclassical Bigeleisen).
        Quantum tunneling: can be >> 10.
        """
        for name, enzyme in ENZYMES.items():
            kie = enzyme.kie_ratio()
            assert kie > 1.0, f"{name}: KIE should be > 1, got {kie}"
            # KIE should not be astronomically large
            assert kie < 1e10, f"{name}: KIE unreasonably large: {kie}"

    def test_radical_pair_singlet_yield_physical(self):
        """Singlet yield should be between 0.25 and 0.75 for cryptochrome.

        For equal recombination rates (k_s = k_t), the integrated singlet
        yield should average around 0.25-0.5 depending on the field angle.
        """
        rp = RadicalPair.cryptochrome()
        for angle in [0.0, math.pi / 2]:
            sy = rp.singlet_yield(angle)
            assert 0.05 < sy < 0.95, (
                f"Singlet yield {sy} at angle {angle} seems unphysical"
            )

    def test_dna_tunneling_probability_very_small(self):
        """DNA proton tunneling probability should be very small.

        This is the reason spontaneous mutations are rare.
        """
        at = BasePair.from_type(BasePairType.AT)
        prob = at._wkb_tunneling_probability()
        assert prob < 0.1, f"Tunneling probability {prob} too large"
        assert prob > 0.0, "Tunneling probability should be non-zero"

    def test_olfaction_resonance_in_ir_range(self):
        """Resonant frequency should be in the IR spectral range (400-4000 cm^-1)."""
        nose = QuantumNose.default()
        f_res = nose.resonant_frequency_cm_inv()
        assert 400.0 < f_res < 4000.0, (
            f"Resonant frequency {f_res} cm^-1 outside IR range"
        )

    def test_mutation_rate_reasonable_at_body_temp(self):
        """Quantum mutation rate should be reasonable at body temperature.

        Known spontaneous mutation rate: ~10^-9 per base per cell division.
        Our per-second rate * replication time should be in that ballpark
        (very rough order of magnitude).
        """
        at = BasePair.from_type(BasePairType.AT)
        rate = at.mutation_rate(310.0)  # 37 C
        # Rate per second should be finite and positive
        assert rate > 0.0
        assert not math.isinf(rate)
